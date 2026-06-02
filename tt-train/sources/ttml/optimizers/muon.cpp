// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "muon.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "serialization/serializable.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::optimizers {

namespace {

// Newton-Schulz coefficients (a, b, c) for the polynomial iteration
//   X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k
// "Aggressive" coefficients drive the singular values rapidly towards 1, the
// "stabilize" coefficients pin them precisely at 1 in the final steps.
constexpr std::array<float, 3> kAggressiveCoeffs = {3.4445F, -4.7750F, 2.0315F};
constexpr std::array<float, 3> kStabilizeCoeffs = {2.0F, -1.5F, 0.5F};

// A parameter is orthogonalizable by Muon iff it is a genuine 2D matrix, i.e. a
// rank-4 tensor [1, 1, n, m] with n > 1 and m > 1.
bool is_matrix_parameter(const tt::tt_metal::Tensor& tensor) {
    const auto shape = tensor.logical_shape();
    if (shape.rank() != 4) {
        return false;
    }
    const auto [b, n_outer, n, m] = shape.to_array_4D();
    return b == 1U && n_outer == 1U && n > 1U && m > 1U;
}

// Approximately orthogonalize a [1, 1, n, m] matrix to U V^T (from its SVD
// U S V^T) via the hybrid Newton-Schulz iteration. The input is first
// normalized by its Frobenius norm so that the largest singular value is <= 1,
// which is the convergence precondition for the iteration.
tt::tt_metal::Tensor newton_schulz(tt::tt_metal::Tensor matrix, uint32_t ns_steps, uint32_t ns_stabilize_steps) {
    // Frobenius norm: sqrt(sum(X^2)) over all elements (leading dims are 1).
    auto squared = ttnn::square(matrix);
    auto frob_sq = ttnn::sum(
        squared,
        /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2, 3},
        /* keep_dim */ true,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
    auto frob_norm = ttnn::add(ttnn::sqrt(frob_sq), 1e-7F);
    auto x = ttnn::divide(matrix, frob_norm);  // broadcast [1,1,n,m] / [1,1,1,1]

    // The iteration converges fastest when rows <= cols; transpose tall matrices
    // and undo it at the end.
    const auto shape = matrix.logical_shape().to_array_4D();
    const uint32_t n = shape[2];
    const uint32_t m = shape[3];
    const bool transposed = n > m;
    if (transposed) {
        x = ttnn::transpose(x, -2, -1);
    }

    const uint32_t stabilize_from = (ns_steps > ns_stabilize_steps) ? (ns_steps - ns_stabilize_steps) : 0U;
    for (uint32_t step = 0; step < ns_steps; ++step) {
        const auto& coeffs = (step < stabilize_from) ? kAggressiveCoeffs : kStabilizeCoeffs;
        const auto [a, bb, cc] = coeffs;

        auto gram = ttml::ttnn_fixed::matmul(x, x, /* transpose_a */ false, /* transpose_b */ true);  // X X^T
        auto gram_x = ttml::ttnn_fixed::matmul(gram, x);                                              // (X X^T) X
        auto gram2_x = ttml::ttnn_fixed::matmul(gram, gram_x);                                        // (X X^T)^2 X

        auto term = ttnn::add(ttnn::multiply(x, a), ttnn::multiply(gram_x, bb));
        x = ttnn::add(term, ttnn::multiply(gram2_x, cc));
    }

    if (transposed) {
        x = ttnn::transpose(x, -2, -1);
    }
    return x;
}

}  // namespace

Muon::Muon(serialization::NamedParameters parameters, const MuonConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_momentum.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
        }
    }
}

void Muon::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void Muon::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    m_steps++;
    for (auto& [name, momentum_ptr] : m_momentum) {
        const auto& tensor_ptr = m_parameters.at(name);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }

        auto gradients = tensor_ptr->get_grad();

        // M_t = mu * M_{t-1} + G_t
        auto momentum = ttnn::add(
            ttnn::multiply(momentum_ptr->get_value(autograd::PreferredPrecision::FULL), m_config.momentum), gradients);
        momentum_ptr->set_value(momentum);

        // Nesterov look-ahead direction: mu * M_t + G_t (else just M_t).
        auto update = m_config.nesterov ? ttnn::add(ttnn::multiply(momentum, m_config.momentum), gradients) : momentum;

        const auto value = tensor_ptr->get_value(autograd::PreferredPrecision::FULL);
        if (is_matrix_parameter(value)) {
            const auto shape = value.logical_shape().to_array_4D();
            const uint32_t n = shape[2];
            const uint32_t m = shape[3];
            update = newton_schulz(update, m_config.ns_steps, m_config.ns_stabilize_steps);
            // Rescale the update RMS so AdamW-tuned learning rates transfer.
            const float scale = std::sqrt(static_cast<float>(std::max(n, m))) * m_config.update_rms_scale;
            update = ttnn::multiply(update, scale);
        } else if (!m_config.fallback_on_non_matrix) {
            throw std::runtime_error(fmt::format(
                "Muon received non-matrix parameter '{}' with shape {}. Route 1D/scalar parameters (norm gains, "
                "biases, gating factors) to AdamW, or set fallback_on_non_matrix=true.",
                name,
                value.logical_shape()));
        }
        // else: plain momentum update for non-matrix parameters (no orthogonalization).

        // Decoupled weight decay then update: W = W * (1 - lr * wd) - lr * O.
        auto new_value = value;
        if (m_config.weight_decay != 0.0F) {
            new_value = ttnn::multiply(new_value, 1.0F - m_config.lr * m_config.weight_decay);
        }
        new_value = ttnn::subtract(new_value, ttnn::multiply(update, m_config.lr));
        tensor_ptr->set_value(new_value);
    }
}

serialization::StateDict Muon::get_state_dict() const {
    serialization::StateDict dict;
    dict["momentum"] = m_momentum;
    dict["steps"] = m_steps;
    return dict;
}

void Muon::set_state_dict(const serialization::StateDict& dict) {
    m_momentum = std::get<serialization::NamedParameters>(dict.at("momentum"));
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
}

size_t Muon::get_steps() const {
    return m_steps;
}

void Muon::set_steps(size_t steps) {
    m_steps = steps;
}

}  // namespace ttml::optimizers
