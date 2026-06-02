// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "muon_v4.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/newton_schulz_op.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

namespace {

// A parameter is orthogonalizable by Muon iff it is a genuine 2D matrix: a
// rank-4 tensor [1, 1, n, m] with n > 1 and m > 1.
bool is_matrix_parameter(const tt::tt_metal::Tensor& tensor) {
    const auto shape = tensor.logical_shape();
    if (shape.rank() != 4) {
        return false;
    }
    return shape[0] == 1U && shape[1] == 1U && shape[2] > 1U && shape[3] > 1U;
}

// Muon's Newton-Schulz update is not elementwise, so it cannot be applied to a
// single mesh shard independently (it would differ from the full-weight result).
bool param_is_sharded(const autograd::Tensor& tensor) {
    const auto& placements = tensor.get_value(autograd::PreferredPrecision::HALF).tensor_topology().placements();
    for (const auto& p : placements) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(p)) {
            return true;
        }
    }
    return false;
}

}  // namespace

MuonV4::MuonV4(serialization::NamedParameters parameters, const MuonV4Config& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            if (param_is_sharded(*tensor_ptr)) {
                throw std::runtime_error(
                    "MuonV4: parameter '" + name +
                    "' appears to be sharded on a mesh axis. Muon's Newton-Schulz update is not elementwise and "
                    "cannot be applied to a shard independently.");
            }
            m_momentum_buffer.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                    /* requires_grad */ false));
        }
    }
}

void MuonV4::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void MuonV4::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    for (auto& [name, buffer_ptr] : m_momentum_buffer) {
        auto buffer = buffer_ptr->get_value(autograd::PreferredPrecision::HALF);
        const auto& tensor_ptr = m_parameters.at(name);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }

        const auto gradients = tensor_ptr->get_grad();

        // M_t = mu * M_{t-1} + G_t
        if (m_steps > 0 && m_config.momentum != 0.0F) {
            buffer = ttnn::add(ttnn::multiply(buffer, m_config.momentum), gradients);
        } else {
            buffer = gradients;
        }
        buffer_ptr->set_value(buffer);

        // Nesterov look-ahead direction: mu * M_t + G_t (else just M_t).
        auto update = m_config.nesterov ? ttnn::add(ttnn::multiply(buffer, m_config.momentum), gradients) : buffer;

        const auto value = tensor_ptr->get_value(autograd::PreferredPrecision::HALF);
        if (is_matrix_parameter(value)) {
            const auto shape = value.logical_shape();
            const uint32_t n = shape[-2];
            const uint32_t m = shape[-1];
            update = ops::newtonschulz_hybrid(update, m_config.ns_steps, m_config.ns_stabilize_steps, 1e-7F);
            // Rescale the update RMS so AdamW-tuned learning rates transfer.
            const float scale = std::sqrt(static_cast<float>(std::max(n, m))) * m_config.update_rms_scale;
            update = ttnn::multiply(update, scale);
        } else if (!m_config.fallback_on_non_matrix) {
            throw std::runtime_error(fmt::format(
                "MuonV4 received non-matrix parameter '{}' with shape {}. Route 1D/scalar parameters (norm gains, "
                "biases, gating factors) to AdamW, or set fallback_on_non_matrix=true.",
                name,
                value.logical_shape()));
        }
        // else: plain momentum update for non-matrix parameters (no orthogonalization).

        // Decoupled weight decay, then the update: W = W * (1 - lr * wd) - lr * O.
        auto new_value = value;
        if (m_config.weight_decay != 0.0F) {
            new_value = ttnn::multiply(new_value, 1.0F - m_config.lr * m_config.weight_decay);
        }
        new_value = ttnn::subtract(new_value, ttnn::multiply(update, m_config.lr));
        tensor_ptr->set_value(new_value);
    }
    m_steps++;
}

serialization::StateDict MuonV4::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    dict["lr"] = m_config.lr;
    dict["momentum"] = m_config.momentum;
    dict["weight_decay"] = m_config.weight_decay;
    dict["ns_steps"] = m_config.ns_steps;
    dict["ns_stabilize_steps"] = m_config.ns_stabilize_steps;
    dict["update_rms_scale"] = m_config.update_rms_scale;
    dict["momentum_buffer"] = m_momentum_buffer;
    return dict;
}

void MuonV4::set_state_dict(const serialization::StateDict& dict) {
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
    set_lr(serialization::get_value_type<float>(dict, "lr"));
    m_config.momentum = serialization::get_value_type<float>(dict, "momentum");
    m_config.weight_decay = serialization::get_value_type<float>(dict, "weight_decay");
    m_config.ns_steps = serialization::get_value_type<int>(dict, "ns_steps");
    m_config.ns_stabilize_steps = serialization::get_value_type<int>(dict, "ns_stabilize_steps");
    m_config.update_rms_scale = serialization::get_value_type<float>(dict, "update_rms_scale");
    m_momentum_buffer = std::get<serialization::NamedParameters>(dict.at("momentum_buffer"));
}

size_t MuonV4::get_steps() const {
    return m_steps;
}

void MuonV4::set_steps(size_t steps) {
    m_steps = steps;
}

}  // namespace ttml::optimizers
