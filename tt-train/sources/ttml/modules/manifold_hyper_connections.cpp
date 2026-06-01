// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manifold_hyper_connections.hpp"

#include <fmt/format.h>

#include <core/ttnn_all_includes.hpp>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"

namespace ttml::modules {

namespace detail {

// Bound a raw parameter to (0, 1) via the logistic sigmoid: used for A_l.
tt::tt_metal::Tensor sigmoid_gate(const tt::tt_metal::Tensor& raw) {
    return ttnn::sigmoid(raw);
}

// Bound a raw parameter to (0, 2): used for the output gate C_l.
tt::tt_metal::Tensor double_sigmoid_gate(const tt::tt_metal::Tensor& raw) {
    return ttnn::multiply(ttnn::sigmoid(raw), 2.0F);
}

// Project a raw [.., n, n] score matrix onto the manifold of doubly-stochastic
// matrices (the Birkhoff polytope) via Sinkhorn-Knopp:
//   M^(0) = exp(raw);  then alternate row- and column-normalization for t_max
//   iterations. The result B satisfies B 1 = 1 and 1^T B = 1^T, so ||B||_2 <= 1
//   and the residual mixing is non-expansive (the core mHC stability property).
//
// This forward computation is correct and reusable as-is. Making it
// differentiable end-to-end currently needs an autograd-level exp op (see the
// header's IMPLEMENTATION STATUS note).
tt::tt_metal::Tensor sinkhorn_knopp(const tt::tt_metal::Tensor& raw, uint32_t iters) {
    auto m = ttnn::exp(raw);  // ensure positivity
    for (uint32_t i = 0; i < iters; ++i) {
        // Row normalization: divide by the sum over the last dim.
        auto row_sums = ttnn::sum(m, /* dim_arg */ -1, /* keep_dim */ true);
        m = ttnn::divide(m, row_sums);
        // Column normalization: divide by the sum over the second-to-last dim.
        auto col_sums = ttnn::sum(m, /* dim_arg */ -2, /* keep_dim */ true);
        m = ttnn::divide(m, col_sums);
    }
    return m;
}

}  // namespace detail

ManifoldHyperConnections::ManifoldHyperConnections(
    const ManifoldHyperConnectionsConfig& config, autograd::ModuleBasePtr inner_layer) :
    m_config(config), m_inner_layer(std::move(inner_layer)) {
    if (m_config.hidden_dim == 0U) {
        throw std::invalid_argument("ManifoldHyperConnectionsConfig::hidden_dim must be set to the layer hidden size.");
    }

    const uint32_t n = m_config.num_streams;
    const uint32_t d = m_config.hidden_dim;
    const uint32_t flat = n * d;  // size of vec(X_l) per token

    auto* device = &autograd::ctx().get_device();

    // Dynamic generators: X_hat @ W -> raw A/B/C components (no bias; the static
    // S_* tensors below play the role of the bias).
    m_w_pre = std::make_shared<LinearLayer>(flat, n, /* has_bias */ false);
    m_w_res = std::make_shared<LinearLayer>(flat, n * n, /* has_bias */ false);
    m_w_post = std::make_shared<LinearLayer>(flat, n, /* has_bias */ false);

    // Static biases, initialized to zero.
    m_s_pre = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, n}), device));
    m_s_res = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, n, n}), device));
    m_s_post = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, n, 1}), device));

    // Learnable gating factors, initialized small so the layer starts close to a
    // plain residual connection.
    m_alpha_pre = autograd::create_tensor();
    m_alpha_res = autograd::create_tensor();
    m_alpha_post = autograd::create_tensor();
    init::constant_init(m_alpha_pre, ttnn::Shape({1, 1, 1, 1}), m_config.alpha_init);
    init::constant_init(m_alpha_res, ttnn::Shape({1, 1, 1, 1}), m_config.alpha_init);
    init::constant_init(m_alpha_post, ttnn::Shape({1, 1, 1, 1}), m_config.alpha_init);

    create_name("manifold_hyper_connections");
    register_module(m_inner_layer, "inner_layer");
    register_module(m_w_pre, "w_pre");
    register_module(m_w_res, "w_res");
    register_module(m_w_post, "w_post");
    register_tensor(m_s_pre, "s_pre");
    register_tensor(m_s_res, "s_res");
    register_tensor(m_s_post, "s_post");
    register_tensor(m_alpha_pre, "alpha_pre");
    register_tensor(m_alpha_res, "alpha_res");
    register_tensor(m_alpha_post, "alpha_post");
}

autograd::TensorPtr ManifoldHyperConnections::operator()(const autograd::TensorPtr& expanded_residual) {
    // The constraint math (detail::sigmoid_gate / double_sigmoid_gate /
    // sinkhorn_knopp) and all parameters are in place. Wiring the full forward
    // X_{l+1} = B X_l + C F(A X_l) through autograd still requires three small
    // ops that tt-train does not expose yet:
    //
    //   1. ops::sigmoid(TensorPtr) and ops::exp(TensorPtr) so the A/C gates and
    //      the Sinkhorn positivity step are differentiable. (Standard unary ops:
    //      d/dx sigmoid = s(1-s); d/dx exp = exp.)
    //   2. slice / concat (or a small matmul) along the N dimension to mix the
    //      n_hc residual streams with A (1 x n_hc), B (n_hc x n_hc) and C
    //      (n_hc x 1). n_hc is tiny (4), so an explicit per-stream weighted sum
    //      is sufficient.
    //
    // Once those land, this body becomes a direct transcription of the equations
    // in the header. Failing loudly here is intentional: returning a silently
    // partial result would corrupt training.
    (void)expanded_residual;
    throw std::runtime_error(
        "ManifoldHyperConnections::operator() is not yet wired end-to-end. The Sinkhorn/gate constraints and "
        "parameters are implemented (see manifold_hyper_connections.cpp); finishing requires autograd ops::sigmoid, "
        "ops::exp, and N-dimension stream mixing. See the header IMPLEMENTATION STATUS note.");
}

}  // namespace ttml::modules
