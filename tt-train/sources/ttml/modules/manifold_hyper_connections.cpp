// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manifold_hyper_connections.hpp"

#include <fmt/format.h>

#include <core/ttnn_all_includes.hpp>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/matmul_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

namespace {

// Project a raw [1, 1, n, n] score matrix onto the manifold of doubly-stochastic
// matrices (the Birkhoff polytope) via Sinkhorn-Knopp:
//   M^(0) = exp(raw); then alternate row- and column-normalization for t_max
//   iterations. The result B satisfies B 1 = 1 and 1^T B = 1^T, so ||B||_2 <= 1
//   and the residual mixing is non-expansive (the core mHC stability property).
// Fully differentiable: every step is an autograd op, so gradients reach raw.
autograd::TensorPtr sinkhorn_knopp(const autograd::TensorPtr& raw, uint32_t iters) {
    const auto full_shape = raw->get_value().logical_shape();
    auto m = ops::exp(raw);  // ensure positivity
    for (uint32_t i = 0; i < iters; ++i) {
        // Row normalization: divide each row by its sum over the last dim.
        auto row_sums = ops::sum(m, /* dim */ -1, /* keep_dim */ true);
        m = ops::div(m, ops::broadcast_to(row_sums, full_shape));
        // Column normalization: divide each column by its sum over the -2 dim.
        auto col_sums = ops::sum(m, /* dim */ -2, /* keep_dim */ true);
        m = ops::div(m, ops::broadcast_to(col_sums, full_shape));
    }
    return m;
}

}  // namespace

ManifoldHyperConnections::ManifoldHyperConnections(
    const ManifoldHyperConnectionsConfig& config, autograd::ModuleBasePtr inner_layer) :
    m_config(config), m_inner_layer(std::move(inner_layer)) {
    if (m_config.hidden_dim == 0U) {
        throw std::invalid_argument("ManifoldHyperConnectionsConfig::hidden_dim must be set to the layer hidden size.");
    }
    if (m_config.dynamic_parameterization) {
        throw std::runtime_error(
            "ManifoldHyperConnections: dynamic (per-token) parameterization is not implemented yet; it requires a "
            "permute op to form vec(X_l) and per-token Sinkhorn. Use dynamic_parameterization=false (static).");
    }

    const uint32_t n = m_config.num_streams;
    auto* device = &autograd::ctx().get_device();

    // Raw parameters, initialized to zero. At init this gives A = 0.5, C = 1, and
    // B = a uniform doubly-stochastic matrix (every entry 1/n), i.e. a gentle,
    // well-conditioned starting point close to an averaging residual.
    m_raw_a = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, n}), device));
    m_raw_b = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, n, n}), device));
    m_raw_c = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, n, 1}), device));

    create_name("manifold_hyper_connections");
    register_module(m_inner_layer, "inner_layer");
    register_tensor(m_raw_a, "raw_a");
    register_tensor(m_raw_b, "raw_b");
    register_tensor(m_raw_c, "raw_c");
}

autograd::TensorPtr ManifoldHyperConnections::operator()(const autograd::TensorPtr& expanded_residual) {
    const auto x_shape = expanded_residual->get_value().logical_shape().to_array_4D();
    const uint32_t batch = x_shape[0];
    const uint32_t streams = x_shape[1];
    const uint32_t seq = x_shape[2];
    const uint32_t dim = x_shape[3];
    const uint32_t n = m_config.num_streams;

    if (streams != n || dim != m_config.hidden_dim) {
        throw std::invalid_argument(fmt::format(
            "mHC expected residual shape [B, {}, S, {}], got [{}, {}, {}, {}].",
            n,
            m_config.hidden_dim,
            batch,
            streams,
            seq,
            dim));
    }

    // Manifold constraints (differentiable): A in (0,1), C in (0,2), B doubly-stochastic.
    auto a = ops::sigmoid(m_raw_a);                             // [1, 1, 1, n]
    auto c = ops::mul(ops::sigmoid(m_raw_c), 2.0F);             // [1, 1, n, 1]
    auto b = sinkhorn_knopp(m_raw_b, m_config.sinkhorn_iters);  // [1, 1, n, n]

    // Broadcast the per-stream mixing matrices over the batch so matmul batch
    // dims line up and gradients reduce back correctly (broadcast_batch sums the
    // gradient over the batch dimension in its backward pass).
    auto a_b = ops::broadcast_batch(a, batch);  // [B, 1, 1, n]
    auto c_b = ops::broadcast_batch(c, batch);  // [B, 1, n, 1]
    auto b_b = ops::broadcast_batch(b, batch);  // [B, 1, n, n]

    // Collapse (S, d) into a single free dimension so the n_hc streams sit in the
    // second-to-last position and can be contracted by matmul. [B, n, S, d] is
    // row-major contiguous as [B, 1, n, S*d].
    const uint32_t sd = seq * dim;
    auto x_r = ops::reshape(expanded_residual, ttnn::Shape({batch, 1, n, sd}));  // [B, 1, n, S*d]

    // A X: combine the n streams into one. [B,1,1,n] @ [B,1,n,S*d] -> [B,1,1,S*d].
    auto ax = ops::matmul_op(a_b, x_r);
    auto ax_r = ops::reshape(ax, ttnn::Shape({batch, 1, seq, dim}));  // [B, 1, S, d]

    // F(A X): the wrapped layer operates on the single combined stream.
    auto f_out = (*m_inner_layer)(ax_r);                             // [B, 1, S, d]
    auto f_r = ops::reshape(f_out, ttnn::Shape({batch, 1, 1, sd}));  // [B, 1, 1, S*d]

    // C F(A X): expand back to n streams. [B,1,n,1] @ [B,1,1,S*d] -> [B,1,n,S*d].
    auto cf = ops::matmul_op(c_b, f_r);
    auto cf_r = ops::reshape(cf, ttnn::Shape({batch, n, seq, dim}));  // [B, n, S, d]

    // B X: mix the n streams. [B,1,n,n] @ [B,1,n,S*d] -> [B,1,n,S*d].
    auto bx = ops::matmul_op(b_b, x_r);
    auto bx_r = ops::reshape(bx, ttnn::Shape({batch, n, seq, dim}));  // [B, n, S, d]

    // X_{l+1} = B X + C F(A X).
    return ops::add(bx_r, cf_r);
}

}  // namespace ttml::modules
