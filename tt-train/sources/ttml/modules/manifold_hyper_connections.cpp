// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "manifold_hyper_connections.hpp"

#include <fmt/format.h>

#include <core/ttnn_all_includes.hpp>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/binary_ops.hpp"
#include "ops/matmul_op.hpp"
#include "ops/permute_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

namespace {

// Swap the N (streams) and S (sequence) axes of a rank-4 tensor.
const ttnn::SmallVector<int64_t> kSwapStreamsSeq = {0, 2, 1, 3};

// Project a raw [.., .., n, n] score matrix onto the manifold of doubly-stochastic
// matrices (the Birkhoff polytope) via Sinkhorn-Knopp. Following the mHC paper
// (Xie et al., 2026, eq 9), M^(0) = exp(raw) and each iteration is
// M^(t) = T_r( T_c( M^(t-1) ) ): column-normalize first, then row-normalize, for
// t_max iterations. The result B satisfies B 1 = 1 and 1^T B = 1^T, so
// ||B||_2 <= 1 and the residual mixing is non-expansive (the core mHC stability
// property). Fully differentiable: every step is an autograd op, so gradients
// reach raw. Works for any leading dims (global [1,1,n,n] or per-token [M,1,n,n]).
autograd::TensorPtr sinkhorn_knopp(const autograd::TensorPtr& raw, uint32_t iters) {
    const auto full_shape = raw->get_value().logical_shape();
    auto m = ops::exp(raw);  // ensure positivity
    for (uint32_t i = 0; i < iters; ++i) {
        // T_c: column normalization (divide each column by its sum over dim -2).
        auto col_sums = ops::sum(m, /* dim */ -2, /* keep_dim */ true);
        m = ops::div(m, ops::broadcast_to(col_sums, full_shape));
        // T_r: row normalization (divide each row by its sum over the last dim).
        auto row_sums = ops::sum(m, /* dim */ -1, /* keep_dim */ true);
        m = ops::div(m, ops::broadcast_to(row_sums, full_shape));
    }
    return m;
}

}  // namespace

ManifoldHyperConnections::ManifoldHyperConnections(
    const ManifoldHyperConnectionsConfig& config, ModuleBasePtr inner_layer) :
    m_config(config), m_inner_layer(std::move(inner_layer)) {
    if (m_config.hidden_dim == 0U) {
        throw std::invalid_argument("ManifoldHyperConnectionsConfig::hidden_dim must be set to the layer hidden size.");
    }

    const uint32_t n = m_config.num_streams;
    const uint32_t nd = n * m_config.hidden_dim;
    auto* device = &autograd::ctx().get_device();

    create_name("manifold_hyper_connections");
    register_module(m_inner_layer, "inner_layer");

    if (m_config.layer_norm) {
        m_layer_norm = std::make_shared<RMSNormLayer>(m_config.hidden_dim);
        register_module(m_layer_norm, "layer_norm");
    }

    if (!m_config.dynamic_parameterization) {
        // Static: raw parameters initialized to zero. At init this gives A = 0.5,
        // C = 1, and B = a uniform doubly-stochastic matrix (every entry 1/n),
        // i.e. a gentle, well-conditioned starting point close to averaging.
        m_raw_a = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, n}), device));
        m_raw_b = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, n, n}), device));
        m_raw_c = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, n, 1}), device));
        register_tensor(m_raw_a, "raw_a");
        register_tensor(m_raw_b, "raw_b");
        register_tensor(m_raw_c, "raw_c");
        return;
    }

    // Dynamic: per-token generators from X_hat = RMSNorm(vec(X_l)) in R^{n*d}.
    m_vec_norm = std::make_shared<RMSNormLayer>(nd);
    m_w_pre = std::make_shared<LinearLayer>(nd, n, /* has_bias */ false);
    m_w_res = std::make_shared<LinearLayer>(nd, n * n, /* has_bias */ false);
    m_w_post = std::make_shared<LinearLayer>(nd, n, /* has_bias */ false);

    // Static biases (zero) and small gating factors so the layer starts close to
    // a plain averaging residual.
    m_s_pre = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, n}), device));
    m_s_res = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, n, n}), device));
    m_s_post = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, n}), device));
    m_alpha_pre = autograd::create_tensor();
    m_alpha_res = autograd::create_tensor();
    m_alpha_post = autograd::create_tensor();
    init::constant_init(m_alpha_pre, ttnn::Shape({1, 1, 1, 1}), m_config.alpha_init);
    init::constant_init(m_alpha_res, ttnn::Shape({1, 1, 1, 1}), m_config.alpha_init);
    init::constant_init(m_alpha_post, ttnn::Shape({1, 1, 1, 1}), m_config.alpha_init);

    register_module(m_vec_norm, "vec_norm");
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
    return forward(expanded_residual, std::nullopt);
}

autograd::TensorPtr ManifoldHyperConnections::operator()(
    const autograd::TensorPtr& expanded_residual, const std::optional<autograd::TensorPtr>& mask) {
    return forward(expanded_residual, mask);
}

autograd::TensorPtr ManifoldHyperConnections::forward(
    const autograd::TensorPtr& expanded_residual, const std::optional<autograd::TensorPtr>& mask) {
    const auto x_shape = expanded_residual->get_value().logical_shape().to_array_4D();
    const uint32_t streams = x_shape[1];
    const uint32_t dim = x_shape[3];
    if (streams != m_config.num_streams || dim != m_config.hidden_dim) {
        throw std::invalid_argument(fmt::format(
            "mHC expected residual shape [B, {}, S, {}], got [{}, {}, {}, {}].",
            m_config.num_streams,
            m_config.hidden_dim,
            x_shape[0],
            streams,
            x_shape[2],
            dim));
    }
    return m_config.dynamic_parameterization ? compute_dynamic(expanded_residual, mask)
                                             : compute_static(expanded_residual, mask);
}

autograd::TensorPtr ManifoldHyperConnections::run_inner(
    const autograd::TensorPtr& x, const std::optional<autograd::TensorPtr>& mask) {
    // Optional pre-sublayer RMSNorm on the combined A*X (the V4 attn_norm/ffn_norm).
    auto inp = m_layer_norm ? (*m_layer_norm)(x) : x;
    return mask.has_value() ? (*m_inner_layer)(inp, mask) : (*m_inner_layer)(inp);
}

autograd::TensorPtr ManifoldHyperConnections::compute_static(
    const autograd::TensorPtr& x, const std::optional<autograd::TensorPtr>& mask) {
    const auto x_shape = x->get_value().logical_shape().to_array_4D();
    const uint32_t batch = x_shape[0];
    const uint32_t seq = x_shape[2];
    const uint32_t dim = x_shape[3];
    const uint32_t n = m_config.num_streams;

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
    auto x_r = ops::reshape(x, ttnn::Shape({batch, 1, n, sd}));  // [B, 1, n, S*d]

    // A X: combine the n streams into one. [B,1,1,n] @ [B,1,n,S*d] -> [B,1,1,S*d].
    auto ax = ops::matmul_op(a_b, x_r);
    auto ax_r = ops::reshape(ax, ttnn::Shape({batch, 1, seq, dim}));  // [B, 1, S, d]

    // F(A X): the wrapped layer operates on the single combined stream.
    auto f_out = run_inner(ax_r, mask);                              // [B, 1, S, d]
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

autograd::TensorPtr ManifoldHyperConnections::compute_dynamic(
    const autograd::TensorPtr& x, const std::optional<autograd::TensorPtr>& mask) {
    const auto x_shape = x->get_value().logical_shape().to_array_4D();
    const uint32_t batch = x_shape[0];
    const uint32_t seq = x_shape[2];
    const uint32_t dim = x_shape[3];
    const uint32_t n = m_config.num_streams;
    const uint32_t tokens = batch * seq;  // M: every (batch, seq) position is an independent token
    const uint32_t nd = n * dim;

    // Move streams next to the feature dim so vec(X_l) is contiguous per token.
    auto x_perm = ops::permute(x, kSwapStreamsSeq);                    // [B, S, n, d]
    auto vec = ops::reshape(x_perm, ttnn::Shape({tokens, 1, 1, nd}));  // [M, 1, 1, n*d]
    auto x_hat = (*m_vec_norm)(vec);                                   // [M, 1, 1, n*d]

    // Per-token raw parameters: raw = alpha * (X_hat @ W) + S.
    auto gen_a = (*m_w_pre)(x_hat);                                        // [M, 1, 1, n]
    auto gen_b = (*m_w_res)(x_hat);                                        // [M, 1, 1, n*n]
    auto gen_c = (*m_w_post)(x_hat);                                       // [M, 1, 1, n]
    auto gen_b_mat = ops::reshape(gen_b, ttnn::Shape({tokens, 1, n, n}));  // [M, 1, n, n]

    auto raw_a = ops::add(ops::mul(gen_a, ops::broadcast_to(m_alpha_pre, gen_a->get_value().logical_shape())), m_s_pre);
    auto raw_b =
        ops::add(ops::mul(gen_b_mat, ops::broadcast_to(m_alpha_res, gen_b_mat->get_value().logical_shape())), m_s_res);
    auto raw_c =
        ops::add(ops::mul(gen_c, ops::broadcast_to(m_alpha_post, gen_c->get_value().logical_shape())), m_s_post);

    // Manifold constraints (per token).
    auto a = ops::sigmoid(raw_a);                               // [M, 1, 1, n]
    auto c0 = ops::mul(ops::sigmoid(raw_c), 2.0F);              // [M, 1, 1, n]
    auto c = ops::reshape(c0, ttnn::Shape({tokens, 1, n, 1}));  // [M, 1, n, 1]
    auto b = sinkhorn_knopp(raw_b, m_config.sinkhorn_iters);    // [M, 1, n, n]

    // Per-token residual state [M, 1, n, d].
    auto x_tok = ops::reshape(x_perm, ttnn::Shape({tokens, 1, n, dim}));

    // A X -> [M,1,1,d], reshaped/permuted to [B,1,S,d] for the inner layer.
    auto ax = ops::matmul_op(a, x_tok);                                                                 // [M, 1, 1, d]
    auto ax_bsd = ops::permute(ops::reshape(ax, ttnn::Shape({batch, seq, 1, dim})), kSwapStreamsSeq);   // [B,1,S,d]
    auto f_out = run_inner(ax_bsd, mask);                                                               // [B, 1, S, d]
    auto f_tok = ops::reshape(ops::permute(f_out, kSwapStreamsSeq), ttnn::Shape({tokens, 1, 1, dim}));  // [M,1,1,d]

    // C F(A X) and B X.
    auto cf = ops::matmul_op(c, f_tok);  // [M, 1, n, d]
    auto bx = ops::matmul_op(b, x_tok);  // [M, 1, n, d]
    auto y_tok = ops::add(bx, cf);       // [M, 1, n, d]

    // Back to [B, n, S, d].
    auto y_perm = ops::reshape(y_tok, ttnn::Shape({batch, seq, n, dim}));  // [B, S, n, d]
    return ops::permute(y_perm, kSwapStreamsSeq);                          // [B, n, S, d]
}

}  // namespace ttml::modules
