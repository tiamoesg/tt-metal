// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_compressor.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/reshape_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

KVCompressor::KVCompressor(const KVCompressorConfig& config) : m_config(config) {
    if (m_config.dim == 0U || m_config.compressed_dim == 0U || m_config.compression_rate == 0U) {
        throw std::invalid_argument("KVCompressorConfig: dim, compressed_dim and compression_rate must all be set.");
    }

    create_name("kv_compressor");

    m_w_kv = std::make_shared<LinearLayer>(m_config.dim, m_config.compressed_dim, /* has_bias */ false);
    m_w_z = std::make_shared<LinearLayer>(m_config.dim, m_config.compressed_dim, /* has_bias */ false);
    // Positional bias B in R^{m' x c}, broadcast over every segment. Zero-init so
    // each segment starts as a uniform average over its m' tokens.
    m_pos_bias = autograd::create_tensor(core::zeros(
        ttnn::Shape({1, 1, m_config.compression_rate, m_config.compressed_dim}), &autograd::ctx().get_device()));

    register_module(m_w_kv, "w_kv");
    register_module(m_w_z, "w_z");
    register_tensor(m_pos_bias, "pos_bias");
}

autograd::TensorPtr KVCompressor::operator()(const autograd::TensorPtr& hidden) {
    const auto shape = hidden->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t m = m_config.compression_rate;
    const uint32_t c = m_config.compressed_dim;

    if (seq % m != 0U) {
        throw std::invalid_argument(
            fmt::format("KVCompressor: sequence length {} must be a multiple of compression_rate {}.", seq, m));
    }
    const uint32_t groups = seq / m;
    const uint32_t folded = batch * groups;  // segments folded into the batch dimension

    auto c_entries = (*m_w_kv)(hidden);  // C = H W^{KV}: [B, 1, n, c]
    auto z_weights = (*m_w_z)(hidden);   // Z = H W^{Z}:  [B, 1, n, c]

    // Fold each m'-token segment into the batch: [B, 1, n, c] -> [B*G, 1, m', c].
    auto c_seg = ops::reshape(c_entries, ttnn::Shape({folded, 1, m, c}));
    auto z_seg = ops::reshape(z_weights, ttnn::Shape({folded, 1, m, c}));

    // Z_seg + B, with B broadcast over the folded batch.
    auto biased = ops::add(z_seg, ops::broadcast_to(m_pos_bias, ttnn::Shape({folded, 1, m, c})));

    // Per-channel softmax over the m' tokens of each segment, then weighted sum.
    auto weights = ops::softmax(biased, /* dim */ -2);               // [B*G, 1, m', c]
    auto pooled = ops::sum(ops::mul(weights, c_seg), /* dim */ -2);  // [B*G, 1, 1, c]

    // Back to [B, 1, G, c].
    return ops::reshape(pooled, ttnn::Shape({batch, 1, groups, c}));
}

}  // namespace ttml::modules
