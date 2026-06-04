// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "overlapping_kv_compressor.hpp"

#include <fmt/format.h>

#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/concat_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/slice_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

OverlappingKVCompressor::OverlappingKVCompressor(const OverlappingKVCompressorConfig& config) : m_config(config) {
    if (m_config.dim == 0U || m_config.compressed_dim == 0U || m_config.compression_rate == 0U) {
        throw std::invalid_argument(
            "OverlappingKVCompressorConfig: dim, compressed_dim and compression_rate must all be set.");
    }

    create_name("overlapping_kv_compressor");

    m_w_akv = std::make_shared<LinearLayer>(m_config.dim, m_config.compressed_dim, /* has_bias */ false);
    m_w_bkv = std::make_shared<LinearLayer>(m_config.dim, m_config.compressed_dim, /* has_bias */ false);
    m_w_az = std::make_shared<LinearLayer>(m_config.dim, m_config.compressed_dim, /* has_bias */ false);
    m_w_bz = std::make_shared<LinearLayer>(m_config.dim, m_config.compressed_dim, /* has_bias */ false);

    const auto bias_shape = ttnn::Shape({1, 1, m_config.compression_rate, m_config.compressed_dim});
    m_bias_a = autograd::create_tensor(core::zeros(bias_shape, &autograd::ctx().get_device()));
    m_bias_b = autograd::create_tensor(core::zeros(bias_shape, &autograd::ctx().get_device()));

    register_module(m_w_akv, "w_akv");
    register_module(m_w_bkv, "w_bkv");
    register_module(m_w_az, "w_az");
    register_module(m_w_bz, "w_bz");
    register_tensor(m_bias_a, "bias_a");
    register_tensor(m_bias_b, "bias_b");
}

autograd::TensorPtr OverlappingKVCompressor::operator()(const autograd::TensorPtr& hidden) {
    const auto shape = hidden->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t m = m_config.compression_rate;
    const uint32_t c = m_config.compressed_dim;
    if (seq % m != 0U) {
        throw std::invalid_argument(fmt::format(
            "OverlappingKVCompressor: sequence length {} must be a multiple of compression_rate {}.", seq, m));
    }
    const uint32_t groups = seq / m;
    const uint32_t folded = batch * groups;
    auto* device = &autograd::ctx().get_device();

    auto c_a = (*m_w_akv)(hidden);  // [B, 1, n, c]
    auto c_b = (*m_w_bkv)(hidden);
    auto z_a = (*m_w_az)(hidden);
    auto z_b = (*m_w_bz)(hidden);

    // "b" series shifted one segment forward: entry i uses the previous segment.
    // Padding segment: zeros for C^b, -inf for Z^b (so its softmax weight is 0).
    const auto seg_shape = ttnn::Shape({batch, 1, m, c});
    auto zeros_seg = autograd::create_tensor(core::zeros(seg_shape, device));
    auto neg_seg = autograd::create_tensor(
        core::from_vector(std::vector<float>(static_cast<size_t>(batch) * m * c, -1.0e9F), seg_shape, device));
    // Drop the last segment (keep the first n - m tokens), then prepend the pad.
    auto c_b_head = ops::slice(c_b, {0U, 0U, 0U, 0U}, {batch, 1U, seq - m, c});
    auto z_b_head = ops::slice(z_b, {0U, 0U, 0U, 0U}, {batch, 1U, seq - m, c});
    auto c_b_shift = ops::concat({zeros_seg, c_b_head}, /* dim */ 2);  // [B, 1, n, c]
    auto z_b_shift = ops::concat({neg_seg, z_b_head}, /* dim */ 2);

    // Fold each segment into the batch dimension: [B, 1, n, c] -> [B*G, 1, m, c].
    auto za_seg = ops::reshape(z_a, ttnn::Shape({folded, 1, m, c}));
    auto ca_seg = ops::reshape(c_a, ttnn::Shape({folded, 1, m, c}));
    auto zb_seg = ops::reshape(z_b_shift, ttnn::Shape({folded, 1, m, c}));
    auto cb_seg = ops::reshape(c_b_shift, ttnn::Shape({folded, 1, m, c}));

    // Add the learnable positional biases (broadcast over the folded batch).
    za_seg = ops::add(za_seg, ops::broadcast_to(m_bias_a, ttnn::Shape({folded, 1, m, c})));
    zb_seg = ops::add(zb_seg, ops::broadcast_to(m_bias_b, ttnn::Shape({folded, 1, m, c})));

    // Joint softmax over the 2m window, then weighted sum of the values.
    auto z_window = ops::concat({za_seg, zb_seg}, /* dim */ 2);  // [B*G, 1, 2m, c]
    auto c_window = ops::concat({ca_seg, cb_seg}, /* dim */ 2);  // [B*G, 1, 2m, c]
    auto weights = ops::softmax(z_window, /* dim */ -2);
    auto pooled = ops::sum(ops::mul(weights, c_window), /* dim */ -2);  // [B*G, 1, 1, c]

    return ops::reshape(pooled, ttnn::Shape({batch, 1, groups, c}));  // [B, 1, G, c]
}

}  // namespace ttml::modules
