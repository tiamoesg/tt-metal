// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_output_projection.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "ops/concat_op.hpp"
#include "ops/permute_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/slice_op.hpp"

namespace ttml::modules {

namespace {
const ttnn::SmallVector<int64_t> kHeadsToFeatures = {0, 2, 1, 3};  // [B,H,S,c] <-> [B,S,H,c]
}  // namespace

GroupedOutputProjection::GroupedOutputProjection(const GroupedOutputProjectionConfig& config) : m_config(config) {
    if (m_config.num_heads == 0U || m_config.head_dim == 0U || m_config.num_groups == 0U ||
        m_config.group_inter_dim == 0U || m_config.out_dim == 0U) {
        throw std::invalid_argument("GroupedOutputProjectionConfig: all fields must be set.");
    }
    if (m_config.num_heads % m_config.num_groups != 0U) {
        throw std::invalid_argument(fmt::format(
            "GroupedOutputProjection: num_heads ({}) must be divisible by num_groups ({}).",
            m_config.num_heads,
            m_config.num_groups));
    }

    create_name("grouped_output_projection");

    const uint32_t group_feature_dim = (m_config.num_heads / m_config.num_groups) * m_config.head_dim;  // c*n_h/g
    m_group_proj.reserve(m_config.num_groups);
    for (uint32_t i = 0; i < m_config.num_groups; ++i) {
        auto proj = std::make_shared<LinearLayer>(group_feature_dim, m_config.group_inter_dim, /* has_bias */ false);
        register_module(proj, fmt::format("group_proj_{}", i));
        m_group_proj.push_back(std::move(proj));
    }
    m_out_proj = std::make_shared<LinearLayer>(
        m_config.num_groups * m_config.group_inter_dim, m_config.out_dim, /* has_bias */ false);
    register_module(m_out_proj, "out_proj");
}

autograd::TensorPtr GroupedOutputProjection::operator()(const autograd::TensorPtr& attn_out) {
    const auto shape = attn_out->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t heads = shape[1];
    const uint32_t seq = shape[2];
    const uint32_t head_dim = shape[3];
    if (heads != m_config.num_heads || head_dim != m_config.head_dim) {
        throw std::invalid_argument(fmt::format(
            "GroupedOutputProjection expected [B, {}, S, {}], got [{}, {}, {}, {}].",
            m_config.num_heads,
            m_config.head_dim,
            batch,
            heads,
            seq,
            head_dim));
    }

    // Fuse all heads (head-major) into the feature dimension: [B, H, S, c] -> [B, 1, S, H*c].
    auto fused =
        ops::reshape(ops::permute(attn_out, kHeadsToFeatures), ttnn::Shape({batch, 1U, seq, heads * head_dim}));

    // Each group owns a contiguous block of (n_h/g) heads = c*n_h/g feature dims.
    const uint32_t group_feature_dim = (heads / m_config.num_groups) * head_dim;
    std::vector<autograd::TensorPtr> group_outputs;
    group_outputs.reserve(m_config.num_groups);
    for (uint32_t i = 0; i < m_config.num_groups; ++i) {
        auto group =
            ops::slice(fused, {0U, 0U, 0U, i * group_feature_dim}, {batch, 1U, seq, (i + 1U) * group_feature_dim});
        group_outputs.push_back((*m_group_proj[i])(group));  // [B, 1, S, d_g]
    }

    auto concatenated = ops::concat(group_outputs, /* dim */ 3);  // [B, 1, S, g*d_g]
    return (*m_out_proj)(concatenated);                           // [B, 1, S, d]
}

}  // namespace ttml::modules
