// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compressed_sparse_attention.hpp"

#include <core/ttnn_all_includes.hpp>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/attention_masks.hpp"
#include "ops/concat_op.hpp"
#include "ops/permute_op.hpp"
#include "ops/reshape_op.hpp"
#include "ops/sink_attention.hpp"

namespace ttml::modules {

namespace {
const ttnn::SmallVector<int64_t> kHeadsToFront = {0, 2, 1, 3};  // [B,S,H,c] -> [B,H,S,c]
}  // namespace

CompressedSparseAttention::CompressedSparseAttention(const CompressedSparseAttentionConfig& config) : m_config(config) {
    if (m_config.dim == 0U || m_config.num_heads == 0U || m_config.head_dim == 0U || m_config.query_comp_dim == 0U ||
        m_config.compression_rate == 0U || m_config.index_head_dim == 0U || m_config.num_index_heads == 0U ||
        m_config.top_k == 0U || m_config.num_groups == 0U || m_config.group_inter_dim == 0U) {
        throw std::invalid_argument("CompressedSparseAttentionConfig: all fields must be set.");
    }

    create_name("compressed_sparse_attention");

    m_kv_compressor = std::make_shared<OverlappingKVCompressor>(OverlappingKVCompressorConfig{
        .dim = m_config.dim, .compressed_dim = m_config.head_dim, .compression_rate = m_config.compression_rate});
    m_idx_key_compressor = std::make_shared<OverlappingKVCompressor>(OverlappingKVCompressorConfig{
        .dim = m_config.dim, .compressed_dim = m_config.index_head_dim, .compression_rate = m_config.compression_rate});
    m_indexer = std::make_shared<LightningIndexer>(LightningIndexerConfig{
        .dim = m_config.dim,
        .index_query_dim = m_config.query_comp_dim,
        .index_head_dim = m_config.index_head_dim,
        .num_index_heads = m_config.num_index_heads,
        .top_k = m_config.top_k,
        .compression_rate = m_config.compression_rate});
    m_w_dq = std::make_shared<LinearLayer>(m_config.dim, m_config.query_comp_dim, /* has_bias */ false);
    m_w_uq = std::make_shared<LinearLayer>(
        m_config.query_comp_dim, m_config.num_heads * m_config.head_dim, /* has_bias */ false);
    m_q_norm = std::make_shared<RMSNormLayer>(m_config.head_dim);
    m_kv_norm = std::make_shared<RMSNormLayer>(m_config.head_dim);
    m_sink_logits =
        autograd::create_tensor(core::zeros(ttnn::Shape({1, m_config.num_heads, 1, 1}), &autograd::ctx().get_device()));
    m_out_proj = std::make_shared<GroupedOutputProjection>(GroupedOutputProjectionConfig{
        .num_heads = m_config.num_heads,
        .head_dim = m_config.head_dim,
        .num_groups = m_config.num_groups,
        .group_inter_dim = m_config.group_inter_dim,
        .out_dim = m_config.dim});

    register_module(m_kv_compressor, "kv_compressor");
    register_module(m_idx_key_compressor, "idx_key_compressor");
    register_module(m_indexer, "indexer");
    register_module(m_w_dq, "w_dq");
    register_module(m_w_uq, "w_uq");
    register_module(m_q_norm, "q_norm");
    register_module(m_kv_norm, "kv_norm");
    register_tensor(m_sink_logits, "sink_logits");
    register_module(m_out_proj, "out_proj");

    if (m_config.sliding_window > 0U) {
        m_w_win = std::make_shared<LinearLayer>(m_config.dim, m_config.head_dim, /* has_bias */ false);
        m_win_norm = std::make_shared<RMSNormLayer>(m_config.head_dim);
        register_module(m_w_win, "w_win");
        register_module(m_win_norm, "win_norm");
    }
}

autograd::TensorPtr CompressedSparseAttention::operator()(const autograd::TensorPtr& hidden) {
    const auto shape = hidden->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t heads = m_config.num_heads;
    const uint32_t head_dim = m_config.head_dim;

    // Compressed, normalized shared KV entries, and the compressed indexer keys.
    auto kv = (*m_kv_norm)((*m_kv_compressor)(hidden));  // [B, 1, G, c]
    auto idx_keys = (*m_idx_key_compressor)(hidden);     // [B, 1, G, c^I]

    // Shared low-rank query latent c^Q = h W^DQ (eq 13), used by BOTH the indexer
    // queries (eq 14) and the main attention queries (eq 18).
    auto latent = (*m_w_dq)(hidden);  // [B, 1, S, d_c]

    // Lightning indexer -> differentiable scores (for the aux loss) and 0/1 top-k
    // selection mask (stop-gradient), [B, 1, S, G].
    m_last_index_scores = m_indexer->index_scores(hidden, latent, idx_keys);
    auto selection = m_indexer->selection_mask(m_last_index_scores);

    // Main queries from the same latent, split into heads, per-head RMSNorm.
    auto q = (*m_w_uq)(latent);  // [B, 1, S, n_h*c]
    auto q_heads = ops::permute(ops::reshape(q, ttnn::Shape({batch, seq, heads, head_dim})), kHeadsToFront);
    q_heads = (*m_q_norm)(q_heads);  // [B, n_h, S, c]

    if (m_config.sliding_window == 0U) {
        // Sparse shared-KV MQA with attention sink (mask selects the top-k blocks).
        auto keep_mask = autograd::create_tensor(selection);
        auto attn = ops::shared_kv_mqa_attention(q_heads, kv, m_sink_logits, keep_mask);  // [B, n_h, S, c]
        return (*m_out_proj)(attn);
    }

    // Sliding-window branch: recent uncompressed tokens, concatenated with the
    // selected compressed entries as additional keys/values (§2.3.3).
    auto* device = &autograd::ctx().get_device();
    auto win_kv = (*m_win_norm)((*m_w_win)(hidden));            // [B, 1, S, c]
    auto combined_kv = ops::concat({win_kv, kv}, /* dim */ 2);  // [B, 1, S+G, c]

    // Combined keep mask [B, 1, S, S+G] = [sliding-window-causal | top-k selection].
    auto window_mask = ops::sliding_window_keep(seq, m_config.sliding_window, device);  // [1,1,S,S]
    auto window_mask_b = ttnn::repeat(window_mask, ttnn::Shape({batch, 1, 1, 1}));      // [B,1,S,S]
    auto combined_mask =
        autograd::create_tensor(ttnn::concat(std::vector<ttnn::Tensor>{window_mask_b, selection}, /* dim */ 3));

    auto attn = ops::shared_kv_mqa_attention(q_heads, combined_kv, m_sink_logits, combined_mask);  // [B, n_h, S, c]
    return (*m_out_proj)(attn);                                                                    // [B, 1, S, d]
}

}  // namespace ttml::modules
