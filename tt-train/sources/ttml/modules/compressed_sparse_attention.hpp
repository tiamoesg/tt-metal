// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "autograd/tensor.hpp"
#include "modules/grouped_output_projection.hpp"
#include "modules/lightning_indexer.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/overlapping_kv_compressor.hpp"
#include "modules/rms_norm_module.hpp"

namespace ttml::modules {

// Compressed Sparse Attention (CSA), DeepSeek-V4 (§2.3.1) -- the flagship sparse
// attention. CSA compresses the KV cache (1/m via the overlapping two-series
// compressor) and then applies DeepSeek Sparse Attention: a lightning indexer
// scores the compressed blocks and each query attends to only its top-k blocks.
//
// Pipeline (hidden [B, 1, S, d] -> [B, 1, S, d]):
//   kv       = RMSNorm( OverlapCompress_kv(H) )            [B, 1, G, c]   (G = S/m)
//   idx_keys = OverlapCompress_idx(H)                      [B, 1, G, c^I]
//   I        = LightningIndexer(H, idx_keys)               [B, 1, S, G]   (scores)
//   keep     = top-k(I) AND compressed-causal              [B, 1, S, G]   (0/1 mask)
//   q        = RMSNorm( split_heads(H W^DQ W^UQ) )         [B, n_h, S, c]
//   o        = SharedKV-MQA-with-sink(q, kv, mask=keep)    [B, n_h, S, c]  (sparse)
//   out      = GroupedOutputProjection(o)                  [B, 1, S, d]
//
// last_index_scores() exposes the differentiable scores I so the training loop
// can add the indexer's auxiliary loss (top-k selection itself is non-diff).
//
// Deferred refinements (in the paper): partial RoPE with the -i output trick, the
// sliding-window branch, and sharing the low-rank latent c^Q between the indexer
// queries and the main attention queries (currently separate projections).
struct CompressedSparseAttentionConfig {
    uint32_t dim{0};               // d
    uint32_t num_heads{0};         // n_h
    uint32_t head_dim{0};          // c
    uint32_t query_comp_dim{0};    // d_c
    uint32_t compression_rate{0};  // m
    uint32_t index_head_dim{0};    // c^I
    uint32_t num_index_heads{0};   // n^I_h
    uint32_t top_k{0};             // k
    uint32_t num_groups{0};        // g
    uint32_t group_inter_dim{0};   // d_g
};

class CompressedSparseAttention : public ModuleBase {
public:
    explicit CompressedSparseAttention(const CompressedSparseAttentionConfig& config);

    // hidden: [B, 1, S, d]  ->  [B, 1, S, d].
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& hidden) override;

    // Differentiable index scores from the most recent forward (for the indexer
    // auxiliary loss). Null before the first forward.
    [[nodiscard]] autograd::TensorPtr last_index_scores() const {
        return m_last_index_scores;
    }

private:
    CompressedSparseAttentionConfig m_config;
    std::shared_ptr<OverlappingKVCompressor> m_kv_compressor;
    std::shared_ptr<OverlappingKVCompressor> m_idx_key_compressor;
    std::shared_ptr<LightningIndexer> m_indexer;
    std::shared_ptr<LinearLayer> m_w_dq;  // d -> d_c
    std::shared_ptr<LinearLayer> m_w_uq;  // d_c -> n_h * c
    std::shared_ptr<RMSNormLayer> m_q_norm;
    std::shared_ptr<RMSNormLayer> m_kv_norm;
    autograd::TensorPtr m_sink_logits;  // [1, n_h, 1, 1]
    std::shared_ptr<GroupedOutputProjection> m_out_proj;

    autograd::TensorPtr m_last_index_scores;
};

}  // namespace ttml::modules
