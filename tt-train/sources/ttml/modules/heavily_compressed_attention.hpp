// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "autograd/tensor.hpp"
#include "modules/grouped_output_projection.hpp"
#include "modules/kv_compressor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/rms_norm_module.hpp"

namespace ttml::modules {

// Heavily Compressed Attention (HCA), DeepSeek-V4 (§2.3.2). HCA compresses the KV
// cache aggressively (every m' >> m tokens into one compressed entry) and then
// runs *dense* shared-KV Multi-Query Attention over the compressed entries (no
// sparse top-k selection -- that is CSA's job).
//
// Pipeline (hidden [B, 1, S, d] -> [B, 1, S, d]):
//   kv      = RMSNorm( Compress(H) )                       [B, 1, G, c]   (G = S/m')
//   c^Q     = H W^{DQ};  q = c^Q W^{UQ} -> split heads     [B, n_h, S, c]
//   q       = RMSNorm(q)                                   (per head)
//   o       = SharedKV-MQA-with-sink(q, kv, mask=causal)   [B, n_h, S, c]
//   out     = GroupedOutputProjection(o)                   [B, 1, S, d]
//
// Per §2.3.3, RMSNorm on the queries and compressed KV entries bounds the
// attention logits (so the sink-softmax is computed without max-subtraction), and
// the attention sink keeps causally-empty early rows numerically safe.
//
// When sliding_window > 0, each query additionally attends (causally) to the most
// recent `sliding_window` uncompressed tokens, concatenated with the compressed
// KV entries before the core attention (§2.3.3). sliding_window = 0 disables it.
//
// Not yet wired (paper refinement): partial RoPE with the -i output trick.
struct HeavilyCompressedAttentionConfig {
    uint32_t dim{0};               // d, hidden size
    uint32_t num_heads{0};         // n_h, query heads
    uint32_t head_dim{0};          // c, compressed entry / head dim
    uint32_t query_comp_dim{0};    // d_c, low-rank query dim (W^DQ)
    uint32_t compression_rate{0};  // m', tokens pooled per compressed entry
    uint32_t num_groups{0};        // g, grouped output projection groups
    uint32_t group_inter_dim{0};   // d_g, grouped output intermediate dim
    uint32_t sliding_window{0};    // n_win, recent uncompressed tokens (0 = disabled)
};

class HeavilyCompressedAttention : public ModuleBase {
public:
    explicit HeavilyCompressedAttention(const HeavilyCompressedAttentionConfig& config);

    // hidden: [B, 1, S, d]  ->  [B, 1, S, d].
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& hidden) override;

private:
    HeavilyCompressedAttentionConfig m_config;
    std::shared_ptr<KVCompressor> m_compressor;
    std::shared_ptr<LinearLayer> m_w_dq;  // d -> d_c
    std::shared_ptr<LinearLayer> m_w_uq;  // d_c -> n_h * c
    std::shared_ptr<RMSNormLayer> m_q_norm;
    std::shared_ptr<RMSNormLayer> m_kv_norm;
    std::shared_ptr<LinearLayer> m_w_win;      // sliding-window KV projection: d -> c (optional)
    std::shared_ptr<RMSNormLayer> m_win_norm;  // RMSNorm on window KV (optional)
    autograd::TensorPtr m_sink_logits;         // [1, n_h, 1, 1]
    std::shared_ptr<GroupedOutputProjection> m_out_proj;
};

}  // namespace ttml::modules
