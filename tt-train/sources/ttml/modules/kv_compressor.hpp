// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

// Token-level KV compressor, DeepSeek-V4 (§2.3.2, eqs 20-23) -- the non-overlapping,
// single-series compressor used by Heavily Compressed Attention (HCA). It pools
// every `compression_rate` (m') consecutive token entries into one compressed
// entry via a per-channel softmax over the segment:
//
//   C = H W^{KV},  Z = H W^{Z}                              C, Z in R^{n x c}
//   S_seg = Softmax_over_segment( Z_seg + B )               B in R^{m' x c}
//   C^Comp_i = sum_{j in segment i} S_j (.) C_j             C^Comp in R^{n/m' x c}
//
// The softmax is taken independently per compressed channel c over the m' tokens
// of each segment, so each compressed entry is a learned, position-biased
// weighted average of its segment. (CSA's overlapping two-series compressor,
// §2.3.1 eqs 9-12, is a separate extension built on the same idea.)
//
// To stay within ttml's rank-4 tensor world, segments are folded into the batch
// dimension: [B, 1, n, c] -> [B*G, 1, m', c], softmax/pool over the m' axis, then
// reshape back to [B, 1, G, c] with G = n / m'.
struct KVCompressorConfig {
    uint32_t dim{0};               // d, hidden size of the input
    uint32_t compressed_dim{0};    // c, head dimension of the compressed entries
    uint32_t compression_rate{0};  // m', number of tokens pooled into one entry
};

class KVCompressor : public ModuleBase {
public:
    explicit KVCompressor(const KVCompressorConfig& config);

    // hidden: [B, 1, n, d]  ->  compressed: [B, 1, n / m', c]   (n must be a multiple of m').
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& hidden) override;

private:
    KVCompressorConfig m_config;
    std::shared_ptr<LinearLayer> m_w_kv;  // W^{KV}: d -> c
    std::shared_ptr<LinearLayer> m_w_z;   // W^{Z}:  d -> c (compression weights)
    autograd::TensorPtr m_pos_bias;       // B: [1, 1, m', c] learnable positional bias
};

}  // namespace ttml::modules
