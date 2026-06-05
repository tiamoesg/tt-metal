// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

// Overlapping two-series token-level compressor, DeepSeek-V4 CSA (§2.3.1, eqs 9-12).
//
// Unlike HCA's single-series compressor, each compressed entry C^Comp_i is pooled
// from an overlapping 2m-token window: the current segment via the "a" series and
// the *previous* segment via the "b" series, under one joint softmax over all 2m
// elements:
//
//   C^a = H W^{aKV},  C^b = H W^{bKV},  Z^a = H W^{aZ},  Z^b = H W^{bZ}   (eqs 9-10)
//   [S^a_{seg i}; S^b_{seg i-1}] = Softmax_row( [Z^a_{seg i}+B^a ; Z^b_{seg i-1}+B^b] )  (eq 11)
//   C^Comp_i = sum_{j in seg i} S^a_j (.) C^a_j  +  sum_{j in seg i-1} S^b_j (.) C^b_j   (eq 12)
//
// For i = 0 the (nonexistent) previous segment is padded: Z^b with -inf (so its
// softmax weight is 0) and C^b with zeros. Consecutive compressed entries overlap
// (C^b of entry i and C^a of entry i-1 cover the same tokens), so the sequence is
// still compressed to 1/m. The same module is reused in CSA for both the KV
// entries (output dim c) and the indexer keys (output dim c^I).
//
// Implementation stays rank-4: the previous-segment "b" series is produced by a
// one-segment shift (slice off the last segment, prepend a padding segment), the
// 2m window is formed by concatenating the per-segment "a" and "b" tensors, and
// segments are folded into the batch dimension for the softmax/pool.
struct OverlappingKVCompressorConfig {
    uint32_t dim{0};               // d, input hidden size
    uint32_t compressed_dim{0};    // c (or c^I for indexer keys)
    uint32_t compression_rate{0};  // m, tokens per segment
};

class OverlappingKVCompressor : public ModuleBase {
public:
    explicit OverlappingKVCompressor(const OverlappingKVCompressorConfig& config);

    // hidden: [B, 1, n, d]  ->  compressed: [B, 1, n/m, c]   (n must be a multiple of m).
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& hidden) override;

private:
    OverlappingKVCompressorConfig m_config;
    std::shared_ptr<LinearLayer> m_w_akv;  // W^{aKV}
    std::shared_ptr<LinearLayer> m_w_bkv;  // W^{bKV}
    std::shared_ptr<LinearLayer> m_w_az;   // W^{aZ}
    std::shared_ptr<LinearLayer> m_w_bz;   // W^{bZ}
    autograd::TensorPtr m_bias_a;          // B^a: [1, 1, m, c]
    autograd::TensorPtr m_bias_b;          // B^b: [1, 1, m, c]
};

}  // namespace ttml::modules
