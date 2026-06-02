// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

// Lightning Indexer for Compressed Sparse Attention, DeepSeek-V4 (§2.3.1, eqs 13-17).
//
// Given the query token hidden states and the compressed indexer keys K^IComp
// (one per compressed KV block), the indexer cheaply scores how relevant each
// compressed block is to each query, then selects the top-k blocks per query for
// the (expensive) core attention. The scoring uses low-rank query projections,
// per-head learnable weights and a ReLU non-linearity:
//
//   c^Q_t = h_t W^{DQ}                                          (eq 13)
//   [q^I_{t,1}; ...; q^I_{t,n^I_h}] = c^Q_t W^{IUQ}             (eq 14)
//   [w^I_{t,1}; ...; w^I_{t,n^I_h}] = h_t W^{w}                 (eq 15)
//   I_{t,s} = sum_h w^I_{t,h} * ReLU(q^I_{t,h} . K^IComp_s)     (eq 16)
//   C^SprsComp_t = { C^Comp_s | I_{t,s} in Top-k(I_{t,:}) }     (eq 17)
//
// `index_scores` returns the differentiable scores I (used by the indexer's
// auxiliary training loss). `selection_mask` turns the scores into a 0/1 keep
// mask over compressed blocks (matching ttml's SDPA mask convention), enforcing
// the compressed-causal constraint that a query in block floor(t/m) may only
// attend to strictly earlier compressed blocks. Top-k selection is
// non-differentiable, so the mask carries no gradient.
struct LightningIndexerConfig {
    uint32_t dim{0};               // d, hidden size
    uint32_t index_query_dim{0};   // d_c, low-rank query compression dim (W^DQ)
    uint32_t index_head_dim{0};    // c^I, indexer head dimension
    uint32_t num_index_heads{0};   // n^I_h, number of indexer heads
    uint32_t top_k{0};             // k, compressed blocks kept per query
    uint32_t compression_rate{1};  // m, for the compressed-causal constraint (s < floor(t/m))
};

class LightningIndexer : public ModuleBase {
public:
    explicit LightningIndexer(const LightningIndexerConfig& config);

    // query_hidden: [B, 1, S, d], index_keys (K^IComp): [B, 1, G, c^I].
    // Returns differentiable index scores I: [B, 1, S, G].
    [[nodiscard]] autograd::TensorPtr index_scores(
        const autograd::TensorPtr& query_hidden, const autograd::TensorPtr& index_keys);

    // 0/1 keep-mask [B, 1, S, G] selecting the top-k compressed blocks per query,
    // intersected with the compressed-causal constraint. Stop-gradient.
    [[nodiscard]] tt::tt_metal::Tensor selection_mask(const autograd::TensorPtr& index_scores_tensor) const;

private:
    LightningIndexerConfig m_config;
    std::shared_ptr<LinearLayer> m_w_dq;   // W^{DQ} : d   -> d_c
    std::shared_ptr<LinearLayer> m_w_iuq;  // W^{IUQ}: d_c -> c^I * n^I_h
    std::shared_ptr<LinearLayer> m_w_w;    // W^{w}  : d   -> n^I_h
};

}  // namespace ttml::modules
