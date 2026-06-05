// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "autograd/tensor.hpp"
#include "modules/compressed_sparse_attention.hpp"
#include "modules/heavily_compressed_attention.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

// Hybrid attention selector, DeepSeek-V4 (§2.3, Figure 2). V4 interleaves two
// attention types across layers: Compressed Sparse Attention (CSA) and Heavily
// Compressed Attention (HCA). This module wraps whichever variant a given layer
// uses behind a single [B,1,S,d] -> [B,1,S,d] interface, so a transformer block
// can pick per-layer via `use_sparse` without caring about the internals.
//
// The interleaving *pattern* (which layers are CSA vs HCA) is a model-level
// decision; this module just realizes one layer's choice.
struct HybridAttentionConfig {
    bool use_sparse{true};  // true -> CSA, false -> HCA
    CompressedSparseAttentionConfig csa;
    HeavilyCompressedAttentionConfig hca;
};

class HybridAttention : public ModuleBase {
public:
    explicit HybridAttention(const HybridAttentionConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& hidden) override;

    [[nodiscard]] bool is_sparse() const {
        return m_use_sparse;
    }

    // CSA index scores from the last forward (for the indexer auxiliary loss);
    // null for HCA layers or before the first forward.
    [[nodiscard]] autograd::TensorPtr last_index_scores() const {
        return m_csa ? m_csa->last_index_scores() : nullptr;
    }

private:
    bool m_use_sparse;
    std::shared_ptr<CompressedSparseAttention> m_csa;
    std::shared_ptr<HeavilyCompressedAttention> m_hca;
};

}  // namespace ttml::modules
