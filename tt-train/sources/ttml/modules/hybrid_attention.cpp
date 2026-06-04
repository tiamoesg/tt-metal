// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_attention.hpp"

namespace ttml::modules {

HybridAttention::HybridAttention(const HybridAttentionConfig& config) : m_use_sparse(config.use_sparse) {
    create_name("hybrid_attention");
    if (m_use_sparse) {
        m_csa = std::make_shared<CompressedSparseAttention>(config.csa);
        register_module(m_csa, "attention");
    } else {
        m_hca = std::make_shared<HeavilyCompressedAttention>(config.hca);
        register_module(m_hca, "attention");
    }
}

autograd::TensorPtr HybridAttention::operator()(const autograd::TensorPtr& hidden) {
    return m_use_sparse ? (*m_csa)(hidden) : (*m_hca)(hidden);
}

}  // namespace ttml::modules
