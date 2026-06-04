// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_v4_block.hpp"

#include <stdexcept>

namespace ttml::modules {

DeepSeekV4Block::DeepSeekV4Block(const DeepSeekV4BlockConfig& config) {
    if (config.dim == 0U || config.ffn_inter_dim == 0U || config.num_streams == 0U) {
        throw std::invalid_argument("DeepSeekV4BlockConfig: dim, ffn_inter_dim and num_streams must be set.");
    }

    create_name("deepseek_v4_block");

    const ManifoldHyperConnectionsConfig mhc_cfg{
        .num_streams = config.num_streams,
        .hidden_dim = config.dim,
        .sinkhorn_iters = config.sinkhorn_iters,
        .alpha_init = config.alpha_init,
        .dynamic_parameterization = true,
        .layer_norm = true};

    auto attention = std::make_shared<HybridAttention>(config.attn);
    m_attn_mhc = std::make_shared<ManifoldHyperConnections>(mhc_cfg, attention);

    auto ffn = std::make_shared<SwiGLUMLP>(SwiGLUMLPConfig{.dim = config.dim, .inter_dim = config.ffn_inter_dim});
    m_ffn_mhc = std::make_shared<ManifoldHyperConnections>(mhc_cfg, ffn);

    register_module(m_attn_mhc, "attn_mhc");
    register_module(m_ffn_mhc, "ffn_mhc");
}

autograd::TensorPtr DeepSeekV4Block::operator()(const autograd::TensorPtr& x) {
    auto h = (*m_attn_mhc)(x);  // mHC-wrapped (norm -> attention)
    return (*m_ffn_mhc)(h);     // mHC-wrapped (norm -> SwiGLU FFN)
}

}  // namespace ttml::modules
