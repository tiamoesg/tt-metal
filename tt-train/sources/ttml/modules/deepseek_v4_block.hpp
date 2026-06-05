// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "autograd/tensor.hpp"
#include "modules/deepseek_moe.hpp"
#include "modules/hybrid_attention.hpp"
#include "modules/manifold_hyper_connections.hpp"
#include "modules/module_base.hpp"
#include "modules/swiglu_mlp.hpp"

namespace ttml::modules {

// One DeepSeek-V4 transformer block (model.py `Block`). The residual stream is
// expanded to n_hc copies and the two sublayers (attention, FFN) are wrapped in
// Manifold-Constrained Hyper-Connections with a pre-sublayer RMSNorm:
//
//   x = mHC_attn( attn_norm , HybridAttention )(x)     # hc_pre -> norm -> attn -> hc_post
//   x = mHC_ffn ( ffn_norm  , SwiGLU FFN     )(x)      # hc_pre -> norm -> ffn  -> hc_post
//
// Operates on the expanded residual [B, n_hc, S, d]. The attention is CSA or HCA
// per `attn.use_sparse` (the per-layer hybrid choice). The FFN here is a dense
// SwiGLU MLP; swapping in a full DeepSeekMoE is a drop-in extension.
struct DeepSeekV4BlockConfig {
    uint32_t dim{0};               // d, hidden size
    uint32_t num_streams{4};       // n_hc (hc_mult)
    uint32_t ffn_inter_dim{0};     // SwiGLU intermediate size
    uint32_t sinkhorn_iters{20};   // mHC t_max
    float alpha_init{1e-2F};       // mHC gating init
    HybridAttentionConfig attn{};  // CSA/HCA configuration for this layer
    // FFN: dense SwiGLU (default) or a routed DeepSeekMoE when use_moe is set.
    bool use_moe{false};
    DeepSeekMoEConfig moe{};
};

class DeepSeekV4Block : public ModuleBase {
public:
    explicit DeepSeekV4Block(const DeepSeekV4BlockConfig& config);

    // expanded residual: [B, n_hc, S, d] -> [B, n_hc, S, d].
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& x) override;

private:
    std::shared_ptr<ManifoldHyperConnections> m_attn_mhc;
    std::shared_ptr<ManifoldHyperConnections> m_ffn_mhc;
};

}  // namespace ttml::modules
