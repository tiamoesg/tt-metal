// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "autograd/tensor.hpp"
#include "modules/deepseek_v4_block.hpp"
#include "modules/embedding_module.hpp"
#include "modules/hybrid_attention.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/rms_norm_module.hpp"

namespace ttml::models::deepseek_v4 {

// A DeepSeek-V4 transformer (model.py `Transformer`), wiring all the V4 components:
//   tokens -> Embedding -> expand to n_hc residual streams
//          -> [DeepSeekV4Block] x L      (mHC-wrapped hybrid CSA/HCA + SwiGLU FFN)
//          -> reduce streams -> RMSNorm -> LM head -> logits [B, 1, S, vocab]
//
// Per-layer attention type follows `layer_sparse` (true = CSA, false = HCA); if
// empty, every layer uses `attn.use_sparse`. (This nano configuration uses a dense
// SwiGLU FFN; a full DeepSeekMoE FFN is a drop-in replacement.)
struct DeepSeekV4Config {
    uint32_t vocab_size{0};
    uint32_t dim{0};
    uint32_t num_streams{4};  // n_hc / hc_mult
    uint32_t num_layers{0};
    uint32_t ffn_inter_dim{0};
    uint32_t sinkhorn_iters{20};
    float alpha_init{1e-2F};
    modules::HybridAttentionConfig attn{};  // csa and hca sub-configs both filled
    std::vector<bool> layer_sparse{};       // per-layer CSA(true)/HCA(false)
};

class DeepSeekV4Transformer : public modules::ModuleBase {
public:
    explicit DeepSeekV4Transformer(const DeepSeekV4Config& config);

    // tokens: [B, 1, 1, S] (token ids)  ->  logits: [B, 1, S, vocab].
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tokens) override;

private:
    DeepSeekV4Config m_config;
    std::shared_ptr<modules::Embedding> m_embedding;
    std::vector<std::shared_ptr<modules::DeepSeekV4Block>> m_blocks;
    std::shared_ptr<modules::RMSNormLayer> m_final_norm;
    std::shared_ptr<modules::LinearLayer> m_head;
};

}  // namespace ttml::models::deepseek_v4
