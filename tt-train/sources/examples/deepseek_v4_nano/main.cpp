// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal runnable trainer for a small *dense* DeepSeek-V4 transformer (dense HCA
// attention + dense SwiGLU FFN). It overfits a synthetic next-token task
// (target = (id + 1) mod vocab) and prints the cross-entropy loss, demonstrating
// that the V4 stack -- mHC residuals, hybrid attention, SwiGLU FFN -- trains
// end-to-end. Swap AdamW for MuonV4, set use_sparse/use_moe, etc. to explore.

#include <fmt/format.h>

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/deepseek_v4.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"

int main() {
    auto& ctx = ttml::autograd::ctx();
    ctx.open_device();
    auto* device = &ctx.get_device();

    const uint32_t vocab = 64U;
    const uint32_t dim = 64U;
    const uint32_t batch = 2U;
    const uint32_t seq = 128U;

    // Dense V4 config: HCA attention (no sparse selection), dense SwiGLU FFN.
    ttml::models::deepseek_v4::DeepSeekV4Config cfg;
    cfg.vocab_size = vocab;
    cfg.dim = dim;
    cfg.num_streams = 4U;
    cfg.num_layers = 2U;
    cfg.ffn_inter_dim = 128U;
    cfg.attn.use_sparse = false;
    cfg.attn.hca.dim = dim;
    cfg.attn.hca.num_heads = 4U;
    cfg.attn.hca.head_dim = 32U;
    cfg.attn.hca.query_comp_dim = 32U;
    cfg.attn.hca.compression_rate = 4U;
    cfg.attn.hca.num_groups = 2U;
    cfg.attn.hca.group_inter_dim = 16U;
    auto model = std::make_shared<ttml::models::deepseek_v4::DeepSeekV4Transformer>(cfg);

    // Synthetic next-token task: predict (id + 1) mod vocab.
    std::vector<uint32_t> ids(static_cast<size_t>(batch) * seq);
    std::vector<uint32_t> targets(static_cast<size_t>(batch) * seq);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<uint32_t>(i % vocab);
        targets[i] = (ids[i] + 1U) % vocab;
    }
    auto tokens = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        ids, ttnn::Shape({batch, 1U, 1U, seq}), device, ttnn::Layout::ROW_MAJOR));
    auto target = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        targets, ttnn::Shape({batch, seq}), device, ttnn::Layout::ROW_MAJOR));

    ttml::optimizers::AdamWConfig opt_cfg;
    opt_cfg.lr = 3e-3F;
    auto optimizer = ttml::optimizers::AdamW(model->parameters(), opt_cfg);

    const int steps = 200;
    fmt::println(
        "Training a nano dense DeepSeek-V4 model ({} layers, dim {}, vocab {})...", cfg.num_layers, dim, vocab);
    for (int step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto logits = (*model)(tokens);
        auto loss = ttml::ops::cross_entropy_loss(logits, target);
        const float loss_value = ttml::core::to_vector(loss->get_value())[0];
        if (step % 10 == 0 || step == steps - 1) {
            fmt::println("step {:4d} | loss {:.4f}", step, loss_value);
        }
        loss->backward();
        optimizer.step();
        ctx.reset_graph();
    }

    ctx.close_device();
    return 0;
}
