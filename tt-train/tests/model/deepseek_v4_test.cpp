// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "models/deepseek_v4.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/hybrid_attention.hpp"
#include "ops/losses.hpp"
#include "optimizers/muon_v4.hpp"

class DeepSeekV4ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

namespace {
ttml::models::deepseek_v4::DeepSeekV4Config make_config(uint32_t vocab, uint32_t dim) {
    ttml::models::deepseek_v4::DeepSeekV4Config cfg;
    cfg.vocab_size = vocab;
    cfg.dim = dim;
    cfg.num_streams = 4;
    cfg.num_layers = 2;
    cfg.ffn_inter_dim = 128;
    // All layers dense HCA (no top-k size constraints; trains at small S).
    cfg.attn.use_sparse = false;
    cfg.attn.hca.dim = dim;
    cfg.attn.hca.num_heads = 4;
    cfg.attn.hca.head_dim = 32;
    cfg.attn.hca.query_comp_dim = 32;
    cfg.attn.hca.compression_rate = 4;
    cfg.attn.hca.num_groups = 2;
    cfg.attn.hca.group_inter_dim = 16;
    return cfg;
}
}  // namespace

// Forward produces logits [B, 1, S, vocab].
TEST_F(DeepSeekV4ModelTest, ForwardShape) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t vocab = 64;
    const uint32_t dim = 64;
    const uint32_t batch = 2;
    const uint32_t seq = 128;

    auto model = ttml::models::deepseek_v4::DeepSeekV4Transformer(make_config(vocab, dim));

    std::vector<uint32_t> ids(static_cast<size_t>(batch) * seq);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<uint32_t>(i % vocab);
    }
    auto tokens = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        ids, ttnn::Shape({batch, 1, 1, seq}), device, ttnn::Layout::ROW_MAJOR));

    auto logits = model(tokens);
    const auto shape = logits->get_value().logical_shape();
    EXPECT_EQ(shape[0], batch);
    EXPECT_EQ(shape[2], seq);
    EXPECT_EQ(shape[3], vocab);
}

// End-to-end trainability: overfit a tiny deterministic next-token task
// (target = (id + 1) mod vocab) with Muon-V4; the cross-entropy loss must drop.
TEST_F(DeepSeekV4ModelTest, TrainsWithMuonV4) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t vocab = 64;
    const uint32_t dim = 64;
    const uint32_t batch = 2;
    const uint32_t seq = 128;

    auto model = std::make_shared<ttml::models::deepseek_v4::DeepSeekV4Transformer>(make_config(vocab, dim));

    std::vector<uint32_t> ids(static_cast<size_t>(batch) * seq);
    std::vector<uint32_t> targets(static_cast<size_t>(batch) * seq);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<uint32_t>(i % vocab);
        targets[i] = (ids[i] + 1U) % vocab;
    }
    auto tokens = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        ids, ttnn::Shape({batch, 1, 1, seq}), device, ttnn::Layout::ROW_MAJOR));
    auto target = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        targets, ttnn::Shape({batch, seq}), device, ttnn::Layout::ROW_MAJOR));

    ttml::optimizers::MuonV4Config opt_cfg;
    opt_cfg.lr = 2e-2F;
    opt_cfg.fallback_on_non_matrix = true;  // single optimizer drives every parameter
    auto optimizer = ttml::optimizers::MuonV4(model->parameters(), opt_cfg);

    std::vector<float> losses;
    const size_t steps = 40;
    losses.reserve(steps);
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto logits = (*model)(tokens);
        auto loss = ttml::ops::cross_entropy_loss(logits, target);
        losses.push_back(ttml::core::to_vector(loss->get_value())[0]);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }

    // The model must fit this trivial mapping: loss decreases substantially.
    EXPECT_LT(losses.back(), losses.front());
    EXPECT_LT(losses.back(), losses.front() * 0.7F);
}
