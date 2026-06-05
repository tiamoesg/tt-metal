// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/lightning_indexer.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class LightningIndexerTest : public ::testing::Test {
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
// G = 64 (power of two, >= 64) keeps ttnn::topk within its supported last-dim sizes.
constexpr uint32_t kBatch = 1;
constexpr uint32_t kRate = 2;               // m
constexpr uint32_t kGroups = 64;            // G compressed blocks
constexpr uint32_t kSeq = kGroups * kRate;  // 128 query tokens
constexpr uint32_t kDim = 16;               // d
constexpr uint32_t kQueryDim = 8;           // d_c
constexpr uint32_t kHeadDim = 4;            // c^I
constexpr uint32_t kHeads = 2;              // n^I_h
constexpr uint32_t kTopK = 8;               // k

ttml::modules::LightningIndexerConfig make_config() {
    ttml::modules::LightningIndexerConfig config;
    config.dim = kDim;
    config.index_query_dim = kQueryDim;
    config.index_head_dim = kHeadDim;
    config.num_index_heads = kHeads;
    config.top_k = kTopK;
    config.compression_rate = kRate;
    return config;
}
}  // namespace

// Index scores must be [B, 1, S, G] and feed gradients back to all three indexer
// projections.
TEST_F(LightningIndexerTest, IndexScoresShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto indexer = ttml::modules::LightningIndexer(make_config());

    auto hidden = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kSeq, kDim}), device), true);
    auto latent =
        ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kSeq, kQueryDim}), device), true);
    auto keys =
        ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kGroups, kHeadDim}), device), true);

    auto scores = indexer.index_scores(hidden, latent, keys);
    const auto shape = scores->get_value().logical_shape();
    EXPECT_EQ(shape[0], kBatch);
    EXPECT_EQ(shape[1], 1U);
    EXPECT_EQ(shape[2], kSeq);
    EXPECT_EQ(shape[3], kGroups);

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(shape, device));
    auto loss = ttml::ops::mse_loss(scores, target);
    loss->backward();

    auto params = indexer.parameters();
    EXPECT_TRUE(params.at("lightning_indexer/w_iuq/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("lightning_indexer/w_w/weight")->is_grad_initialized());
    EXPECT_TRUE(latent->is_grad_initialized());
}

// The selection mask must be 0/1, respect the compressed-causal constraint
// (block s visible to query t iff s < floor(t/m)), and keep at most top_k blocks.
TEST_F(LightningIndexerTest, SelectionMaskCausalAndTopK) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto indexer = ttml::modules::LightningIndexer(make_config());

    auto hidden = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kSeq, kDim}), device), true);
    auto latent =
        ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kSeq, kQueryDim}), device), true);
    auto keys =
        ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kGroups, kHeadDim}), device), true);

    auto scores = indexer.index_scores(hidden, latent, keys);
    auto mask = indexer.selection_mask(scores);
    auto values = ttml::core::to_vector(mask);  // [S, G] row-major

    for (uint32_t t = 0; t < kSeq; ++t) {
        const uint32_t allowed = (t + 1U) / kRate;
        uint32_t row_sum = 0;
        for (uint32_t s = 0; s < kGroups; ++s) {
            const float v = values[static_cast<size_t>(t) * kGroups + s];
            EXPECT_TRUE(v == 0.0F || v == 1.0F) << "mask must be 0/1 at (" << t << "," << s << ")";
            if (s >= allowed) {
                EXPECT_EQ(v, 0.0F) << "non-causal block kept at (" << t << "," << s << ")";
            }
            row_sum += static_cast<uint32_t>(v);
        }
        // Never keep more than top_k, never more than the causally-available blocks.
        EXPECT_LE(row_sum, kTopK);
        EXPECT_LE(row_sum, allowed);
        // When blocks are available, at least one must be kept.
        if (allowed > 0) {
            EXPECT_GE(row_sum, 1U);
        }
    }
}
