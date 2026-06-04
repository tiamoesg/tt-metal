// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/sink_attention.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class SinkAttentionTest : public ::testing::Test {
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
// Tile-friendly dimensions for on-device matmul.
constexpr uint32_t kBatch = 1;
constexpr uint32_t kHeads = 2;
constexpr uint32_t kSeq = 32;
constexpr uint32_t kGroups = 32;
constexpr uint32_t kHeadDim = 32;
}  // namespace

// q [B,H,S,c], kv [B,1,G,c], sink [1,H,1,1] -> out [B,H,S,c]; gradients reach all
// three inputs.
TEST_F(SinkAttentionTest, ShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto q =
        ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, kHeads, kSeq, kHeadDim}), device), true);
    auto kv =
        ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kGroups, kHeadDim}), device), true);
    auto sink = ttml::autograd::create_tensor(ttml::core::zeros(ttnn::Shape({1, kHeads, 1, 1}), device), true);

    auto out = ttml::ops::shared_kv_mqa_attention(q, kv, sink);
    const auto shape = out->get_value().logical_shape();
    EXPECT_EQ(shape[0], kBatch);
    EXPECT_EQ(shape[1], kHeads);
    EXPECT_EQ(shape[2], kSeq);
    EXPECT_EQ(shape[3], kHeadDim);

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(shape, device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();

    EXPECT_TRUE(q->is_grad_initialized());
    EXPECT_TRUE(kv->is_grad_initialized());
    EXPECT_TRUE(sink->is_grad_initialized());
}

// A query row whose entire selection mask is zero must produce ~0 output (the
// attention sink absorbs all the probability mass instead of dividing by zero).
TEST_F(SinkAttentionTest, FullyMaskedRowIsZero) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto q = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, kHeads, kSeq, kHeadDim}), device));
    auto kv = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({kBatch, 1, kGroups, kHeadDim}), device));
    auto sink = ttml::autograd::create_tensor(ttml::core::zeros(ttnn::Shape({1, kHeads, 1, 1}), device));

    // Mask out every block for row 0, keep all blocks for the rest.
    std::vector<float> mask_data(static_cast<size_t>(kSeq) * kGroups, 1.0F);
    for (uint32_t s = 0; s < kGroups; ++s) {
        mask_data[s] = 0.0F;  // row 0
    }
    auto mask = ttml::autograd::create_tensor(
        ttml::core::from_vector(mask_data, ttnn::Shape({kBatch, 1, kSeq, kGroups}), device));

    auto out = ttml::ops::shared_kv_mqa_attention(q, kv, sink, mask);
    auto values = ttml::core::to_vector(out->get_value());  // [H, S, c]

    for (uint32_t h = 0; h < kHeads; ++h) {
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            const size_t idx = (static_cast<size_t>(h) * kSeq + 0) * kHeadDim + d;  // row s = 0
            EXPECT_NEAR(values[idx], 0.0F, 1e-2F);
        }
    }
}
