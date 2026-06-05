// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/grouped_output_projection.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class GroupedOutputProjectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// [B, n_h, S, c] -> [B, 1, S, d]; gradients must reach every per-group projection
// and the final output projection.
TEST_F(GroupedOutputProjectionTest, ForwardShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 1;
    const uint32_t heads = 4;
    const uint32_t seq = 32;
    const uint32_t head_dim = 32;
    const uint32_t groups = 2;
    const uint32_t group_inter_dim = 16;
    const uint32_t out_dim = 64;

    ttml::modules::GroupedOutputProjectionConfig config;
    config.num_heads = heads;
    config.head_dim = head_dim;
    config.num_groups = groups;
    config.group_inter_dim = group_inter_dim;
    config.out_dim = out_dim;
    auto proj = ttml::modules::GroupedOutputProjection(config);

    auto attn_out = ttml::autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({batch, heads, seq, head_dim}), device), /* requires_grad */ true);

    auto out = proj(attn_out);
    const auto shape = out->get_value().logical_shape();
    EXPECT_EQ(shape[0], batch);
    EXPECT_EQ(shape[1], 1U);
    EXPECT_EQ(shape[2], seq);
    EXPECT_EQ(shape[3], out_dim);

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(shape, device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();

    auto params = proj.parameters();
    EXPECT_TRUE(params.at("grouped_output_projection/group_proj_0/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("grouped_output_projection/group_proj_1/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("grouped_output_projection/out_proj/weight")->is_grad_initialized());
    EXPECT_TRUE(attn_out->is_grad_initialized());
}
