// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <numeric>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "ops/partial_rope_op.hpp"
#include "ops/rope_op.hpp"

class PartialRopeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// partial_rope rotates only the last `rope_dim` dims: the leading no-PE prefix is
// returned unchanged, the shape is preserved, and gradients flow to the input.
TEST_F(PartialRopeTest, LeavesPrefixUnchangedAndBackprops) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t seq = 32;
    const uint32_t rope_dim = 64;
    const uint32_t nope = 32;
    const uint32_t full = nope + rope_dim;  // 96

    std::vector<float> data(static_cast<size_t>(seq) * full);
    std::iota(data.begin(), data.end(), 1.0F);
    auto input = ttml::autograd::create_tensor(
        ttml::core::from_vector(data, ttnn::Shape({1, 1, seq, full}), device), /* requires_grad */ true);

    auto params = ttml::ops::build_rope_params(seq, rope_dim);
    auto out = ttml::ops::partial_rope(input, params, /* token_position */ 0);

    EXPECT_EQ(out->get_value().logical_shape(), input->get_value().logical_shape());

    // The leading `nope` dimensions must be passed through untouched.
    auto in_v = ttml::core::to_vector(input->get_value());
    auto out_v = ttml::core::to_vector(out->get_value());
    for (uint32_t s = 0; s < seq; ++s) {
        for (uint32_t d = 0; d < nope; ++d) {
            const size_t idx = static_cast<size_t>(s) * full + d;
            EXPECT_NEAR(out_v[idx], in_v[idx], 1e-2F) << "prefix changed at s=" << s << " d=" << d;
        }
    }

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(out->get_value().logical_shape(), device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();
    EXPECT_TRUE(input->is_grad_initialized());
}
