// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <core/ttnn_all_includes.hpp>
#include <numeric>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/concat_op.hpp"
#include "ops/slice_op.hpp"

class SliceConcatTest : public ::testing::Test {
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
constexpr uint32_t kRows = 64;  // dim 2
constexpr uint32_t kCols = 32;  // dim 3
}  // namespace

// Slicing rows [32, 64) must return exactly that block, and its backward must
// route the gradient only to those rows (the rest stay zero).
TEST_F(SliceConcatTest, SliceForwardAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(static_cast<size_t>(kRows) * kCols);
    std::iota(data.begin(), data.end(), 0.0F);  // 0, 1, 2, ...
    auto input = ttml::autograd::create_tensor(
        ttml::core::from_vector(data, ttnn::Shape({1, 1, kRows, kCols}), device), /* requires_grad */ true);

    auto out = ttml::ops::slice(input, {0, 0, 32, 0}, {1, 1, kRows, kCols});  // rows [32, 64)
    EXPECT_EQ(out->get_value().logical_shape()[2], 32U);

    auto out_values = ttml::core::to_vector(out->get_value());
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < kCols; ++c) {
            const float expected = static_cast<float>((r + 32) * kCols + c);
            EXPECT_NEAR(out_values[static_cast<size_t>(r) * kCols + c], expected, 1e-3F);
        }
    }

    out->backward();  // seeds dL/dout = 1
    auto grad = ttml::core::to_vector(input->get_grad());
    // Rows [0,32) untouched -> zero gradient; rows [32,64) -> one.
    for (uint32_t r = 0; r < kRows; ++r) {
        const float expected = (r >= 32) ? 1.0F : 0.0F;
        EXPECT_NEAR(grad[static_cast<size_t>(r) * kCols + 0], expected, 1e-3F);
    }
}

// Concatenating two [.,.,32,.] tensors along dim 2 yields [.,.,64,.], and the
// backward routes the top/bottom halves of the gradient to each input.
TEST_F(SliceConcatTest, ConcatForwardAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> a_data(32U * kCols, 2.0F);
    std::vector<float> b_data(32U * kCols, 5.0F);
    auto a =
        ttml::autograd::create_tensor(ttml::core::from_vector(a_data, ttnn::Shape({1, 1, 32, kCols}), device), true);
    auto b =
        ttml::autograd::create_tensor(ttml::core::from_vector(b_data, ttnn::Shape({1, 1, 32, kCols}), device), true);

    auto out = ttml::ops::concat({a, b}, /* dim */ 2);
    EXPECT_EQ(out->get_value().logical_shape()[2], kRows);

    auto out_values = ttml::core::to_vector(out->get_value());
    EXPECT_NEAR(out_values[0], 2.0F, 1e-3F);                                // top half = a
    EXPECT_NEAR(out_values[static_cast<size_t>(32) * kCols], 5.0F, 1e-3F);  // bottom half = b

    out->backward();  // dL/dout = 1 everywhere
    EXPECT_TRUE(a->is_grad_initialized());
    EXPECT_TRUE(b->is_grad_initialized());
    EXPECT_NEAR(ttml::core::to_vector(a->get_grad())[0], 1.0F, 1e-3F);
    EXPECT_NEAR(ttml::core::to_vector(b->get_grad())[0], 1.0F, 1e-3F);
}
