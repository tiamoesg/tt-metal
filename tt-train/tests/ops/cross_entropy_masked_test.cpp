// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class CrossEntropyMaskedTest : public ::testing::Test {
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
constexpr uint32_t N = 2;
constexpr uint32_t H = 4;
constexpr uint32_t W = 8;

ttml::autograd::TensorPtr make_logits() {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> logits(N * H * W);
    for (uint32_t i = 0; i < logits.size(); ++i) {
        logits[i] = std::sin(static_cast<float>(i) * 0.3F);  // deterministic, varied
    }
    return ttml::autograd::create_tensor(ttml::core::from_vector(logits, ttnn::Shape({N, 1, H, W}), device));
}

ttml::autograd::TensorPtr make_targets() {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<uint32_t> targets(N * H);
    for (uint32_t i = 0; i < targets.size(); ++i) {
        targets[i] = i % W;
    }
    return ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        targets, ttnn::Shape({N, H}), device, ttnn::Layout::ROW_MAJOR));
}

ttml::autograd::TensorPtr make_mask(const std::vector<float>& values) {
    auto* device = &ttml::autograd::ctx().get_device();
    return ttml::autograd::create_tensor(ttml::core::from_vector(values, ttnn::Shape({N, 1, H, 1}), device));
}
}  // namespace

// With an all-ones mask, masked cross-entropy must equal the standard mean
// cross-entropy.
TEST_F(CrossEntropyMaskedTest, AllOnesMaskMatchesUnmasked) {
    auto unmasked = ttml::ops::cross_entropy_loss(make_logits(), make_targets());
    float unmasked_value = ttml::core::to_vector(unmasked->get_value())[0];
    ttml::autograd::ctx().reset_graph();

    auto mask = make_mask(std::vector<float>(N * H, 1.0F));
    auto masked = ttml::ops::cross_entropy_loss_masked(make_logits(), make_targets(), mask);
    float masked_value = ttml::core::to_vector(masked->get_value())[0];

    EXPECT_NEAR(masked_value, unmasked_value, 1e-2F);
}

// Masking a position must zero its gradient and exclude it from the average.
TEST_F(CrossEntropyMaskedTest, MaskedPositionHasZeroGradient) {
    auto logits = make_logits();

    // Mask out the first position of the first sample (n=0, h=0).
    std::vector<float> mask_values(N * H, 1.0F);
    mask_values[0] = 0.0F;
    auto mask = make_mask(mask_values);

    auto loss = ttml::ops::cross_entropy_loss_masked(logits, make_targets(), mask);
    loss->backward();

    auto grad = ttml::core::to_vector(logits->get_grad());  // [N * H * W]
    // The W gradient entries for (n=0, h=0) must all be ~0.
    for (uint32_t w = 0; w < W; ++w) {
        EXPECT_NEAR(grad[w], 0.0F, 1e-5F);
    }
    // A non-masked position should have at least one non-zero gradient entry.
    float sum_abs_next = 0.0F;
    for (uint32_t w = 0; w < W; ++w) {
        sum_abs_next += std::abs(grad[W + w]);
    }
    EXPECT_GT(sum_abs_next, 0.0F);
}
