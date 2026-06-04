// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/attention_masks.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

class AttentionMasksTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// compressed_causal_keep: block s visible to query t iff s < floor(t/rate).
TEST_F(AttentionMasksTest, CompressedCausal) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t seq = 8;
    const uint32_t groups = 4;
    const uint32_t rate = 2;
    auto mask = ttml::ops::compressed_causal_keep(seq, groups, rate, device);
    auto v = ttml::core::to_vector(mask);  // [S, G]
    for (uint32_t t = 0; t < seq; ++t) {
        const uint32_t allowed = t / rate;
        for (uint32_t s = 0; s < groups; ++s) {
            const float expected = (s < allowed) ? 1.0F : 0.0F;
            EXPECT_EQ(v[static_cast<size_t>(t) * groups + s], expected) << "t=" << t << " s=" << s;
        }
    }
}

// sliding_window_keep: token j visible to query t iff t - window < j <= t.
TEST_F(AttentionMasksTest, SlidingWindow) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t seq = 8;
    const uint32_t window = 3;
    auto mask = ttml::ops::sliding_window_keep(seq, window, device);
    auto v = ttml::core::to_vector(mask);  // [S, S]
    for (uint32_t t = 0; t < seq; ++t) {
        for (uint32_t j = 0; j < seq; ++j) {
            const bool visible = (j <= t) && (j + window > t);
            EXPECT_EQ(v[static_cast<size_t>(t) * seq + j], visible ? 1.0F : 0.0F) << "t=" << t << " j=" << j;
        }
    }
}
