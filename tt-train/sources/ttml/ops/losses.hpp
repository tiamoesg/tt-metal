// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

enum ReduceType : uint8_t { MEAN = 0, SUM = 1 };

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce = ReduceType::MEAN);

autograd::TensorPtr cross_entropy_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce = ReduceType::MEAN);

// Cross-entropy averaged over only the positions selected by `mask`.
// Shapes: prediction [N, 1, H, W] logits, target [N, H] token ids,
// mask [N, 1, H, 1] with 1.0 = include this position in the loss, 0.0 = ignore.
// The loss is sum(mask * per_position_ce) / sum(mask), and gradients are zero at
// masked positions. This is the primitive for supervised fine-tuning, where the
// loss is computed on response tokens only (prompt/padding masked out).
autograd::TensorPtr cross_entropy_loss_masked(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, const autograd::TensorPtr& mask);

autograd::TensorPtr nll_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce = ReduceType::MEAN);

}  // namespace ttml::ops
