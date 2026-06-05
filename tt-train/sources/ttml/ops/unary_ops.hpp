// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor);
autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw = false);
autograd::TensorPtr sigmoid(const autograd::TensorPtr& tensor);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
// autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
// Sum over a single dimension. keep_dim must be true: the backward pass relies
// on the reduced axis remaining size-1 so it can be replicated back.
autograd::TensorPtr sum(const autograd::TensorPtr& tensor, int dim, bool keep_dim = true);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
// Broadcast (via repeat) to a target shape. Backward sums the gradient back over
// the broadcasted dimensions.
autograd::TensorPtr broadcast_to(const autograd::TensorPtr& tensor, const ttnn::Shape& target_shape);
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);
// Numerically stable softmax over a single dimension (autograd-aware).
autograd::TensorPtr softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr exp(const autograd::TensorPtr& tensor);
autograd::TensorPtr clip(const autograd::TensorPtr& tensor, float lo, float hi);
autograd::TensorPtr sqrt(const autograd::TensorPtr& tensor);
autograd::TensorPtr softplus(const autograd::TensorPtr& tensor);
}  // namespace ttml::ops
