// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor);
autograd::TensorPtr silu(const autograd::TensorPtr& tensor);
autograd::TensorPtr sigmoid(const autograd::TensorPtr& tensor);
autograd::TensorPtr exp(const autograd::TensorPtr& tensor);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
// Sum over a single dimension. keep_dim must be true: the backward pass relies
// on broadcasting the reduced (size-1) dimension back to the input shape.
autograd::TensorPtr sum(const autograd::TensorPtr& tensor, int dim, bool keep_dim = true);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
// Broadcast (repeat) size-1 dimensions of a rank-4 tensor up to target_shape.
// Backward sums the gradient back over the broadcasted dimensions.
autograd::TensorPtr broadcast_to(const autograd::TensorPtr& tensor, const ttnn::Shape& target_shape);
autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim);
autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim);
}  // namespace ttml::ops
