// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Differentiable reshape. The logical volume of `new_shape` must match the
// input's. Backward simply reshapes the upstream gradient back to the original
// shape (reshape is a pure view/layout change of the data).
autograd::TensorPtr reshape(const autograd::TensorPtr& tensor, const ttnn::Shape& new_shape);

}  // namespace ttml::ops
