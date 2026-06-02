// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Differentiable permutation of tensor dimensions. `dims` is the new ordering
// of axes (same convention as ttnn::permute). Backward applies the inverse
// permutation to the upstream gradient.
autograd::TensorPtr permute(const autograd::TensorPtr& tensor, const ttnn::SmallVector<int64_t>& dims);

}  // namespace ttml::ops
