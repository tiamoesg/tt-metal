// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Autograd-aware concatenation along `dim` (ttnn::concat has no backward).
// `dim` may be negative. Backward slices the upstream gradient back into each
// input's contribution.
autograd::TensorPtr concat(const std::vector<autograd::TensorPtr>& tensors, int dim);

}  // namespace ttml::ops
