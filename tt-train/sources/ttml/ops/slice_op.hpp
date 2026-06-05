// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Autograd-aware slice (ttnn::slice has no backward). `start`/`end` are rank-4,
// half-open ([start, end)). Backward pads the upstream gradient with zeros back
// to the original shape, placing it at the sliced position.
autograd::TensorPtr slice(
    const autograd::TensorPtr& tensor, const std::array<uint32_t, 4>& start, const std::array<uint32_t, 4>& end);

}  // namespace ttml::ops
