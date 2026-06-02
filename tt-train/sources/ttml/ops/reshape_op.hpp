// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr reshape(const autograd::TensorPtr& tensor, std::span<uint32_t> shape);

// Convenience overload taking a ttnn::Shape directly.
autograd::TensorPtr reshape(const autograd::TensorPtr& tensor, const ttnn::Shape& shape);

}  // namespace ttml::ops
