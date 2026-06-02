// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ttnn/tensor/tensor.hpp>

namespace ttml::ops {

tt::tt_metal::Tensor newtonschulz5(const tt::tt_metal::Tensor& G, int steps = 5, float eps = 1e-7f);

tt::tt_metal::Tensor newtonschulz(const tt::tt_metal::Tensor& G, int steps, float eps, float a, float b, float c);

// DeepSeek-V4 hybrid Newton-Schulz: one Frobenius normalization, then `steps`
// iterations where the first (steps - stabilize_steps) use the aggressive
// coefficients (3.4445, -4.7750, 2.0315) to drive the singular values toward 1,
// and the final `stabilize_steps` use the stabilizing coefficients (2, -1.5, 0.5)
// to pin them precisely at 1. DeepSeek-V4 uses steps=10, stabilize_steps=2.
tt::tt_metal::Tensor newtonschulz_hybrid(
    const tt::tt_metal::Tensor& G, int steps = 10, int stabilize_steps = 2, float eps = 1e-7f);

}  // namespace ttml::ops
