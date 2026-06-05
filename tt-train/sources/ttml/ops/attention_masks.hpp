// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttml::ops {

// 0/1 keep masks for DeepSeek-V4 attention (constants; no autograd). 1 = attend.

// Compressed-causal mask [1, 1, S, G]: a query token t (in compressed block
// floor(t/rate)) may attend to compressed block s iff s < floor(t/rate) -- i.e.
// strictly earlier blocks only (§2.3.1/§2.3.3).
tt::tt_metal::Tensor compressed_causal_keep(
    uint32_t seq, uint32_t groups, uint32_t rate, ttnn::distributed::MeshDevice* device);

// Sliding-window causal mask [1, 1, S, S]: query token t may attend to
// uncompressed token j iff t - window < j <= t (the recent `window` tokens,
// causal). This is the local branch of CSA/HCA (§2.3.3).
tt::tt_metal::Tensor sliding_window_keep(uint32_t seq, uint32_t window, ttnn::distributed::MeshDevice* device);

}  // namespace ttml::ops
