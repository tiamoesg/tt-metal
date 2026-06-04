// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ops/rope_op.hpp"

namespace ttml::ops {

// Partial Rotary Positional Embedding (DeepSeek-V4 §2.3.3). Applies RoPE to the
// last `rope_params.head_dim` dimensions of the input's last axis and leaves the
// leading "no-PE" dimensions unchanged, then concatenates the two parts:
//
//   partial_rope(x) = concat( x[..., :d_nope] , rope(x[..., d_nope:]) )
//
// where d_nope = (last dim) - rope_params.head_dim. V4 rotates the last 64 dims of
// each query / KV-entry vector. The input's last dim must be >= the rope dim, and
// both the untouched prefix and the rotated suffix must be tile-aligned (the
// underlying slice/concat operate on the last axis).
//
// `token_position` is the position offset for the first sequence element (forwarded
// to rope), so passing a negative-rotation `rope_params` (built from the
// neg_cos/neg_sin caches) implements the V4 "-i" output trick.
autograd::TensorPtr partial_rope(
    const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params, uint32_t token_position);

}  // namespace ttml::ops
