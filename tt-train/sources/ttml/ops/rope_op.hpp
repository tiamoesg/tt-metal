// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

// Forward declaration to avoid heavy include in files that only need the pointer type
namespace ttnn::distributed {
class TensorToMesh;
}  // namespace ttnn::distributed

namespace ttml::ops {

struct RopeScalingParams {
    uint32_t original_context_length = 2048U;
    float scaling_factor = 1.0F;
    float high_freq_factor = 1.0F;
    float low_freq_factor = 1.0F;
};

struct RotaryEmbeddingParams {
    ttnn::Tensor cos_cache{};
    ttnn::Tensor sin_cache{};
    ttnn::Tensor neg_cos_cache{};
    ttnn::Tensor neg_sin_cache{};
    ttnn::Tensor trans_mat{};

    uint32_t sequence_length = 0U;
    uint32_t head_dim = 0U;
    float theta = 10000.0F;

    RopeScalingParams rope_scaling_params;
};

autograd::TensorPtr rope(
    const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params, const uint32_t token_position);

// --- DeepSeek-V4 rope-params transforms (§2.3.3) -------------------------------
// All three return a new params view (sharing trans_mat) with sliced/swapped trig
// caches; the underlying rope() op is unchanged.

// Prefix: the first `count` positions [0, count). Use for queries / window KV /
// outputs whose positions are the contiguous token indices 0..count-1.
RotaryEmbeddingParams rope_params_prefix(const RotaryEmbeddingParams& params, uint32_t count);

// Strided: positions {0, stride, 2*stride, ...} for `count` entries -- a compressed
// block s is rotated at its representative token position s*stride (model.py
// `freqs_cis[:cutoff:ratio]`).
RotaryEmbeddingParams rope_params_strided(const RotaryEmbeddingParams& params, uint32_t stride, uint32_t count);

// Inverse: rotate by -theta (apply_rotary_emb(..., inverse=True)). Used for the
// "-i" trick on the core-attention outputs so the value-carried absolute positions
// become relative.
RotaryEmbeddingParams rope_params_inverse(const RotaryEmbeddingParams& params);

std::pair<ttnn::Tensor, ttnn::Tensor> gen_freqs(
    uint32_t head_dim,
    uint32_t sequence_length,
    float theta,
    const RopeScalingParams& rope_scaling_params,
    const ttnn::distributed::TensorToMesh* mesh_mapper = nullptr);

ttnn::Tensor gen_trans_mat();

RotaryEmbeddingParams build_rope_params(
    uint32_t sequence_length,
    uint32_t head_dim,
    float theta = 10000.0F,
    RopeScalingParams rope_scaling_params = RopeScalingParams{});
// Throws an exception if the input is bad, parameters are bad, or the two are
// incompatible with one another.
void validate_rope_input_and_params(const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params);

}  // namespace ttml::ops
