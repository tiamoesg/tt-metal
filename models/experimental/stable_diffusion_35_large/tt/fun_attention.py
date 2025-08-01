# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .fun_linear import sd_linear, TtLinearParameters
from .fun_normalization import sd_rms_norm, TtRmsNormParameters
from .substate import has_substate, substate
from .parallel_config import DiTParallelConfig, StableDiffusionParallelManager
from .utils import unpadded_all_gather_async


@dataclass
class TtAttentionPartParameters:
    qkv_proj: TtLinearParameters
    norm_q: TtRmsNormParameters
    norm_k: TtRmsNormParameters
    out_proj: TtLinearParameters | None


@dataclass
class TtAttentionParameters:
    spatial: TtAttentionPartParameters
    prompt: TtAttentionPartParameters | None
    eps: float = 1e-6

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        num_heads: int,
        unpadded_num_heads: int,
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_config: DiTParallelConfig,
    ) -> TtAttentionParameters:
        spatial_qkv_proj = _merge_qkv_proj(substate(state, "to_q"), substate(state, "to_k"), substate(state, "to_v"))
        prompt_qkv_proj = _merge_qkv_proj(
            substate(state, "add_q_proj"), substate(state, "add_k_proj"), substate(state, "add_v_proj")
        )
        n_local_heads = num_heads // parallel_config.tensor_parallel.factor
        return cls(
            spatial=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch_col_parallel(
                    state=spatial_qkv_proj,
                    n_local_heads=n_local_heads,
                    unpadded_num_heads=unpadded_num_heads,
                    hidden_dim_padding=hidden_dim_padding,
                    dtype=dtype,
                    device=device,
                    parallel_config=parallel_config,
                ),
                norm_q=TtRmsNormParameters.from_torch(
                    substate(state, "norm_q"), dtype=dtype, device=device, eps=cls.eps
                ),
                norm_k=TtRmsNormParameters.from_torch(
                    substate(state, "norm_k"), dtype=dtype, device=device, eps=cls.eps
                ),
                out_proj=TtLinearParameters.from_torch(
                    substate(state, "to_out.0"),
                    dtype=dtype,
                    device=device,
                    shard_dim=-1,
                    hidden_dim_padding=hidden_dim_padding,
                    parallel_config=parallel_config,
                ),
            ),
            prompt=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch_col_parallel(
                    state=prompt_qkv_proj,
                    n_local_heads=n_local_heads,
                    unpadded_num_heads=unpadded_num_heads,
                    hidden_dim_padding=hidden_dim_padding,
                    dtype=dtype,
                    device=device,
                    parallel_config=parallel_config,
                ),
                norm_q=TtRmsNormParameters.from_torch(
                    substate(state, "norm_added_q"), dtype=dtype, device=device, eps=cls.eps
                ),
                norm_k=TtRmsNormParameters.from_torch(
                    substate(state, "norm_added_k"), dtype=dtype, device=device, eps=cls.eps
                ),
                out_proj=TtLinearParameters.from_torch(
                    substate(state, "to_add_out"),
                    dtype=dtype,
                    device=device,
                    shard_dim=-1,
                    hidden_dim_padding=hidden_dim_padding,
                    parallel_config=parallel_config,
                )
                if has_substate(state, "to_add_out")
                else None,
            )
            if has_substate(state, "add_q_proj")
            else None,
        )


def _merge_qkv_proj(
    q_state: dict[str, torch.Tensor],
    k_state: dict[str, torch.Tensor],
    v_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.cat([q_state["weight"], k_state["weight"], v_state["weight"]]) if "weight" in q_state else None,
        "bias": torch.cat([q_state["bias"], k_state["bias"], v_state["bias"]]) if "bias" in q_state else None,
    }


def sd_attention_qkv(
    x: ttnn.Tensor,
    parameters: TtAttentionPartParameters,
    parallel_config: DiTParallelConfig,
    *,
    num_heads: int,
    deallocate: bool,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    # tracy.signpost("enter TtAttentionPart")

    _batch_size, sequence_length, _embedding_dim = x.shape

    # Input sharding
    if sequence_length > 1024:
        # sharding leads to worse PCC, so disable it until further investigation
        mm_a_x = 8
        mm_a_y = 8
        mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG
    elif sequence_length >= 512:
        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG
        deallocate = True
    else:
        mm_a_x = 8
        mm_a_y = 6
        mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG

    qkv = sd_linear(
        x,
        parameters.qkv_proj,
        memory_config=mm_a_x_memory_config,
        core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        dtype=ttnn.bfloat16,
        deallocate=deallocate,
    )

    num_local_heads = num_heads // parallel_config.tensor_parallel.factor
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=num_local_heads, transpose_key=False
    )

    # ttnn.deallocate(qkv)

    q = sd_rms_norm(q, parameters.norm_q, deallocate=True)
    k = sd_rms_norm(k, parameters.norm_k, deallocate=True)

    return q, k, v


def sd_attention_out_proj(x: ttnn.Tensor, parameters: TtAttentionPartParameters) -> ttnn.Tensor:
    if parameters.out_proj is None:
        return x

    grid_size = x.device().compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)
    return sd_linear(x, parameters.out_proj, core_grid=core_grid)


def sd_joint_attention(
    *,
    spatial: ttnn.Tensor,
    prompt: ttnn.Tensor | None = None,
    parameters: TtAttentionParameters,
    parallel_manager: StableDiffusionParallelManager,
    deallocate: bool = False,
    num_heads: int,  # TODO: should be a model parameter
    N: int,
    L: int,
    cfg_index: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
    """
    spatial: N ⊗ S1 ⊗ (H * E1)
    prompt: N ⊗ S2 ⊗ (H * E2)
    """
    device = spatial.device()

    spatial = ttnn.squeeze(spatial, 1)
    prompt = ttnn.squeeze(prompt, 1)

    q, k, v = sd_attention_qkv(
        spatial,
        parameters=parameters.spatial,
        parallel_config=parallel_manager.dit_parallel_config,
        num_heads=num_heads,
        deallocate=deallocate,
    )

    full_grid = device.compute_with_storage_grid_size()
    sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_worker_grid,
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=False,  # NOTE: False is more correct
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        # MathFidelity.LoFi results in bad image quality.
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
    )

    if prompt is None:
        assert True, "This has not been updated for SP + TP changes"
        # operands must be in DRAM
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        # ttnn.deallocate(q)
        # ttnn.deallocate(k)
        # ttnn.deallocate(v)

        concatenated_attn = ttnn.transformer.concatenate_heads(
            attn,
        )
        # ttnn.deallocate(attn)

        spatial = sd_attention_out_proj(concatenated_attn, parameters.spatial)

        spatial = ttnn.unsqueeze(spatial, 1)
        return spatial, None

    assert parameters.prompt is not None

    q2, k2, v2 = sd_attention_qkv(
        prompt,
        parameters=parameters.prompt,
        parallel_config=parallel_manager.dit_parallel_config,
        num_heads=num_heads,
        deallocate=deallocate,
    )

    if parallel_manager.is_ring_parallel:
        spatial, prompt, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            q2,
            k2,
            v2,
            persistent_output_buffer_k=parallel_manager.persistent_buffers[cfg_index]["K_gathered"],
            persistent_output_buffer_v=parallel_manager.persistent_buffers[cfg_index]["V_gathered"],
            joint_strategy="rear",
            logical_n=N,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=parallel_manager.cfg_semaphores[cfg_index]["ring_sdpa"],
            num_links=parallel_manager.num_links,
            cluster_axis=parallel_manager.dit_parallel_config.ring_parallel.mesh_axis,
            mesh_device=device,
            topology=parallel_manager.dit_parallel_config.topology,
            subdevice_id=parallel_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(0, sdpa_worker_grid[1]),
        )
    else:
        # TODO: Check that unpadded text seqlen is logical shape of joint tensors.
        spatial, prompt = ttnn.transformer.joint_scaled_dot_product_attention(
            q,
            k,
            v,
            q2,
            k2,
            v2,
            joint_strategy="rear",
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    spatial = ttnn.transformer.concatenate_heads(
        spatial,
    )
    prompt = ttnn.transformer.concatenate_heads(
        prompt,
    )

    spatial = ttnn.unsqueeze(spatial, 1)
    prompt = ttnn.unsqueeze(prompt, 1)

    if parallel_manager.is_ulysses_parallel:
        spatial = ttnn.experimental.all_gather_async(
            spatial,
            dim=3,
            num_links=parallel_manager.num_links,
            cluster_axis=parallel_manager.dit_parallel_config.ulysses_parallel.mesh_axis,
            mesh_device=device,
            topology=parallel_manager.dit_parallel_config.topology,
            persistent_output_tensor=parallel_manager.get_ping_pong_buffer(cfg_index, "spatial_buffer"),
            multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(cfg_index),
        )
        prompt = unpadded_all_gather_async(
            prompt,
            dim=3,
            num_links=parallel_manager.num_links,
            cluster_axis=parallel_manager.dit_parallel_config.ulysses_parallel.mesh_axis,
            mesh_device=device,
            topology=parallel_manager.dit_parallel_config.topology,
            persistent_output_tensor=parallel_manager.get_ping_pong_buffer(cfg_index, "prompt_buffer"),
            multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(cfg_index),
        )

    spatial = sd_attention_out_proj(spatial, parameters.spatial)
    prompt = sd_attention_out_proj(prompt, parameters.prompt)

    return spatial, prompt
