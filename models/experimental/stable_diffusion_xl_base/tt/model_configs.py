# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import re


class ModelOptimisations:
    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat16,
        attention_weights_dtype=ttnn.bfloat16,
        ff_weights_dtype=ttnn.bfloat8_b,
    ):
        self.conv_configs = {}
        self.conv_output_dtype = conv_act_dtype
        self.matmul_configs = {}
        self.compute_configs = {}
        self.prepared_weights = False
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b
        self.attention_weights_dtype = attention_weights_dtype
        self.ff_weights_dtype = ff_weights_dtype

        # HEIGHT SHARDED
        self.conv_configs["ABH_256_ADB"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )
        self.conv_configs["ABH_128_NO_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )

        # BLOCK SHARDED
        self.conv_configs["ABH_0_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )

        self.conv_configs["ABH_0_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )

        self.conv_configs["ABH_0_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )
        self.conv_configs["ABH_32_NO_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32,
        )
        self.conv_configs["ABH_64_NO_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
        )
        self.conv_configs["ABH_64_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
        )
        self.conv_configs["ABH_64_NO_ADB_WDB_MOVE_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
        )
        self.conv_configs["ABH_64_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
        )

        self.conv_configs["ABH_0_ADB_WDB_BS_BF16"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )

        self.conv_configs["ABH_128_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_128_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_128_NO_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_128_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_128_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_128_NO_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_128_ADB_WDB_MOVE_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_256_NO_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )
        self.conv_configs["ABH_256_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )
        self.conv_configs["ABH_256_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.conv_configs["ABH_256_ADB_WDB_BS_NO_MOVE"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.conv_configs["ABH_256_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.conv_configs["ABH_512_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
        )

        self.conv_configs["ABH_512_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
        )

        self.conv_configs["ABH_512_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
        )

        # WIDTH SHARDED
        self.conv_configs["ABH_256_NO_ADB_WS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )
        self.conv_configs["ABH_512_NO_ADB_WS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
        )

        # DEFAULT CONF
        self.conv_configs["DEFAULT"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )

        # DRAM CONF
        self.conv_configs["ABH_64_NO_ADB_DRAM"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.conv_configs["ABH_128_NO_ADB_DRAM"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.conv_configs["ABH_512_NO_ADB_DRAM"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.conv_configs["DEFAULT_DRAM"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.conv_configs["ABH_256_NO_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_4096"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=1,  # max is 20, 1 seems optimal?
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=8,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # max is 40, 4 seems optimal?
            per_core_M=4,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF2_SEQ_LEN_1024"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=10,  # max is 20, 10 seems optimal
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=4,
            per_core_N=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF2_SEQ_LEN_4096"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=10,  # max is 10
            out_subblock_h=1,
            out_subblock_w=3,
            per_core_M=16,
            per_core_N=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_RESNET_LINEAR"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=10,  # max is 40, 10 seems optimal
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        in_0_block_w_geglu_640 = 5
        per_core_M_geglu_640 = 16
        per_core_N_geglu_640 = 10
        out_subblock_h_geglu_640 = 1
        out_subblock_w_geglu_640 = 5
        self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=in_0_block_w_geglu_640,
            per_core_M=per_core_M_geglu_640,
            per_core_N=per_core_N_geglu_640,
            out_subblock_h=out_subblock_h_geglu_640,
            out_subblock_w=out_subblock_w_geglu_640,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT_GELU"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=in_0_block_w_geglu_640,
            per_core_M=per_core_M_geglu_640,
            per_core_N=per_core_N_geglu_640,
            out_subblock_h=out_subblock_h_geglu_640,
            out_subblock_w=out_subblock_w_geglu_640,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, True],
        )

        in_0_block_w_geglu_1280 = 5  # max is 5
        per_core_M_geglu_1280 = 4
        per_core_N_geglu_1280 = 20
        out_subblock_h_geglu_1280 = 1
        out_subblock_w_geglu_1280 = 5
        self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=in_0_block_w_geglu_1280,
            per_core_M=per_core_M_geglu_1280,
            per_core_N=per_core_N_geglu_1280,
            out_subblock_h=out_subblock_h_geglu_1280,
            out_subblock_w=out_subblock_w_geglu_1280,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT_GELU"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=in_0_block_w_geglu_1280,
            per_core_M=per_core_M_geglu_1280,
            per_core_N=per_core_N_geglu_1280,
            out_subblock_h=out_subblock_h_geglu_1280,
            out_subblock_w=out_subblock_w_geglu_1280,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, True],
        )

        self.matmul_configs["2D_TM_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=8,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_TM_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=5,
            per_core_M=4,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_ATTN_OUT_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=2,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_CONV_320_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=2,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        self.matmul_configs["2D_ATTN_OUT_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=5,
            per_core_M=4,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_CONV_640_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            per_core_M=4,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_ATTN_QKV_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            per_core_M=16,
            per_core_N=8,
            out_subblock_h=1,
            out_subblock_w=8,
            transpose_mcast=False,
            fused_activation=None,
        )

        # 21 cores, [1, 1, 96, 2048] x [1, 1, 2048, 640]
        self.matmul_configs["2D_ATTEN_K_V_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # max is 64, 4 seems optimal
            per_core_M=1,
            per_core_N=3,
            out_subblock_h=1,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        # 40 cores, [1, 1, 96, 2048] x [1, 1, 2048, 1280]
        self.matmul_configs["1D_ATTEN_K_V_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=16,  # max is 64, 16 seems optimal
            out_subblock_h=3,
            out_subblock_w=1,
            per_core_M=3,
            per_core_N=1,
            mcast_in0=True,
            fuse_batch=True,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_CONV_2560_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            per_core_M=4,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_ATTN_QKV_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            per_core_M=4,
            per_core_N=15,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        self.matmul_configs["2D_RESNET_CONV_1920_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=5,
            per_core_M=4,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        self.matmul_configs["2D_RESNET_CONV_1920_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=3,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=2,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_CONV_1280_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=2,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_CONV_960_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=1,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_RESNET_CONV_960_320"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=8,
            per_core_N=10,
            mcast_in0=False,
            gather_in0=False,
            fuse_batch=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_RESNET_CONV_640_320"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=8,
            per_core_N=10,
            mcast_in0=False,
            gather_in0=False,
            fuse_batch=False,
            fused_activation=None,
        )

        self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_configs["MATH_APPROX_MM_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def get_matmul_config(self, matmul_path):
        if matmul_path is None:
            return None

        if not ("decoder" in matmul_path):
            # # # RESNET CONV MM # # #
            if "conv_shortcut" in matmul_path:
                if "down_blocks.1" in matmul_path:
                    return self.matmul_configs["2D_RESNET_CONV_320_640"]
                if "down_blocks.2" in matmul_path:
                    return self.matmul_configs["2D_RESNET_CONV_640_1280"]
                if "up_blocks.0.resnets.0" in matmul_path or "up_blocks.0.resnets.1" in matmul_path:
                    return self.matmul_configs["2D_RESNET_CONV_2560_1280"]
                if "up_blocks.0.resnets.2" in matmul_path:
                    return self.matmul_configs["2D_RESNET_CONV_1920_1280"]
                if "up_blocks.1.resnets.0" in matmul_path:
                    return self.matmul_configs["2D_RESNET_CONV_1920_640"]
                if "up_blocks.1.resnets.1" in matmul_path:
                    return self.matmul_configs["2D_RESNET_CONV_1280_640"]
                if "up_blocks.1.resnets.2" in matmul_path:
                    return self.matmul_configs["2D_RESNET_CONV_960_640"]
                if "up_blocks.2.resnets.0" in matmul_path:
                    return self.matmul_configs["1D_RESNET_CONV_960_320"]
                if "up_blocks.2.resnets.1" in matmul_path or "up_blocks.2.resnets.2" in matmul_path:
                    return self.matmul_configs["1D_RESNET_CONV_640_320"]
                else:
                    return None

            # # # GEGLU # # #
            if "net.0.proj" in matmul_path:
                if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                    if "gelu" in matmul_path:
                        return self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT_GELU"]
                    else:
                        return self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT"]

                else:
                    if "gelu" in matmul_path:
                        return self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT_GELU"]
                    else:
                        return self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT"]

            # # # TM LINEAR # # #
            if "proj_in" in matmul_path or "proj_out" in matmul_path:
                if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                    return self.matmul_configs["2D_TM_LINEAR_640"]
                else:
                    return self.matmul_configs["2D_TM_LINEAR_1280"]

            # # # ATTN OUT LINEAR # # #
            if "attn1.to_out" in matmul_path or "attn2.to_out" in matmul_path or "attn2.to_q" in matmul_path:
                if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                    return self.matmul_configs["2D_ATTN_OUT_LINEAR_640"]
                else:
                    return self.matmul_configs["2D_ATTN_OUT_LINEAR_1280"]
            if "attn1.to_q" in matmul_path:
                if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                    return self.matmul_configs["2D_ATTN_QKV_LINEAR_640"]
                else:
                    return self.matmul_configs["2D_ATTN_QKV_LINEAR_1280"]
            if (
                "attn1.to_k" in matmul_path
                or "attn1.to_v" in matmul_path
                or "attn2.to_k" in matmul_path
                or "attn2.to_v" in matmul_path
            ):
                if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                    return self.matmul_configs["2D_ATTEN_K_V_LINEAR_640"]
                else:
                    return self.matmul_configs["1D_ATTEN_K_V_LINEAR_1280"]

            # # # Down block 1 # # #
            pattern_downn_block_1_dense_out = re.compile(
                r"down_blocks\.1\.attentions\.[01]\.transformer_blocks\.[01]\.attn[12]\.dense_out"
            )

            # 8 occurences
            if pattern_downn_block_1_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_4096"]

            pattern_down_blocks_1_ff2 = re.compile(
                r"down_blocks\.1\.attentions\.[01]\.transformer_blocks\.[01]\.ff\.net\.2"
            )

            # 4 occurences
            if pattern_down_blocks_1_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_4096"]

            # # # Down block 2 # # #
            pattern_down_blocks_2_dense_out = re.compile(
                r"down_blocks\.2\.attentions\.[01]\.transformer_blocks\.[0123456789]\.attn[12]\.dense_out"
            )

            # 40 occurences
            if pattern_down_blocks_2_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"]

            pattern_down_blockcs_2_ff2 = re.compile(
                r"down_blocks\.2\.attentions\.[01]\.transformer_blocks\.[0123456789]\.ff\.net\.2"
            )

            # 20 occurences
            if pattern_down_blockcs_2_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

            # # # Mid block  # # #
            pattern_mid_block_ff2 = re.compile(
                r"mid_block\.attentions\.0\.transformer_blocks\.[0123456789]\.ff\.net\.2"
            )

            # 10 occurences
            if pattern_mid_block_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

            pattern_mid_block_dense_out = re.compile(
                r"mid_block\.attentions\.0\.transformer_blocks\.[0123456789]\.attn[12]\.dense_out"
            )

            # 20 occurences
            if pattern_mid_block_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"]

            # # # Up block 0 # # #
            pattern_up_blocks_0_dense_out = re.compile(
                r"up_blocks\.0\.attentions\.[012]\.transformer_blocks\.[0123456789]\.attn[12]\.dense_out"
            )

            # 60 occurences
            if pattern_up_blocks_0_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"]

            pattern_up_blocks_0_ff2 = re.compile(
                r"up_blocks\.0\.attentions\.[012]\.transformer_blocks\.[0123456789]\.ff\.net\.2"
            )

            # 30 occurences
            if pattern_up_blocks_0_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

            # # # Up block 1 # # #
            pattern_up_blocks_1_dense_out = re.compile(
                r"up_blocks\.1\.attentions\.[012]\.transformer_blocks\.[01]\.attn[12]\.dense_out"
            )

            # 12 occurences
            if pattern_up_blocks_1_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_4096"]

            pattern_up_blocks_1_ff2 = re.compile(
                r"up_blocks\.1\.attentions\.[012]\.transformer_blocks\.[01]\.ff\.net\.2"
            )

            # 6 occurences
            if pattern_up_blocks_1_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_4096"]

            pattern_resnet_linear = re.compile(
                r"(down_blocks\.[012]\.resnets\.[01]\.linear|up_blocks\.[012]\.resnets\.[012]\.linear|mid_block\.resnets\.[01]\.linear)"
            )

            if pattern_resnet_linear.search(matmul_path):
                return self.matmul_configs["1D_RESNET_LINEAR"]
        return None

    def get_mm_compute_config(self, module_path):
        # for now, return default config
        if ".to_q" in module_path:
            return self.compute_configs["MATH_APPROX_MM_COMPUTE_CONFIG"]
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

    def get_conv_config(self, conv_path):
        if conv_path is None:
            return None

        if not ("decoder" in conv_path):
            if "conv_in" == conv_path:
                return self.conv_configs["ABH_256_ADB"]

            # DOWN BLOCK 0
            elif ("down_blocks.0.resnets" in conv_path) and ("conv2" in conv_path):
                return self.conv_configs["ABH_512_ADB_WDB_BS"]
            elif "down_blocks.0.resnets" in conv_path:
                return self.conv_configs["ABH_256_ADB_WDB_BS_NO_MOVE"]
            elif "down_blocks.0.downsamplers.0" == conv_path:
                return self.conv_configs["ABH_512_ADB_WDB_NO_DEALLOC_BS"]

            # DOWN BLOCK 1
            elif "down_blocks.1.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_512_ADB_WDB_BS"]
            elif ("down_blocks.1.resnets.0.conv2" == conv_path) or ("down_blocks.1.resnets.1" in conv_path):
                return self.conv_configs["ABH_0_ADB_WDB_BS"]
            elif "down_blocks.1.downsamplers.0" == conv_path:
                return self.conv_configs["ABH_0_ADB_WDB_NO_DEALLOC_BS"]

            # DOWN BLOCK 2
            elif "down_blocks.2.resnets.1.conv1" == conv_path:
                return self.conv_configs["ABH_0_ADB_WDB_BS"]
            elif "down_blocks.2.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_0_ADB_WDB_BS"]
            elif ("down_blocks.2.resnets.0.conv2" == conv_path) or ("down_blocks.2.resnets.1.conv2" == conv_path):
                return self.conv_configs["ABH_0_ADB_WDB_BS"]

            # MID BLOCK
            elif "mid_block" in conv_path:
                return self.conv_configs["ABH_0_ADB_WDB_BS"]

            # UP BLOCK 0
            elif ("up_blocks.0.resnets.0.conv1" == conv_path) or ("up_blocks.0.resnets.1.conv1" == conv_path):
                return self.conv_configs["ABH_0_ADB_WDB_BS"]
            elif "up_blocks.0.upsamplers.0" == conv_path:
                return self.conv_configs["ABH_256_ADB_WDB_BS"]
            elif ("up_blocks.0.resnets" in conv_path) and ("conv2" in conv_path):
                return self.conv_configs["ABH_0_ADB_WDB_BS"]
            elif "up_blocks.0.resnets.2.conv1" == conv_path:
                return self.conv_configs["ABH_0_ADB_WDB_BS"]

            # UP BLOCK 1
            elif "up_blocks.1.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_64_ADB_WDB_BS"]
            elif "up_blocks.1.resnets.1.conv1" == conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_BS"]
            elif "up_blocks.1.resnets.2.conv1" == conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_MOVE_BS"]
            elif ("up_blocks.1.resnets" in conv_path) and ("conv2" in conv_path):
                return self.conv_configs["ABH_0_ADB_WDB_BS"]
            elif "up_blocks.1.upsamplers.0" == conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_BS"]

            # UP BLOCK 2
            elif "up_blocks.2.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_256_ADB_WDB_BS"]
            elif ("up_blocks.2.resnets" in conv_path) and ("conv2" in conv_path):
                return self.conv_configs["ABH_512_ADB_WDB_BS"]
            elif ("up_blocks.2.resnets.1.conv1" == conv_path) or ("up_blocks.2.resnets.2.conv1" == conv_path):
                return self.conv_configs["ABH_64_ADB_WDB_BS"]

            elif "conv_out" == conv_path:
                return self.conv_configs["ABH_128_NO_ADB_HS"]
            else:
                return self.conv_configs["DEFAULT"]
        else:
            # VAE
            if "decoder.conv_in" == conv_path:
                return self.conv_configs["ABH_256_NO_ADB_HS"]
            elif ("decoder.up_blocks.2.resnet.0" in conv_path) and ("conv1" in conv_path):
                return self.conv_configs["ABH_128_NO_ADB_DRAM"]
            elif ("decoder.up_blocks.2.resnet" in conv_path) and ("conv1" in conv_path):
                return self.conv_configs["ABH_64_NO_ADB_DRAM"]  # should be 128, OOM in demo
            # elif ("decoder.up_blocks.2.resnet" in conv_path) and ("conv2" in conv_path):
            #     return self.conv_configs["ABH_32_NO_ADB_DRAM"] # Note: ABH should be 128 (OOM)
            elif "decoder.up_blocks.2.upsamplers.0" == conv_path:
                return self.conv_configs["ABH_64_NO_ADB_DRAM"]  # should be 128, OOM in demo
            elif "decoder.conv_out" == conv_path:
                return self.conv_configs["ABH_512_NO_ADB_DRAM"]
            else:
                return self.conv_configs["DEFAULT_DRAM"]

    def get_conv_output_dtype(self):
        return self.conv_output_dtype
