# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.nightly.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # Normal reduction cases are when channels <= 8 * 32 and kernel_hw <= 16
        # Wide reduction cases channels > 8 * 32
        # Large reduction cases (channels < 32 and kernel_hw > 16) or (channels > 32 and kernel_hw > 32)
        [2, 32, 16, 16],
        [1, 512, 112, 32],
        [1, 320, 48, 48],
        [1, 320, 47, 47],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        # Wide and normal reductions go to normal kernels
        # Large reductions go to large kernels
        # Reductions which are large and wide at the same time
        # go to large kernels
        (3, 3),
        (9, 9),
        (36, 36),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize(
    "padding",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "count_include_pad",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "divisor_override",
    [
        None,
        5,
    ],
)
@pytest.mark.parametrize(
    "shard_scheme",
    [
        None,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "use_program_cache",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
def test_avg_pool2d_post_commit(
    device,
    tensor_map,
    input_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    use_program_cache,
    divisor_override,
    count_include_pad,
    shard_scheme,
    dtype,
):
    # we only want to test the largest kernel size with a specific input shape
    # to test otherwise untouched paths in the large kernel, other shapes run OOM
    # or will just slow the test down doing redundant work
    if kernel_size == (36, 36) and input_shape != [1, 320, 48, 48] and input_shape != [1, 320, 47, 47]:
        pytest.skip("Skipping, only run shapes [1, 320, 48, 48] and [1, 320, 47, 47] with kernel size (36, 36)")
    if dtype == ttnn.bfloat8_b and input_shape != [1, 320, 48, 48] and input_shape != [1, 320, 47, 47]:
        pytest.skip("Skipping, only run shapes [1, 320, 48, 48] and [1, 320, 47, 47] with bfloat8_b dtype")
    run_avg_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        shard_scheme=shard_scheme,
        dtype=dtype,
    )
