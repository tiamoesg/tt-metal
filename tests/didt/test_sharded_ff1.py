# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Legacy file calling new test files to keep old commands behavior.

import pytest

import ttnn
from tests.didt.test_ff1_matmul import test_ff1_matmul, test_specific_chip_ff1_matmul
from models.utility_functions import skip_for_blackhole, is_blackhole

MATH_FIDELITY = ttnn.MathFidelity.HiFi2
GELU = True


@pytest.mark.parametrize(
    # to keep the legacy commands that used this parametrization id working
    "ff1_hang_dummy_param",
    (None,),
    ids=["ff1-hang"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
    ],
    indirect=["mesh_device"],
)
def test_reproduce_matmul_2d_hang(mesh_device, ff1_hang_dummy_param, didt_workload_iterations):
    if is_blackhole() and mesh_device.get_num_devices() > 1:
        pytest.skip("Multi-chip Blackhole has not been tested")
    test_ff1_matmul(mesh_device, GELU, MATH_FIDELITY, didt_workload_iterations, -1)


@skip_for_blackhole("Multi-chip Blackhole has not been tested")
@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_specific_chip_reproduce_matmul_2d_hang(mesh_device, logical_chip_index, didt_workload_iterations):
    test_specific_chip_ff1_matmul(
        mesh_device, logical_chip_index, GELU, MATH_FIDELITY, didt_workload_iterations, -1, False
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
    ],
    indirect=["mesh_device"],
)
def test_determinism(mesh_device, didt_workload_iterations, determinism_check_interval):
    if is_blackhole() and mesh_device.get_num_devices() > 1:
        pytest.skip("Multi-chip Blackhole has not been tested")

    if determinism_check_interval == -1:
        determinism_check_interval = 1

    test_ff1_matmul(mesh_device, GELU, MATH_FIDELITY, didt_workload_iterations, determinism_check_interval, False)


@skip_for_blackhole("Multi-chip Blackhole has not been tested")
@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_determinism_specific_chip(
    mesh_device, logical_chip_index, didt_workload_iterations, determinism_check_interval
):
    if determinism_check_interval == -1:
        determinism_check_interval = 1

    test_specific_chip_ff1_matmul(
        mesh_device,
        logical_chip_index,
        GELU,
        MATH_FIDELITY,
        didt_workload_iterations,
        determinism_check_interval,
        False,
    )
