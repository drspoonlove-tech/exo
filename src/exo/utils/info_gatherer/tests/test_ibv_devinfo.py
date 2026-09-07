from pathlib import Path

import pytest

from exo.utils.info_gatherer.ibv_devinfo import (
    IbvDevinfoCommandResult,
    IbvDevinfoStatus,
    claimed_rdma_device_names_from_sysfs,
    evaluate_ibv_devinfo,
    parse_ibv_devinfo_output,
)

ACTIVE_DEVICE_STDOUT = """\
hca_id:	rdma_en2
	transport:			InfiniBand (0)
	fw_ver:				1.0.0
	phys_port_cnt:			1
		port:	1
			state:			PORT_ACTIVE (4)
			max_mtu:		4096 (5)
			link_layer:		InfiniBand
"""

DOWN_PORT_STDOUT = """\
hca_id:	rdma_en2
	transport:			InfiniBand (0)
	phys_port_cnt:			1
		port:	1
			state:			PORT_DOWN (1)
			link_layer:		Ethernet
"""

TWO_DEVICE_STDOUT = """\
hca_id: mlx5_0
	transport:			InfiniBand (0)
		port:	1
			state:			PORT_ACTIVE (4)
			link_layer:		InfiniBand

hca_id: mlx5_1
	transport:			InfiniBand (0)
		port:	1
			state:			PORT_ACTIVE (4)
			link_layer:		InfiniBand
"""


def test_parse_ibv_devinfo_output_extracts_device_and_port():
    devices = parse_ibv_devinfo_output(ACTIVE_DEVICE_STDOUT)
    assert len(devices) == 1
    assert devices[0].name == "rdma_en2"
    assert devices[0].transport == "InfiniBand (0)"
    assert len(devices[0].ports) == 1
    assert devices[0].ports[0].port_number == 1
    assert devices[0].ports[0].state == "PORT_ACTIVE"
    assert devices[0].ports[0].link_layer == "InfiniBand"


def test_parse_ibv_devinfo_output_empty_text():
    assert parse_ibv_devinfo_output("") == ()
    assert parse_ibv_devinfo_output("No IB devices found\n") == ()


def test_evaluate_happy_path_with_claimed_device():
    status = evaluate_ibv_devinfo(
        binary_path="/usr/bin/ibv_devinfo",
        command_result=IbvDevinfoCommandResult(
            returncode=0, stdout=ACTIVE_DEVICE_STDOUT.encode(), stderr=b""
        ),
        claimed_device_names=("rdma_en2",),
    )
    assert status.ok is True
    assert status.failure is None
    assert status.missing_claimed_devices == ()
    assert status.devices[0].name == "rdma_en2"


def test_evaluate_missing_binary():
    status = evaluate_ibv_devinfo(
        binary_path=None,
        command_result=None,
        claimed_device_names=("rdma_en2",),
    )
    assert status.ok is False
    assert status.failure == "ibv_devinfo not found in PATH"
    assert status.devices == []
    assert status.missing_claimed_devices == ("rdma_en2",)


def test_evaluate_empty_devices_nonzero_exit():
    status = evaluate_ibv_devinfo(
        binary_path="/usr/bin/ibv_devinfo",
        command_result=IbvDevinfoCommandResult(
            returncode=1,
            stdout=b"",
            stderr=b"No IB devices found\n",
        ),
    )
    assert status.ok is False
    assert status.failure is not None
    assert "no RDMA devices" in status.failure
    assert "No IB devices found" in status.failure


def test_evaluate_bad_port():
    status = evaluate_ibv_devinfo(
        binary_path="/usr/bin/ibv_devinfo",
        command_result=IbvDevinfoCommandResult(
            returncode=0, stdout=DOWN_PORT_STDOUT.encode(), stderr=b""
        ),
        claimed_device_names=("rdma_en2",),
    )
    assert status.ok is False
    assert status.failure is not None
    assert "no usable port" in status.failure
    assert "PORT_DOWN" in status.failure
    assert status.devices[0].ports[0].state == "PORT_DOWN"


def test_evaluate_claimed_device_absent_from_ibv_devinfo():
    status = evaluate_ibv_devinfo(
        binary_path="/usr/bin/ibv_devinfo",
        command_result=IbvDevinfoCommandResult(
            returncode=0, stdout=TWO_DEVICE_STDOUT.encode(), stderr=b""
        ),
        claimed_device_names=("rdma_en2",),
    )
    assert status.ok is False
    assert status.missing_claimed_devices == ("rdma_en2",)
    assert status.failure is not None
    assert "rdma_en2" in status.failure


def test_claimed_rdma_device_names_from_sysfs(tmp_path: Path):
    missing = tmp_path / "missing"
    assert claimed_rdma_device_names_from_sysfs(missing) == ()

    infiniband = tmp_path / "infiniband"
    infiniband.mkdir()
    (infiniband / "mlx5_1").mkdir()
    (infiniband / "mlx5_0").mkdir()
    (infiniband / "readme.txt").write_text("not a device")
    assert claimed_rdma_device_names_from_sysfs(infiniband) == ("mlx5_0", "mlx5_1")


@pytest.mark.anyio
async def test_gather_happy_path_mocked_process():
    async def run_ibv_devinfo(_binary_path: str) -> IbvDevinfoCommandResult:
        return IbvDevinfoCommandResult(
            returncode=0, stdout=ACTIVE_DEVICE_STDOUT.encode(), stderr=b""
        )

    gathered = await IbvDevinfoStatus.gather(
        claimed_device_names=("rdma_en2",),
        locate_binary=lambda _name: "/usr/bin/ibv_devinfo",
        run_ibv_devinfo=run_ibv_devinfo,
    )
    assert gathered.status.ok is True
    assert gathered.status.devices[0].name == "rdma_en2"


@pytest.mark.anyio
async def test_gather_missing_binary():
    gathered = await IbvDevinfoStatus.gather(
        claimed_device_names=("rdma_en2",),
        locate_binary=lambda _name: None,
    )
    assert gathered.status.ok is False
    assert gathered.status.failure == "ibv_devinfo not found in PATH"


@pytest.mark.anyio
async def test_gather_empty_devices():
    async def run_ibv_devinfo(_binary_path: str) -> IbvDevinfoCommandResult:
        return IbvDevinfoCommandResult(returncode=0, stdout=b"", stderr=b"")

    gathered = await IbvDevinfoStatus.gather(
        locate_binary=lambda _name: "/usr/bin/ibv_devinfo",
        run_ibv_devinfo=run_ibv_devinfo,
    )
    assert gathered.status.ok is False
    assert gathered.status.failure == "ibv_devinfo reported no RDMA devices"


@pytest.mark.anyio
async def test_gather_bad_port():
    async def run_ibv_devinfo(_binary_path: str) -> IbvDevinfoCommandResult:
        return IbvDevinfoCommandResult(
            returncode=0, stdout=DOWN_PORT_STDOUT.encode(), stderr=b""
        )

    gathered = await IbvDevinfoStatus.gather(
        claimed_device_names=("rdma_en2",),
        locate_binary=lambda _name: "/usr/bin/ibv_devinfo",
        run_ibv_devinfo=run_ibv_devinfo,
    )
    assert gathered.status.ok is False
    assert gathered.status.failure is not None
    assert "PORT_DOWN" in gathered.status.failure


@pytest.mark.anyio
async def test_gather_timeout_is_failure():
    async def run_ibv_devinfo(_binary_path: str) -> IbvDevinfoCommandResult:
        raise TimeoutError

    gathered = await IbvDevinfoStatus.gather(
        locate_binary=lambda _name: "/usr/bin/ibv_devinfo",
        run_ibv_devinfo=run_ibv_devinfo,
    )
    assert gathered.status.ok is False
    assert gathered.status.failure == "ibv_devinfo failed to run"
