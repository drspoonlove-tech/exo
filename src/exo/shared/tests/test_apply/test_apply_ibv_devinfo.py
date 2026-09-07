from datetime import datetime, timezone

from exo.shared.apply import apply_node_gathered_info, apply_node_timed_out
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeGatheredInfo, NodeTimedOut
from exo.shared.types.profiling import (
    IbvDevicePort,
    IbvDeviceSummary,
    NodeIbvDevinfoStatus,
)
from exo.shared.types.state import State
from exo.utils.info_gatherer.ibv_devinfo import IbvDevinfoStatus


def test_apply_stores_ibv_devinfo_status_and_clears_on_timeout():
    node_id = NodeId()
    gathered = IbvDevinfoStatus(
        status=NodeIbvDevinfoStatus(
            ok=True,
            devices=[
                IbvDeviceSummary(
                    name="rdma_en2",
                    transport="InfiniBand (0)",
                    ports=[
                        IbvDevicePort(
                            port_number=1, state="PORT_ACTIVE", link_layer="InfiniBand"
                        )
                    ],
                )
            ],
        )
    )
    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=node_id,
            when=datetime.now(timezone.utc).isoformat(),
            info=gathered,
        ),
        State(),
    )
    stored = state.node_ibv_devinfo[node_id]
    assert stored.ok is True
    assert stored.devices[0].name == "rdma_en2"

    cleared = apply_node_timed_out(NodeTimedOut(node_id=node_id), state)
    assert node_id not in cleared.node_ibv_devinfo
