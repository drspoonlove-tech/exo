from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.events import IndexedEvent, InstanceReplacedAtomically
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _instance(neighbor_ip: str) -> MlxRingInstance:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    runner_a = RunnerId("runner-a")
    runner_b = RunnerId("runner-b")
    card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_bytes(1),
        n_layers=2,
        hidden_size=8,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    return MlxRingInstance(
        instance_id=InstanceId("inst-1"),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            runner_to_shard={
                runner_a: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=2,
                    start_layer=0,
                    end_layer=1,
                    n_layers=2,
                ),
                runner_b: PipelineShardMetadata(
                    model_card=card,
                    device_rank=1,
                    world_size=2,
                    start_layer=1,
                    end_layer=2,
                    n_layers=2,
                ),
            },
            node_to_runner={node_a: runner_a, node_b: runner_b},
        ),
        hosts_by_node={
            node_a: [
                Host(ip="0.0.0.0", port=50000),
                Host(ip=neighbor_ip, port=50000),
            ],
            node_b: [
                Host(ip="10.0.0.1", port=50000),
                Host(ip="0.0.0.0", port=50000),
            ],
        },
        ephemeral_port=50000,
    )


def test_instance_replaced_atomically_overwrites_same_id_and_keeps_tasks() -> None:
    current = _instance("10.0.0.2")
    replacement = _instance("169.254.1.2")
    task_id = TaskId("task-1")
    task = TextGeneration(
        task_id=task_id,
        command_id=CommandId("cmd-1"),
        instance_id=current.instance_id,
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )
    state = apply(
        State(
            instances={current.instance_id: current},
            tasks={task_id: task},
        ),
        IndexedEvent(idx=0, event=InstanceReplacedAtomically(instance=replacement)),
    )

    assert state.instances[current.instance_id] == replacement
    assert state.tasks[task_id] == task
