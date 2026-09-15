# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Prove a late-arriving prompt does not stall in-flight decode.

Uses injectable clocks and a stand-in mlx BatchGenerator — no cluster, no
downloaded weights. ``prefill()`` is the blocking path that used to run
inside ``submit()``; if it runs while decode is already in flight, the
clock jumps by ``PREFILL_DURATION`` and the test fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest

import exo.worker.engines.mlx.generator.batch_generate as batch_generate_module
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import PrefillCancelled
from exo.worker.engines.mlx.patches.opt_batch_gen import BatchTopKLogprobs
from exo.worker.engines.mlx.types import Model

PREFILL_DURATION = 100.0
DECODE_DURATION = 1.0


@dataclass
class FakeClock:
    now: float = 0.0
    events: list[tuple[float, str]] = field(default_factory=list)

    def advance(self, delta: float, label: str) -> None:
        self.now += delta
        self.events.append((self.now, label))


@dataclass
class FakeTokenArray:
    tokens: list[int]

    @property
    def shape(self) -> tuple[int]:
        return (len(self.tokens),)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, key: int | slice) -> FakeTokenArray | int:
        if isinstance(key, slice):
            return FakeTokenArray(list(self.tokens[key]))
        return self.tokens[key]

    def tolist(self) -> list[int]:
        return list(self.tokens)


@dataclass
class FakePromptResponse:
    uid: int
    progress: tuple[int, int] = (0, 0)


@dataclass
class FakeGenerationResponse:
    uid: int
    token: int = 7
    finish_reason: str | None = None
    logprobs: object | None = None


class FakeGenerationBatch:
    def __init__(self, size: int = 0) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class FakeMlxBatchGenerator:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._generation_batch = FakeGenerationBatch(0)
        self._unprocessed_sequences: list[list[int]] = []
        self._prompt_batch = FakeGenerationBatch(0)
        self.insert_calls: list[list[list[int]]] = []
        self.next_generation: list[FakeGenerationResponse] = []
        self.next_prompts: list[FakePromptResponse] = []
        self.clock: FakeClock | None = None

    def insert(
        self,
        prompts: list[list[int]],
        **_kwargs: object,
    ) -> list[int]:
        self.insert_calls.append(prompts)
        self._unprocessed_sequences.append(prompts[0])
        return [len(self.insert_calls) - 1]

    def next(
        self,
    ) -> tuple[list[FakePromptResponse], list[FakeGenerationResponse]]:
        if self.clock is not None:
            self.clock.advance(DECODE_DURATION, "decode")
        prompts = list(self.next_prompts)
        generation = list(self.next_generation)
        self.next_prompts.clear()
        self.next_generation.clear()
        return prompts, generation

    def remove(self, uids: list[int]) -> dict[int, object]:
        return {uid: None for uid in uids}

    def close(self) -> None:
        return None


class FakeDetokenizer:
    last_segment = "x"

    def add_token(self, _token: int) -> None:
        return None

    def finalize(self) -> None:
        return None


class FakeTokenizer:
    detokenizer = FakeDetokenizer()
    eos_token_ids: list[int] = [0]


class FakeModel:
    layers: list[object] = []


def _task_params() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test/model"),
        input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        max_output_tokens=8,
        temperature=0.0,
    )


@pytest.fixture
def patched_batch_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeClock, list[str], FakeMlxBatchGenerator]:
    clock = FakeClock()
    prefill_calls: list[str] = []
    mlx_gen = FakeMlxBatchGenerator()
    mlx_gen.clock = clock
    prompt_tokens = FakeTokenArray([10, 11, 12, 13, 14, 15, 16, 17])

    def fake_prefill(
        *_args: object, **_kwargs: object
    ) -> tuple[float, int, list[object]]:
        prefill_calls.append("prefill")
        clock.advance(PREFILL_DURATION, "blocking_prefill")
        return 12.0, 8, []

    def fake_remote_prefill(
        *_args: object, **_kwargs: object
    ) -> tuple[float, int, list[object]]:
        prefill_calls.append("remote_prefill")
        clock.advance(PREFILL_DURATION, "blocking_remote_prefill")
        return 12.0, 8, []

    def fake_mlx_ctor(*_args: object, **_kwargs: object) -> FakeMlxBatchGenerator:
        return mlx_gen

    monkeypatch.setattr(batch_generate_module, "prefill", fake_prefill)
    monkeypatch.setattr(batch_generate_module, "remote_prefill", fake_remote_prefill)
    monkeypatch.setattr(batch_generate_module, "MlxBatchGenerator", fake_mlx_ctor)
    monkeypatch.setattr(
        batch_generate_module,
        "encode_prompt",
        lambda _tokenizer, _prompt: prompt_tokens,
    )
    monkeypatch.setattr(
        batch_generate_module,
        "fix_unmatched_think_end_tokens",
        lambda tokens, _tokenizer: tokens,
    )
    monkeypatch.setattr(batch_generate_module, "make_kv_cache", lambda _model: [])
    monkeypatch.setattr(
        batch_generate_module, "make_sampler", lambda **_kwargs: (lambda logits: logits)
    )
    monkeypatch.setattr(
        batch_generate_module, "make_logits_processors", lambda **_kwargs: []
    )
    monkeypatch.setattr(
        batch_generate_module, "eos_ids_from_tokenizer", lambda _tokenizer: [0]
    )
    monkeypatch.setattr(
        batch_generate_module, "has_pipeline_communication_layer", lambda _model: False
    )
    monkeypatch.setattr(batch_generate_module, "set_needs_topk", lambda *_a, **_k: None)
    monkeypatch.setattr(
        batch_generate_module,
        "take_ready_topk",
        lambda _batch: BatchTopKLogprobs(),
    )
    monkeypatch.setattr(batch_generate_module.mx.random, "seed", lambda _seed: None)

    return clock, prefill_calls, mlx_gen


def _make_generator() -> ExoBatchGenerator:
    return ExoBatchGenerator(
        model=cast(Model, FakeModel()),
        tokenizer=cast(Any, FakeTokenizer()),
        group=None,
        kv_prefix_cache=None,
    )


def test_first_submit_still_runs_blocking_prefill(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
) -> None:
    clock, prefill_calls, mlx_gen = patched_batch_generate
    generator = _make_generator()

    uid = generator.submit(task_params=_task_params(), prompt="hello")

    assert uid == 0
    assert prefill_calls == ["prefill"]
    assert clock.now == PREFILL_DURATION
    assert mlx_gen.insert_calls == [[[16, 17]]]


def test_submit_skips_blocking_prefill_when_decode_is_in_flight(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
) -> None:
    clock, prefill_calls, mlx_gen = patched_batch_generate
    generator = _make_generator()
    generator.submit(task_params=_task_params(), prompt="first")
    mlx_gen._generation_batch.size = 1
    clock_after_first = clock.now
    prefill_calls.clear()

    uid = generator.submit(task_params=_task_params(), prompt="second")

    assert uid == 1
    assert prefill_calls == []
    assert clock.now == clock_after_first
    assert mlx_gen.insert_calls[-1] == [[10, 11, 12, 13, 14, 15, 16, 17]]
    assert generator._active_tasks[uid].interleaved_prefill is True


def test_decode_token_emitted_without_waiting_for_new_prompt_prefill(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
) -> None:
    clock, prefill_calls, mlx_gen = patched_batch_generate
    generator = _make_generator()

    first_uid = generator.submit(task_params=_task_params(), prompt="first")
    mlx_gen._generation_batch.size = 1
    prefill_calls.clear()

    generator.submit(task_params=_task_params(), prompt="second")
    mlx_gen.next_generation = [FakeGenerationResponse(uid=first_uid, token=42)]

    results = generator.step()

    assert prefill_calls == []
    decode_times = [when for when, label in clock.events if label == "decode"]
    prefill_times = [
        when for when, label in clock.events if label.startswith("blocking_")
    ]
    assert decode_times == [PREFILL_DURATION + DECODE_DURATION]
    assert prefill_times == [PREFILL_DURATION]
    assert len(results) == 1
    assert results[0][0] == first_uid
    assert results[0][1].token == 42
    assert clock.now < PREFILL_DURATION * 2


def test_pipeline_parallel_still_blocks_while_decode_is_in_flight(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, prefill_calls, mlx_gen = patched_batch_generate
    monkeypatch.setattr(
        batch_generate_module, "has_pipeline_communication_layer", lambda _model: True
    )
    generator = _make_generator()
    generator.submit(task_params=_task_params(), prompt="first")
    mlx_gen._generation_batch.size = 1
    prefill_calls.clear()
    clock_after_first = clock.now

    generator.submit(task_params=_task_params(), prompt="second")

    assert prefill_calls == ["prefill"]
    assert clock.now == clock_after_first + PREFILL_DURATION


def test_interleaved_prefill_progress_does_not_block_decode(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
) -> None:
    _clock, prefill_calls, mlx_gen = patched_batch_generate
    generator = _make_generator()
    generator.submit(task_params=_task_params(), prompt="first")
    mlx_gen._generation_batch.size = 1
    prefill_calls.clear()

    progress: list[tuple[int, int]] = []
    second_uid = generator.submit(
        task_params=_task_params(),
        prompt="second",
        on_prefill_progress=lambda processed, total: progress.append(
            (processed, total)
        ),
    )
    mlx_gen.next_prompts = [FakePromptResponse(uid=second_uid, progress=(4, 8))]
    mlx_gen.next_generation = [FakeGenerationResponse(uid=0, token=3)]

    results = generator.step()

    assert progress == [(4, 8)]
    assert prefill_calls == []
    assert results[0][0] == 0


def test_prefill_cancel_during_interleaved_progress_drops_only_that_request(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
) -> None:
    _clock, _prefill_calls, mlx_gen = patched_batch_generate
    generator = _make_generator()
    first_uid = generator.submit(task_params=_task_params(), prompt="first")
    mlx_gen._generation_batch.size = 1

    def raise_cancelled() -> None:
        raise PrefillCancelled()

    second_uid = generator.submit(
        task_params=_task_params(),
        prompt="second",
        distributed_prompt_progress_callback=raise_cancelled,
    )
    mlx_gen.next_prompts = [FakePromptResponse(uid=second_uid, progress=(2, 8))]
    mlx_gen.next_generation = [FakeGenerationResponse(uid=first_uid, token=9)]

    results = generator.step()

    assert second_uid not in generator._active_tasks
    assert first_uid in generator._active_tasks
    assert results[0][0] == first_uid


def _text_task(task_id: str) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(task_id),
        instance_id=InstanceId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        command_id=CommandId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        task_params=_task_params().model_copy(update={"bench": True}),
    )


def test_engine_step_keeps_decode_moving_when_a_new_prompt_is_queued(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner-shaped loop: start A, decode, queue B, decode again.

    ``BatchGenerator.step`` admits queued tasks before it steps the engine.
    After A is already decoding, admitting B must not run blocking prefill.
    """
    import exo.worker.runner.llm_inference.batch_generator as engine_module
    from exo.worker.runner.llm_inference.batch_generator import BatchGenerator

    clock, prefill_calls, mlx_gen = patched_batch_generate
    monkeypatch.setattr(
        engine_module, "apply_chat_template", lambda *_a, **_k: "prompt"
    )
    monkeypatch.setattr(
        engine_module, "_check_for_debug_prompts", lambda *_a, **_k: None
    )

    engine = BatchGenerator(
        model=cast(Model, FakeModel()),
        tokenizer=cast(Any, FakeTokenizer()),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=ModelId("test/model"),
        device_rank=0,
        cancel_receiver=cast(Any, type("R", (), {"collect": staticmethod(list)})()),
        event_sender=cast(
            Any, type("S", (), {"send": staticmethod(lambda _e: None)})()
        ),
    )

    engine._queue.append(_text_task("task-a"))
    mlx_gen.next_generation = [FakeGenerationResponse(uid=0, token=1)]
    first_results = list(engine.step())
    assert any(chunk[0] == TaskId("task-a") for chunk in first_results)

    mlx_gen._generation_batch.size = 1
    prefill_calls.clear()
    mlx_gen.next_generation = [FakeGenerationResponse(uid=0, token=2)]
    engine._queue.append(_text_task("task-b"))
    second_results = list(engine.step())

    assert prefill_calls == []
    assert clock.now == PREFILL_DURATION + DECODE_DURATION + DECODE_DURATION
    assert any(chunk[0] == TaskId("task-a") for chunk in second_results)
    assert engine._gen._active_tasks[1].interleaved_prefill is True


def test_on_generation_token_still_runs_for_in_flight_request(
    patched_batch_generate: tuple[FakeClock, list[str], FakeMlxBatchGenerator],
) -> None:
    _clock, _prefill_calls, mlx_gen = patched_batch_generate
    generator = _make_generator()
    ticks: list[str] = []
    first_uid = generator.submit(
        task_params=_task_params(),
        prompt="first",
        on_generation_token=lambda: ticks.append("decode"),
    )
    mlx_gen._generation_batch.size = 1
    generator.submit(task_params=_task_params(), prompt="second")
    mlx_gen.next_generation = [FakeGenerationResponse(uid=first_uid, token=1)]

    generator.step()

    assert ticks == ["decode"]
