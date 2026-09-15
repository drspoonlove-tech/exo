from dataclasses import dataclass
from typing import Literal, final

PrefillStrategy = Literal["block_until_complete", "interleave_with_decode"]
AdmissionReason = Literal[
    "no_in_flight_decode",
    "pipeline_parallel",
    "vision_embeddings",
    "keep_decode_moving",
]


@final
@dataclass(frozen=True)
class PromptAdmission:
    """Whether a newly arrived prompt may run a blocking prefill.

    ``block_until_complete`` is the legacy path: ``submit()`` runs the full
    custom prefill (prefix cache, remote, pipeline, vision) before returning.
    ``interleave_with_decode`` inserts remaining tokens into mlx-lm's
    incremental prefill so an in-flight generation batch can keep decoding.
    """

    strategy: PrefillStrategy
    reason: AdmissionReason


def admit_new_prompt(
    *,
    in_flight_decode_sequences: int,
    pipeline_parallel: bool,
    requires_synchronous_embedding_patch: bool,
) -> PromptAdmission:
    """Choose how to admit a prompt relative to an in-flight decode batch.

    Pipeline-parallel prefill and vision embedding patches must wrap the
    entire prompt in one call; those stay blocking. Every other new prompt
    that arrives while decode is running is interleaved so the current batch
    is not stalled.
    """
    if in_flight_decode_sequences <= 0:
        return PromptAdmission(
            strategy="block_until_complete",
            reason="no_in_flight_decode",
        )
    if pipeline_parallel:
        return PromptAdmission(
            strategy="block_until_complete",
            reason="pipeline_parallel",
        )
    if requires_synchronous_embedding_patch:
        return PromptAdmission(
            strategy="block_until_complete",
            reason="vision_embeddings",
        )
    return PromptAdmission(
        strategy="interleave_with_decode",
        reason="keep_decode_moving",
    )
