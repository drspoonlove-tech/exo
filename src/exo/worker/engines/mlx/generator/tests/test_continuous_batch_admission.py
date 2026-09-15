from exo.worker.engines.mlx.generator.continuous_batch import admit_new_prompt


def test_first_prompt_uses_blocking_prefill() -> None:
    admission = admit_new_prompt(
        in_flight_decode_sequences=0,
        pipeline_parallel=False,
        requires_synchronous_embedding_patch=False,
    )
    assert admission.strategy == "block_until_complete"
    assert admission.reason == "no_in_flight_decode"


def test_new_prompt_interleaves_while_decode_is_in_flight() -> None:
    admission = admit_new_prompt(
        in_flight_decode_sequences=3,
        pipeline_parallel=False,
        requires_synchronous_embedding_patch=False,
    )
    assert admission.strategy == "interleave_with_decode"
    assert admission.reason == "keep_decode_moving"


def test_pipeline_parallel_stays_blocking() -> None:
    admission = admit_new_prompt(
        in_flight_decode_sequences=2,
        pipeline_parallel=True,
        requires_synchronous_embedding_patch=False,
    )
    assert admission.strategy == "block_until_complete"
    assert admission.reason == "pipeline_parallel"


def test_vision_embedding_patch_stays_blocking() -> None:
    admission = admit_new_prompt(
        in_flight_decode_sequences=2,
        pipeline_parallel=False,
        requires_synchronous_embedding_patch=True,
    )
    assert admission.strategy == "block_until_complete"
    assert admission.reason == "vision_embeddings"
