"""Local completeness rule for copied / air-gapped models (TODO.md item 8)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiofiles
import aiofiles.os as aios

from exo.download.download_utils import (
    fetch_file_list_with_cache,
    is_model_directory_complete,
    resolve_existing_model,
)
from exo.shared.types.common import ModelId

MODEL_ID = ModelId("test-org/copied-model")
NORMALIZED = MODEL_ID.normalize()


def _write_index_and_weights(model_dir: Path, *, include_weights: bool) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "metadata": {"total_size": 1024},
        "weight_map": {"layer.weight": "model.safetensors"},
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    if include_weights:
        (model_dir / "model.safetensors").write_bytes(b"weights")


class TestIsModelDirectoryComplete:
    def test_empty_directory_is_incomplete(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "empty"
        model_dir.mkdir()
        assert is_model_directory_complete(model_dir) is False

    def test_missing_directory_is_incomplete(self, tmp_path: Path) -> None:
        assert is_model_directory_complete(tmp_path / "does-not-exist") is False

    def test_non_empty_without_partial_is_complete(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "copied"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "model.safetensors").write_bytes(b"weights")
        assert is_model_directory_complete(model_dir) is True

    def test_top_level_partial_is_incomplete(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "partial"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors.partial").write_bytes(b"in progress")
        assert is_model_directory_complete(model_dir) is False

    def test_nested_partial_is_incomplete(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "nested"
        nested = model_dir / "transformer"
        nested.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (nested / "weights.safetensors.partial").write_bytes(b"in progress")
        assert is_model_directory_complete(model_dir) is False

    def test_only_partial_files_is_incomplete(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "only-partial"
        model_dir.mkdir()
        (model_dir / "model.safetensors.partial").write_bytes(b"in progress")
        assert is_model_directory_complete(model_dir) is False

    def test_index_with_all_weights_is_complete(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "indexed"
        _write_index_and_weights(model_dir, include_weights=True)
        assert is_model_directory_complete(model_dir) is True

    def test_index_missing_weight_is_incomplete(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "indexed-missing"
        _write_index_and_weights(model_dir, include_weights=False)
        assert is_model_directory_complete(model_dir) is False


class TestResolveCopiedModelWithoutIndex:
    def test_finds_copied_model_without_safetensors_index(
        self, tmp_path: Path
    ) -> None:
        writable = tmp_path / "models"
        model_dir = writable / NORMALIZED
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "model.safetensors").write_bytes(b"weights")
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == model_dir

    def test_skips_copied_model_with_nested_partial(self, tmp_path: Path) -> None:
        writable = tmp_path / "models"
        model_dir = writable / NORMALIZED
        nested = model_dir / "vae"
        nested.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (nested / "diffusion.safetensors.partial").write_bytes(b"in progress")
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) is None


class TestFetchFileListCopiedModelOffline:
    async def test_lists_copied_model_without_cache_or_index(
        self, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        model_dir = models_dir / NORMALIZED
        await aios.makedirs(model_dir, exist_ok=True)
        async with aiofiles.open(model_dir / "config.json", "w") as handle:
            await handle.write('{"model_type": "test"}')
        async with aiofiles.open(model_dir / "model.safetensors", "wb") as handle:
            await handle.write(b"x" * 64)

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (models_dir,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models_dir),
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            result = await fetch_file_list_with_cache(
                MODEL_ID, "main", skip_internet=True
            )

        mock_fetch.assert_not_called()
        paths = {entry.path for entry in result}
        assert "config.json" in paths
        assert "model.safetensors" in paths
        assert all(entry.size is not None for entry in result)
