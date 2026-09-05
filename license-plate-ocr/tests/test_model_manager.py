"""Behavioral tests for deterministic detector model management."""

from hashlib import sha256
from pathlib import Path

import pytest

from src import model_manager


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_model_manager_imports_from_this_project_src_directory():
    expected_src = Path(__file__).resolve().parents[1] / "src"

    assert Path(model_manager.__file__).resolve().parent == expected_src


def test_existing_model_is_reused_when_hash_matches(tmp_path, monkeypatch):
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"valid detector bytes")
    monkeypatch.setattr(model_manager, "MODEL_SHA256", _sha256(model_path))

    assert model_manager.ensure_detector_model(model_path) == model_path


def test_existing_model_with_wrong_hash_is_rejected(tmp_path, monkeypatch):
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"tampered detector bytes")
    monkeypatch.setattr(model_manager, "MODEL_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model_manager.ensure_detector_model(model_path)


def test_missing_model_downloads_pinned_file_and_verifies_it(tmp_path, monkeypatch):
    payload = b"downloaded detector bytes"
    cached_path = tmp_path / "cache" / model_manager.MODEL_FILENAME
    model_path = tmp_path / "models" / model_manager.MODEL_FILENAME
    calls = {}

    def fake_hf_hub_download(*, repo_id, filename, revision):
        calls.update(repo_id=repo_id, filename=filename, revision=revision)
        cached_path.parent.mkdir()
        cached_path.write_bytes(payload)
        return str(cached_path)

    monkeypatch.setattr(model_manager, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(model_manager, "MODEL_SHA256", sha256(payload).hexdigest())

    assert model_manager.ensure_detector_model(model_path) == model_path
    assert model_path.read_bytes() == payload
    assert calls == {
        "repo_id": model_manager.MODEL_REPO,
        "filename": model_manager.MODEL_FILENAME,
        "revision": model_manager.MODEL_REVISION,
    }
