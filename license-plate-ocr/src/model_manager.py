"""Download and validate the detector weight required by the OCR pipeline.

The model is fetched from an immutable Hugging Face revision and checked
against the expected SHA-256 digest before it is returned to callers. This
keeps large binary weights out of Git while preventing accidental use of a
different or corrupted detector file.
"""

from hashlib import sha256
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download


MODEL_REPO = "morsetechlab/yolov11-license-plate-detection"
MODEL_FILENAME = "license-plate-finetune-v1l.onnx"
MODEL_REVISION = "0f8dc03"
MODEL_SHA256 = "5efdfbe4909bfa6c895bed48676b7de695bf71788932e095e7bc74b8b52b75d8"


def _calculate_sha256(model_path: Path) -> str:
    digest = sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(model_path: Path) -> None:
    actual_sha256 = _calculate_sha256(model_path)
    if actual_sha256 != MODEL_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {model_path}: "
            f"expected {MODEL_SHA256}, got {actual_sha256}"
        )


def ensure_detector_model(model_path: Path) -> Path:
    """Download the pinned Hugging Face model if absent and verify SHA-256.

    Args:
        model_path: Local path where the detector ONNX file must be available.

    Returns:
        The validated ``model_path``.

    Raises:
        ValueError: If the existing or downloaded file has the wrong digest.
        FileNotFoundError: If the Hugging Face downloader returns no file.
    """
    model_path = Path(model_path)
    if model_path.is_file():
        _verify_sha256(model_path)
        return model_path

    downloaded_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    )
    if not downloaded_path.is_file():
        raise FileNotFoundError(f"Hugging Face download did not produce {downloaded_path}")

    _verify_sha256(downloaded_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_path.resolve() != model_path.resolve():
        shutil.copyfile(downloaded_path, model_path)
    _verify_sha256(model_path)
    return model_path
