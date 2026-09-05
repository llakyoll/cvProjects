"""Behavioral tests for stable OCR consensus per ByteTrack identifier."""

import pytest

from src.pipeline import PlateResult


class Crop:
    """Small copyable crop stand-in."""

    size = 1

    def __init__(self, name: str) -> None:
        self.name = name

    def copy(self) -> "Crop":
        return Crop(self.name)


class Frame:
    """Return a named crop for every valid NumPy-style slice."""

    shape = (100, 100, 3)

    def __init__(self, name: str) -> None:
        self.name = name

    def __getitem__(self, key: object) -> Crop:
        return Crop(self.name)


def result(text: str, ocr: float, detection: float, track_id: int = 7) -> PlateResult:
    """Build one tracked OCR result."""
    return PlateResult((10, 10, 30, 20), detection, text, ocr, None, track_id)


def test_consensus_chooses_highest_weighted_text_and_its_best_crop():
    from src.track_consensus import TrackConsensusStore

    store = TrackConsensusStore(max_records=5)
    store.update(Frame("weak"), [result("34ABC123", 0.60, 0.80)])
    records = store.update(Frame("best"), [result("34ABC123", 0.95, 0.90)])
    records = store.update(Frame("wrong"), [result("34A8C123", 0.70, 0.80)])

    assert records[0].text == "34ABC123"
    assert records[0].crop.name == "best"
    assert records[0].consensus_confidence > 0.5


def test_consensus_keeps_only_five_most_recent_tracks():
    from src.track_consensus import TrackConsensusStore

    store = TrackConsensusStore(max_records=5)
    for track_id in range(6):
        store.update(Frame(str(track_id)), [result("34ABC123", 0.9, 0.9, track_id)])

    assert [record.track_id for record in store.records()] == [5, 4, 3, 2, 1]


def test_consensus_keeps_tracked_crop_when_ocr_text_is_empty():
    from src.track_consensus import TrackConsensusStore

    store = TrackConsensusStore(max_records=5)
    records = store.update(Frame("pending"), [result("", 0.0, 0.91, track_id=12)])

    assert records[0].track_id == 12
    assert records[0].text == "READING..."
    assert records[0].crop.name == "pending"


def test_consensus_preserves_actual_last_seen_frame_while_plate_is_absent():
    from src.track_consensus import TrackConsensusStore

    store = TrackConsensusStore(max_records=5)
    store.update(Frame("seen"), [result("34ABC123", 0.9, 0.9, track_id=4)])
    records = store.update(Frame("absent"), [])

    assert records[0].last_seen_frame == 1


def test_consensus_rejects_unresolved_results_instead_of_silently_dropping_them():
    from src.track_consensus import TrackConsensusStore

    unresolved = result("34ABC123", 0.9, 0.9, track_id=None)

    with pytest.raises(ValueError, match="application-level track ID"):
        TrackConsensusStore().update(Frame("unresolved"), [unresolved])
