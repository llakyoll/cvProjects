"""Behavioral tests for application-level plate identity association."""

from src.pipeline import PlateResult


def result(
    bbox: tuple[int, int, int, int],
    text: str = "34ECK651",
    track_id: int | None = None,
) -> PlateResult:
    """Build one OCR result for association tests."""
    return PlateResult(bbox, 0.8, text, 0.9, None, track_id)


def test_associator_assigns_identity_when_ultralytics_id_is_missing() -> None:
    from src.plate_associator import PlateAssociator

    associated = PlateAssociator().update([result((10, 10, 100, 40))])

    assert associated[0].track_id == 1


def test_associator_keeps_identity_across_non_overlapping_fast_motion() -> None:
    from src.plate_associator import PlateAssociator

    associator = PlateAssociator()
    first = associator.update([result((1049, 711, 1143, 745))])
    second = associator.update([result((1080, 752, 1181, 789))])
    third = associator.update([result((1114, 798, 1216, 839))])

    assert [first[0].track_id, second[0].track_id, third[0].track_id] == [1, 1, 1]


def test_associator_uses_motion_when_one_ocr_reading_changes_completely() -> None:
    from src.plate_associator import PlateAssociator

    associator = PlateAssociator()
    associator.update([result((1156, 851, 1259, 888), "74EIK551")])
    previous = associator.update([result((1192, 910, 1304, 947), "34ELK651")])
    changed = associator.update([result((1243, 971, 1357, 1012), "14LEX697")])

    assert changed[0].track_id == previous[0].track_id


def test_associator_keeps_simultaneous_distant_plates_separate() -> None:
    from src.plate_associator import PlateAssociator

    associator = PlateAssociator()
    first = associator.update(
        [
            result((10, 10, 100, 40), "34ABC123"),
            result((800, 400, 900, 440), "06XYZ987"),
        ]
    )
    second = associator.update(
        [
            result((830, 430, 930, 470), "06XYZ987"),
            result((45, 45, 145, 80), "34ABC123"),
        ]
    )

    assert [item.track_id for item in first] == [1, 2]
    assert [item.track_id for item in second] == [2, 1]


def test_associator_prefers_existing_ultralytics_identity_mapping() -> None:
    from src.plate_associator import PlateAssociator

    associator = PlateAssociator()
    first = associator.update([result((10, 10, 100, 40), track_id=42)])
    second = associator.update([result((900, 700, 1000, 740), track_id=42)])

    assert first[0].track_id == 42
    assert second[0].track_id == 42


def test_associator_expires_stale_identity() -> None:
    from src.plate_associator import PlateAssociator

    associator = PlateAssociator(max_missed_frames=1)
    first = associator.update([result((10, 10, 100, 40))])
    associator.update([])
    associator.update([])
    later = associator.update([result((10, 10, 100, 40))])

    assert first[0].track_id == 1
    assert later[0].track_id == 2
