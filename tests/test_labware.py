import dataclasses

import pytest

from angular_spectrum.labware import (
    DEFAULT_LABCYTE_PLATE_ID,
    LABCYTE_PLATES,
    LABWARE_CATALOGUE_URL,
    get_labcyte_plate,
    labcyte_plate_choice_ids,
    labcyte_plate_choice_label,
    labcyte_plate_choices,
    labcyte_plates_for_family,
)


def test_pp_0200_is_default_with_authoritative_geometry() -> None:
    plate = get_labcyte_plate()

    assert DEFAULT_LABCYTE_PLATE_ID == "PP-0200"
    assert LABCYTE_PLATES[0] is plate
    assert plate.id == "PP-0200"
    assert plate.part_number == "PP-0200"
    assert plate.guid == "2a09adfa-a468-4327-b5b1-5f8296136782"
    assert plate.family == "PP-0200"
    assert plate.material == "polypropylene"
    assert plate.well_count == 384
    assert plate.bottom_thickness_mm == pytest.approx(0.78)
    assert plate.well_depth_mm == pytest.approx(10.91)
    assert plate.well_top_width_mm == pytest.approx(3.8)
    assert plate.well_bottom_width_mm == pytest.approx(3.6)
    assert plate.well_pitch_mm == pytest.approx(4.5)
    assert plate.well_volume_ul == pytest.approx(65.0)
    assert plate.raw_longitudinal_speed == pytest.approx(2_732_049.037)
    assert plate.inferred_longitudinal_speed_m_s == pytest.approx(2732.049037)
    assert plate.source_url.endswith(f"/{plate.guid}.json")
    assert LABWARE_CATALOGUE_URL in plate.provenance
    assert len(plate.limitations) >= 3


def test_commercial_variants_are_grouped_into_three_physical_profiles() -> None:
    expected = {"PP-0200": 4, "LP-0200": 6, "LP-0400": 5}

    assert len(LABCYTE_PLATES) == sum(expected.values()) == 15
    assert {
        family: len(labcyte_plates_for_family(family)) for family in expected
    } == expected

    for family in expected:
        variants = labcyte_plates_for_family(family.lower())
        profile_values = {
            (
                plate.material,
                plate.well_count,
                plate.bottom_thickness_mm,
                plate.well_depth_mm,
                plate.well_top_width_mm,
                plate.well_bottom_width_mm,
                plate.well_pitch_mm,
                plate.well_volume_ul,
                plate.raw_longitudinal_speed,
            )
            for plate in variants
        }
        assert len(profile_values) == 1


def test_ids_guids_and_ui_choices_are_unique_and_resolvable() -> None:
    ids = [plate.id for plate in LABCYTE_PLATES]
    guids = [plate.guid for plate in LABCYTE_PLATES]
    choices = labcyte_plate_choices()

    assert len(ids) == len(set(ids))
    assert len(guids) == len(set(guids))
    assert [identifier for identifier, _label in choices] == ids
    assert labcyte_plate_choice_ids() == tuple(ids)
    assert choices[0][0] == DEFAULT_LABCYTE_PLATE_ID
    assert len({label for _identifier, label in choices}) == len(choices)

    for plate in LABCYTE_PLATES:
        assert get_labcyte_plate(plate.id.lower()) is plate
        assert get_labcyte_plate(plate.guid.upper()) is plate
        assert labcyte_plate_choice_label(plate.id) == plate.display_label
        assert plate.id in plate.display_label


def test_flagged_duplicate_and_inconsistent_lpl_records_are_omitted() -> None:
    ids = [plate.id for plate in LABCYTE_PLATES]
    guids = {plate.guid for plate in LABCYTE_PLATES}

    assert ids.count("LPL-0200") == 1
    assert "LPL-0200-BC" not in ids
    assert "4e923fcc-156b-40f8-9d62-776d5cd63980" in guids
    assert "fa110ac9-b87a-4abe-9a7e-c6619a4f6225" not in guids
    assert "def0b118-c1b5-4e6b-8507-845cf4d244da" not in guids


def test_snapshot_is_immutable_and_keeps_speed_inference_explicit() -> None:
    for plate in LABCYTE_PLATES:
        assert plate.inferred_longitudinal_speed_m_s == pytest.approx(
            plate.raw_longitudinal_speed / 1000.0
        )
        assert any("assumes" in limitation for limitation in plate.limitations)
        assert any("density" in limitation for limitation in plate.limitations)
        assert any("CMM" in limitation for limitation in plate.limitations)

    assert any(
        "1.530 mm" in limitation
        for limitation in get_labcyte_plate("LP-0200").limitations
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        LABCYTE_PLATES[0].bottom_thickness_mm = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("identifier", ["", "unknown", "LPL-0200-BC"])
def test_unknown_lookup_has_a_clear_error(identifier: str) -> None:
    with pytest.raises(KeyError, match="Labcyte"):
        get_labcyte_plate(identifier)


def test_unknown_family_has_a_clear_error() -> None:
    with pytest.raises(KeyError, match="available"):
        labcyte_plates_for_family("not-a-family")
