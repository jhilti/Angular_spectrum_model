"""Offline Labcyte plate metadata for the Streamlit plate selector.

The source catalogue is maintained by UK Robotics at
``https://labware.ukrobotics.app``.  This module intentionally snapshots the
small, simulation-relevant subset instead of making the application depend on
network access.  Commercial variants that differ only by colour, sterility,
or barcode share one of three physical bottom profiles.

The catalogue provides geometry and a raw longitudinal sound-speed number. It
does not provide the density, shear-wave speed, attenuation, manufacturing
tolerances, or frequency/temperature dependence needed to construct a fully
specified elastic plate.  Callers must therefore keep those missing material
parameters explicit rather than treating this metadata as a complete acoustic
calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


LABWARE_CATALOGUE_URL = "https://labware.ukrobotics.app/SBSPlatesFlat.json"
LABWARE_SNAPSHOT_DATE = "2026-08-01"
DEFAULT_LABCYTE_PLATE_ID = "PP-0200"

_SPEED_INFERENCE = (
    "The site declares millimetres as its distance unit but does not state a "
    "time unit for MaterialSpeedOfSound. The SI value assumes the raw number "
    "is in mm/s and divides it by 1000."
)
_MATERIAL_LIMITATION = (
    "The source does not provide density, shear-wave speed, attenuation, "
    "dispersion, or temperature dependence; these require independent values "
    "or measurements before quantitative elastic-plate simulation."
)
_GEOMETRY_LIMITATION = (
    "The source gives nominal well geometry without manufacturing tolerances. "
    "Bottom thickness should be verified on the physical plate for calibrated "
    "timing or amplitude work."
)
_VOLUME_HEIGHT_LIMITATION = (
    "Fill height and volume are converted with an ideal square/diamond "
    "frustum derived from the catalogue well widths and depth. The catalogue "
    "volume field is reported separately and is not used to rescale the "
    "geometry; the estimate does not include rounded corners, meniscus "
    "curvature, corner wetting, dead volume, or manufacturing tolerances."
)
_CMM_PROVENANCE_LIMITATION = (
    "The site labels bottom thickness as CMM-measured but gives the malformed "
    "date '00.12.2025' and no measurement uncertainty."
)
_LP_0200_WIDTH_LIMITATION = (
    "For the LP-0200 family, the site says a datasheet value of 1.530 mm for "
    "the bottom width appeared wrong and substitutes 2.432 mm by observation."
)


@dataclass(frozen=True, slots=True)
class LabcytePlate:
    """One commercial Labcyte plate record and its acoustic profile.

    ``id`` is the manufacturer part number used as the stable UI value.
    ``family`` identifies variants with the same simulation-relevant bottom
    geometry and material.  The longitudinal speed in SI units is explicitly
    labelled as inferred because the source does not declare a time unit.
    """

    id: str
    name: str
    guid: str
    family: str
    material: str
    well_shape: str
    well_count: int
    bottom_thickness_mm: float
    well_depth_mm: float
    well_top_width_mm: float
    well_bottom_width_mm: float
    well_pitch_mm: float
    well_volume_ul: float
    raw_longitudinal_speed: float
    inferred_longitudinal_speed_m_s: float
    source_url: str
    provenance: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        text_fields = {
            "id": self.id,
            "name": self.name,
            "guid": self.guid,
            "family": self.family,
            "material": self.material,
            "well_shape": self.well_shape,
            "source_url": self.source_url,
            "provenance": self.provenance,
        }
        for field_name, value in text_fields.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.well_count <= 0:
            raise ValueError("well_count must be > 0")
        if self.well_shape not in {"square", "diamond"}:
            raise ValueError("well_shape must be 'square' or 'diamond'")
        positive_values = {
            "bottom_thickness_mm": self.bottom_thickness_mm,
            "well_depth_mm": self.well_depth_mm,
            "well_top_width_mm": self.well_top_width_mm,
            "well_bottom_width_mm": self.well_bottom_width_mm,
            "well_pitch_mm": self.well_pitch_mm,
            "well_volume_ul": self.well_volume_ul,
            "raw_longitudinal_speed": self.raw_longitudinal_speed,
            "inferred_longitudinal_speed_m_s": (
                self.inferred_longitudinal_speed_m_s
            ),
        }
        for field_name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and > 0")
        expected_speed = self.raw_longitudinal_speed / 1000.0
        if abs(self.inferred_longitudinal_speed_m_s - expected_speed) > (
            1.0e-9 * expected_speed
        ):
            raise ValueError(
                "inferred_longitudinal_speed_m_s must equal "
                "raw_longitudinal_speed / 1000"
            )
        if not self.limitations or any(
            not item.strip() for item in self.limitations
        ):
            raise ValueError("limitations must contain non-empty caveats")

    @property
    def part_number(self) -> str:
        """Alias for the commercial ``id``."""

        return self.id

    @property
    def display_label(self) -> str:
        """Compact label suitable for a Streamlit ``selectbox``."""

        material = (
            "PP" if self.material == "polypropylene" else self.material.upper()
        )
        return (
            f"{self.id} · {self.well_count}-well {material} · "
            f"{self.bottom_thickness_mm:g} mm bottom"
        )

    def estimated_fill_volume_ul(self, fill_height_mm: float) -> float:
        """Estimate per-well volume from the catalogue geometry.

        The square or rotated-square (diamond) well is represented as a
        frustum whose side length changes linearly from
        ``well_bottom_width_mm`` to ``well_top_width_mm``. Because 1 mm³ is
        exactly 1 µL, integrating the ideal cross-sectional area directly
        gives the estimated volume in µL. ``well_volume_ul`` remains a
        separate catalogue capacity field and does not rescale the geometry.
        """

        height = float(fill_height_mm)
        if not math.isfinite(height) or not 0.0 <= height <= self.well_depth_mm:
            raise ValueError(
                "fill_height_mm must be finite and lie between 0 and the "
                "catalogue well depth"
            )
        if height == 0.0:
            return 0.0
        bottom = self.well_bottom_width_mm
        width_slope = (
            self.well_top_width_mm - bottom
        ) / self.well_depth_mm
        return float(
            bottom**2 * height
            + bottom * width_slope * height**2
            + width_slope**2 * height**3 / 3.0
        )

    @property
    def estimated_geometric_capacity_ul(self) -> float:
        """Idealized volume at the catalogue well depth in µL."""

        return self.estimated_fill_volume_ul(self.well_depth_mm)

    def estimated_fill_height_mm(self, fill_volume_ul: float) -> float:
        """Invert :meth:`estimated_fill_volume_ul` by monotone bisection."""

        volume = float(fill_volume_ul)
        if (
            not math.isfinite(volume)
            or not 0.0 <= volume <= self.estimated_geometric_capacity_ul
        ):
            raise ValueError(
                "fill_volume_ul must be finite and lie between 0 and the "
                "estimated geometric well capacity"
            )
        if volume == 0.0:
            return 0.0
        if volume == self.estimated_geometric_capacity_ul:
            return self.well_depth_mm

        lower_mm = 0.0
        upper_mm = self.well_depth_mm
        for _ in range(60):
            midpoint_mm = 0.5 * (lower_mm + upper_mm)
            if self.estimated_fill_volume_ul(midpoint_mm) < volume:
                lower_mm = midpoint_mm
            else:
                upper_mm = midpoint_mm
        return float(0.5 * (lower_mm + upper_mm))


def _source_url(guid: str) -> str:
    return f"https://labware.ukrobotics.app/{guid}.json"


def _plate(
    *,
    id: str,
    name: str,
    guid: str,
    family: str,
    material: str,
    well_shape: str,
    well_count: int,
    bottom_thickness_mm: float,
    well_depth_mm: float,
    well_top_width_mm: float,
    well_bottom_width_mm: float,
    well_pitch_mm: float,
    well_volume_ul: float,
    raw_longitudinal_speed: float,
    extra_limitations: tuple[str, ...] = (),
) -> LabcytePlate:
    return LabcytePlate(
        id=id,
        name=name,
        guid=guid,
        family=family,
        material=material,
        well_shape=well_shape,
        well_count=well_count,
        bottom_thickness_mm=bottom_thickness_mm,
        well_depth_mm=well_depth_mm,
        well_top_width_mm=well_top_width_mm,
        well_bottom_width_mm=well_bottom_width_mm,
        well_pitch_mm=well_pitch_mm,
        well_volume_ul=well_volume_ul,
        raw_longitudinal_speed=raw_longitudinal_speed,
        inferred_longitudinal_speed_m_s=raw_longitudinal_speed / 1000.0,
        source_url=_source_url(guid),
        provenance=(
            f"Offline snapshot ({LABWARE_SNAPSHOT_DATE}) of the resolved "
            f"plate record from {LABWARE_CATALOGUE_URL}."
        ),
        limitations=(
            _SPEED_INFERENCE,
            _MATERIAL_LIMITATION,
            _GEOMETRY_LIMITATION,
            _VOLUME_HEIGHT_LIMITATION,
            _CMM_PROVENANCE_LIMITATION,
        )
        + extra_limitations,
    )


def _pp_0200(
    *,
    id: str,
    name: str,
    guid: str,
) -> LabcytePlate:
    return _plate(
        id=id,
        name=name,
        guid=guid,
        family="PP-0200",
        material="polypropylene",
        well_shape="square",
        well_count=384,
        bottom_thickness_mm=0.78,
        well_depth_mm=10.91,
        well_top_width_mm=3.8,
        well_bottom_width_mm=3.6,
        well_pitch_mm=4.5,
        well_volume_ul=65.0,
        raw_longitudinal_speed=2_732_049.037,
    )


def _lp_0200(
    *,
    id: str,
    name: str,
    guid: str,
) -> LabcytePlate:
    return _plate(
        id=id,
        name=name,
        guid=guid,
        family="LP-0200",
        material="coc",
        well_shape="diamond",
        well_count=384,
        bottom_thickness_mm=1.0,
        well_depth_mm=5.1,
        well_top_width_mm=2.432,
        well_bottom_width_mm=2.432,
        well_pitch_mm=4.5,
        well_volume_ul=21.0,
        raw_longitudinal_speed=2_500_000.0,
        extra_limitations=(_LP_0200_WIDTH_LIMITATION,),
    )


def _lp_0400(
    *,
    id: str,
    name: str,
    guid: str,
) -> LabcytePlate:
    return _plate(
        id=id,
        name=name,
        guid=guid,
        family="LP-0400",
        material="coc",
        well_shape="square",
        well_count=1536,
        bottom_thickness_mm=0.84,
        well_depth_mm=5.15,
        well_top_width_mm=1.7,
        well_bottom_width_mm=1.4,
        well_pitch_mm=2.25,
        well_volume_ul=5.5,
        raw_longitudinal_speed=2_500_000.0,
    )


# Keep the default first so a UI can use these choices without another sort.
# Sterile, barcode, and colour variants deliberately retain their commercial
# identifiers while sharing their family's acoustic profile.
LABCYTE_PLATES: tuple[LabcytePlate, ...] = (
    _pp_0200(
        id="PP-0200",
        name="Echo Qualified 384-Well Polypropylene Microplate, Clear",
        guid="2a09adfa-a468-4327-b5b1-5f8296136782",
    ),
    _pp_0200(
        id="PP-0200-BC",
        name=(
            "Echo Qualified 384-Well Polypropylene Microplate, Clear, "
            "Barcoded"
        ),
        guid="dcf061d9-9470-455b-ba0d-8ea085de5810",
    ),
    _pp_0200(
        id="PPS-0200",
        name=(
            "Echo Qualified 384-Well Polypropylene Microplate, Clear, Sterile"
        ),
        guid="e46a8836-3d31-4306-b85c-41c1dd177241",
    ),
    _pp_0200(
        id="PPS-0200-BC",
        name=(
            "Echo Qualified 384-Well Polypropylene Microplate, Clear, "
            "Sterile, Barcoded"
        ),
        guid="ba78dcb9-cb53-4660-bcbb-29082daf4f98",
    ),
    _lp_0200(
        id="LP-0200",
        name=(
            "Echo Qualified 384-Well COC Source Microplate, Diamond, Low Dead "
            "Volume, Clear, Flat Bottom"
        ),
        guid="050b8fdb-ee4c-4b4a-bef9-b7c1f52bc0a7",
    ),
    _lp_0200(
        id="LP-0200-BC",
        name=(
            "Echo Qualified 384-Well COC Source Microplate, Low Dead Volume, "
            "Clear, Flat Bottom, Barcoded"
        ),
        guid="144bf20b-c8e1-4fb6-9d32-ea7cce04da85",
    ),
    _lp_0200(
        id="LP-0210",
        name=(
            "Echo Qualified 384-Well COC Source Microplate, Low Dead Volume, "
            "Black, Flat Bottom"
        ),
        guid="c2d2c5d2-31e2-433b-8883-a497d6968ff8",
    ),
    _lp_0200(
        id="LP-0210-BC",
        name=(
            "Echo Qualified 384-Well COC Source Microplate, Low Dead Volume, "
            "Black, Flat Bottom, Barcoded"
        ),
        guid="8bad01fc-47fd-4cd5-87e6-158c073880c0",
    ),
    _lp_0200(
        id="LPS-0200",
        name=(
            "Echo Qualified 384-Well COC Source Microplate, Low Dead Volume, "
            "Clear, Flat Bottom, Sterile"
        ),
        guid="f3f8eca4-c243-497f-ac1a-f0ff859b11a3",
    ),
    _lp_0200(
        id="LPL-0200",
        name=(
            "Echo Qualified 384-Well LDV PLUS Source Microplate, Diamond, "
            "COC, Clear, Flat Bottom"
        ),
        guid="4e923fcc-156b-40f8-9d62-776d5cd63980",
    ),
    _lp_0400(
        id="LP-0400",
        name=(
            "Echo Qualified 1536-Well COC Source Microplate, Low Dead Volume, "
            "Clear, Flat Bottom"
        ),
        guid="61bbefb1-65a5-4533-a8e9-cbfba18bbb19",
    ),
    _lp_0400(
        id="LP-0400-BC",
        name=(
            "Echo Qualified 1536-Well COC Source Microplate, Low Dead Volume, "
            "Clear, Flat Bottom, Barcoded"
        ),
        guid="6b43a3f6-9410-4e0c-bf3c-4f7b1f84a7e7",
    ),
    _lp_0400(
        id="LPS-0400",
        name=(
            "Echo Qualified 1536-Well COC Source Microplate, Low Dead Volume, "
            "Clear, Flat Bottom, Sterile"
        ),
        guid="6e16885d-e8ed-4d9c-87e7-0a70909a39af",
    ),
    _lp_0400(
        id="LP-0410",
        name=(
            "Echo Qualified 1536-Well COC Source Microplate, Low Dead Volume, "
            "Black, Flat Bottom"
        ),
        guid="b0542bee-3b3c-49e5-b31b-1858bce8601a",
    ),
    _lp_0400(
        id="LP-0410-BC",
        name=(
            "Echo Qualified 1536-Well COC Source Microplate, Low Dead Volume, "
            "Black, Flat Bottom, Barcoded"
        ),
        guid="b03c3043-12c5-4028-b290-872df3e01f26",
    ),
)

_BY_ID = {plate.id.casefold(): plate for plate in LABCYTE_PLATES}
_BY_GUID = {plate.guid.casefold(): plate for plate in LABCYTE_PLATES}


def get_labcyte_plate(identifier: str = DEFAULT_LABCYTE_PLATE_ID) -> LabcytePlate:
    """Look up a plate by commercial ID or GUID, case-insensitively."""

    if not isinstance(identifier, str) or not identifier.strip():
        raise KeyError("Labcyte plate identifier must be a non-empty string")
    key = identifier.strip().casefold()
    plate = _BY_ID.get(key) or _BY_GUID.get(key)
    if plate is None:
        available = ", ".join(plate.id for plate in LABCYTE_PLATES)
        raise KeyError(
            f"Unknown Labcyte plate {identifier!r}; available IDs: {available}"
        )
    return plate


def labcyte_plate_choices() -> tuple[tuple[str, str], ...]:
    """Return ordered ``(id, display_label)`` choices for a UI widget."""

    return tuple((plate.id, plate.display_label) for plate in LABCYTE_PLATES)


def labcyte_plate_choice_ids() -> tuple[str, ...]:
    """Return IDs for ``selectbox(options=..., format_func=...)``."""

    return tuple(plate.id for plate in LABCYTE_PLATES)


def labcyte_plate_choice_label(identifier: str) -> str:
    """Return the display label for a Streamlit ``format_func``."""

    return get_labcyte_plate(identifier).display_label


def labcyte_plates_for_family(family: str) -> tuple[LabcytePlate, ...]:
    """Return commercial variants sharing one physical bottom profile."""

    if not isinstance(family, str) or not family.strip():
        raise KeyError("Labcyte plate family must be a non-empty string")
    key = family.strip().casefold()
    matches = tuple(
        plate for plate in LABCYTE_PLATES if plate.family.casefold() == key
    )
    if not matches:
        available = ", ".join(dict.fromkeys(p.family for p in LABCYTE_PLATES))
        raise KeyError(f"Unknown Labcyte family {family!r}; available: {available}")
    return matches
