"""Intensity sweep at an optimally focused 80% DMSO-air meniscus."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from angular_spectrum import (
    AngularSpectrumModel,
    CartesianGrid,
    ElasticPlate,
    ElasticSolid,
    Fluid,
    FocusedCircularAperture,
    dmso_water_properties,
    optimal_meniscus_intensity_sweep,
)


FREQUENCY_HZ = 10.0e6
WATER_PATH_M = 25.3e-3
PP_THICKNESS_M = 0.78e-3
DMSO_VOLUME_FRACTION = 0.80
TEMPERATURE_C = 22.0
MIN_HEIGHT_M = 2.0e-3
MAX_HEIGHT_M = 3.0e-3
HEIGHT_SAMPLES = 501
OUTPUT_DIRECTORY = Path("results")


def local_maxima(values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
    ) + 1


def local_minima(values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        (values[1:-1] < values[:-2])
        & (values[1:-1] <= values[2:])
    ) + 1


def main() -> None:
    dmso_properties = dmso_water_properties(
        DMSO_VOLUME_FRACTION,
        basis="volume",
        temperature_c=TEMPERATURE_C,
    )
    water = Fluid("water_22C", 997.77, 1488.4)
    dmso = Fluid(
        "80volpct_DMSO_22C",
        dmso_properties.density_kg_m3,
        dmso_properties.sound_speed_m_s,
    )
    air = Fluid("air_22C", 1.196, 344.0)
    polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=900.0,
        longitudinal_speed_m_s=2732.0,
        poisson_ratio=0.42,
    )
    model = AngularSpectrumModel(
        grid=CartesianGrid(nx=384, ny=384, dx_m=50e-6),
        aperture=FocusedCircularAperture(
            diameter_m=13.0e-3,
            focal_length_m=25.0e-3,
        ),
        incident_fluid=water,
        plate=ElasticPlate(polypropylene, PP_THICKNESS_M),
        transmitted_fluid=dmso,
        water_path_m=WATER_PATH_M,
        plate_radial_samples=4096,
    )
    heights_m = np.linspace(
        MIN_HEIGHT_M,
        MAX_HEIGHT_M,
        HEIGHT_SAMPLES,
    )
    result = optimal_meniscus_intensity_sweep(
        model,
        FREQUENCY_HZ,
        heights_m,
        backing_fluid=air,
    )

    baseline = result.single_pass_intensity_w_m2
    reverberant = result.intensity_with_reverberation_w_m2
    reference = float(np.mean(baseline))
    baseline_relative = baseline / reference
    reverberant_relative = reverberant / reference
    gain_db = 10.0 * np.log10(result.interference_gain)

    maxima = local_maxima(reverberant_relative)
    minima = local_minima(reverberant_relative)
    strongest_maximum = int(maxima[np.argmax(reverberant_relative[maxima])])
    strongest_minimum = int(minima[np.argmin(reverberant_relative[minima])])
    detrended_gain = gain_db - np.polyval(
        np.polyfit(heights_m, gain_db, 2),
        heights_m,
    )
    spatial_frequency_m_inv = np.fft.rfftfreq(
        heights_m.size,
        d=float(heights_m[1] - heights_m[0]),
    )
    gain_spectrum = np.abs(np.fft.rfft(detrended_gain))
    plausible_fringe_band = (
        (spatial_frequency_m_inv > 1.0 / 0.2e-3)
        & (spatial_frequency_m_inv < 1.0 / 0.04e-3)
    )
    dominant_index = np.flatnonzero(plausible_fringe_band)[
        np.argmax(gain_spectrum[plausible_fringe_band])
    ]
    fringe_spacing_m = float(
        1.0 / spatial_frequency_m_inv[dominant_index]
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        heights_m * 1e3,
        reverberant_relative,
        label="mit DMSO–Luft-Reflexion",
        color="#355f8a",
    )
    axes[0].plot(
        heights_m * 1e3,
        baseline_relative,
        label="ohne Mehrfachreflexion",
        color="#d1495b",
        linestyle="--",
    )
    axes[0].scatter(
        heights_m[[strongest_maximum, strongest_minimum]] * 1e3,
        reverberant_relative[[strongest_maximum, strongest_minimum]],
        color="#355f8a",
        zorder=3,
    )
    axes[0].annotate(
        f"Maximum {heights_m[strongest_maximum] * 1e3:.3f} mm",
        (
            heights_m[strongest_maximum] * 1e3,
            reverberant_relative[strongest_maximum],
        ),
        xytext=(8, -18),
        textcoords="offset points",
    )
    axes[0].annotate(
        f"Minimum {heights_m[strongest_minimum] * 1e3:.3f} mm",
        (
            heights_m[strongest_minimum] * 1e3,
            reverberant_relative[strongest_minimum],
        ),
        xytext=(8, 10),
        textcoords="offset points",
    )
    axes[0].set(
        ylabel="Intensität / mittlere Einweg-Intensität",
        title="Ideal auf den Meniskus fokussierte Vorwärtsintensität",
    )
    axes[0].legend()

    axes[1].plot(
        heights_m * 1e3,
        gain_db,
        color="#355f8a",
    )
    axes[1].axhline(0.0, color="0.45", linewidth=1.0, linestyle="--")
    axes[1].set(
        xlabel="DMSO-Füllhöhe [mm]",
        ylabel="Interferenzverstärkung [dB]",
    )
    for axis in axes:
        axis.grid(alpha=0.23)

    figure_path = OUTPUT_DIRECTORY / "meniscus_intensity_sweep.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)

    np.savez_compressed(
        OUTPUT_DIRECTORY / "meniscus_intensity_sweep.npz",
        height_m=heights_m,
        intensity_with_reverberation_w_m2=reverberant,
        single_pass_intensity_w_m2=baseline,
        interference_gain=result.interference_gain,
        interference_gain_db=gain_db,
    )
    with (OUTPUT_DIRECTORY / "meniscus_intensity_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "height_mm",
                "intensity_with_reverberation_w_m2_for_1Pa_aperture",
                "single_pass_intensity_w_m2_for_1Pa_aperture",
                "interference_gain",
                "interference_gain_db",
            ]
        )
        writer.writerows(
            zip(
                heights_m * 1e3,
                reverberant,
                baseline,
                result.interference_gain,
                gain_db,
            )
        )

    summary = {
        "frequency_hz": FREQUENCY_HZ,
        "geometry": {
            "water_path_to_pp_front_m": WATER_PATH_M,
            "pp_thickness_m": PP_THICKNESS_M,
            "dmso_height_range_m": [MIN_HEIGHT_M, MAX_HEIGHT_M],
            "backing": "air",
            "meniscus": "planar",
        },
        "dmso": {
            "volume_fraction": DMSO_VOLUME_FRACTION,
            "temperature_c": TEMPERATURE_C,
            "density_kg_m3": dmso.density_kg_m3,
            "sound_speed_m_s": dmso.sound_speed_m_s,
        },
        "interference": {
            "strongest_maximum_height_m": float(
                heights_m[strongest_maximum]
            ),
            "strongest_minimum_height_m": float(
                heights_m[strongest_minimum]
            ),
            "maximum_gain_db": float(np.max(gain_db)),
            "minimum_gain_db": float(np.min(gain_db)),
            "peak_to_valley_db": float(np.max(gain_db) - np.min(gain_db)),
            "dominant_fringe_period_m": fringe_spacing_m,
            "half_wavelength_m": (
                dmso.sound_speed_m_s / (2.0 * FREQUENCY_HZ)
            ),
        },
        "normalization": (
            "relative curves use the mean optimally focused single-pass "
            "intensity; W/m2 values assume 1 Pa aperture pressure"
        ),
        "assumptions": [
            "ideal phase-only refocusing to every meniscus height",
            "forward-wave intensity immediately below the meniscus",
            "flat DMSO-air interface",
            "zero material attenuation because measured values were not supplied",
        ],
    }
    with (OUTPUT_DIRECTORY / "meniscus_intensity_sweep.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        f"Interferenz: {float(np.min(gain_db)):.2f} bis "
        f"{float(np.max(gain_db)):.2f} dB"
    )
    print(
        f"Fransenabstand: {fringe_spacing_m * 1e3:.4f} mm; "
        f"λ/2: {dmso.sound_speed_m_s / (2.0 * FREQUENCY_HZ) * 1e3:.4f} mm"
    )
    print(f"Plot: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
