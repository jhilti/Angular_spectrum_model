"""Focus sweep for a 4.22 mm layer containing 70-100% DMSO.

Run from the project directory after installing the package:

    python examples/dmso_concentration_sweep.py

Edit the constants below to match the actual experiment.  Concentrations are
interpreted as volume fractions by default; use "mass" for weight percent.
"""

from argparse import Namespace
from pathlib import Path

from angular_spectrum.dmso_sweep import run


# Experiment and sweep settings
CONCENTRATION_BASIS = "volume"  # "volume", "mass", or "mole"
MIN_DMSO_PERCENT = 70.0
MAX_DMSO_PERCENT = 100.0
STEP_PERCENT = 1.0
TEMPERATURE_C = 22.0  # The mixture-property model currently uses 22 degC.

DMSO_HEIGHT_MM = 4.22
WATER_PATH_MM = 20.0
PP_THICKNESS_MM = 0.78
FREQUENCY_MHZ = 10.0

# Numerical settings used for the converged result.
GRID_SIZE = 512
GRID_SPACING_UM = 40.0
PLATE_RADIAL_SAMPLES = 4096
OUTPUT_DIRECTORY = Path("results")


def main() -> None:
    if TEMPERATURE_C != 22.0:
        raise ValueError(
            "This example is fixed to 22 degC. Change dmso_sweep.py as well "
            "if a different interpolation temperature is required."
        )

    settings = Namespace(
        basis=CONCENTRATION_BASIS,
        min_percent=MIN_DMSO_PERCENT,
        max_percent=MAX_DMSO_PERCENT,
        step_percent=STEP_PERCENT,
        dmso_height_mm=DMSO_HEIGHT_MM,
        water_path_mm=WATER_PATH_MM,
        plate_thickness_mm=PP_THICKNESS_MM,
        frequency_mhz=FREQUENCY_MHZ,
        output_dir=OUTPUT_DIRECTORY,
        nx=GRID_SIZE,
        dx_um=GRID_SPACING_UM,
        radial_samples=PLATE_RADIAL_SAMPLES,
    )
    rows = run(settings)

    print(
        "DMSO [%]  c [m/s]  Fokus ab PP [mm]  "
        "Fokus ab Apertur [mm]  unter Oberkante [mm]"
    )
    for row in rows:
        concentration = float(row["dmso_percent"])
        if concentration % 5.0 == 0.0:
            print(
                f"{concentration:7.1f}"
                f"  {float(row['sound_speed_m_s']):7.1f}"
                f"  {float(row['focus_from_plate_exit_mm']):17.3f}"
                f"  {float(row['focus_from_aperture_mm']):22.3f}"
                f"  {float(row['distance_focus_to_mixture_top_mm']):20.3f}"
            )

    print(f"\nErgebnisse: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
