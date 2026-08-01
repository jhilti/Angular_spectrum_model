"""Generate the README cross-section from the same code as the web app."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from angular_spectrum.app_model import SimulationInputs, run_interactive_simulation
from angular_spectrum.labware import get_labcyte_plate
from angular_spectrum.schematic import acoustic_stack_schematic_figure


def main() -> None:
    inputs = SimulationInputs()
    result = run_interactive_simulation(inputs)
    plate = get_labcyte_plate(inputs.plate_part_number)
    figure = acoustic_stack_schematic_figure(
        result.inputs,
        result.focus_from_aperture_mm,
        plate,
    )
    output = Path("results/streamlit_acoustic_stack.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
