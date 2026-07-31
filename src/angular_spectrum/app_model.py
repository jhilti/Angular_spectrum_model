"""High-level calculations used by the interactive Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dmso_mixture import DMSOWaterProperties, dmso_water_properties
from .grid import CartesianGrid
from .materials import ElasticPlate, ElasticSolid, Fluid
from .model import AngularSpectrumModel, FocusedCircularAperture
from .pulse import asymmetric_gaussian_response
from .pulse_echo import simulate_monostatic_pulse_echo, sine_burst


WATER_SOUND_SPEED_M_S = 1488.4
WATER_DENSITY_KG_M3 = 997.77
PP_LONGITUDINAL_SPEED_M_S = 2732.0
PP_DENSITY_KG_M3 = 900.0
PP_POISSON_RATIO = 0.42

TRANSDUCER_CENTER_FREQUENCY_HZ = 9.97e6
TRANSDUCER_PEAK_FREQUENCY_HZ = 11.29e6
TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB_ROUND_TRIP = 1.0822
TRANSDUCER_LOWER_FREQUENCY_6DB_HZ = (
    TRANSDUCER_CENTER_FREQUENCY_HZ
    * (1.0 - TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB_ROUND_TRIP / 2.0)
)
TRANSDUCER_UPPER_FREQUENCY_6DB_HZ = (
    TRANSDUCER_CENTER_FREQUENCY_HZ
    * (1.0 + TRANSDUCER_FRACTIONAL_BANDWIDTH_6DB_ROUND_TRIP / 2.0)
)


@dataclass(frozen=True)
class NumericalPreset:
    """Sampling settings for an interactive calculation."""

    grid_size: int
    grid_spacing_m: float
    radial_samples: int
    sample_rate_hz: float
    relative_spectrum_threshold: float


NUMERICAL_PRESETS = {
    "Fast": NumericalPreset(
        grid_size=256,
        grid_spacing_m=62.5e-6,
        radial_samples=768,
        sample_rate_hz=80.0e6,
        relative_spectrum_threshold=7.0e-3,
    ),
    "Accurate": NumericalPreset(
        grid_size=384,
        grid_spacing_m=50.0e-6,
        radial_samples=2048,
        sample_rate_hz=160.0e6,
        relative_spectrum_threshold=2.0e-3,
    ),
}


@dataclass(frozen=True)
class SimulationInputs:
    """User-editable physical and numerical inputs."""

    dmso_volume_percent: float = 80.0
    temperature_c: float = 22.0
    water_path_mm: float = 25.3
    plate_thickness_mm: float = 0.78
    fluid_height_mm: float = 4.22
    excitation_frequency_mhz: float = 10.0
    excitation_cycles: float = 1.0
    transducer_diameter_mm: float = 13.0
    transducer_focal_length_mm: float = 25.4
    numerical_preset: str = "Fast"

    def validate(self) -> None:
        if not 0.0 <= self.dmso_volume_percent <= 100.0:
            raise ValueError("DMSO volume percent must lie between 0 and 100")
        if not 20.0 <= self.temperature_c <= 40.0:
            raise ValueError(
                "temperature must lie between 20 and 40 °C for the property model"
            )
        positive = {
            "water path": self.water_path_mm,
            "PP thickness": self.plate_thickness_mm,
            "fluid height": self.fluid_height_mm,
            "excitation frequency": self.excitation_frequency_mhz,
            "excitation cycles": self.excitation_cycles,
            "transducer diameter": self.transducer_diameter_mm,
            "transducer focal length": self.transducer_focal_length_mm,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.numerical_preset not in NUMERICAL_PRESETS:
            raise ValueError(
                f"numerical preset must be one of {tuple(NUMERICAL_PRESETS)}"
            )


@dataclass(frozen=True)
class InterfaceArrivals:
    """Geometric pulse-echo arrival times."""

    water_pp_s: float
    pp_fluid_s: float
    fluid_air_s: float

    @property
    def relative_to_water_pp_us(self) -> dict[str, float]:
        return {
            "Water–PP": 0.0,
            "PP–DMSO": (self.pp_fluid_s - self.water_pp_s) * 1e6,
            "DMSO–air": (self.fluid_air_s - self.water_pp_s) * 1e6,
        }


@dataclass(frozen=True)
class InteractiveSimulationResult:
    """Compact simulation result ready for plotting or download."""

    inputs: SimulationInputs
    dmso_properties: DMSOWaterProperties
    arrivals: InterfaceArrivals
    time_relative_us: NDArray[np.float64]
    received_normalized: NDArray[np.float64]
    plate_normalized: NDArray[np.float64]
    surface_normalized: NDArray[np.float64]
    envelope_normalized: NDArray[np.float64]
    frequency_mhz: NDArray[np.float64]
    received_spectrum_db: NDArray[np.float64]
    axial_position_after_pp_mm: NDArray[np.float64]
    axial_intensity_normalized: NDArray[np.float64]
    focus_after_pp_mm: float
    focus_from_aperture_mm: float
    focus_offset_from_meniscus_mm: float
    focus_scan_boundary_limited: bool
    water_path_scan_mm: NDArray[np.float64]
    meniscus_intensity_normalized: NDArray[np.float64]
    optimal_water_path_mm: float
    optimal_water_path_gain_db: float
    simulated_frequency_bin_count: int


def analytic_envelope(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the magnitude of an FFT-based analytic signal."""

    values = np.asarray(signal, dtype=float)
    spectrum = np.fft.fft(values)
    multiplier = np.zeros(values.size)
    multiplier[0] = 1.0
    if values.size % 2 == 0:
        multiplier[1 : values.size // 2] = 2.0
        multiplier[values.size // 2] = 1.0
    else:
        multiplier[1 : (values.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def _quadratic_peak_position(
    coordinate: NDArray[np.float64],
    amplitude: NDArray[np.float64],
) -> tuple[float, bool]:
    index = int(np.argmax(amplitude))
    boundary_limited = index in {0, coordinate.size - 1}
    if boundary_limited:
        return float(coordinate[index]), True
    denominator = (
        amplitude[index - 1]
        - 2.0 * amplitude[index]
        + amplitude[index + 1]
    )
    if denominator == 0.0:
        return float(coordinate[index]), False
    offset = 0.5 * (
        amplitude[index - 1] - amplitude[index + 1]
    ) / denominator
    return (
        float(coordinate[index] + offset * (coordinate[1] - coordinate[0])),
        False,
    )


def interface_arrivals(
    inputs: SimulationInputs,
    *,
    fluid_sound_speed_m_s: float,
) -> InterfaceArrivals:
    """Return geometric echo times for the three interfaces."""

    water_pp_s = (
        2.0 * inputs.water_path_mm * 1e-3 / WATER_SOUND_SPEED_M_S
    )
    pp_fluid_s = (
        water_pp_s
        + 2.0
        * inputs.plate_thickness_mm
        * 1e-3
        / PP_LONGITUDINAL_SPEED_M_S
    )
    fluid_air_s = (
        pp_fluid_s
        + 2.0
        * inputs.fluid_height_mm
        * 1e-3
        / fluid_sound_speed_m_s
    )
    return InterfaceArrivals(water_pp_s, pp_fluid_s, fluid_air_s)


def _build_model(
    inputs: SimulationInputs,
) -> tuple[
    AngularSpectrumModel,
    Fluid,
    Fluid,
    Fluid,
    DMSOWaterProperties,
]:
    preset = NUMERICAL_PRESETS[inputs.numerical_preset]
    dmso_properties = dmso_water_properties(
        inputs.dmso_volume_percent / 100.0,
        basis="volume",
        temperature_c=inputs.temperature_c,
    )
    water = Fluid(
        f"water_{inputs.temperature_c:g}C",
        WATER_DENSITY_KG_M3,
        WATER_SOUND_SPEED_M_S,
    )
    dmso = Fluid(
        f"{inputs.dmso_volume_percent:g}volpct_DMSO",
        dmso_properties.density_kg_m3,
        dmso_properties.sound_speed_m_s,
    )
    air = Fluid("air", 1.196, 344.0)
    polypropylene = ElasticSolid.from_longitudinal_speed_and_poisson(
        name="polypropylene",
        density_kg_m3=PP_DENSITY_KG_M3,
        longitudinal_speed_m_s=PP_LONGITUDINAL_SPEED_M_S,
        poisson_ratio=PP_POISSON_RATIO,
    )
    model = AngularSpectrumModel(
        grid=CartesianGrid(
            nx=preset.grid_size,
            ny=preset.grid_size,
            dx_m=preset.grid_spacing_m,
        ),
        aperture=FocusedCircularAperture(
            diameter_m=inputs.transducer_diameter_mm * 1e-3,
            focal_length_m=inputs.transducer_focal_length_mm * 1e-3,
        ),
        incident_fluid=water,
        plate=ElasticPlate(
            polypropylene,
            inputs.plate_thickness_mm * 1e-3,
        ),
        transmitted_fluid=dmso,
        water_path_m=inputs.water_path_mm * 1e-3,
        plate_radial_samples=preset.radial_samples,
    )
    return model, water, dmso, air, dmso_properties


def _focus_scans(
    model: AngularSpectrumModel,
    inputs: SimulationInputs,
    frequency_hz: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    bool,
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
]:
    fluid_height_m = inputs.fluid_height_mm * 1e-3
    paraxial_focus_after_pp_m = (
        (
            inputs.transducer_focal_length_mm
            - inputs.water_path_mm
        )
        * 1e-3
        * model.transmitted_fluid.sound_speed_m_s
        / model.incident_fluid.sound_speed_m_s
        - model.plate.thickness_m
        * model.transmitted_fluid.sound_speed_m_s
        / model.plate.solid.longitudinal_speed_m_s
    )
    axial_end_m = max(
        8.0e-3,
        1.5 * fluid_height_m,
        paraxial_focus_after_pp_m + 4.0e-3,
    )
    axial_end_m = min(axial_end_m, 35.0e-3)
    axial_z_m = np.linspace(0.0, axial_end_m, 181)
    axial_pressure = model.on_axis_scan_after_plate(
        frequency_hz,
        axial_z_m,
    )
    axial_intensity = np.abs(axial_pressure) ** 2
    focus_after_pp_m, focus_boundary_limited = _quadratic_peak_position(
        axial_z_m,
        axial_intensity,
    )
    axial_scale = max(float(np.max(axial_intensity)), 1e-30)

    estimated_optimum_m = (
        inputs.transducer_focal_length_mm * 1e-3
        - model.incident_fluid.sound_speed_m_s
        * (
            model.plate.thickness_m
            / model.plate.solid.longitudinal_speed_m_s
            + fluid_height_m / model.transmitted_fluid.sound_speed_m_s
        )
    )
    current_water_path_m = inputs.water_path_mm * 1e-3
    lower_m = max(
        0.25e-3,
        min(estimated_optimum_m - 3.0e-3, current_water_path_m - 2.0e-3),
    )
    upper_m = max(
        estimated_optimum_m + 3.0e-3,
        current_water_path_m + 2.0e-3,
        lower_m + 1.0e-3,
    )
    water_path_m = np.linspace(lower_m, upper_m, 91)

    source_spectrum = model.source_spectrum(frequency_hz)
    plate_transfer = model.plate_transfer_map(frequency_hz)
    fluid_propagation = model._propagator(
        model.transmitted_fluid,
        frequency_hz,
        fluid_height_m,
    )
    centre_phase = model.grid.centre_ifft_phase()
    fixed_spectrum = (
        source_spectrum
        * plate_transfer
        * fluid_propagation
        * centre_phase
    )
    normalization = model.grid.nx * model.grid.ny
    meniscus_pressure = np.empty(water_path_m.size, dtype=np.complex128)
    for index, water_path in enumerate(water_path_m):
        water_propagation = model._propagator(
            model.incident_fluid,
            frequency_hz,
            float(water_path),
        )
        meniscus_pressure[index] = (
            np.sum(fixed_spectrum * water_propagation) / normalization
        )
    meniscus_intensity = np.abs(meniscus_pressure) ** 2
    optimal_water_path_m, _ = _quadratic_peak_position(
        water_path_m,
        meniscus_intensity,
    )
    intensity_scale = max(float(np.max(meniscus_intensity)), 1e-30)
    current_intensity = float(
        np.interp(current_water_path_m, water_path_m, meniscus_intensity)
    )
    optimal_gain_db = 10.0 * np.log10(
        intensity_scale / max(current_intensity, 1e-30)
    )

    return (
        axial_z_m * 1e3,
        axial_intensity / axial_scale,
        focus_after_pp_m * 1e3,
        focus_boundary_limited,
        water_path_m * 1e3,
        meniscus_intensity / intensity_scale,
        optimal_water_path_m * 1e3,
        float(optimal_gain_db),
    )


def run_interactive_simulation(
    inputs: SimulationInputs,
) -> InteractiveSimulationResult:
    """Run pulse echo, focus location and meniscus-focus optimization."""

    inputs.validate()
    preset = NUMERICAL_PRESETS[inputs.numerical_preset]
    model, _, dmso, air, dmso_properties = _build_model(inputs)
    arrivals = interface_arrivals(
        inputs,
        fluid_sound_speed_m_s=dmso.sound_speed_m_s,
    )
    record_length_s = max(
        20.0e-6,
        arrivals.fluid_air_s + 4.0e-6,
    )
    time_s, drive = sine_burst(
        center_frequency_hz=inputs.excitation_frequency_mhz * 1e6,
        cycles=inputs.excitation_cycles,
        sample_rate_hz=preset.sample_rate_hz,
        record_length_s=record_length_s,
        start_time_s=0.0,
    )

    def certificate_round_trip_response(
        frequency_hz: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return asymmetric_gaussian_response(
            frequency_hz,
            peak_frequency_hz=TRANSDUCER_PEAK_FREQUENCY_HZ,
            lower_frequency_6db_hz=TRANSDUCER_LOWER_FREQUENCY_6DB_HZ,
            upper_frequency_6db_hz=TRANSDUCER_UPPER_FREQUENCY_6DB_HZ,
        )

    pulse_echo = simulate_monostatic_pulse_echo(
        model,
        time_s,
        drive,
        fluid_layer_thickness_m=inputs.fluid_height_mm * 1e-3,
        backing_fluid=air,
        round_trip_response=certificate_round_trip_response,
        relative_spectrum_threshold=preset.relative_spectrum_threshold,
        minimum_frequency_hz=2.5e6,
        maximum_frequency_hz=20.0e6,
    )
    signal_scale = max(
        float(np.max(np.abs(pulse_echo.received_signal))),
        1e-30,
    )
    received = pulse_echo.received_signal / signal_scale
    plate = pulse_echo.plate_front_signal / signal_scale
    surface = pulse_echo.backing_signal / signal_scale
    envelope = analytic_envelope(received)
    envelope /= max(float(np.max(envelope)), 1e-30)

    spectral_magnitude = np.abs(pulse_echo.received_spectrum)
    spectral_scale = max(float(np.max(spectral_magnitude)), 1e-30)
    spectral_db = 20.0 * np.log10(
        np.maximum(spectral_magnitude / spectral_scale, 1e-6)
    )

    (
        axial_position_mm,
        axial_intensity,
        focus_after_pp_mm,
        focus_boundary_limited,
        water_path_scan_mm,
        meniscus_intensity,
        optimal_water_path_mm,
        optimal_gain_db,
    ) = _focus_scans(
        model,
        inputs,
        inputs.excitation_frequency_mhz * 1e6,
    )
    focus_from_aperture_mm = (
        inputs.water_path_mm
        + inputs.plate_thickness_mm
        + focus_after_pp_mm
    )

    return InteractiveSimulationResult(
        inputs=inputs,
        dmso_properties=dmso_properties,
        arrivals=arrivals,
        time_relative_us=(pulse_echo.time_s - arrivals.water_pp_s) * 1e6,
        received_normalized=received,
        plate_normalized=plate,
        surface_normalized=surface,
        envelope_normalized=envelope,
        frequency_mhz=pulse_echo.frequency_hz * 1e-6,
        received_spectrum_db=spectral_db,
        axial_position_after_pp_mm=axial_position_mm,
        axial_intensity_normalized=axial_intensity,
        focus_after_pp_mm=focus_after_pp_mm,
        focus_from_aperture_mm=focus_from_aperture_mm,
        focus_offset_from_meniscus_mm=(
            focus_after_pp_mm - inputs.fluid_height_mm
        ),
        focus_scan_boundary_limited=focus_boundary_limited,
        water_path_scan_mm=water_path_scan_mm,
        meniscus_intensity_normalized=meniscus_intensity,
        optimal_water_path_mm=optimal_water_path_mm,
        optimal_water_path_gain_db=optimal_gain_db,
        simulated_frequency_bin_count=int(
            np.count_nonzero(pulse_echo.simulated_bin_mask)
        ),
    )
