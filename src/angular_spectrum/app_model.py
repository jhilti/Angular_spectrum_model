"""High-level calculations used by the interactive Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .analysis import fwhm
from .dmso_mixture import (
    DMSOWaterProperties,
    WaterProperties,
    dmso_water_properties,
    water_properties,
)
from .grid import CartesianGrid
from .materials import ElasticPlate, ElasticSolid, Fluid
from .meniscus import (
    MeniscusCavityMetric,
    configured_meniscus_cavity_metric,
)
from .model import (
    AngularSpectrumModel,
    FocusedCircularAperture,
    validate_focused_grid_support,
)
from .pulse import asymmetric_gaussian_response, smooth_dc_block_response
from .pulse_echo import simulate_monostatic_pulse_echo, sine_burst


_WATER_22C = water_properties(22.0)
# Backward-compatible 22 °C aliases. Interactive calculations use the selected
# temperature through ``water_properties`` rather than these constants.
WATER_SOUND_SPEED_M_S = _WATER_22C.sound_speed_m_s
WATER_DENSITY_KG_M3 = _WATER_22C.density_kg_m3
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
    echo_grid_spacing_m: float | None = None
    echo_grid_size: int | None = None


NUMERICAL_PRESETS = {
    "Fast": NumericalPreset(
        grid_size=256,
        grid_spacing_m=80.0e-6,
        radial_samples=768,
        sample_rate_hz=80.0e6,
        relative_spectrum_threshold=7.0e-3,
        # The two-way water path needs a wider FFT window than the focal-field
        # calculation.  Coarser transverse sampling is still well above the
        # highest focused-aperture spatial frequency retained below 25 MHz.
        echo_grid_spacing_m=100.0e-6,
        echo_grid_size=320,
    ),
    "Accurate": NumericalPreset(
        grid_size=384,
        grid_spacing_m=50.0e-6,
        radial_samples=2048,
        sample_rate_hz=160.0e6,
        relative_spectrum_threshold=2.0e-3,
        echo_grid_spacing_m=75.0e-6,
        echo_grid_size=448,
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
    plate_part_number: str = "PP-0200"
    plate_material_name: str = "polypropylene"
    plate_longitudinal_speed_m_s: float = PP_LONGITUDINAL_SPEED_M_S
    plate_density_kg_m3: float = PP_DENSITY_KG_M3
    plate_poisson_ratio: float = PP_POISSON_RATIO
    pp_longitudinal_attenuation_db_per_m: float = 0.0
    pp_shear_attenuation_db_per_m: float = 0.0
    fluid_attenuation_db_per_m: float = 0.0
    attenuation_power: float = 1.0
    fill_height_uncertainty_mm: float = 0.05
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
            "plate thickness": self.plate_thickness_mm,
            "fluid height": self.fluid_height_mm,
            "excitation frequency": self.excitation_frequency_mhz,
            "excitation cycles": self.excitation_cycles,
            "transducer diameter": self.transducer_diameter_mm,
            "transducer focal length": self.transducer_focal_length_mm,
            "plate longitudinal speed": self.plate_longitudinal_speed_m_s,
            "plate density": self.plate_density_kg_m3,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        nonnegative = {
            "plate longitudinal attenuation": (
                self.pp_longitudinal_attenuation_db_per_m
            ),
            "plate shear attenuation": self.pp_shear_attenuation_db_per_m,
            "fluid attenuation": self.fluid_attenuation_db_per_m,
            "attenuation power": self.attenuation_power,
            "fill height uncertainty": self.fill_height_uncertainty_mm,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.fill_height_uncertainty_mm >= self.fluid_height_mm:
            raise ValueError(
                "fill height uncertainty must be smaller than fluid height"
            )
        if not self.plate_part_number.strip():
            raise ValueError("plate part number must not be empty")
        if not self.plate_material_name.strip():
            raise ValueError("plate material name must not be empty")
        if (
            not np.isfinite(self.plate_poisson_ratio)
            or not 0.0 <= self.plate_poisson_ratio < 0.5
        ):
            raise ValueError(
                "plate Poisson ratio must be finite and lie in [0, 0.5)"
            )
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
    def since_excitation_us(self) -> dict[str, float]:
        """Interface arrival times measured from the excitation start."""

        return {
            "Water–plate": self.water_pp_s * 1e6,
            "Plate–DMSO": self.pp_fluid_s * 1e6,
            "DMSO–air": self.fluid_air_s * 1e6,
        }

    @property
    def relative_to_water_pp_us(self) -> dict[str, float]:
        return {
            "Water–plate": 0.0,
            "Plate–DMSO": (self.pp_fluid_s - self.water_pp_s) * 1e6,
            "DMSO–air": (self.fluid_air_s - self.water_pp_s) * 1e6,
        }


@dataclass(frozen=True)
class InteractiveSimulationResult:
    """Compact simulation result ready for plotting or download."""

    inputs: SimulationInputs
    water_properties: WaterProperties
    dmso_properties: DMSOWaterProperties
    arrivals: InterfaceArrivals
    time_since_excitation_us: NDArray[np.float64]
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
    optimal_meniscus_intensity_fwhm_mm: float
    optimal_water_path_gain_db: float
    optimal_water_path_boundary_limited: bool
    simulated_frequency_bin_count: int
    fluid_cavity_echo_count: int
    meniscus_cavity: MeniscusCavityMetric

    @property
    def time_relative_us(self) -> NDArray[np.float64]:
        """Backward-compatible time axis relative to the water–PP echo."""

        return (
            self.time_since_excitation_us
            - self.arrivals.water_pp_s * 1e6
        )


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


def _paraxial_focus_after_plate_m(
    model: AngularSpectrumModel,
    inputs: SimulationInputs,
) -> float:
    """Small-angle focus after water and a longitudinal plate traversal."""

    return float(
        (
            (
                inputs.transducer_focal_length_mm
                - inputs.water_path_mm
            )
            * 1e-3
            * model.incident_fluid.sound_speed_m_s
            - model.plate.thickness_m
            * model.plate.solid.longitudinal_speed_m_s
        )
        / model.transmitted_fluid.sound_speed_m_s
    )


def _paraxial_water_path_for_target_m(
    model: AngularSpectrumModel,
    inputs: SimulationInputs,
) -> float:
    """Small-angle water gap that places the focus at the meniscus."""

    return float(
        inputs.transducer_focal_length_mm * 1e-3
        - (
            model.plate.thickness_m
            * model.plate.solid.longitudinal_speed_m_s
            + inputs.fluid_height_mm
            * 1e-3
            * model.transmitted_fluid.sound_speed_m_s
        )
        / model.incident_fluid.sound_speed_m_s
    )


def interface_arrivals(
    inputs: SimulationInputs,
    *,
    fluid_sound_speed_m_s: float,
    water_sound_speed_m_s: float | None = None,
) -> InterfaceArrivals:
    """Return geometric echo times for the three interfaces."""

    if water_sound_speed_m_s is None:
        water_sound_speed_m_s = water_properties(
            inputs.temperature_c
        ).sound_speed_m_s
    if (
        not np.isfinite(water_sound_speed_m_s)
        or water_sound_speed_m_s <= 0.0
    ):
        raise ValueError("water_sound_speed_m_s must be finite and > 0")
    water_pp_s = (
        2.0 * inputs.water_path_mm * 1e-3 / water_sound_speed_m_s
    )
    pp_fluid_s = (
        water_pp_s
        + 2.0
        * inputs.plate_thickness_mm
        * 1e-3
        / inputs.plate_longitudinal_speed_m_s
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
    *,
    grid_spacing_m: float | None = None,
    grid_size: int | None = None,
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
    water_data = water_properties(inputs.temperature_c)
    water = Fluid(
        f"water_{inputs.temperature_c:g}C",
        water_data.density_kg_m3,
        water_data.sound_speed_m_s,
    )
    dmso = Fluid(
        f"{inputs.dmso_volume_percent:g}volpct_DMSO",
        dmso_properties.density_kg_m3,
        dmso_properties.sound_speed_m_s,
        attenuation_db_per_m=inputs.fluid_attenuation_db_per_m,
        attenuation_power=inputs.attenuation_power,
        attenuation_reference_hz=10.0e6,
    )
    air = Fluid("air", 1.196, 344.0)
    plate_solid = ElasticSolid.from_longitudinal_speed_and_poisson(
        name=inputs.plate_material_name,
        density_kg_m3=inputs.plate_density_kg_m3,
        longitudinal_speed_m_s=inputs.plate_longitudinal_speed_m_s,
        poisson_ratio=inputs.plate_poisson_ratio,
        longitudinal_attenuation_db_per_m=(
            inputs.pp_longitudinal_attenuation_db_per_m
        ),
        shear_attenuation_db_per_m=inputs.pp_shear_attenuation_db_per_m,
        attenuation_power=inputs.attenuation_power,
        attenuation_reference_hz=10.0e6,
    )
    model = AngularSpectrumModel(
        grid=CartesianGrid(
            nx=preset.grid_size if grid_size is None else grid_size,
            ny=preset.grid_size if grid_size is None else grid_size,
            dx_m=(
                preset.grid_spacing_m
                if grid_spacing_m is None
                else grid_spacing_m
            ),
        ),
        aperture=FocusedCircularAperture(
            diameter_m=inputs.transducer_diameter_mm * 1e-3,
            focal_length_m=inputs.transducer_focal_length_mm * 1e-3,
        ),
        incident_fluid=water,
        plate=ElasticPlate(
            plate_solid,
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
    bool,
    float,
]:
    fluid_height_m = inputs.fluid_height_mm * 1e-3
    paraxial_focus_after_pp_m = _paraxial_focus_after_plate_m(model, inputs)
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

    estimated_optimum_m = _paraxial_water_path_for_target_m(model, inputs)
    current_water_path_m = inputs.water_path_mm * 1e-3
    lower_m = max(
        min(0.25e-3, current_water_path_m),
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
    meniscus_spectrum_without_water = (
        source_spectrum * plate_transfer * fluid_propagation
    )
    fixed_spectrum = meniscus_spectrum_without_water * centre_phase
    normalization = model.grid.nx * model.grid.ny
    meniscus_pressure = np.empty(water_path_m.size, dtype=np.complex128)
    for index, water_path in enumerate(water_path_m):
        water_propagation = model._propagator(
            model.incident_fluid,
            frequency_hz,
            float(water_path),
        )
        combined_support = model._combined_bandlimit_mask(
            frequency_hz,
            (
                (model.incident_fluid, float(water_path)),
                (model.transmitted_fluid, fluid_height_m),
            ),
        )
        meniscus_pressure[index] = (
            np.sum(fixed_spectrum * water_propagation * combined_support)
            / normalization
        )
    meniscus_intensity = np.abs(meniscus_pressure) ** 2
    optimal_water_path_m, optimum_boundary_limited = _quadratic_peak_position(
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
    optimal_water_propagation = model._propagator(
        model.incident_fluid,
        frequency_hz,
        optimal_water_path_m,
    )
    optimal_support = model._combined_bandlimit_mask(
        frequency_hz,
        (
            (model.incident_fluid, optimal_water_path_m),
            (model.transmitted_fluid, fluid_height_m),
        ),
    )
    optimal_field = np.fft.ifft2(
        meniscus_spectrum_without_water
        * optimal_water_propagation
        * optimal_support
    )
    centre_y, _ = model.grid.centre_index
    optimal_intensity_fwhm_m = fwhm(
        model.grid.x_m,
        np.abs(optimal_field[centre_y, :]) ** 2,
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
        optimum_boundary_limited,
        float(optimal_intensity_fwhm_m * 1e3),
    )


def run_interactive_simulation(
    inputs: SimulationInputs,
) -> InteractiveSimulationResult:
    """Run pulse echo, focus location and meniscus-focus optimization."""

    inputs.validate()
    preset = NUMERICAL_PRESETS[inputs.numerical_preset]
    model, water, dmso, air, dmso_properties = _build_model(inputs)
    echo_spacing_m = (
        preset.grid_spacing_m
        if preset.echo_grid_spacing_m is None
        else preset.echo_grid_spacing_m
    )
    echo_grid_size = (
        preset.grid_size
        if preset.echo_grid_size is None
        else preset.echo_grid_size
    )
    if (
        echo_spacing_m == preset.grid_spacing_m
        and echo_grid_size == preset.grid_size
    ):
        echo_model = model
    else:
        echo_model = _build_model(
            inputs,
            grid_spacing_m=echo_spacing_m,
            grid_size=echo_grid_size,
        )[0]
    maximum_frequency_hz = max(
        25.0e6,
        1.5 * inputs.excitation_frequency_mhz * 1e6,
    )
    estimated_water_path_m = _paraxial_water_path_for_target_m(
        model,
        inputs,
    )
    current_water_path_m = inputs.water_path_mm * 1e-3
    largest_focus_scan_water_path_m = max(
        estimated_water_path_m + 3.0e-3,
        current_water_path_m + 2.0e-3,
        1.25e-3,
    )
    paraxial_focus_m = _paraxial_focus_after_plate_m(model, inputs)
    largest_axial_scan_m = min(
        max(
            8.0e-3,
            1.5 * inputs.fluid_height_mm * 1e-3,
            paraxial_focus_m + 4.0e-3,
        ),
        35.0e-3,
    )
    validate_focused_grid_support(
        model,
        maximum_frequency_hz=maximum_frequency_hz,
        propagation_segments=(
            (
                "one-way water path and water-gap focus scan",
                water,
                largest_focus_scan_water_path_m,
            ),
            (
                "one-way liquid focal scan",
                dmso,
                largest_axial_scan_m,
            ),
        ),
    )
    validate_focused_grid_support(
        echo_model,
        maximum_frequency_hz=maximum_frequency_hz,
        propagation_segments=(
            (
                "water pulse-echo round trip",
                water,
                2.0 * current_water_path_m,
            ),
            (
                "liquid-layer cavity round trip",
                dmso,
                2.0 * inputs.fluid_height_mm * 1e-3,
            ),
        ),
    )
    arrivals = interface_arrivals(
        inputs,
        fluid_sound_speed_m_s=dmso.sound_speed_m_s,
        water_sound_speed_m_s=water.sound_speed_m_s,
    )
    cavity_round_trip_s = (
        2.0
        * inputs.fluid_height_mm
        * 1e-3
        / dmso.sound_speed_m_s
    )
    drive_duration_s = (
        inputs.excitation_cycles
        / (inputs.excitation_frequency_mhz * 1e6)
    )
    record_length_s = max(
        20.0e-6,
        # Leave a causal guard after the first liquid-surface return. The app
        # deliberately retains only that robust surface order on this grid.
        arrivals.fluid_air_s
        + 2.0 * drive_duration_s
        + max(8.0e-6, cavity_round_trip_s),
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
        ) * smooth_dc_block_response(frequency_hz)

    pulse_echo = simulate_monostatic_pulse_echo(
        echo_model,
        time_s,
        drive,
        fluid_layer_thickness_m=inputs.fluid_height_mm * 1e-3,
        backing_fluid=air,
        round_trip_response=certificate_round_trip_response,
        relative_spectrum_threshold=preset.relative_spectrum_threshold,
        minimum_frequency_hz=0.0,
        maximum_frequency_hz=maximum_frequency_hz,
        fluid_cavity_echo_count=1,
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
        optimum_boundary_limited,
        optimal_meniscus_intensity_fwhm_mm,
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
    meniscus_cavity = configured_meniscus_cavity_metric(
        model,
        inputs.excitation_frequency_mhz * 1e6,
        inputs.fluid_height_mm * 1e-3,
        backing_fluid=air,
        excitation_cycles=inputs.excitation_cycles,
        height_uncertainty_m=inputs.fill_height_uncertainty_mm * 1e-3,
        height_sensitivity_samples=9,
        cavity_relative_tolerance=1.0e-6,
        maximum_cavity_orders=128,
    )

    return InteractiveSimulationResult(
        inputs=inputs,
        water_properties=water_properties(inputs.temperature_c),
        dmso_properties=dmso_properties,
        arrivals=arrivals,
        time_since_excitation_us=pulse_echo.time_s * 1e6,
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
        optimal_meniscus_intensity_fwhm_mm=(
            optimal_meniscus_intensity_fwhm_mm
        ),
        optimal_water_path_gain_db=optimal_gain_db,
        optimal_water_path_boundary_limited=optimum_boundary_limited,
        simulated_frequency_bin_count=int(
            np.count_nonzero(pulse_echo.simulated_bin_mask)
        ),
        fluid_cavity_echo_count=pulse_echo.fluid_cavity_echo_count,
        meniscus_cavity=meniscus_cavity,
    )
