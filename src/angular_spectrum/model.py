"""FFT angular-spectrum propagation for the focused transducer case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .grid import CartesianGrid
from .materials import ElasticPlate, Fluid
from .plate import (
    elastic_plate_transfer_map,
    fluid_interface_scattering,
    vertical_wavenumber,
)


ComplexField = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def validate_focused_grid_support(
    model: "AngularSpectrumModel",
    *,
    maximum_frequency_hz: float,
    propagation_segments: Iterable[tuple[str, Fluid, float]],
) -> None:
    """Reject an FFT grid that must clip the intended focused beam.

    The aperture-edge ray defines the largest transverse wavenumber required
    by the prescribed focusing phase. It must fit both the transverse Nyquist
    limit and the band-limited angular-spectrum support of each propagation
    segment. The listed segments are consecutive parts of one optical/acoustic
    path, so their aperture-edge walkoff is also checked cumulatively. This
    check targets the focused beam itself; convergence of sharp aperture-edge
    diffraction and plate resonances still requires a separate grid-refinement
    study.
    """

    if not np.isfinite(maximum_frequency_hz) or maximum_frequency_hz <= 0.0:
        raise ValueError("maximum_frequency_hz must be finite and > 0")
    sine_edge_angle = model.aperture.radius_m / np.hypot(
        model.aperture.focal_length_m,
        model.aperture.radius_m,
    )
    q_edge = (
        2.0
        * np.pi
        * maximum_frequency_hz
        / model.incident_fluid.sound_speed_m_s
        * sine_edge_angle
    )
    q_nyquist = min(
        np.pi / model.grid.dx_m,
        np.pi / model.grid.dy_m,
    )
    if q_edge >= q_nyquist:
        raise ValueError(
            "the selected grid undersamples the focused-aperture phase at "
            f"{maximum_frequency_hz / 1e6:g} MHz; use a finer custom grid or "
            "reduce aperture/steep focusing"
        )

    segments = tuple(propagation_segments)
    cumulative_edge_walkoff_m = 0.0
    for label, medium, distance_m in segments:
        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError(
                f"propagation distance for {label} must be finite and >= 0"
            )
        if distance_m == 0.0 or not model.bandlimit:
            continue
        k_real = 2.0 * np.pi * maximum_frequency_hz / medium.sound_speed_m_s
        if q_edge >= k_real:
            raise ValueError(
                f"the focused aperture edge is evanescent in the {label}"
            )
        cumulative_edge_walkoff_m += (
            distance_m * q_edge / np.sqrt(k_real**2 - q_edge**2)
        )
        supported_q = min(
            k_real
            / np.sqrt(
                1.0
                + (2.0 * distance_m / model.grid.extent_x_m) ** 2
            ),
            k_real
            / np.sqrt(
                1.0
                + (2.0 * distance_m / model.grid.extent_y_m) ** 2
            ),
        )
        if q_edge > supported_q:
            raise ValueError(
                f"the selected FFT window is too small for the {label}; "
                "the band-limited propagator would clip rays from the "
                "aperture edge. Use a larger-window custom grid, shorten the "
                "path, or reduce aperture/steep focusing"
            )
    half_window_m = 0.5 * min(
        model.grid.extent_x_m,
        model.grid.extent_y_m,
    )
    if model.bandlimit and cumulative_edge_walkoff_m > half_window_m:
        raise ValueError(
            "the selected FFT window is too small for the combined layered "
            "path: cumulative aperture-edge walkoff exceeds half the window; "
            "use a larger-window grid, shorten the path, or reduce aperture/"
            "steep focusing"
        )


@dataclass(frozen=True)
class FocusedCircularAperture:
    """Planar equivalent of a uniformly driven focused circular transducer."""

    diameter_m: float
    focal_length_m: float
    pressure_amplitude_pa: float = 1.0
    apodization: str = "uniform"

    def __post_init__(self) -> None:
        if not np.isfinite(self.diameter_m) or self.diameter_m <= 0.0:
            raise ValueError("diameter_m must be finite and > 0")
        if not np.isfinite(self.focal_length_m) or self.focal_length_m <= 0.0:
            raise ValueError("focal_length_m must be finite and > 0")
        if (
            not np.isfinite(self.pressure_amplitude_pa)
            or self.pressure_amplitude_pa <= 0.0
        ):
            raise ValueError("pressure_amplitude_pa must be finite and > 0")
        if self.apodization not in {"uniform", "hann"}:
            raise ValueError("apodization must be 'uniform' or 'hann'")

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    def pressure_field(
        self,
        grid: CartesianGrid,
        frequency_hz: float,
        launch_medium: Fluid,
    ) -> ComplexField:
        """Return the prescribed complex pressure at the aperture plane.

        The exact spherical focusing phase is used rather than its paraxial
        quadratic approximation.
        """

        x, y = grid.spatial_mesh()
        radius = np.hypot(x, y)
        inside = radius <= self.radius_m
        path_excess = np.sqrt(self.focal_length_m**2 + radius**2) - (
            self.focal_length_m
        )
        focusing_phase = np.exp(
            -1j
            * (2.0 * np.pi * frequency_hz / launch_medium.sound_speed_m_s)
            * path_excess
        )
        if self.apodization == "uniform":
            weight = inside.astype(float)
        else:
            radial_weight = 0.5 * (
                1.0 + np.cos(np.pi * radius / self.radius_m)
            )
            weight = np.where(inside, radial_weight, 0.0)
        return (
            self.pressure_amplitude_pa * weight * focusing_phase
        ).astype(np.complex128)


@dataclass(frozen=True)
class AngularSpectrumModel:
    """Exact plane-wave propagation through fluid–elastic-plate–fluid."""

    grid: CartesianGrid
    aperture: FocusedCircularAperture
    incident_fluid: Fluid
    plate: ElasticPlate
    transmitted_fluid: Fluid
    water_path_m: float
    bandlimit: bool = True
    plate_radial_samples: int | None = 4096

    def __post_init__(self) -> None:
        if not np.isfinite(self.water_path_m) or self.water_path_m < 0.0:
            raise ValueError("water_path_m must be finite and >= 0")
        if self.aperture.diameter_m >= min(
            self.grid.extent_x_m, self.grid.extent_y_m
        ):
            raise ValueError(
                "the aperture must be smaller than both transverse grid extents"
            )
        if (
            self.plate_radial_samples is not None
            and self.plate_radial_samples < 128
        ):
            raise ValueError("plate_radial_samples must be >= 128 or None")

    def _propagator(
        self,
        medium: Fluid,
        frequency_hz: float,
        distance_m: float,
    ) -> ComplexField:
        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("propagation distance must be finite and >= 0")
        kx, ky, q = self.grid.spectral_mesh()
        kz = vertical_wavenumber(medium.wavenumber(frequency_hz), q)
        transfer = np.exp(1j * kz * distance_m)
        if self.bandlimit and distance_m > 0.0:
            k_real = 2.0 * np.pi * frequency_hz / medium.sound_speed_m_s
            kx_limit = k_real / np.sqrt(
                1.0 + (2.0 * distance_m / self.grid.extent_x_m) ** 2
            )
            ky_limit = k_real / np.sqrt(
                1.0 + (2.0 * distance_m / self.grid.extent_y_m) ** 2
            )
            supported = (np.abs(kx) <= kx_limit) & (np.abs(ky) <= ky_limit)
            transfer = np.where(supported, transfer, 0.0)
        return transfer.astype(np.complex128)

    def _combined_bandlimit_mask(
        self,
        frequency_hz: float,
        propagation_segments: Iterable[tuple[Fluid, float]],
    ) -> NDArray[np.bool_]:
        """Band limit a composed layered path by cumulative phase walkoff.

        Multiplying separately band-limited segment propagators is not enough:
        lateral phase gradients add across layers. This mask retains only FFT
        modes whose cumulative x/y walkoff remains inside half the transverse
        window. It reduces to the total-distance mask when all segments use
        one medium.
        """

        kx, ky, q = self.grid.spectral_mesh()
        if not self.bandlimit:
            return np.ones(q.shape, dtype=bool)
        walkoff_x = np.zeros(q.shape, dtype=float)
        walkoff_y = np.zeros(q.shape, dtype=float)
        supported = np.ones(q.shape, dtype=bool)
        for medium, distance_m in propagation_segments:
            if not np.isfinite(distance_m) or distance_m < 0.0:
                raise ValueError(
                    "combined propagation distances must be finite and >= 0"
                )
            if distance_m == 0.0:
                continue
            k_real = 2.0 * np.pi * frequency_hz / medium.sound_speed_m_s
            propagating = q < k_real
            kz_real = np.sqrt(np.maximum(k_real**2 - q**2, 0.0))
            supported &= propagating
            with np.errstate(divide="ignore", invalid="ignore"):
                walkoff_x += np.where(
                    propagating,
                    distance_m * np.abs(kx) / kz_real,
                    np.inf,
                )
                walkoff_y += np.where(
                    propagating,
                    distance_m * np.abs(ky) / kz_real,
                    np.inf,
                )
        supported &= walkoff_x <= 0.5 * self.grid.extent_x_m
        supported &= walkoff_y <= 0.5 * self.grid.extent_y_m
        return supported

    def source_pressure(self, frequency_hz: float) -> ComplexField:
        return self.aperture.pressure_field(
            self.grid, frequency_hz, self.incident_fluid
        )

    def source_spectrum(self, frequency_hz: float) -> ComplexField:
        return np.fft.fft2(self.source_pressure(frequency_hz))

    def incident_spectrum_at_plate(self, frequency_hz: float) -> ComplexField:
        return self.source_spectrum(frequency_hz) * self._propagator(
            self.incident_fluid, frequency_hz, self.water_path_m
        )

    def plate_transfer_map(self, frequency_hz: float) -> ComplexField:
        _, _, q = self.grid.spectral_mesh()
        return elastic_plate_transfer_map(
            q,
            frequency_hz,
            self.incident_fluid,
            self.plate,
            self.transmitted_fluid,
            radial_samples=self.plate_radial_samples,
        )

    def transmitted_spectrum_at_plate_exit(
        self, frequency_hz: float
    ) -> ComplexField:
        return self.incident_spectrum_at_plate(
            frequency_hz
        ) * self.plate_transfer_map(frequency_hz)

    def reference_spectrum_at_exit(self, frequency_hz: float) -> ComplexField:
        """Plate-free reference with the PP thickness replaced by water."""

        _, _, q = self.grid.spectral_mesh()
        water_distance = self.water_path_m + self.plate.thickness_m
        interface_transmission = fluid_interface_scattering(
            q,
            frequency_hz,
            self.incident_fluid,
            self.transmitted_fluid,
        )[1]
        return (
            self.source_spectrum(frequency_hz)
            * self._propagator(
                self.incident_fluid, frequency_hz, water_distance
            )
            * interface_transmission
        )

    def field_after_plate(
        self, frequency_hz: float, z_after_plate_m: float
    ) -> ComplexField:
        spectrum = self.transmitted_spectrum_at_plate_exit(frequency_hz)
        propagated = spectrum * self._propagator(
            self.transmitted_fluid, frequency_hz, z_after_plate_m
        )
        propagated *= self._combined_bandlimit_mask(
            frequency_hz,
            (
                (self.incident_fluid, self.water_path_m),
                (self.transmitted_fluid, z_after_plate_m),
            ),
        )
        return np.fft.ifft2(propagated)

    def reference_field(
        self, frequency_hz: float, z_after_exit_m: float
    ) -> ComplexField:
        spectrum = self.reference_spectrum_at_exit(frequency_hz)
        propagated = spectrum * self._propagator(
            self.transmitted_fluid, frequency_hz, z_after_exit_m
        )
        propagated *= self._combined_bandlimit_mask(
            frequency_hz,
            (
                (
                    self.incident_fluid,
                    self.water_path_m + self.plate.thickness_m,
                ),
                (self.transmitted_fluid, z_after_exit_m),
            ),
        )
        return np.fft.ifft2(propagated)

    def _on_axis_scan_from_exit_spectrum(
        self,
        exit_spectrum: ComplexField,
        frequency_hz: float,
        z_values_m: FloatArray,
        *,
        incident_path_m: float | None = None,
        chunk_size: int = 8,
    ) -> ComplexField:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        _, _, q = self.grid.spectral_mesh()
        kz = vertical_wavenumber(
            self.transmitted_fluid.wavenumber(frequency_hz), q
        )
        centre_phase = self.grid.centre_ifft_phase()
        weighted_spectrum = exit_spectrum * centre_phase
        result = np.empty(z_values_m.size, dtype=np.complex128)
        normalization = self.grid.nx * self.grid.ny
        if incident_path_m is None:
            incident_path_m = self.water_path_m

        for start in range(0, z_values_m.size, chunk_size):
            z_chunk = z_values_m[start : start + chunk_size]
            propagation = np.exp(1j * z_chunk[:, None, None] * kz[None, :, :])
            if self.bandlimit:
                kx, ky, _ = self.grid.spectral_mesh()
                k_real = (
                    2.0
                    * np.pi
                    * frequency_hz
                    / self.transmitted_fluid.sound_speed_m_s
                )
                for local_index, distance in enumerate(z_chunk):
                    if distance > 0.0:
                        kx_limit = k_real / np.sqrt(
                            1.0
                            + (2.0 * distance / self.grid.extent_x_m) ** 2
                        )
                        ky_limit = k_real / np.sqrt(
                            1.0
                            + (2.0 * distance / self.grid.extent_y_m) ** 2
                        )
                        propagation[local_index] *= (
                            (np.abs(kx) <= kx_limit)
                            & (np.abs(ky) <= ky_limit)
                        )
                    propagation[local_index] *= self._combined_bandlimit_mask(
                        frequency_hz,
                        (
                            (self.incident_fluid, incident_path_m),
                            (self.transmitted_fluid, float(distance)),
                        ),
                    )
            result[start : start + z_chunk.size] = np.sum(
                propagation * weighted_spectrum[None, :, :], axis=(1, 2)
            ) / normalization
        return result

    def on_axis_scan_after_plate(
        self,
        frequency_hz: float,
        z_values_m: Iterable[float],
    ) -> ComplexField:
        z = np.asarray(tuple(z_values_m), dtype=float)
        if z.ndim != 1 or z.size == 0:
            raise ValueError("z_values_m must be a non-empty 1D sequence")
        if np.any(~np.isfinite(z)) or np.any(z < 0.0):
            raise ValueError("z_values_m must be finite and >= 0")
        spectrum = self.transmitted_spectrum_at_plate_exit(frequency_hz)
        return self._on_axis_scan_from_exit_spectrum(
            spectrum, frequency_hz, z
        )

    def reference_on_axis_scan(
        self,
        frequency_hz: float,
        z_values_m: Iterable[float],
    ) -> ComplexField:
        z = np.asarray(tuple(z_values_m), dtype=float)
        if z.ndim != 1 or z.size == 0:
            raise ValueError("z_values_m must be a non-empty 1D sequence")
        if np.any(~np.isfinite(z)) or np.any(z < 0.0):
            raise ValueError("z_values_m must be finite and >= 0")
        spectrum = self.reference_spectrum_at_exit(frequency_hz)
        return self._on_axis_scan_from_exit_spectrum(
            spectrum,
            frequency_hz,
            z,
            incident_path_m=self.water_path_m + self.plate.thickness_m,
        )

    def on_axis_value_after_plate(
        self, frequency_hz: float, z_after_plate_m: float
    ) -> complex:
        return complex(
            self.on_axis_scan_after_plate(
                frequency_hz, [z_after_plate_m]
            )[0]
        )

    def sampling_report(self, frequency_hz: float) -> dict[str, float]:
        wavelength = (
            self.incident_fluid.sound_speed_m_s / frequency_hz
        )
        edge_angle = np.arctan2(
            self.aperture.radius_m, self.aperture.focal_length_m
        )
        return {
            "water_wavelength_m": wavelength,
            "samples_per_wavelength_x": wavelength / self.grid.dx_m,
            "samples_per_wavelength_y": wavelength / self.grid.dy_m,
            "aperture_edge_angle_deg": float(np.degrees(edge_angle)),
            "window_x_m": self.grid.extent_x_m,
            "window_y_m": self.grid.extent_y_m,
        }
