"""Cartesian FFT sampling grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CartesianGrid:
    """Even-sized, centred Cartesian grid and its FFT wavenumbers."""

    nx: int = 512
    ny: int = 512
    dx_m: float = 40.0e-6
    dy_m: float | None = None

    def __post_init__(self) -> None:
        dy = self.dx_m if self.dy_m is None else self.dy_m
        object.__setattr__(self, "dy_m", dy)
        if self.nx < 4 or self.ny < 4 or self.nx % 2 or self.ny % 2:
            raise ValueError("nx and ny must be even integers >= 4")
        if not np.isfinite(self.dx_m) or self.dx_m <= 0.0:
            raise ValueError("dx_m must be finite and > 0")
        if not np.isfinite(dy) or dy <= 0.0:
            raise ValueError("dy_m must be finite and > 0")

    @property
    def x_m(self) -> NDArray[np.float64]:
        return (np.arange(self.nx) - self.nx // 2) * self.dx_m

    @property
    def y_m(self) -> NDArray[np.float64]:
        return (np.arange(self.ny) - self.ny // 2) * self.dy_m

    @property
    def kx_rad_m(self) -> NDArray[np.float64]:
        return 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx_m)

    @property
    def ky_rad_m(self) -> NDArray[np.float64]:
        return 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy_m)

    @property
    def extent_x_m(self) -> float:
        return self.nx * self.dx_m

    @property
    def extent_y_m(self) -> float:
        return self.ny * self.dy_m

    @property
    def centre_index(self) -> tuple[int, int]:
        return self.ny // 2, self.nx // 2

    def spatial_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.meshgrid(self.x_m, self.y_m, indexing="xy")

    def spectral_mesh(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        kx, ky = np.meshgrid(self.kx_rad_m, self.ky_rad_m, indexing="xy")
        return kx, ky, np.hypot(kx, ky)

    def centre_ifft_phase(self) -> NDArray[np.complex128]:
        """Phase factor for evaluating an inverse FFT at the grid centre."""

        kx, ky, _ = self.spectral_mesh()
        x_index_m = (self.nx // 2) * self.dx_m
        y_index_m = (self.ny // 2) * self.dy_m
        return np.exp(1j * (kx * x_index_m + ky * y_index_m))
