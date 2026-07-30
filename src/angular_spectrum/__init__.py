"""Fluid–elastic-solid–fluid angular-spectrum ultrasound model."""

from .analysis import fwhm
from .grid import CartesianGrid
from .materials import ElasticSolid, Fluid, ElasticPlate
from .dmso_mixture import (
    DMSOWaterProperties,
    dmso_concentration_to_mole_fraction,
    dmso_water_properties,
)
from .model import AngularSpectrumModel, FocusedCircularAperture
from .plate import (
    elastic_plate_scattering,
    elastic_plate_transfer_map,
    fluid_interface_scattering,
    normal_power_transmission,
)
from .pulse import (
    PulseResult,
    gaussian_transducer_response,
    propagate_pulse_on_axis,
    square_burst,
)

__all__ = [
    "AngularSpectrumModel",
    "CartesianGrid",
    "DMSOWaterProperties",
    "ElasticPlate",
    "ElasticSolid",
    "Fluid",
    "FocusedCircularAperture",
    "PulseResult",
    "elastic_plate_scattering",
    "elastic_plate_transfer_map",
    "dmso_concentration_to_mole_fraction",
    "dmso_water_properties",
    "fluid_interface_scattering",
    "fwhm",
    "gaussian_transducer_response",
    "normal_power_transmission",
    "propagate_pulse_on_axis",
    "square_burst",
]
