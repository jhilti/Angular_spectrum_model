"""Print linear length/time scales for the current ADE candidate geometry."""

from angular_spectrum import dmso_water_properties, linear_ade_screening


FREQUENCY_HZ = 10.0e6
DMSO_VOLUME_FRACTION = 0.80
TEMPERATURE_C = 22.0
FOCAL_LENGTH_M = 25.4e-3
APERTURE_DIAMETER_M = 13.0e-3
LIQUID_HEIGHT_M = 4.22e-3
TONE_CYCLES = 1.0


properties = dmso_water_properties(
    DMSO_VOLUME_FRACTION,
    basis="volume",
    temperature_c=TEMPERATURE_C,
)
metrics = linear_ade_screening(
    frequency_hz=FREQUENCY_HZ,
    sound_speed_m_s=properties.sound_speed_m_s,
    focal_length_m=FOCAL_LENGTH_M,
    aperture_diameter_m=APERTURE_DIAMETER_M,
    liquid_height_m=LIQUID_HEIGHT_M,
    tone_cycles=TONE_CYCLES,
)

print(f"Liquid sound speed: {properties.sound_speed_m_s:.1f} m/s")
print(f"Wavelength: {metrics.wavelength_m * 1e6:.1f} µm")
print(f"F-number: {metrics.f_number:.3f}")
print(
    "Homogeneous-medium intensity-FWHM proxy: "
    f"{metrics.diffraction_spot_diameter_m * 1e3:.3f} mm"
)
print(
    "Spot-equivalent sphere volume scale (not predicted drop volume): "
    f"{metrics.spot_equivalent_sphere_volume_m3 * 1e12:.2f} nL"
)
print(f"Tone duration: {metrics.tone_duration_s * 1e6:.3f} µs")
print(
    "Liquid cavity round trip: "
    f"{metrics.liquid_cavity_round_trip_s * 1e6:.3f} µs "
    f"({metrics.liquid_cavity_round_trip_cycles:.1f} cycles)"
)
print(
    "Nominal 2h/c delay exceeds ideal electrical-burst duration: "
    f"{metrics.nominal_cavity_delay_exceeds_ideal_burst}"
)
print(
    "These are linear screening scales only. Ejection threshold, drop volume, "
    "velocity, satellites, cavitation, and heating require calibration and a "
    "nonlinear free-surface model or measurements. Measured acoustic "
    "ring-down, not this ideal boolean, determines pulse overlap."
)
