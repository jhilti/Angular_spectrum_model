import unittest

import numpy as np

from angular_spectrum import (
    ContactLineModel,
    FreeSurfaceLiquid,
    axisymmetric_surface_modes,
    equilibrium_meniscus_profile,
    radiation_stress_from_incident_pressure_envelope,
    raised_cosine_tone_envelope,
    simulate_axisymmetric_free_surface,
)


class FreeSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.radius = np.linspace(0.0, 1.5e-3, 65)
        self.liquid = FreeSurfaceLiquid(
            density_kg_m3=1050.0,
            dynamic_viscosity_pa_s=2.0e-3,
            surface_tension_n_m=0.045,
        )

    def test_radiation_stress_uses_peak_pressure_squared(self) -> None:
        pressure = np.array([0.0, 1.0e6, 2.0e6])
        stress = radiation_stress_from_incident_pressure_envelope(
            pressure,
            density_kg_m3=1000.0,
            sound_speed_m_s=1500.0,
        )
        np.testing.assert_allclose(
            stress,
            pressure**2 / (1000.0 * 1500.0**2),
        )
        self.assertAlmostEqual(stress[2] / stress[1], 4.0)
        transmitted = radiation_stress_from_incident_pressure_envelope(
            1.0e6,
            density_kg_m3=1000.0,
            sound_speed_m_s=1500.0,
            reflected_intensity_fraction=0.25,
            transmitted_intensity_fraction=0.50,
            backing_sound_speed_m_s=500.0,
        )
        intensity = 1.0e6**2 / (2.0 * 1000.0 * 1500.0)
        self.assertAlmostEqual(
            float(transmitted),
            intensity * (1.25 / 1500.0 - 0.50 / 500.0),
        )

    def test_raised_cosine_tone_envelope(self) -> None:
        time = np.linspace(0.0, 10.0e-6, 101)
        envelope = raised_cosine_tone_envelope(
            time,
            start_time_s=2.0e-6,
            duration_s=6.0e-6,
            edge_time_s=1.0e-6,
        )
        self.assertEqual(envelope[0], 0.0)
        self.assertAlmostEqual(envelope[30], 1.0)
        self.assertAlmostEqual(envelope[50], 1.0)
        self.assertAlmostEqual(envelope[80], 0.0, places=12)

    def test_nonlinear_equilibrium_enforces_contact_angle_and_volume(self) -> None:
        profile = equilibrium_meniscus_profile(
            self.radius,
            liquid=self.liquid,
            equilibrium_contact_angle_deg=70.0,
        )
        wall_slope = (profile[-1] - profile[-2]) / (
            self.radius[-1] - self.radius[-2]
        )
        normalized_wall_slope = wall_slope / np.sqrt(1.0 + wall_slope**2)
        self.assertAlmostEqual(
            normalized_wall_slope,
            np.cos(np.deg2rad(70.0)),
            delta=5.0e-3,
        )
        radial_integrand = profile * self.radius
        volume = np.sum(
            0.5
            * (radial_integrand[1:] + radial_integrand[:-1])
            * np.diff(self.radius)
        )
        self.assertAlmostEqual(volume, 0.0, delta=2.0e-12)

    def test_fixed_angle_first_mode_matches_cylindrical_bessel_root(self) -> None:
        modes = axisymmetric_surface_modes(
            self.radius,
            liquid=self.liquid,
            liquid_depth_m=3.0e-3,
            contact_line=ContactLineModel("fixed_contact_angle", 90.0),
            mode_count=4,
        )
        # J1(kR)=0 for a fixed-angle perturbation; first root = 3.83170597.
        self.assertAlmostEqual(
            modes.radial_wavenumber_m_inv[0] * self.radius[-1],
            3.83170597,
            delta=0.01,
        )
        self.assertTrue(np.all(modes.natural_frequency_hz > 0.0))

    def test_fixed_angle_modes_conserve_volume_and_pinned_is_rejected(self) -> None:
        modes = axisymmetric_surface_modes(
            self.radius,
            liquid=self.liquid,
            liquid_depth_m=3.0e-3,
            contact_line=ContactLineModel("fixed_contact_angle", 80.0),
            mode_count=5,
        )
        volume = np.zeros(modes.mode_shapes_m_inv.shape[1])
        for left in range(self.radius.size - 1):
            right = left + 1
            r0, r1 = self.radius[left], self.radius[right]
            for xi in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)):
                shape = np.array(
                    [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)]
                )
                r_gauss = shape @ np.array([r0, r1])
                phi_gauss = shape @ modes.mode_shapes_m_inv[[left, right]]
                volume += 0.5 * (r1 - r0) * r_gauss * phi_gauss
        np.testing.assert_allclose(volume, 0.0, atol=2.0e-14)
        with self.assertRaisesRegex(NotImplementedError, "rigid-wall"):
            axisymmetric_surface_modes(
                self.radius,
                liquid=self.liquid,
                liquid_depth_m=3.0e-3,
                contact_line=ContactLineModel("pinned_contact_line", 80.0),
                mode_count=5,
            )

    def test_uniform_pressure_is_a_gauge_load_in_closed_well(self) -> None:
        time = np.linspace(0.0, 40.0e-6, 81)
        uniform_stress = np.full((time.size, self.radius.size), 250.0)
        result = simulate_axisymmetric_free_surface(
            time,
            self.radius,
            uniform_stress,
            liquid=self.liquid,
            liquid_depth_m=3.0e-3,
            contact_line=ContactLineModel("fixed_contact_angle", 90.0),
            mode_count=8,
        )
        self.assertLess(np.max(np.abs(result.dynamic_elevation_m)), 1.0e-15)
        self.assertLess(np.max(np.abs(result.volume_residual_m3)), 1.0e-18)
        self.assertFalse(result.can_predict_detached_drop_volume)

    def test_local_load_is_linear_and_conserves_volume(self) -> None:
        time = np.linspace(0.0, 80.0e-6, 161)
        envelope = raised_cosine_tone_envelope(
            time,
            start_time_s=5.0e-6,
            duration_s=15.0e-6,
            edge_time_s=1.0e-6,
        )
        radial_profile = np.exp(-(self.radius / 0.25e-3) ** 2)
        stress = 100.0 * envelope[:, None] * radial_profile[None, :]
        kwargs = dict(
            liquid=self.liquid,
            liquid_depth_m=3.0e-3,
            contact_line=ContactLineModel("fixed_contact_angle", 85.0),
            mode_count=10,
        )
        result = simulate_axisymmetric_free_surface(
            time, self.radius, stress, **kwargs
        )
        doubled = simulate_axisymmetric_free_surface(
            time, self.radius, 2.0 * stress, **kwargs
        )
        np.testing.assert_allclose(
            doubled.dynamic_elevation_m,
            2.0 * result.dynamic_elevation_m,
            rtol=2.0e-12,
            atol=1.0e-18,
        )
        self.assertLess(np.max(np.abs(result.volume_residual_m3)), 1.0e-18)
        self.assertGreater(result.peak_positive_apex_displacement_m, 0.0)

    def test_surface_tension_and_viscosity_change_modal_dynamics(self) -> None:
        low_sigma = FreeSurfaceLiquid(1050.0, 1.0e-3, 0.030)
        high_sigma = FreeSurfaceLiquid(1050.0, 4.0e-3, 0.060)
        kwargs = dict(
            radius_m=self.radius,
            liquid_depth_m=3.0e-3,
            contact_line=ContactLineModel("fixed_contact_angle", 90.0),
            mode_count=4,
        )
        low_modes = axisymmetric_surface_modes(liquid=low_sigma, **kwargs)
        high_modes = axisymmetric_surface_modes(liquid=high_sigma, **kwargs)
        self.assertTrue(
            np.all(high_modes.natural_frequency_hz > low_modes.natural_frequency_hz)
        )
        self.assertTrue(
            np.all(
                high_modes.viscous_amplitude_decay_rate_s
                > low_modes.viscous_amplitude_decay_rate_s
            )
        )

    def test_safety_flags_cover_static_slope_depth_and_acoustic_feedback(self) -> None:
        time = np.linspace(0.0, 20.0e-6, 41)
        result = simulate_axisymmetric_free_surface(
            time,
            self.radius,
            np.zeros((time.size, self.radius.size)),
            liquid=self.liquid,
            liquid_depth_m=0.10e-3,
            contact_line=ContactLineModel("fixed_contact_angle", 60.0),
            mode_count=6,
            acoustic_wavelength_m=160.0e-6,
            focal_spot_radius_m=150.0e-6,
            equilibrium_slope_limit=0.30,
        )
        self.assertTrue(result.equilibrium_slope_limit_exceeded)
        self.assertTrue(result.nonpositive_depth_reached)
        self.assertTrue(result.frozen_acoustic_feedback_likely)
        self.assertFalse(result.within_reduced_model_validity)

    def test_zero_viscosity_is_available_for_inviscid_benchmark(self) -> None:
        inviscid = FreeSurfaceLiquid(1000.0, 0.0, 0.050)
        modes = axisymmetric_surface_modes(
            self.radius,
            liquid=inviscid,
            liquid_depth_m=3.0e-3,
            mode_count=3,
        )
        np.testing.assert_allclose(
            modes.viscous_amplitude_decay_rate_s, 0.0, atol=0.0
        )

    def test_unused_overdamped_high_modes_do_not_invalidate_result(self) -> None:
        liquid = FreeSurfaceLiquid(1050.0, 10.0e-3, 0.045)
        modes = axisymmetric_surface_modes(
            self.radius,
            liquid=liquid,
            liquid_depth_m=3.0e-3,
            mode_count=24,
        )
        self.assertLess(modes.damping_ratio[0], 0.20)
        self.assertGreater(modes.damping_ratio[-1], 0.20)
        initial = (
            1.0e-6
            * modes.mode_shapes_m_inv[:, 0]
            / modes.mode_shapes_m_inv[0, 0]
        )
        time = np.linspace(0.0, 50.0e-6, 51)
        result = simulate_axisymmetric_free_surface(
            time,
            self.radius,
            np.zeros((time.size, self.radius.size)),
            liquid=liquid,
            liquid_depth_m=3.0e-3,
            mode_count=24,
            initial_dynamic_elevation_m=initial,
            weak_viscosity_limit=0.20,
        )
        self.assertFalse(result.weak_viscosity_limit_exceeded)
        self.assertTrue(result.active_mode_mask[0])
        self.assertFalse(np.any(result.active_mode_mask[1:]))

    def test_free_mode_matches_viscously_damped_oscillator(self) -> None:
        radius = np.linspace(0.0, 0.5e-3, 65)
        liquid = FreeSurfaceLiquid(1000.0, 1.0e-3, 0.050)
        contact_line = ContactLineModel("fixed_contact_angle", 90.0)
        modes = axisymmetric_surface_modes(
            radius,
            liquid=liquid,
            liquid_depth_m=2.0e-3,
            contact_line=contact_line,
            mode_count=4,
        )
        time = np.arange(0.0, 4.0e-3 + 1.0e-6, 2.0e-6)
        initial = (
            1.0e-6
            * modes.mode_shapes_m_inv[:, 0]
            / modes.mode_shapes_m_inv[0, 0]
        )
        result = simulate_axisymmetric_free_surface(
            time,
            radius,
            np.zeros((time.size, radius.size)),
            liquid=liquid,
            liquid_depth_m=2.0e-3,
            contact_line=contact_line,
            mode_count=4,
            initial_dynamic_elevation_m=initial,
        )
        decay = modes.viscous_amplitude_decay_rate_s[0]
        omega = modes.natural_angular_frequency_rad_s[0]
        damped_omega = np.sqrt(omega**2 - decay**2)
        expected = 1.0e-6 * np.exp(-decay * time) * (
            np.cos(damped_omega * time)
            + decay / damped_omega * np.sin(damped_omega * time)
        )
        np.testing.assert_allclose(
            result.apex_dynamic_elevation_m,
            expected,
            rtol=2.0e-4,
            atol=1.0e-10,
        )
        self.assertLess(result.mechanical_energy_j[-1], result.mechanical_energy_j[0])

    def test_invalid_shapes_are_rejected(self) -> None:
        time = np.linspace(0.0, 10.0e-6, 11)
        with self.assertRaisesRegex(ValueError, "shape"):
            simulate_axisymmetric_free_surface(
                time,
                self.radius,
                np.zeros((time.size, self.radius.size - 1)),
                liquid=self.liquid,
                liquid_depth_m=3.0e-3,
            )
        with self.assertRaisesRegex(ValueError, "contact_angle"):
            ContactLineModel("fixed_contact_angle", 179.0)


if __name__ == "__main__":
    unittest.main()
