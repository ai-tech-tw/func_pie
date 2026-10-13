import unittest
import math
from funcpie.physics import constants
from funcpie.physics.mechanics import (
    velocity, acceleration, displacement_constant_acc, final_velocity,
    projectile_motion, force, weight, friction_force, centripetal_force,
    gravitational_force, kinetic_energy, work, power, momentum,
    elastic_collision_1d, simple_pendulum_period, spring_mass_period
)
from funcpie.physics.thermodynamics import (
    ideal_gas_pressure, heat_energy, latent_heat,
    carnot_efficiency, stefan_boltzmann_radiation, wien_displacement_peak_wavelength
)
from funcpie.physics.electromagnetism import (
    coulombs_law, ohms_law_voltage, ohms_law_current, electric_power,
    resistors_series, resistors_parallel, capacitors_series, capacitors_parallel,
    rc_time_constant, lorentz_force
)
from funcpie.physics.waves import (
    wave_speed, wave_frequency, sound_speed_air, snells_law_angle,
    thin_lens_focal_length, sound_level_decibels
)
from funcpie.physics.modern import (
    lorentz_factor, time_dilation, length_contraction,
    rest_energy, photon_energy_from_freq, radioactive_decay_remaining
)
from funcpie.physics.units import convert, celsius_to_fahrenheit, fahrenheit_to_celsius


class TestPhysics(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(constants.c, 299792458.0)
        self.assertAlmostEqual(constants.g, 9.80665)
        self.assertIn("c", constants.ALL_CONSTANTS)

    def test_mechanics(self):
        self.assertEqual(velocity(100, 5), 20.0)
        self.assertEqual(acceleration(0, 20, 4), 5.0)
        self.assertEqual(displacement_constant_acc(0, 10, 2), 20.0)
        self.assertEqual(final_velocity(10, 2, 5), 20.0)
        self.assertEqual(force(10, 2), 20.0)
        self.assertAlmostEqual(weight(10), 98.0665)
        self.assertEqual(friction_force(0.2, 50), 10.0)
        self.assertEqual(centripetal_force(2, 10, 5), 40.0)
        self.assertEqual(kinetic_energy(2, 3), 9.0)
        self.assertEqual(work(10, 5), 50.0)
        self.assertEqual(power(100, 5), 20.0)
        self.assertEqual(momentum(5, 4), 20.0)

        # Elastic collision equal masses
        v1p, v2p = elastic_collision_1d(1.0, 5.0, 1.0, 0.0)
        self.assertAlmostEqual(v1p, 0.0)
        self.assertAlmostEqual(v2p, 5.0)

        # Pendulum
        T = simple_pendulum_period(1.0, gravity=constants.g)
        self.assertAlmostEqual(T, 2.006, places=2)

    def test_thermodynamics(self):
        # 1 mol at 300K in 0.025 m^3
        p = ideal_gas_pressure(1.0, 300.0, 0.025)
        self.assertAlmostEqual(p, (1.0 * constants.R * 300.0) / 0.025)

        self.assertEqual(heat_energy(2.0, 4184.0, 10.0), 83680.0)
        self.assertEqual(carnot_efficiency(300.0, 600.0), 0.5)
        self.assertAlmostEqual(wien_displacement_peak_wavelength(5800.0), 2.897771955e-3 / 5800.0)

    def test_electromagnetism(self):
        self.assertEqual(ohms_law_voltage(2.0, 5.0), 10.0)
        self.assertEqual(ohms_law_current(10.0, 5.0), 2.0)
        self.assertEqual(electric_power(voltage=10.0, current=2.0), 20.0)
        self.assertEqual(resistors_series(10, 20, 30), 60.0)
        self.assertAlmostEqual(resistors_parallel(10, 10), 5.0)
        self.assertAlmostEqual(capacitors_series(10, 10), 5.0)
        self.assertEqual(capacitors_parallel(10, 20), 30.0)
        self.assertEqual(rc_time_constant(1000, 1e-6), 1e-3)
        self.assertEqual(lorentz_force(1.0, 10.0, 2.0), 20.0)

    def test_waves(self):
        self.assertEqual(wave_speed(440.0, 0.77), 440.0 * 0.77)
        self.assertAlmostEqual(sound_speed_air(20.0), 343.2, places=0)
        # Snell: n1=1.0, theta1=30 deg, n2=1.0 -> theta2=30 deg
        self.assertAlmostEqual(snells_law_angle(1.0, 30.0, 1.0), 30.0)
        # Lens: do=20, di=20 -> f=10
        self.assertAlmostEqual(thin_lens_focal_length(20.0, 20.0), 10.0)
        # Decibels: I = 1e-9 W/m^2 -> 30 dB
        self.assertAlmostEqual(sound_level_decibels(1e-9), 30.0)

    def test_modern_physics(self):
        # Lorentz factor at v=0 is 1.0
        self.assertEqual(lorentz_factor(0.0), 1.0)
        # E = m * c^2 for 1 kg
        self.assertEqual(rest_energy(1.0), constants.c ** 2)
        # Decay: after 1 half-life, remaining is 50%
        self.assertEqual(radioactive_decay_remaining(100.0, 5.0, 5.0), 50.0)

    def test_units(self):
        self.assertAlmostEqual(celsius_to_fahrenheit(0.0), 32.0)
        self.assertAlmostEqual(celsius_to_fahrenheit(100.0), 212.0)
        self.assertAlmostEqual(fahrenheit_to_celsius(32.0), 0.0)
        self.assertAlmostEqual(convert(1.0, "km", "m"), 1000.0)
        self.assertAlmostEqual(convert(1.0, "hr", "s"), 3600.0)
        self.assertAlmostEqual(convert(1.0, "atm", "pa"), 101325.0)


if __name__ == "__main__":
    unittest.main()
