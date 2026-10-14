"""FastMCP tools wrapping FuncPie physics computations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from funcpie.physics import (
    constants,
    electromagnetism,
    mechanics,
    modern,
    thermodynamics,
    units,
    waves,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register_physics_tools(mcp: FastMCP) -> None:
    """Register all physics tools, resources, and prompts onto the given FastMCP server."""

    # =========================================================================
    # 1. Mechanics Tools
    # =========================================================================

    @mcp.tool()
    def physics_projectile_motion(
        v0: float,
        angle_deg: float,
        gravity: float = 9.80665,
    ) -> dict[str, float]:
        """Calculate complete 2D projectile motion summary.

        Args:
            v0: Initial launch speed in m/s.
            angle_deg: Launch angle in degrees above horizontal.
            gravity: Gravitational acceleration in m/s^2 (default: 9.80665).

        Returns:
            Dictionary with max_height (m), flight_time (s), horizontal_range (m), impact_velocity (m/s).
        """
        summary = mechanics.projectile_motion(v0=v0, angle_deg=angle_deg, gravity=gravity)
        return {
            "max_height": summary.max_height,
            "flight_time": summary.flight_time,
            "horizontal_range": summary.horizontal_range,
        }

    @mcp.tool()
    def physics_gravitational_force(m1: float, m2: float, r: float) -> float:
        """Calculate Newtonian gravitational force between two point masses.

        Args:
            m1: Mass of first body in kg.
            m2: Mass of second body in kg.
            r: Distance between center of masses in meters.

        Returns:
            Gravitational attractive force in Newtons (N).
        """
        if r <= 0.0:
            raise ValueError("Distance r must be greater than zero.")
        return mechanics.gravitational_force(m1, m2, r)

    @mcp.tool()
    def physics_escape_velocity(mass: float, radius: float) -> float:
        """Calculate escape velocity from a spherical body.

        Args:
            mass: Mass of the celestial body in kg.
            radius: Radius from center of mass in meters.

        Returns:
            Escape velocity in m/s.
        """
        if radius <= 0.0:
            raise ValueError("Radius must be greater than zero.")
        return mechanics.escape_velocity(mass, radius)

    @mcp.tool()
    def physics_orbital_velocity(mass: float, radius: float) -> float:
        """Calculate circular orbital velocity around a celestial body.

        Args:
            mass: Mass of the central body in kg.
            radius: Orbital radius in meters.

        Returns:
            Circular orbital speed in m/s.
        """
        if radius <= 0.0:
            raise ValueError("Radius must be greater than zero.")
        return mechanics.orbital_velocity(mass, radius)

    @mcp.tool()
    def physics_kinetic_energy(mass: float, velocity: float) -> float:
        """Calculate classical kinetic energy (1/2 * m * v^2).

        Args:
            mass: Mass in kg.
            velocity: Speed in m/s.

        Returns:
            Kinetic energy in Joules (J).
        """
        return mechanics.kinetic_energy(mass, velocity)

    @mcp.tool()
    def physics_work_and_power(
        force_val: float,
        distance: float,
        time: float,
        angle_deg: float = 0.0,
    ) -> dict[str, float]:
        """Calculate work done by a force and average power output.

        Args:
            force_val: Force magnitude in Newtons (N).
            distance: Displacement in meters (m).
            time: Duration in seconds (s).
            angle_deg: Angle between force and displacement in degrees (default: 0.0).

        Returns:
            Dictionary with work_joules and power_watts.
        """
        if time <= 0.0:
            raise ValueError("Time must be positive.")
        theta_rad = math.radians(angle_deg)
        w = mechanics.work(force_val, distance, theta_rad)
        p = mechanics.power(w, time)
        return {"work_joules": w, "power_watts": p}

    @mcp.tool()
    def physics_elastic_collision_1d(
        m1: float,
        v1: float,
        m2: float,
        v2: float,
    ) -> dict[str, float]:
        """Calculate post-collision velocities for a 1D perfectly elastic collision.

        Args:
            m1: Mass of body 1 in kg.
            v1: Initial velocity of body 1 in m/s.
            m2: Mass of body 2 in kg.
            v2: Initial velocity of body 2 in m/s.

        Returns:
            Dictionary with v1_final and v2_final in m/s.
        """
        v1_f, v2_f = mechanics.elastic_collision_1d(m1, v1, m2, v2)
        return {"v1_final": v1_f, "v2_final": v2_f}

    @mcp.tool()
    def physics_pendulum_period(length: float, g: float = 9.80665) -> float:
        """Calculate small-angle period of a simple pendulum.

        Args:
            length: Length of pendulum in meters.
            g: Gravitational acceleration in m/s^2 (default: 9.80665).

        Returns:
            Period T in seconds.
        """
        return mechanics.simple_pendulum_period(length, g)

    @mcp.tool()
    def physics_spring_period(mass: float, spring_constant: float) -> float:
        """Calculate oscillation period of an ideal spring-mass system.

        Args:
            mass: Mass in kg.
            spring_constant: Spring stiffness k in N/m.

        Returns:
            Oscillation period in seconds.
        """
        return mechanics.spring_mass_period(mass, spring_constant)

    # =========================================================================
    # 2. Thermodynamics Tools
    # =========================================================================

    @mcp.tool()
    def physics_ideal_gas_solver(
        pressure: float | None = None,
        volume: float | None = None,
        moles: float | None = None,
        temperature_kelvin: float | None = None,
    ) -> dict[str, float]:
        """Solve for the unknown variable in the Ideal Gas Law (PV = nRT).

        Exactly three of (pressure, volume, moles, temperature_kelvin) must be provided.

        Args:
            pressure: Pressure in Pascals (Pa).
            volume: Volume in cubic meters (m^3).
            moles: Amount of gas in moles (mol).
            temperature_kelvin: Absolute temperature in Kelvin (K).

        Returns:
            Dictionary with all 4 resolved state variables.
        """
        r = constants.R
        none_count = sum(1 for x in (pressure, volume, moles, temperature_kelvin) if x is None)
        if none_count != 1:
            raise ValueError("Exactly one parameter must be None to solve for.")

        if pressure is None:
            assert volume is not None and moles is not None and temperature_kelvin is not None
            pressure = thermodynamics.ideal_gas_pressure(moles, temperature_kelvin, volume, r)
        elif volume is None:
            assert pressure is not None and moles is not None and temperature_kelvin is not None
            volume = thermodynamics.ideal_gas_volume(moles, temperature_kelvin, pressure, r)
        elif moles is None:
            assert pressure is not None and volume is not None and temperature_kelvin is not None
            moles = thermodynamics.ideal_gas_moles(pressure, volume, temperature_kelvin, r)
        else:
            assert pressure is not None and volume is not None and moles is not None
            temperature_kelvin = thermodynamics.ideal_gas_temperature(pressure, volume, moles, r)

        return {
            "pressure_pa": pressure,
            "volume_m3": volume,
            "moles": moles,
            "temperature_k": temperature_kelvin,
        }

    @mcp.tool()
    def physics_carnot_efficiency(th_kelvin: float, tc_kelvin: float) -> float:
        """Calculate theoretical maximum Carnot thermal efficiency (1 - Tc/Th).

        Args:
            th_kelvin: Hot reservoir temperature in Kelvin (> 0).
            tc_kelvin: Cold reservoir temperature in Kelvin (> 0).

        Returns:
            Carnot efficiency as a dimensionless fraction (0.0 to 1.0).
        """
        return thermodynamics.carnot_efficiency(t_cold_k=tc_kelvin, t_hot_k=th_kelvin)

    @mcp.tool()
    def physics_heat_conduction(
        thermal_conductivity: float,
        area: float,
        delta_temp: float,
        thickness: float,
    ) -> float:
        """Calculate steady-state heat conduction rate (Fourier's Law).

        Args:
            thermal_conductivity: Thermal conductivity k in W/(m K).
            area: Heat transfer cross-sectional area in m^2.
            delta_temp: Temperature difference delta T in Kelvin or Celsius.
            thickness: Material thickness in meters.

        Returns:
            Heat conduction rate Q/t in Watts (W).
        """
        return thermodynamics.heat_conduction_rate(
            thermal_conductivity, area, delta_temp, thickness
        )

    @mcp.tool()
    def physics_stefan_boltzmann_radiation(
        area: float,
        temperature_kelvin: float,
        emissivity: float = 1.0,
    ) -> float:
        """Calculate total thermal radiation power emitted by a blackbody or graybody.

        Args:
            area: Surface area in m^2.
            temperature_kelvin: Absolute surface temperature in Kelvin.
            emissivity: Emissivity factor between 0.0 and 1.0 (default: 1.0).

        Returns:
            Radiant power P in Watts (W).
        """
        return thermodynamics.stefan_boltzmann_radiation(area, temperature_kelvin, emissivity)

    @mcp.tool()
    def physics_thermal_expansion_linear(
        initial_length: float,
        alpha: float,
        delta_temp: float,
    ) -> dict[str, float]:
        """Calculate 1D linear thermal expansion (delta_L = L0 * alpha * delta_T).

        Args:
            initial_length: Initial length L0 in meters.
            alpha: Coefficient of linear thermal expansion in 1/K.
            delta_temp: Temperature change in Kelvin.

        Returns:
            Dictionary with delta_length_m and final_length_m.
        """
        dl = thermodynamics.thermal_expansion_linear(initial_length, alpha, delta_temp)
        return {"delta_length_m": dl, "final_length_m": initial_length + dl}

    # =========================================================================
    # 3. Electromagnetism Tools
    # =========================================================================

    @mcp.tool()
    def physics_coulombs_law(q1: float, q2: float, r: float) -> float:
        """Calculate electrostatic force between two point charges (Coulomb's Law).

        Args:
            q1: Electric charge 1 in Coulombs (C).
            q2: Electric charge 2 in Coulombs (C).
            r: Distance between charges in meters (> 0).

        Returns:
            Electrostatic force magnitude in Newtons (N). Positive = repulsive, negative = attractive.
        """
        return electromagnetism.coulombs_law(q1, q2, r)

    @mcp.tool()
    def physics_ohms_law(
        voltage: float | None = None,
        current: float | None = None,
        resistance: float | None = None,
    ) -> dict[str, float]:
        """Solve for the unknown in Ohm's Law (V = I * R).

        Exactly two parameters must be supplied.

        Args:
            voltage: Voltage in Volts (V).
            current: Current in Amperes (A).
            resistance: Resistance in Ohms (Omega).

        Returns:
            Dictionary with resolved voltage_v, current_a, resistance_ohm.
        """
        none_count = sum(1 for x in (voltage, current, resistance) if x is None)
        if none_count != 1:
            raise ValueError("Exactly two of (voltage, current, resistance) must be provided.")

        if voltage is None:
            assert current is not None and resistance is not None
            voltage = electromagnetism.ohms_law_voltage(current, resistance)
        elif current is None:
            assert voltage is not None and resistance is not None
            current = electromagnetism.ohms_law_current(voltage, resistance)
        else:
            assert voltage is not None and current is not None
            resistance = electromagnetism.ohms_law_resistance(voltage, current)

        return {
            "voltage_v": voltage,
            "current_a": current,
            "resistance_ohm": resistance,
        }

    @mcp.tool()
    def physics_electric_power(
        voltage: float | None = None,
        current: float | None = None,
        resistance: float | None = None,
    ) -> float:
        """Calculate DC electric power (P = V*I = I^2*R = V^2/R).

        Provide any two of (voltage, current, resistance).

        Args:
            voltage: Voltage in Volts (V).
            current: Current in Amperes (A).
            resistance: Resistance in Ohms.

        Returns:
            Power in Watts (W).
        """
        if voltage is not None and current is not None:
            return electromagnetism.electric_power(voltage, current)
        elif current is not None and resistance is not None:
            return current * current * resistance
        elif voltage is not None and resistance is not None:
            if resistance == 0.0:
                raise ValueError("Resistance cannot be zero.")
            return (voltage * voltage) / resistance
        else:
            raise ValueError("At least two of (voltage, current, resistance) must be provided.")

    @mcp.tool()
    def physics_rc_circuit(
        resistance: float,
        capacitance: float,
        v0: float = 1.0,
        time: float | None = None,
    ) -> dict[str, float]:
        """Analyze RC circuit time constant and transient voltages.

        Args:
            resistance: Resistance in Ohms (> 0).
            capacitance: Capacitance in Farads (> 0).
            v0: Supply or initial voltage in Volts (default: 1.0).
            time: Observation time in seconds (optional).

        Returns:
            Dictionary with time_constant_seconds, and optionally charging_voltage and discharging_voltage.
        """
        tau = electromagnetism.rc_time_constant(resistance, capacitance)
        res: dict[str, float] = {"time_constant_seconds": tau}
        if time is not None:
            res["charging_voltage"] = electromagnetism.rc_charging_voltage(
                v0, time, resistance, capacitance
            )
            res["discharging_voltage"] = electromagnetism.rc_discharging_voltage(
                v0, time, resistance, capacitance
            )
        return res

    @mcp.tool()
    def physics_lorentz_force(
        charge: float,
        velocity: float,
        magnetic_field: float,
        angle_deg: float = 90.0,
    ) -> float:
        """Calculate magnetic force on a moving charge: F = q * v * B * sin(theta).

        Args:
            charge: Electric charge q in Coulombs (C).
            velocity: Velocity magnitude v in m/s.
            magnetic_field: Magnetic field B in Tesla (T).
            angle_deg: Angle between v and B in degrees (default: 90.0).

        Returns:
            Force in Newtons (N).
        """
        theta_rad = math.radians(angle_deg)
        return electromagnetism.lorentz_force(charge, velocity, magnetic_field, theta_rad)

    @mcp.tool()
    def physics_magnetic_field_wire(current: float, distance: float) -> float:
        """Calculate magnetic field B generated by a long straight wire carrying current.

        Args:
            current: Current in Amperes (A).
            distance: Perpendicular distance from wire in meters (r > 0).

        Returns:
            Magnetic field strength in Tesla (T).
        """
        return electromagnetism.magnetic_field_wire(current, distance)

    # =========================================================================
    # 4. Waves & Optics Tools
    # =========================================================================

    @mcp.tool()
    def physics_wave_speed(
        frequency: float | None = None,
        wavelength: float | None = None,
        speed: float | None = None,
    ) -> dict[str, float]:
        """Solve wave relationship (v = f * lambda). Exactly two arguments must be given.

        Args:
            frequency: Frequency f in Hz.
            wavelength: Wavelength lambda in meters.
            speed: Propagation speed v in m/s.

        Returns:
            Dictionary with frequency_hz, wavelength_m, speed_mps.
        """
        none_count = sum(1 for x in (frequency, wavelength, speed) if x is None)
        if none_count != 1:
            raise ValueError("Exactly two of (frequency, wavelength, speed) must be given.")

        if speed is None:
            assert frequency is not None and wavelength is not None
            speed = waves.wave_speed(frequency, wavelength)
        elif frequency is None:
            assert speed is not None and wavelength is not None
            frequency = waves.wave_frequency(speed, wavelength)
        else:
            assert speed is not None and frequency is not None
            wavelength = waves.wave_length(speed, frequency)

        return {
            "frequency_hz": frequency,
            "wavelength_m": wavelength,
            "speed_mps": speed,
        }

    @mcp.tool()
    def physics_doppler_effect(
        source_freq: float,
        v_sound: float,
        v_receiver: float,
        v_source: float,
    ) -> float:
        """Calculate observed frequency under the Doppler effect.

        Conventions:
        - v_receiver > 0 when receiver moves towards source.
        - v_source > 0 when source moves towards receiver.

        Args:
            source_freq: Emitted wave frequency in Hz.
            v_sound: Wave propagation speed in medium in m/s.
            v_receiver: Receiver speed towards source in m/s.
            v_source: Source speed towards receiver in m/s.

        Returns:
            Observed frequency in Hz.
        """
        return waves.doppler_effect(source_freq, v_sound, v_receiver, v_source)

    @mcp.tool()
    def physics_snells_law(
        n1: float,
        theta1_deg: float,
        n2: float,
    ) -> dict[str, float]:
        """Calculate refraction angle and critical angle via Snell's Law (n1*sin(t1) = n2*sin(t2)).

        Args:
            n1: Refractive index of incident medium.
            theta1_deg: Incident angle in degrees from normal.
            n2: Refractive index of refracted medium.

        Returns:
            Dictionary with refracted_angle_deg, and critical_angle_deg (if n1 > n2).
        """
        t2_deg = waves.snells_law_angle(n1, theta1_deg, n2)
        res: dict[str, float] = {"refracted_angle_deg": t2_deg}
        if n1 > n2:
            res["critical_angle_deg"] = waves.critical_angle(n1, n2)
        return res

    @mcp.tool()
    def physics_thin_lens(
        focal_length: float | None = None,
        object_distance: float | None = None,
        image_distance: float | None = None,
    ) -> dict[str, float]:
        """Solve thin lens equation (1/f = 1/do + 1/di). Exactly two must be provided.

        Args:
            focal_length: Focal length f in meters (positive for converging, negative for diverging).
            object_distance: Object distance do in meters (usually > 0).
            image_distance: Image distance di in meters (positive for real, negative for virtual).

        Returns:
            Dictionary with focal_length, object_distance, image_distance, and magnification.
        """
        none_count = sum(1 for x in (focal_length, object_distance, image_distance) if x is None)
        if none_count != 1:
            raise ValueError("Exactly two of (focal_length, object_distance, image_distance) required.")

        if focal_length is None:
            assert object_distance is not None and image_distance is not None
            focal_length = waves.thin_lens_focal_length(object_distance, image_distance)
        elif image_distance is None:
            assert focal_length is not None and object_distance is not None
            image_distance = waves.thin_lens_image_distance(object_distance, focal_length)
        else:
            assert focal_length is not None and image_distance is not None
            # 1/do = 1/f - 1/di
            diff = (1.0 / focal_length) - (1.0 / image_distance)
            if math.isclose(diff, 0.0):
                raise ValueError("Object at infinity.")
            object_distance = 1.0 / diff

        m = waves.thin_lens_magnification(object_distance, image_distance)
        return {
            "focal_length": focal_length,
            "object_distance": object_distance,
            "image_distance": image_distance,
            "magnification": m,
        }

    @mcp.tool()
    def physics_sound_level_decibels(
        intensity: float,
        reference_intensity: float = 1e-12,
    ) -> float:
        """Convert sound intensity (W/m^2) to sound level in decibels (dB).

        Args:
            intensity: Sound intensity in W/m^2 (> 0).
            reference_intensity: Reference threshold intensity (default: 1e-12 W/m^2).

        Returns:
            Sound level in dB.
        """
        return waves.sound_level_decibels(intensity, reference_intensity)

    # =========================================================================
    # 5. Modern Physics Tools
    # =========================================================================

    @mcp.tool()
    def physics_lorentz_factor(velocity: float) -> float:
        """Calculate relativistic Lorentz factor gamma = 1 / sqrt(1 - (v/c)^2).

        Args:
            velocity: Particle speed in m/s (|v| < c).

        Returns:
            Lorentz factor gamma (dimensionless, >= 1.0).
        """
        return modern.lorentz_factor(velocity)

    @mcp.tool()
    def physics_time_dilation(proper_time: float, velocity: float) -> float:
        """Calculate time dilation: t = gamma * t0.

        Args:
            proper_time: Proper time interval in seconds measured in rest frame.
            velocity: Relative velocity in m/s.

        Returns:
            Dilated time interval in seconds.
        """
        return modern.time_dilation(proper_time, velocity)

    @mcp.tool()
    def physics_rest_energy(mass: float) -> float:
        """Calculate mass rest energy via Einstein's mass-energy equivalence (E = m * c^2).

        Args:
            mass: Rest mass in kg.

        Returns:
            Rest energy in Joules (J).
        """
        return modern.rest_energy(mass)

    @mcp.tool()
    def physics_photon_energy(
        frequency: float | None = None,
        wavelength: float | None = None,
    ) -> dict[str, float]:
        """Calculate photon energy from frequency (E = h*f) or wavelength (E = h*c/lambda).

        Provide either frequency (Hz) or wavelength (meters).

        Args:
            frequency: Frequency in Hz.
            wavelength: Wavelength in meters.

        Returns:
            Dictionary with energy_joules and energy_electronvolts (eV).
        """
        if frequency is not None:
            e_joules = modern.photon_energy_from_freq(frequency)
        elif wavelength is not None:
            e_joules = modern.photon_energy_from_wavelength(wavelength)
        else:
            raise ValueError("Either frequency or wavelength must be supplied.")

        e_ev = e_joules / constants.e
        return {"energy_joules": e_joules, "energy_electronvolts": e_ev}

    @mcp.tool()
    def physics_radioactive_decay(
        initial_amount: float,
        half_life: float,
        time_elapsed: float,
    ) -> dict[str, float]:
        """Calculate remaining radioactive substance after a given elapsed time.

        Args:
            initial_amount: Initial quantity (e.g. grams, moles, particle count).
            half_life: Half-life of the radionuclide in seconds (or matching time unit).
            time_elapsed: Elapsed duration in matching time unit.

        Returns:
            Dictionary with remaining_amount, decay_constant, fraction_remaining.
        """
        if half_life <= 0.0:
            raise ValueError("Half-life must be greater than zero.")
        lam = modern.decay_constant(half_life)
        rem = modern.radioactive_decay_remaining(initial_amount, half_life, time_elapsed)
        return {
            "remaining_amount": rem,
            "decay_constant": lam,
            "fraction_remaining": rem / initial_amount if initial_amount != 0.0 else 0.0,
        }

    # =========================================================================
    # 6. Physical Constants & Unit Conversion
    # =========================================================================

    @mcp.tool()
    def physics_get_constant(symbol_or_name: str) -> dict[str, str | float]:
        """Lookup standard CODATA physical constant by symbol or common key.

        Examples: 'c' (speed of light), 'h' (Planck), 'G' (gravitational), 'k_B' (Boltzmann),
        'N_A' (Avogadro), 'R' (gas constant), 'm_e' (electron mass), 'e' (elementary charge).

        Args:
            symbol_or_name: Constant symbol (e.g. 'c', 'G', 'h', 'k_B', 'e', 'g').

        Returns:
            Dictionary containing symbol, name, value, and unit.
        """
        key = symbol_or_name.strip()
        if key in constants.ALL_CONSTANTS:
            const = constants.ALL_CONSTANTS[key]
            return {
                "symbol": const.symbol,
                "name": const.name,
                "value": const.value,
                "unit": const.unit,
            }
        # Try matching by name
        for const in constants.ALL_CONSTANTS.values():
            if key.lower() in const.name.lower():
                return {
                    "symbol": const.symbol,
                    "name": const.name,
                    "value": const.value,
                    "unit": const.unit,
                }
        raise KeyError(
            f"Physical constant '{symbol_or_name}' not found. Available symbols: {list(constants.ALL_CONSTANTS.keys())}"
        )

    @mcp.tool()
    def physics_convert_unit(value: float, from_unit: str, to_unit: str) -> float:
        """Convert a numerical quantity between physical units.

        Supports length, mass, time, energy, pressure, force, velocity, area, volume.
        Examples:
        - km to mi, m to ft, light_year to km
        - kg to lb, g to oz
        - joule to ev, kwh to j, cal to j
        - atm to pa, psi to bar

        Args:
            value: Numerical quantity to convert.
            from_unit: Source unit string.
            to_unit: Target unit string.

        Returns:
            Converted numerical value.
        """
        return units.convert(value, from_unit, to_unit)

    @mcp.tool()
    def physics_convert_temperature(
        value: float,
        from_scale: str,
        to_scale: str,
    ) -> float:
        """Convert temperature between Celsius ('C'), Fahrenheit ('F'), and Kelvin ('K').

        Args:
            value: Numerical temperature value.
            from_scale: Source scale ('C', 'F', 'K').
            to_scale: Target scale ('C', 'F', 'K').

        Returns:
            Converted temperature value.
        """
        return units.convert_temperature(value, from_scale, to_scale)

    # =========================================================================
    # Resources & Prompts
    # =========================================================================

    @mcp.resource("physics://constants")
    def resource_constants_list() -> str:
        """Return formatted markdown table of all CODATA standard physical constants."""
        lines = [
            "# CODATA Recommended Physical Constants",
            "| Symbol | Name | Value | Unit |",
            "| --- | --- | --- | --- |",
        ]
        for const in constants.ALL_CONSTANTS.values():
            lines.append(f"| `{const.symbol}` | {const.name} | `{const.value:g}` | {const.unit} |")
        return "\n".join(lines)

    @mcp.prompt()
    def physics_problem_solver(problem_statement: str) -> str:
        """Prompt template guiding an AI model to systematically solve a physics problem."""
        return (
            f"Please solve the following physics problem using FuncPie physics tools:\n\n"
            f"Problem Statement:\n{problem_statement}\n\n"
            f"Instructions:\n"
            f"1. Identify the given physical parameters and state their units clearly.\n"
            f"2. Use the available `physics_*` tools for accurate numerical calculations.\n"
            f"3. Verify units and physical boundary conditions (e.g., speed of light limit, absolute zero).\n"
            f"4. State the final answer with appropriate significant figures and standard SI units.\n"
        )
