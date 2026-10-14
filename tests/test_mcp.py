"""Tests for FastMCP server and physics tools."""

from __future__ import annotations

import asyncio
import unittest

from funcpie.mcp import create_mcp_server

try:
    from funcpie.mcp.server import get_fastmcp_class

    get_fastmcp_class()
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False


@unittest.skipUnless(HAS_FASTMCP, "FastMCP is not installed")
class TestMCPServer(unittest.TestCase):
    """Test suite for FastMCP server and its registered physics tools."""

    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_FASTMCP:
            return
        cls.server = create_mcp_server("test-physics-mcp", legacy_tools=True)

    async def call_tool_float(
        self,
        name: str,
        arguments: dict[str, str | float | int | bool | None],
    ) -> float:
        tool = await self.server.get_tool(name)
        assert tool is not None
        result = await tool.run(arguments)
        assert result.structured_content is not None
        val = result.structured_content["result"]  # pyright: ignore[reportUnknownVariableType,reportIndexIssue]
        assert isinstance(val, (int, float))
        return float(val)

    async def call_tool_dict(
        self,
        name: str,
        arguments: dict[str, str | float | int | bool | None],
    ) -> dict[str, float]:
        tool = await self.server.get_tool(name)
        assert tool is not None
        result = await tool.run(arguments)
        assert result.structured_content is not None
        data = result.structured_content  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(data, dict)
        return data  # pyright: ignore[reportUnknownVariableType]

    def test_server_creation(self) -> None:
        self.assertEqual(self.server.name, "test-physics-mcp")

    def test_tools_registered(self) -> None:
        async def _test() -> None:
            tools = await self.server.list_tools()
            tool_names = {t.name for t in tools}
            expected = {
                "physics_projectile_motion",
                "physics_gravitational_force",
                "physics_escape_velocity",
                "physics_orbital_velocity",
                "physics_kinetic_energy",
                "physics_work_and_power",
                "physics_elastic_collision_1d",
                "physics_pendulum_period",
                "physics_spring_period",
                "physics_ideal_gas_solver",
                "physics_carnot_efficiency",
                "physics_heat_conduction",
                "physics_stefan_boltzmann_radiation",
                "physics_thermal_expansion_linear",
                "physics_coulombs_law",
                "physics_ohms_law",
                "physics_electric_power",
                "physics_rc_circuit",
                "physics_lorentz_force",
                "physics_magnetic_field_wire",
                "physics_wave_speed",
                "physics_doppler_effect",
                "physics_snells_law",
                "physics_thin_lens",
                "physics_sound_level_decibels",
                "physics_lorentz_factor",
                "physics_time_dilation",
                "physics_rest_energy",
                "physics_photon_energy",
                "physics_radioactive_decay",
                "physics_get_constant",
                "physics_convert_unit",
                "physics_convert_temperature",
            }
            self.assertTrue(expected.issubset(tool_names))

        asyncio.run(_test())

    def test_mechanics_tools(self) -> None:
        async def _test() -> None:
            # Kinetic energy: 0.5 * 10 * 5^2 = 125 J
            res_ke = await self.call_tool_float("physics_kinetic_energy", {"mass": 10.0, "velocity": 5.0})
            self.assertAlmostEqual(res_ke, 125.0)

            # Projectile motion
            data_pm = await self.call_tool_dict("physics_projectile_motion", {"v0": 50.0, "angle_deg": 45.0})
            self.assertGreater(data_pm["max_height"], 60.0)
            self.assertGreater(data_pm["horizontal_range"], 250.0)

            # Gravitational force
            res_gf = await self.call_tool_float("physics_gravitational_force", {"m1": 5.972e24, "m2": 70.0, "r": 6.371e6})
            self.assertAlmostEqual(res_gf, 686.0, delta=10.0)

            # Elastic collision 1D: m1=2, v1=3, m2=1, v2=-2
            data_col = await self.call_tool_dict("physics_elastic_collision_1d", {"m1": 2.0, "v1": 3.0, "m2": 1.0, "v2": -2.0})
            self.assertAlmostEqual(data_col["v1_final"], -0.333, delta=0.01)
            self.assertAlmostEqual(data_col["v2_final"], 4.666, delta=0.01)

        asyncio.run(_test())

    def test_thermodynamics_tools(self) -> None:
        async def _test() -> None:
            # Carnot efficiency: 1 - 300 / 600 = 0.5
            res_carnot = await self.call_tool_float("physics_carnot_efficiency", {"th_kelvin": 600.0, "tc_kelvin": 300.0})
            self.assertAlmostEqual(res_carnot, 0.5)

            # Ideal gas law solver: solve for pressure P = nRT/V
            data_gas = await self.call_tool_dict("physics_ideal_gas_solver", {
                "pressure": None,
                "volume": 0.024,
                "moles": 1.0,
                "temperature_kelvin": 300.0,
            })
            self.assertAlmostEqual(data_gas["pressure_pa"], 103930.78, delta=10.0)

        asyncio.run(_test())

    def test_electromagnetism_tools(self) -> None:
        async def _test() -> None:
            # Ohm's law: V = I * R = 2 * 10 = 20 V
            data_ohm = await self.call_tool_dict("physics_ohms_law", {"voltage": None, "current": 2.0, "resistance": 10.0})
            self.assertAlmostEqual(data_ohm["voltage_v"], 20.0)

            # Electric power: V * I = 20 * 2 = 40 W
            res_power = await self.call_tool_float("physics_electric_power", {"voltage": 20.0, "current": 2.0})
            self.assertAlmostEqual(res_power, 40.0)

            # Coulomb's law: 1e-6 C at 1m
            res_coulomb = await self.call_tool_float("physics_coulombs_law", {"q1": 1e-6, "q2": 1e-6, "r": 1.0})
            self.assertAlmostEqual(res_coulomb, 8.98755e-3, delta=0.001)

        asyncio.run(_test())

    def test_waves_and_optics_tools(self) -> None:
        async def _test() -> None:
            # Wave speed: v = f * lambda = 440 * 0.772 ~ 340 m/s
            data_wave = await self.call_tool_dict("physics_wave_speed", {"frequency": 440.0, "wavelength": 0.7727, "speed": None})
            self.assertAlmostEqual(data_wave["speed_mps"], 340.0, delta=1.0)

            # Thin lens: f=0.1, do=0.2 => di = 0.2, m = -1.0
            data_lens = await self.call_tool_dict("physics_thin_lens", {
                "focal_length": 0.1,
                "object_distance": 0.2,
                "image_distance": None,
            })
            self.assertAlmostEqual(data_lens["image_distance"], 0.2, places=4)
            self.assertAlmostEqual(data_lens["magnification"], -1.0, places=4)

        asyncio.run(_test())

    def test_modern_physics_tools(self) -> None:
        async def _test() -> None:
            # Lorentz factor at 0.6c => 1 / sqrt(1 - 0.36) = 1.25
            c = 299792458.0
            res_lorentz = await self.call_tool_float("physics_lorentz_factor", {"velocity": 0.6 * c})
            self.assertAlmostEqual(res_lorentz, 1.25, places=4)

            # Rest energy of 1 kg => E = c^2 ~ 8.987e16 J
            res_e = await self.call_tool_float("physics_rest_energy", {"mass": 1.0})
            self.assertAlmostEqual(res_e, c * c, places=2)

            # Radioactive decay: N0=100, t_half=10, t=20 => N=25
            data_decay = await self.call_tool_dict("physics_radioactive_decay", {
                "initial_amount": 100.0,
                "half_life": 10.0,
                "time_elapsed": 20.0,
            })
            self.assertAlmostEqual(data_decay["remaining_amount"], 25.0, places=4)
            self.assertAlmostEqual(data_decay["fraction_remaining"], 0.25, places=4)

        asyncio.run(_test())

    def test_constants_and_units_tools(self) -> None:
        async def _test() -> None:
            # Constant lookup: speed of light
            tool = await self.server.get_tool("physics_get_constant")
            assert tool is not None
            res = await tool.run({"symbol_or_name": "c"})
            assert res.structured_content is not None
            self.assertEqual(res.structured_content["symbol"], "c")  # pyright: ignore[reportIndexIssue]
            self.assertEqual(res.structured_content["value"], 299792458.0)  # pyright: ignore[reportIndexIssue]

            # Unit convert: 1 km to m
            res_unit = await self.call_tool_float("physics_convert_unit", {"value": 1.0, "from_unit": "km", "to_unit": "m"})
            self.assertAlmostEqual(res_unit, 1000.0)

            # Temperature convert: 100 C to F
            res_temp = await self.call_tool_float("physics_convert_temperature", {"value": 100.0, "from_scale": "C", "to_scale": "F"})
            self.assertAlmostEqual(res_temp, 212.0)

        asyncio.run(_test())

    def test_resources_and_prompts(self) -> None:
        async def _test() -> None:
            resources = await self.server.list_resources()
            uris = [str(r.uri) for r in resources]
            self.assertIn("physics://constants", uris)

            prompts = await self.server.list_prompts()
            p_names = [p.name for p in prompts]
            self.assertIn("physics_problem_solver", p_names)

        asyncio.run(_test())

    def test_argon2_authenticator(self) -> None:
        import os
        from funcpie.mcp.auth import APIKeyAuthenticator, HAS_ARGON2

        if HAS_ARGON2:
            from argon2 import PasswordHasher
            ph = PasswordHasher()
            argon_hash = ph.hash("gpu-secret-pass")

            auth = APIKeyAuthenticator(argon_hash)
            self.assertTrue(auth.verify("gpu-secret-pass"))
            self.assertFalse(auth.verify("wrong-password"))

            # Test ${VAR} syntax
            os.environ["MCP_TEST_KEY_ENV"] = argon_hash
            auth_env = APIKeyAuthenticator("${MCP_TEST_KEY_ENV}")
            self.assertTrue(auth_env.verify("gpu-secret-pass"))
            self.assertFalse(auth_env.verify("wrong-password"))

        # Plain key fallback
        plain_auth = APIKeyAuthenticator("plain-secret")
        self.assertTrue(plain_auth.verify("plain-secret"))
        self.assertFalse(plain_auth.verify("wrong-key"))

    def test_http_api_key_middleware(self) -> None:
        from funcpie.mcp.auth import APIKeyAuthenticator, APIKeyMiddleware, HAS_STARLETTE

        if not HAS_STARLETTE:
            self.skipTest("Starlette is not installed")

        from starlette.middleware import Middleware
        from starlette.testclient import TestClient

        auth = APIKeyAuthenticator("test-secret-key")
        app = self.server.http_app(middleware=[Middleware(APIKeyMiddleware, authenticator=auth)])
        client = TestClient(app)

        # 1. No auth -> 401
        r1 = client.get("/")
        self.assertEqual(r1.status_code, 401)
        self.assertEqual(r1.text, "unauthorized")

        # 2. Wrong key -> 401
        r2 = client.get("/", headers={"x-api-key": "invalid"})
        self.assertEqual(r2.status_code, 401)

        # 3. Correct x-api-key -> not 401
        r3 = client.get("/", headers={"x-api-key": "test-secret-key"})
        self.assertNotEqual(r3.status_code, 401)

        # 4. Correct Authorization: Bearer -> not 401
        r4 = client.get("/", headers={"Authorization": "Bearer test-secret-key"})
        self.assertNotEqual(r4.status_code, 401)


if __name__ == "__main__":
    unittest.main()
