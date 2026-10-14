# FuncPie

[正體中文](README.zh-TW.md) | English

[![CI](https://github.com/ai-tech-tw/func_pie/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-tech-tw/func_pie/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked](https://img.shields.io/badge/basedpyright-checked-blue)](https://github.com/DetachHead/basedpyright)

> A lightweight, comprehensive mathematics and physical functions library designed for Python.

FuncPie is built with **zero external dependencies (Pure Python Standard Library)**, fully type-annotated, ready out-of-the-box, and natively supports elegant Unix-style functional pipelines.

---

## Key Features

- **Cross-Platform GPU Acceleration (AMD HIP & Apple Silicon Metal)**: Native hardware acceleration support via AMD ROCm HIP (`libamdhip64.so` / CUDA compatibility) on Linux/Windows and Apple Metal (`Metal.framework` via MSL compute shaders) on macOS Apple Silicon (M1/M2/M3/M4), delivering high-throughput parallel compute for matrix operations, N-Body simulations, PDEs, and mathematical conjectures, with seamless automatic fallback to CPU.
- **Zero External Dependencies**: Pure Python standard library implementation with zero third-party packages. Extremely lightweight, instant installation, and consistent across platforms.
- **Comprehensive Mathematics Suite**:
  - **Basic Arithmetic & Chaining**: `Calculation` class with method chaining, running accumulations, and safe division/modulo operations.
  - **Algebra & Number Theory**: Polynomial evaluation, calculus, and root finding; GCD, LCM, prime sieving, prime factorization, combinations, and permutations.
  - **Calculus & Numerical Methods**: Numerical differentiation (forward, backward, central differences), numerical integration (Simpson's rule, trapezoidal rule), root finding (Newton-Raphson, bisection), and multivariable gradient computation.
  - **Linear Algebra**: 2D and 3D vectors with geometric operations, matrix arithmetic, determinants, Gaussian elimination, matrix inversion, and linear system solving ($Ax = b$).
  - **Statistics & Probability**: Descriptive statistics (mean, variance, standard deviation, percentiles, quartiles, skewness, kurtosis), Pearson correlation, least-squares linear regression, and common distributions (Normal, Binomial, Poisson).
  - **Geometry & Coordinates**: Polygon area via Shoelace formula, perimeter, volume, surface area of 2D/3D shapes, and coordinate transformations (polar, spherical, cylindrical).
  - **Special & Activation Functions**: Gamma, Beta, Error function ($\text{erf}$), and machine learning activations (Sigmoid, ReLU, Leaky ReLU, Softmax).
- **Full Physics Suite**:
  - **CODATA Physical Constants**: Accurate scientific constants ($c$, $G$, $h$, $\hbar$, $k_B$, $e$, $m_e$, etc.).
  - **Classical Mechanics**: Kinematics equations, projectile trajectory simulation, Newton's law of universal gravitation, escape velocity, kinetic energy, work, momentum, elastic/inelastic collisions, simple pendulum, and harmonic oscillators.
  - **Thermodynamics**: Ideal gas law ($PV=nRT$), thermal conduction, thermal expansion, Carnot cycle efficiency, Stefan-Boltzmann blackbody radiation, and Wien's displacement law.
  - **Electromagnetism & Circuits**: Coulomb's law, Ohm's law, series/parallel equivalent resistance and capacitance, RC circuit charging/discharging transients, Lorentz force, and solenoid magnetic fields.
  - **Waves & Optics**: Temperature-dependent speed of sound, Doppler effect, Snell's law, total internal reflection critical angle, thin-lens equation, and decibel sound level conversions.
  - **Modern & Quantum Physics**: Special relativity Lorentz factor ($\gamma$), time dilation, length contraction, mass-energy equivalence ($E=mc^2$), photon energy, de Broglie matter wavelength, and radioactive half-life decay.
  - **Unit Converter**: Conversions across length, mass, time, pressure, energy, velocity, angle, and temperature scales.
- **Functional Pipelines**: Native support for Unix-style pipelining syntax `value | f | g`, along with functional utilities `pipe()`, `compose()`, and `curry()`.
- **Mathematical Problems & Conjectures Suite**: High-throughput GPU/CPU solvers for famous mathematical problems and conjectures:
  - **Collatz Conjecture ($3n+1$)**: Parallel trajectory verification, stopping times, and peak value searches.
  - **Goldbach's Conjecture**: Batch prime pair partitions ($2k = p_1 + p_2$) and partition counting.
  - **Mersenne & Twin Primes**: Deterministic Lucas-Lehmer primality testing for $M_p = 2^p - 1$ and segmented twin prime sieves.
  - **Riemann Hypothesis**: Critical line $s = 1/2 + it$ zero approximation via Dirichlet Eta expansion.
  - **Chaotic Dynamical Systems**: Poincaré Three-Body gravitational integration (Figure-8 orbit) and Lorenz attractor with Lyapunov exponent estimation.
  - **Navier-Stokes Fluid Dynamics**: 2D incompressible fluid solver with pressure Poisson equation for cavity flow.
  - **NP-Hard Optimization**: Multi-walker parallel simulated annealing for the Traveling Salesperson Problem (TSP).

---

## Installation

### Standard Installation (from source)
```bash
git clone https://github.com/ai-tech-tw/func_pie.git
cd func_pie
pip install .
```

### Developer Mode (Editable)
```bash
git clone https://github.com/ai-tech-tw/func_pie.git
cd func_pie
pip install -e .
```

---

## Quickstart

### 1. Basic Arithmetic & Functional Pipelines
```python
from funcpie import Calculation
from funcpie.core import pipeable

# Method chaining with Calculation
calc = Calculation(10)
calc.add(15).mul(2).sub(10).div(2)
print(calc.value)  # ((10 + 15) * 2 - 10) / 2 = 20.0

# Unix-style pipe operations
@pipeable
def add_ten(x):
    return x + 10

@pipeable
def square(x):
    return x ** 2

result = 5 | add_ten | square
print(result)  # (5 + 10)^2 = 225
```

### 2. Vectors & Linear Algebra
```python
from funcpie.math import Vector2D, Matrix

# 2D Vector operations
v1 = Vector2D(3, 4)
v2 = Vector2D(1, 2)
print(v1.magnitude)  # 5.0
print(v1.dot(v2))    # 11.0

# Matrix operations and solving linear systems:
#  2x +  y = 4
#   x + 3y = 7
A = Matrix([[2, 1], [1, 3]])
x = A.solve([4, 7])
print(x)  # [1.0, 2.0]
```

### 3. Calculus & Numerical Analysis
```python
from funcpie.math import derivative, integrate, find_root_newton

# Numerical differentiation: derivative of f(x) = x^3 at x = 2
df = derivative(lambda x: x ** 3, 2.0)  # ~ 12.0

# Numerical integration: definite integral of f(x) = x^2 from 0 to 3
area = integrate(lambda x: x ** 2, 0.0, 3.0)  # 9.0

# Newton-Raphson root finding: solve x^2 - 4 = 0
root = find_root_newton(lambda x: x ** 2 - 4.0, x0=3.0)  # 2.0
```

### 4. Classical Mechanics & Projectile Motion
```python
from funcpie.physics import mechanics

# Projectile motion simulation (initial velocity 50 m/s, launch angle 45 deg)
summary = mechanics.projectile_motion(v0=50.0, angle_deg=45.0)
print(f"Max Height:        {summary.max_height:.2f} m")
print(f"Total Flight Time: {summary.flight_time:.2f} s")
print(f"Horizontal Range:  {summary.horizontal_range:.2f} m")

# Earth escape velocity
v_esc = mechanics.escape_velocity(mass=5.972e24, radius=6.371e6)
print(f"Escape Velocity:   {v_esc / 1000.0:.2f} km/s")
```

### 5. Modern Physics & Relativity
```python
from funcpie.physics import constants, modern

# Lorentz factor and time dilation at 0.8c
v = 0.8 * constants.c
gamma = modern.lorentz_factor(v)
dilated_time = modern.time_dilation(proper_time=1.0, velocity=v)
print(f"Lorentz Factor: {gamma:.4f}")
print(f"Dilated Time:   {dilated_time:.4f} s")

# Mass-energy equivalence for 1 kg rest mass
print(f"E = mc^2:       {modern.rest_energy(1.0):.2e} J")
```

### 6. Unit Conversions
```python
from funcpie.physics import units

print(units.convert(100, "km/h", "mph"))         # 62.14 mph
print(units.celsius_to_fahrenheit(100))          # 212.0 °F
print(units.convert(1, "atm", "psi"))            # 14.70 psi
print(units.convert(1, "light_year", "km"))      # 9.46e12 km
```

### 7. Mathematical Problems & Conjectures
```python
from funcpie.problems import (
    collatz_batch_verify,
    goldbach_partition,
    lucas_lehmer_mersenne,
    riemann_zeta_critical_line,
    tsp_simulated_annealing,
)

# Collatz conjecture verification
collatz_results = collatz_batch_verify(start=1, count=1000)

# Goldbach prime partition: 1000 = 3 + 997
p1, p2 = goldbach_partition(1000)
print(f"Goldbach partition for 1000: {p1} + {p2}")

# Lucas-Lehmer deterministic primality for Mersenne number M_13 = 2^13 - 1 = 8191
is_prime = lucas_lehmer_mersenne(13)             # True

# Solve TSP using multi-walker simulated annealing
cities = [(0.0, 0.0), (0.0, 4.0), (3.0, 4.0), (3.0, 0.0)]
best_tour, best_dist = tsp_simulated_annealing(cities, num_walkers=8)
```

### 8. OpenAI/math Theoretical Verification Suite
```python
from funcpie.problems import (
    EllipticCurve,
    bsd_ratio_estimate,
    goldfeld_twist_batch,
    ramanujan_tau,
    verify_deligne_bound,
    lll_reduction,
    is_ramsey_graph,
)

# 1. Birch and Swinnerton-Dyer (BSD) Conjecture on curve y^2 = x^3 - x
curve = EllipticCurve(-1.0, 0.0)
primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
bsd_ratio = bsd_ratio_estimate(-1, 0, primes)   # L(E, 1) / Omega_E
twists = goldfeld_twist_batch(-1, 0, [-3, -1, 1, 3, 5], primes)

# 2. Modular forms: Ramanujan tau function and Deligne bound (|tau(p)| <= 2 * p^(11/2))
tau_2 = ramanujan_tau(2)                        # -24
valid_bound, tau_p, bound = verify_deligne_bound(7)  # True, tau(7)=-16744

# 3. LLL polynomial-time lattice basis reduction
basis = [[1.0, 1.0, 1.0], [-1.0, 0.0, 2.0], [3.0, 5.0, 6.0]]
reduced_basis = lll_reduction(basis, delta=0.75)

# 4. Extremal Combinatorics: Ramsey graph validation (C_5 proves R(3, 3) > 5)
c5 = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
]
valid_ramsey = is_ramsey_graph(c5, k=3, l=3)     # True
```

---

## Directory Structure

```
func_pie/
├── funcpie/                    # Main package source
│   ├── __init__.py             # Public API exports & __version__
│   ├── py.typed                # PEP 561 typing marker
│   ├── backend/                # Compute backends (CPU fallback, AMD HIP & Apple Metal)
│   │   ├── __init__.py
│   │   ├── cpu.py              # Pure Python standard CPU operations
│   │   ├── hip.py              # AMD HIP runtime acceleration via ctypes
│   │   └── metal.py            # Apple Silicon Metal acceleration via ctypes
│   ├── core/                   # Core functional utilities
│   │   ├── __init__.py
│   │   └── pipeline.py         # Functional pipeline (pipe, compose, curry, Pipeable)
│   ├── math/                   # Mathematics modules
│   │   ├── __init__.py
│   │   ├── basic.py            # Basic arithmetic & safe operations
│   │   ├── algebra.py          # Polynomials, roots, number theory, combinatorics
│   │   ├── calculus.py         # Differentiation, integration, root finding
│   │   ├── linalg.py           # Vectors (2D/3D), matrices, linear systems
│   │   ├── statistics.py       # Descriptive statistics, regression, distributions
│   │   ├── geometry.py         # 2D/3D geometry, coordinate transforms
│   │   └── special.py          # Special functions (Gamma, Beta, Erf) & activations
│   └── physics/                # Physics modules
│       ├── __init__.py
│       ├── constants.py        # CODATA physical constants
│       ├── mechanics.py        # Kinematics, dynamics, gravitation, collisions
│       ├── thermodynamics.py   # Ideal gas, conduction, radiation, heat engines
│       ├── electromagnetism.py # Coulomb's law, circuits, Lorentz force
│       ├── waves.py            # Sound speed, Doppler, Snell's law, optics
│       ├── modern.py           # Special relativity, photons, decay law
│       └── units.py            # Comprehensive physical unit converter
│   └── problems/               # Mathematical problems & famous conjectures suite
│       ├── __init__.py
│       ├── collatz.py          # Collatz 3n+1 conjecture batch verifier
│       ├── goldbach.py         # Goldbach's conjecture prime partitions
│       ├── primes.py           # Mersenne Lucas-Lehmer & twin primes sieve
│       ├── riemann.py          # Riemann Hypothesis critical line zeros
│       ├── three_body.py       # Chaotic Three-Body gravitational orbits
│       ├── lorenz.py           # Lorenz attractor & Lyapunov exponent
│       ├── navier_stokes.py    # 2D Navier-Stokes fluid cavity solver
│       ├── tsp.py              # NP-hard TSP parallel simulated annealing
│       ├── bsd.py              # Birch and Swinnerton-Dyer (BSD) & Elliptic curves
│       ├── modular.py          # Modular forms & Ramanujan tau function
│       ├── lattice.py          # LLL lattice basis reduction & shortest vector
│       └── combinatorics_open.py # Ramsey graph checking & stochastic search
├── tests/                      # Full unit test suite (89+ test cases)
├── examples/                   # Executable example scripts
│   ├── quickstart.py
│   ├── physics_simulation.py
│   ├── hip_benchmark.py
│   ├── math_problems_showcase.py
│   └── openai_math_verifier.py
├── pyproject.toml              # PEP 517/621 build configuration
├── setup.py                    # Setup script for packaging
└── LICENSE                     # MIT License
```

---

## Running Tests & Type Checking

### Running Tests
Run the test suite using `pytest`:
```bash
pytest -v
```
Or use Python's built-in `unittest` runner:
```bash
python3 -m unittest discover tests -v
```

### Static Type Checking
Perform static type checking using `basedpyright`:
```bash
npx basedpyright
```

### Running Examples & Benchmarks
Execute example and benchmark scripts:
```bash
python3 examples/quickstart.py
python3 examples/physics_simulation.py
python3 examples/hip_benchmark.py
```

---

## FastMCP Physics Server (Model Context Protocol)

FuncPie integrates with [FastMCP](https://github.com/jlowin/fastmcp) to expose comprehensive physics computation tools (Mechanics, Thermodynamics, Electromagnetism, Waves & Optics, Modern Physics, Units & Constants) to AI assistants (e.g., Claude Desktop, Cursor, and custom Bots):

### Running with uvx (Direct from GitHub, Zero Installation)
```bash
uvx --from "git+https://github.com/ai-tech-tw/func_pie[mcp]" funcpie-mcp
```

### Client Configuration (Claude Desktop, Cursor, AI Bots)
Configure your client (such as `claude_desktop_config.json`) to invoke `uvx`, automatically provisioning the latest environment on demand:
```json
{
  "mcpServers": {
    "funcpie-physics": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/ai-tech-tw/func_pie[mcp]",
        "funcpie-mcp"
      ]
    }
  }
}
```

### Remote GPU Compute Server Mode (HTTP / SSE with Argon2 Auth)
When your GPU-accelerated node and AI Bot reside on different physical machines, run the MCP server over HTTP/SSE protected by API Key / Argon2 authentication (matching `aurora-mcp` with `${ENV_VAR}` syntax support):

```bash
# Launch on the GPU compute host (supports Argon2 hash or plain API key)
funcpie-mcp --transport sse --host 0.0.0.0 --port 8080 --api-key "$argon2id$v=19$..."

# Or specify via environment variable
export FUNCPie_API_KEY="my-secret-key"
funcpie-mcp --transport sse --host 0.0.0.0 --port 8080
```

Remote AI Bot Client Configuration (Claude Desktop, Cursor, Antigravity):
```json
{
  "mcpServers": {
    "funcpie-physics": {
      "serverUrl": "http://gpu-host.internal:8080/sse",
      "headers": {
        "x-api-key": "my-secret-key"
      }
    }
  }
}
```
### Modern Agent Architecture: Discovery & Execution Engine
To prevent LLM context bloating and tool flooding, FuncPie MCP exposes 4 streamlined core tools covering all mathematics, physics, and arithmetic operations:
- `funcpie_search(query, category=None)`: Discover functions, formulas, and physical constants.
- `funcpie_list(category=None)`: Browse functions organized by module (`math.basic`, `physics.mechanics`, etc.).
- `funcpie_execute(expression, variables=None)`: Safely evaluate mathematical, arithmetic, and physical formulas using Google Common Expression Language (CEL), offering non-Turing complete execution immune to arbitrary code injection (RCE).

*(For legacy granular tools, pass `--legacy-tools`)*

---

## License

This project is licensed under the [MIT License](LICENSE).
