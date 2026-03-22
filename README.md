# MagRot: Rotational-Vector Framework for Magnetic Field Dynamics

A dimensionless **rotational parameter** R(x) that encodes the local tendency
toward collapse, expansion, or equilibrium in magnetic field configurations.
Derived from field-line geometry (curvature, tension, pressure balance) rather
than raw magnitudes or current densities.

| R Value | Regime | Description |
|---------|--------|-------------|
| R < 1 | **Expanding** | Outward pressure dominates; field dissipates |
| R = 1 | **Equilibrium** | Curvature-driven tension balances pressure |
| R > 1 | **Contracting** | Inward tension dominates; field collapses |

**Author:** WCNEGENTROPY HOLDINGS LLC
**License:** MIT
**Version:** 0.2.0-dev
**Status:** Active R&D -- v1 validated (Tier 1 & 2), v2 thermodynamic framework validated (1D + full 3D)

---

## Metrics

The framework implements four complementary R definitions:

| Metric | Name | Formula | Strength |
|--------|------|---------|----------|
| A | Curvature Ratio (R_kappa) | kappa / kappa_eq | Best overall performer |
| B | Collapse Indicator (R_C) | 1 + C(x) | Physically grounded |
| C | Misalignment (R_J) | \|B x curl(B)\| / \|B\|^2 | Works from B alone |
| Universal | Force Ratio | \|f_inward\| / \|f_outward\| | Self-normalizing |

## Installation

```bash
pip install -e .
```

### Dependencies

- **Core:** numpy, scipy, matplotlib
- **Optional:** jax, jaxlib, diffrax (GPU acceleration), pyvista (3D viz)
- **Dev:** pytest, jupyter

## Package Structure

```
magrot/
├── magrot/                     # Core Python package
│   ├── __init__.py
│   ├── numerics.py             # 4th-order finite differences
│   ├── fields/
│   │   ├── analytic.py         # Wire, Z-pinch, theta-pinch, dipole (2D + 3D)
│   │   ├── grid.py             # Cylindrical & Cartesian grids (full 3D)
│   │   └── io.py               # Field data import/export
│   ├── geometry/
│   │   ├── curvature.py        # kappa = (b . nabla) b (full 3D cylindrical)
│   │   └── fieldlines.py       # Field-line tracing (RK45)
│   ├── stress/
│   │   ├── maxwell.py          # Conservative J x B forces (full 3D curl)
│   │   └── decompose.py        # Tension / pressure separation
│   ├── rotation/
│   │   ├── metrics.py          # All 4 R definitions
│   │   └── normalize.py        # Convention toggle
│   ├── thermodynamics/         # v2: Entropy-based state flow
│   │   ├── free_energy.py      # F[B, p] functional computation
│   │   ├── entropy.py          # Local entropy production rate s_dot(x)
│   │   ├── constraints.py      # Helicity K, flux Phi, mass M conservation
│   │   ├── diagnostics.py      # F(sigma) tracking, Lyapunov verification
│   │   └── state_flow.py       # Variational relaxation engine
│   ├── stability/              # v2: Entropic hypothesis tests
│   │   ├── hessian.py          # Multi-axis perturbation -> Hessian eigenvalues
│   │   ├── entropy_audit.py    # Entropy production at equilibrium
│   │   ├── manifold.py         # Constraint boundary mapping (3D volume-avg)
│   │   └── attractors.py       # Basin-of-attraction characterization
│   ├── dynamics/               # v1 legacy time-based models
│   │   ├── mhd_1d.py           # 1D Z-pinch thin-shell model
│   │   ├── em_wave.py          # EM wave mode
│   │   └── evolve.py           # Time-stepper interface
│   ├── viz/
│   │   ├── fields_2d.py        # 2D cross-section plots
│   │   ├── fields_3d.py        # 3D matplotlib viz (polar, r-z, quiver)
│   │   ├── rotation_map.py     # R(x) heatmaps
│   │   └── time_series.py      # R(t) evolution plots
│   ├── validation/
│   │   ├── zpinch.py           # Z-pinch benchmarks
│   │   ├── theta_pinch.py      # Theta-pinch benchmarks
│   │   ├── bennett.py          # Bennett equilibrium
│   │   └── wave.py             # Plane wave verification
│   └── tests/
│       ├── test_fields.py
│       ├── test_geometry.py
│       ├── test_stress.py
│       ├── test_rotation.py
│       ├── test_validation.py
│       ├── test_thermodynamics.py  # v2: Entropy & free energy tests
│       └── test_stability.py       # v2: Stability analysis tests
├── simulations/                # Standalone simulation scripts
│   ├── magrot_sim_v1.py        # Original MVP (all tests)
│   ├── magrot_sim_v2.py        # Fixed dynamics + analysis
│   ├── magrot_sim_v3.py        # Engineering fixes (production)
│   ├── earth_dipole.py         # Earth dipole + radiation belts
│   ├── magrot_tokamak.py       # Tokamak (Solov'ev, ITER-scale)
│   ├── magrot_v2_thermodynamic_suite.py  # v2 1D validation (33 checks)
│   └── magrot_v2_3d_suite.py   # v2 full 3D validation (37 checks)
├── results/                    # Generated plots & data
│   ├── v1_mvp/                 # 7 plots from initial validation
│   ├── v3_engineering/         # 6 plots with all fixes applied
│   ├── earth_dipole/           # 2D R map + L-shell profiles
│   ├── tokamak/                # Equilibrium maps, beta sweep, etc.
│   ├── v2_thermodynamic/       # 7 plots + RESULTS.md (1D thermo suite)
│   └── v2_3d_thermodynamic/    # 7 plots + RESULTS.md (full 3D suite)
├── docs/                       # Reports & specifications
│   ├── MAGROT_Framework_Spec.md
│   ├── RESULTS_SUMMARY.txt
│   ├── MagRot_Validation_Report_v3.docx
│   ├── MagRot_Tokamak_Validation_Report_v1.docx
│   └── MagRot_Tokamak_Validation_Report_v1.pdf
├── MAGROT_v2_Plan.md           # v2 development roadmap
├── pyproject.toml
└── README.md
```

## Validation Status

### Tier 1: Analytic Verification (All PASS)

| Test | Configuration | Key Result |
|------|--------------|------------|
| 1.1 | Infinite wire | R_universal = 1.0 everywhere (vacuum self-equilibrium) |
| 1.2 | Z-pinch Bennett | R_kappa(r=a) = 1.006, correct inside/outside asymmetry |
| 1.3 | Z-pinch vs theta-pinch | Clean curvature separation (10^11 : 1 ratio) |
| 1.4 | Screw pinch sweep | Smooth transition across pitch angles |
| 1.5 | Dipole 2D map | Spatial R variation, no singularity artifacts |
| 1.6 | Z-pinch dynamics | Oscillation around Bennett eq., R tracks state |

### Tier 2: Cross-Validation

| Test | Configuration | Key Result |
|------|--------------|------------|
| 2.1 | Earth dipole | R = 1.0 at equator (universal across L-shells) |
| 2.2 | Tokamak (ITER) | R maps equilibrium, detects beta limits |

### v2 Thermodynamic Validation — 1D (28/33 checks pass)

Entropy-parameterized state flow replaces clock-time dynamics. Headline result:
entropy-parameterized Z-pinch converges to R = 1.000000 exactly; time-based method
still oscillates at R = 1.05 after 15 periods. 5 known findings (ODE kinetic energy,
test calibration), no framework bugs. See `results/v2_thermodynamic/RESULTS.md`.

### v2 Thermodynamic Validation — Full 3D (26/37 checks pass)

Full 3D upgrade of all physics modules (curl, curvature, forces) and all 7 tests.
New validations: angular symmetry (std = 0.00 for axisymmetric fields), div(B) ~ 0
to machine precision, 3D Cartesian dipole with azimuthal symmetry to 1e-15. 11
known findings (8 inherited ODE dynamics, 3 physics tolerances), zero 3D-specific
failures. Key discovery: Hessian classification changes from saddle-point (1D) to
minimum (3D) — uniform perturbations wash out mode-specific instabilities,
identifying mode-resolved perturbations as the next development target.
See `results/v2_3d_thermodynamic/RESULTS.md`.

### Engineering Fixes (v3)

1. **epsilon-floor regularization** -- eliminates R_universal overflow
2. **Conservative J x B** -- avoids catastrophic cancellation in forces
3. **Dynamic L0 = min(1/kappa, L_char)** -- stabilizes Metric B at boundaries
4. **Softened dipole origin** -- removes masking artifacts
5. **Damped dynamics** -- demonstrates resistive relaxation to R = 1

## Quick Start

```python
from magrot.fields.grid import CylindricalGrid
from magrot.fields.analytic import field_zpinch_bennett
from magrot.rotation.metrics import compute_all_metrics
import numpy as np

# Set up a 3D Z-pinch (works with any Ntheta, Nz — including 1D)
grid = CylindricalGrid(Nr=80, Ntheta=32, Nz=20, r_range=(0.0003, 0.05))
B, p_mat = field_zpinch_bennett(grid, I=1e4, a=0.01)

# Define equilibrium curvature for Metric A
def kappa_eq(g):
    return np.full(g.R.shape, 1.0 / 0.01)

# Compute all metrics (full 3D curl, curvature, forces)
results = compute_all_metrics(B, grid, p_mat=p_mat,
                              kappa_eq_func=kappa_eq, L_char=0.01)

print(f"R_kappa at r=a: {results['R_kappa'][40, 0, 0]:.4f}")
print(f"R_universal range: [{results['R_universal'].min():.2f}, "
      f"{results['R_universal'].max():.2f}]")
```

## Running Tests

```bash
pytest magrot/tests/ -v
```

## Running Simulations

Each script in `simulations/` is self-contained and generates plots:

```bash
python simulations/magrot_sim_v3.py                # v1 test suite with fixes
python simulations/earth_dipole.py                 # Earth dipole analysis
python simulations/magrot_tokamak.py               # Tokamak equilibrium
python simulations/magrot_v2_thermodynamic_suite.py # v2 thermo (1D, 33 checks)
python simulations/magrot_v2_3d_suite.py           # v2 thermo (3D, 37 checks)
```

## v2: Thermodynamic State Flow

Version 2 replaces the v1 time-based dynamics paradigm with entropy-parameterized
state evolution. The evolution parameter is entropy produced (sigma), not elapsed
time (t), yielding guaranteed convergence via the second law and physical
interpretation at every step.

Key additions:

- **Free energy functional** F[B, p] and its connection to R (|R - 1| maps
  local free energy density)
- **Entropy production** s_dot(x) = eta|J|^2/T with Spitzer resistivity
- **Constraint conservation** -- magnetic helicity K, flux Phi, mass M
- **Variational relaxation** engine with steepest descent, L-BFGS, and
  Onsager linear response schemes
- **Entropic hypothesis tests** (Tests A--D) to determine whether R = 1 is a
  conditional entropy maximum, a negentropic state, or a saddle point
- **Full 3D physics pipeline** -- curl(B), curvature kappa = (b.nabla)b, and
  Maxwell forces all support full 3D cylindrical (r, theta, z) coordinates
  with conditional derivative terms when Ntheta > 1 or Nz > 1
- **3D Cartesian dipole** -- field_dipole_cartesian_3d() with full 3D curvature
- **3D visualization** -- polar heatmaps, r-z cross sections, quiver plots,
  orthogonal slice views via matplotlib (no PyVista dependency)
- **Divergence-free validation** -- div(B) checks confirm field generators
  satisfy Maxwell's equations to machine precision on 3D grids

## Future Directions

- **Mode-resolved 3D stability:** Extend Hessian analysis with spatially
  structured perturbations (m=0 sausage, m=1 kink, m=2 elliptical) instead
  of uniform parameter sweeps, which wash out mode-specific instabilities
  in 3D volume integrals (see v2 3D Finding #12)
- **Kinetic energy tracking:** Add E_kinetic to state flow F_total for strict
  monotonicity in inertial systems (resolves dynamics F non-monotonicity)
- **Tokamak 3D applications:** Apply full 3D pipeline to Solov'ev equilibrium,
  q0 tuning, X-point geometry, disruption precursor identification
- **Non-axisymmetric equilibria:** Stellarators, 3D islands -- the full 3D
  curl and curvature operators are ready for non-axisymmetric fields
- **Geometric Algebra:** Unify static/dynamic R via Faraday bivector F = E + IcB
- **Fusion control:** Real-time R computation as feedback for Z-pinch experiments
- **Topological extension:** Combine local R with magnetic winding metrics


## ⚖️ Licensing and Copyright

The MAGROT project utilizes a split-licensing model to encourage open scientific collaboration while protecting the commercial engineering implementations of the framework. All intellectual property is held by WCNEGENTROPY HOLDINGS LLC.

Software & Simulators (Root, magrot/, simulations/, tests/): All source code, JAX-accelerated Forge optimizers, and simulation engines are licensed under the GNU Affero General Public License v3.0 (AGPLv3).

Research & Documentation (docs/ and results/ directories): All framework specifications, validation reports, mathematical derivations, and generated visual results are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. You are free to share, adapt, and commercially publish this theoretical and visual work, provided you give appropriate credit to WCNegentropy Holdings LLC and link to this repository.

## 💼 Commercial Licensing: 

> The AGPLv3 requires that any modified versions or network services (SaaS) running this code be open-sourced. If your organization wishes to integrate the MAGROT Forge optimizer or plasma confinement simulators into a proprietary, closed-source commercial backend, you must obtain a commercial license.
Contact: contact@wcnegentropy.com for commercial licensing inquiries.
