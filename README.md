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

**Author:** Mikeal Clark / WCNEGENTROPY HOLDINGS LLC
**License:** MIT
**Status:** Active R&D -- validated through Tier 1 & 2 test cases

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
│   │   ├── analytic.py         # Wire, Z-pinch, theta-pinch, dipole
│   │   ├── grid.py             # Cylindrical & Cartesian grids
│   │   └── io.py               # Field data import/export
│   ├── geometry/
│   │   ├── curvature.py        # kappa = (b . nabla) b
│   │   └── fieldlines.py       # Field-line tracing (RK45)
│   ├── stress/
│   │   ├── maxwell.py          # Conservative J x B forces
│   │   └── decompose.py        # Tension / pressure separation
│   ├── rotation/
│   │   ├── metrics.py          # All 4 R definitions
│   │   └── normalize.py        # Convention toggle
│   ├── dynamics/
│   │   ├── mhd_1d.py           # 1D Z-pinch thin-shell model
│   │   ├── em_wave.py          # EM wave mode (Phase 2)
│   │   └── evolve.py           # Time-stepper interface
│   ├── viz/
│   │   ├── fields_2d.py        # 2D cross-section plots
│   │   ├── fields_3d.py        # 3D rendering (PyVista)
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
│       └── test_validation.py
├── simulations/                # Standalone simulation scripts
│   ├── magrot_sim_v1.py        # Original MVP (all tests)
│   ├── magrot_sim_v2.py        # Fixed dynamics + analysis
│   ├── magrot_sim_v3.py        # Engineering fixes (production)
│   ├── earth_dipole.py         # Earth dipole + radiation belts
│   └── magrot_tokamak.py       # Tokamak (Solov'ev, ITER-scale)
├── results/                    # Generated plots & data
│   ├── v1_mvp/                 # 7 plots from initial validation
│   ├── v3_engineering/         # 6 plots with all fixes applied
│   ├── earth_dipole/           # 2D R map + L-shell profiles
│   └── tokamak/                # Equilibrium maps, beta sweep, etc.
├── docs/                       # Reports & specifications
│   ├── MAGROT_Framework_Spec.md
│   ├── RESULTS_SUMMARY.txt
│   ├── MagRot_Validation_Report_v3.docx
│   ├── MagRot_Tokamak_Validation_Report_v1.docx
│   └── MagRot_Tokamak_Validation_Report_v1.pdf
├── MAGROT_Framework_Spec.md    # Framework spec (root copy)
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

# Set up a Z-pinch
grid = CylindricalGrid(Nr=400, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
B, p_mat = field_zpinch_bennett(grid, I=1e4, a=0.01)

# Define equilibrium curvature for Metric A
def kappa_eq(g):
    return np.full(g.R.shape, 1.0 / 0.01)

# Compute all metrics
results = compute_all_metrics(B, grid, p_mat=p_mat,
                              kappa_eq_func=kappa_eq, L_char=0.01)

print(f"R_kappa at r=a: {results['R_kappa'][200, 0, 0]:.4f}")
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
python simulations/magrot_sim_v3.py      # Full test suite with fixes
python simulations/earth_dipole.py       # Earth dipole analysis
python simulations/magrot_tokamak.py     # Tokamak equilibrium
```

## Future Directions

- **Geometric Algebra:** Unify static/dynamic R via Faraday bivector F = E + IcB
- **Fusion control:** Real-time R computation as feedback for Z-pinch experiments
- **Topological extension:** Combine local R with magnetic winding metrics
- **Phase 2:** Time-dependent EM wave mode, full FDTD integration
