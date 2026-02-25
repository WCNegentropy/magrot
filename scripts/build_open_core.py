#!/usr/bin/env python3
"""
Build the magrot-core open-source distribution.

Creates a clean directory containing only the open-core files, with all
modifications applied (trimmed __init__.py, stub modules, test split,
patched output paths).  Produces a zip archive ready for a fresh public
repository.

Usage:
    python scripts/build_open_core.py [--output-dir ./magrot_open]
"""

import argparse
import os
import re
import shutil
import textwrap
import zipfile
from pathlib import Path

VERSION = "0.1.0"
PACKAGE_NAME = "magrot-core"
ROOT = Path(__file__).resolve().parent.parent  # repo root

# ── File manifest ────────────────────────────────────────────────────────────
# Files to copy verbatim (source path relative to ROOT → dest path in output).

COPY_FILES = [
    # Core package
    "magrot/numerics.py",
    # fields
    "magrot/fields/__init__.py",
    "magrot/fields/analytic.py",
    "magrot/fields/grid.py",
    "magrot/fields/io.py",
    # geometry
    "magrot/geometry/__init__.py",
    "magrot/geometry/curvature.py",
    "magrot/geometry/fieldlines.py",
    # stress
    "magrot/stress/__init__.py",
    "magrot/stress/maxwell.py",
    "magrot/stress/decompose.py",
    # rotation
    "magrot/rotation/__init__.py",
    "magrot/rotation/metrics.py",
    "magrot/rotation/normalize.py",
    # thermodynamics (open subset)
    "magrot/thermodynamics/free_energy.py",
    "magrot/thermodynamics/entropy.py",
    "magrot/thermodynamics/constraints.py",
    # dynamics (legacy)
    "magrot/dynamics/__init__.py",
    "magrot/dynamics/mhd_1d.py",
    "magrot/dynamics/em_wave.py",
    "magrot/dynamics/evolve.py",
    # viz
    "magrot/viz/__init__.py",
    "magrot/viz/fields_2d.py",
    "magrot/viz/fields_3d.py",
    "magrot/viz/rotation_map.py",
    "magrot/viz/time_series.py",
    # validation
    "magrot/validation/__init__.py",
    "magrot/validation/zpinch.py",
    "magrot/validation/theta_pinch.py",
    "magrot/validation/bennett.py",
    "magrot/validation/wave.py",
    # tests (verbatim subset)
    "magrot/tests/__init__.py",
    "magrot/tests/test_fields.py",
    "magrot/tests/test_geometry.py",
    "magrot/tests/test_stress.py",
    "magrot/tests/test_rotation.py",
    "magrot/tests/test_validation.py",
    # simulations (self-contained — v3 and earth_dipole patched for output paths)
    "simulations/magrot_sim_v1.py",
    "simulations/magrot_sim_v2.py",
    "simulations/magrot_sim_v3.py",
    "simulations/earth_dipole.py",
    # docs
    "docs/MAGROT_Framework_Spec.md",
    "docs/RESULTS_SUMMARY.txt",
    # .gitignore
    ".gitignore",
]

# Directories to copy recursively (results plots).
COPY_DIRS = [
    "results/v1_mvp",
    "results/v3_engineering",
    "results/earth_dipole",
]


# ── Generated content ────────────────────────────────────────────────────────

INIT_PY = textwrap.dedent('''\
    """
    MagRot Core: Rotational-Vector Framework for Magnetic Field Dynamics
    ====================================================================

    A dimensionless rotational parameter R(x) that encodes the local tendency
    toward collapse, expansion, or equilibrium in magnetic field configurations.

    Metrics:
        R_kappa     -- Metric A: curvature ratio (kappa / kappa_eq)
        R_collapse  -- Metric B: collapse indicator (1 + C)
        R_misalign  -- Metric C: current-field misalignment |B x curl(B)| / |B|^2
        R_universal -- Universal: |f_inward| / |f_outward|

    Convention (force-ratio):
        R < 1  ->  Expanding (outward forces dominate)
        R = 1  ->  Equilibrium (force balance)
        R > 1  ->  Contracting (inward forces dominate)

    Thermodynamics:
        free_energy      -- F[B, p] functional computation
        entropy          -- Local entropy production rate s_dot(x)
        constraints      -- Helicity K, flux Phi, mass M conservation

    Author: WCNEGENTROPY HOLDINGS LLC
    License: MIT
    """

    __version__ = "{version}"
''').format(version=VERSION)

THERMO_INIT_PY = textwrap.dedent('''\
    """
    Thermodynamic diagnostic module.

    Provides free energy computation, entropy production analysis, and
    conservation constraint evaluation for magnetic field configurations.

    Submodules
    ----------
    free_energy   -- F[B, p] functional computation
    entropy       -- Local entropy production rate density s_dot(x)
    constraints   -- Magnetic helicity K, total flux Phi, total mass M
    """
''')

STUB_STATE_FLOW = textwrap.dedent('''\
    """
    State flow engine — available in MagRot Pro.

    The entropy-parameterized variational relaxation solver is part of the
    MagRot Pro commercial product.
    """
    raise ImportError(
        "magrot.thermodynamics.state_flow is not included in magrot-core. "
        "The state flow engine is available in MagRot Pro."
    )
''')

STUB_STABILITY_INIT = textwrap.dedent('''\
    """
    Stability analysis module — available in MagRot Pro.

    The Hessian eigenvalue solver, entropy audit framework, constraint
    manifold mapping, and basin-of-attraction characterization are part
    of the MagRot Pro commercial product.
    """
    raise ImportError(
        "magrot.stability is not included in magrot-core. "
        "The stability analysis suite is available in MagRot Pro."
    )
''')

PYPROJECT_TOML = textwrap.dedent('''\
    [build-system]
    requires = ["setuptools>=68.0", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "{name}"
    version = "{version}"
    description = "Rotational-Vector Diagnostic Framework for Magnetic Field Analysis"
    readme = "README.md"
    license = {{text = "MIT"}}
    authors = [
        {{name = "WCNEGENTROPY HOLDINGS LLC"}},
    ]
    requires-python = ">=3.10"
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ]
    dependencies = [
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ]

    [project.optional-dependencies]
    viz3d = [
        "pyvista>=0.43",
    ]
    dev = [
        "pytest>=7.0",
        "jupyter>=1.0",
    ]

    [tool.setuptools.packages.find]
    include = ["magrot*"]

    [tool.pytest.ini_options]
    testpaths = ["magrot/tests"]
''').format(name=PACKAGE_NAME, version=VERSION)

LICENSE_TEXT = textwrap.dedent('''\
    MIT License

    Copyright (c) 2024-2026 WCNEGENTROPY HOLDINGS LLC

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
''')

README_MD = textwrap.dedent('''\
    # MagRot Core: Rotational-Vector Framework for Magnetic Field Dynamics

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
    **Version:** {version}

    ---

    ## Metrics

    The framework implements four complementary R definitions:

    | Metric | Name | Formula | Strength |
    |--------|------|---------|----------|
    | A | Curvature Ratio (R_kappa) | kappa / kappa_eq | Best overall performer |
    | B | Collapse Indicator (R_C) | 1 + C(x) | Physically grounded |
    | C | Misalignment (R_J) | \\|B x curl(B)\\| / \\|B\\|^2 | Works from B alone |
    | Universal | Force Ratio | \\|f_inward\\| / \\|f_outward\\| | Self-normalizing |

    ## Installation

    ```bash
    pip install magrot-core
    ```

    Or install from source:

    ```bash
    pip install -e .
    ```

    ### Dependencies

    - **Core:** numpy, scipy, matplotlib
    - **Optional:** pyvista (3D viz)
    - **Dev:** pytest, jupyter

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

    print(f"R_kappa at r=a: {{results['R_kappa'][200, 0, 0]:.4f}}")
    print(f"R_universal range: [{{results['R_universal'].min():.2f}}, "
          f"{{results['R_universal'].max():.2f}}]")
    ```

    ## Package Structure

    ```
    magrot/
    ├── __init__.py
    ├── numerics.py             # 4th-order finite differences
    ├── fields/
    │   ├── analytic.py         # Wire, Z-pinch, theta-pinch, dipole
    │   ├── grid.py             # Cylindrical & Cartesian grids
    │   └── io.py               # Field data import/export
    ├── geometry/
    │   ├── curvature.py        # kappa = (b . nabla) b
    │   └── fieldlines.py       # Field-line tracing (RK45)
    ├── stress/
    │   ├── maxwell.py          # Conservative J x B forces
    │   └── decompose.py        # Tension / pressure separation
    ├── rotation/
    │   ├── metrics.py          # All 4 R definitions
    │   └── normalize.py        # Convention toggle
    ├── thermodynamics/
    │   ├── free_energy.py      # F[B, p] functional computation
    │   ├── entropy.py          # Local entropy production s_dot(x)
    │   └── constraints.py      # Helicity K, flux Phi, mass M
    ├── dynamics/
    │   └── mhd_1d.py           # 1D Z-pinch thin-shell model
    ├── viz/
    │   └── fields_2d.py        # 2D cross-section plots
    └── validation/
        ├── zpinch.py           # Z-pinch benchmarks
        └── theta_pinch.py      # Theta-pinch benchmarks
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

    ## Running Tests

    ```bash
    pytest magrot/tests/ -v
    ```

    ## Running Simulations

    Each script in `simulations/` is self-contained and generates plots:

    ```bash
    python simulations/magrot_sim_v3.py      # Full test suite with fixes
    python simulations/earth_dipole.py       # Earth dipole analysis
    ```

    ## MagRot Pro

    Advanced modules for fusion research and reactor design are available in
    MagRot Pro, including:

    - **State flow engine** -- Entropy-parameterized variational relaxation solver
    - **Stability analysis** -- Hessian eigenvalue spectrum, entropy audit,
      constraint manifold mapping, basin-of-attraction characterization
    - **Tokamak optimization** -- ITER-scale equilibrium analysis and disruption
      precursor detection

    ## Citation

    If you use MagRot Core in academic work, please cite:

    ```bibtex
    @software{{magrot_core,
      title  = {{MagRot Core: Rotational-Vector Framework for Magnetic Field Dynamics}},
      author = {{WCNEGENTROPY HOLDINGS LLC}},
      year   = {{2025}},
      url    = {{https://github.com/WCNegentropy/magrot-core}},
    }}
    ```
''').format(version=VERSION)

CI_WORKFLOW = textwrap.dedent('''\
    name: Tests
    on: [push, pull_request]
    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: ['3.10', '3.11', '3.12']
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: ${{ matrix.python-version }}
          - run: pip install -e ".[dev]"
          - run: pytest magrot/tests/ -v
''')


# ── Transformations ──────────────────────────────────────────────────────────

def trim_test_thermodynamics(src: str) -> str:
    """Remove TestDiagnostics and TestStateFlow classes and their imports."""
    lines = src.splitlines(keepends=True)
    out = []
    skip = False
    for line in lines:
        # Remove diagnostics imports
        if "from magrot.thermodynamics.diagnostics import" in line:
            # Skip this line and any continuation lines
            skip = True
            continue
        if skip:
            if line.strip() == ")":
                skip = False
                continue
            if line.strip().startswith(")"):
                skip = False
                continue
            # Still inside the multi-line import
            continue

        # Remove TestDiagnostics class and TestStateFlow class
        if re.match(r"^# ── Diagnostics tests", line):
            break  # Everything from here on is private
        out.append(line)

    # Clean trailing whitespace
    result = "".join(out).rstrip() + "\n"
    return result


def patch_simulation_output_dir(src: str, rel_dir: str) -> str:
    """Replace hardcoded /home/claude/... paths with relative output dirs."""
    # Match OUT_DIR = '/home/claude/...' or OUT_DIR = "/home/claude/..."
    patched = re.sub(
        r"""OUT_DIR\s*=\s*['"][^'"]+['"]""",
        f"OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '{rel_dir}')",
        src,
        count=1,
    )
    return patched


# ── Builder ──────────────────────────────────────────────────────────────────

def build(output_dir: Path):
    """Build the open-core distribution."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"Building {PACKAGE_NAME} v{VERSION} → {output_dir}")

    # 1. Copy verbatim files
    for rel in COPY_FILES:
        src = ROOT / rel
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"  WARN: {rel} not found, skipping")
            continue
        shutil.copy2(src, dst)
        print(f"  COPY  {rel}")

    # 2. Copy result directories
    for rel in COPY_DIRS:
        src = ROOT / rel
        dst = output_dir / rel
        if src.is_dir():
            shutil.copytree(src, dst)
            print(f"  TREE  {rel}/")
        else:
            print(f"  WARN: {rel}/ not found, skipping")

    # 3. Generate modified files
    def write(rel: str, content: str):
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)
        print(f"  GEN   {rel}")

    write("magrot/__init__.py", INIT_PY)
    write("magrot/thermodynamics/__init__.py", THERMO_INIT_PY)
    write("magrot/thermodynamics/state_flow.py", STUB_STATE_FLOW)
    write("magrot/stability/__init__.py", STUB_STABILITY_INIT)
    write("pyproject.toml", PYPROJECT_TOML)
    write("README.md", README_MD)
    write("LICENSE", LICENSE_TEXT)
    write(".github/workflows/test.yml", CI_WORKFLOW)

    # 4. Transform test_thermodynamics.py (remove private test classes)
    src_test = (ROOT / "magrot/tests/test_thermodynamics.py").read_text()
    write("magrot/tests/test_thermodynamics.py", trim_test_thermodynamics(src_test))

    # 5. Patch simulation output paths
    for script, rel_dir in [
        ("simulations/magrot_sim_v3.py", "results/v3_engineering"),
        ("simulations/earth_dipole.py", "results/earth_dipole"),
    ]:
        src_path = output_dir / script
        if src_path.exists():
            patched = patch_simulation_output_dir(src_path.read_text(), rel_dir)
            src_path.write_text(patched)
            print(f"  PATCH {script}")

    # 6. Verification scan
    print("\n── Verification ──")
    errors = []
    for py_file in output_dir.rglob("*.py"):
        rel = py_file.relative_to(output_dir)
        text = py_file.read_text()

        # Skip stub files (they intentionally reference the module name)
        if "raise ImportError" in text:
            continue

        # Check for private module imports
        for pattern in [
            r"from\s+magrot\.stability\.",
            r"from\s+magrot\.thermodynamics\.diagnostics\s+import",
            r"import\s+magrot\.stability",
        ]:
            if re.search(pattern, text):
                errors.append(f"  LEAK: {rel} references private module")

        # Check for private simulation imports
        for pattern in [
            r"from\s+magrot\.thermodynamics\.state_flow\s+import",
        ]:
            # Allow the stub file itself
            if "state_flow.py" in str(rel):
                continue
            if re.search(pattern, text):
                errors.append(f"  LEAK: {rel} references state_flow")

    if errors:
        for e in errors:
            print(e)
        raise SystemExit("Verification FAILED: private module references found")
    else:
        print("  OK    No private module references detected")

    # Check no private files leaked
    private_should_not_exist = [
        "magrot/thermodynamics/diagnostics.py",
        "magrot/stability/hessian.py",
        "magrot/stability/entropy_audit.py",
        "magrot/stability/manifold.py",
        "magrot/stability/attractors.py",
        "magrot/tests/test_stability.py",
        "simulations/magrot_tokamak.py",
        "simulations/magrot_v2_thermodynamic_suite.py",
        "results/tokamak",
        "results/v2_thermodynamic",
    ]
    for rel in private_should_not_exist:
        if (output_dir / rel).exists():
            errors.append(f"  LEAK: {rel} should not be in open core")

    if errors:
        for e in errors:
            print(e)
        raise SystemExit("Verification FAILED: private files found")
    else:
        print("  OK    No private files detected")

    # 7. Create zip archive
    zip_name = f"{PACKAGE_NAME}-{VERSION}"
    zip_path = output_dir.parent / f"{zip_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(output_dir.rglob("*")):
            if fpath.is_file():
                arcname = f"{zip_name}/{fpath.relative_to(output_dir)}"
                zf.write(fpath, arcname)
    print(f"\n  ZIP   {zip_path}  ({zip_path.stat().st_size / 1024:.0f} KB)")
    print(f"\nDone. Open core ready at: {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "magrot_open",
        help="Output directory (default: ./magrot_open)",
    )
    args = parser.parse_args()
    build(args.output_dir)
