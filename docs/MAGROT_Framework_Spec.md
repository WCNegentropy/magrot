# MagRot: Rotational-Vector Framework for Magnetic Field Dynamics

## Framework Specification & Simulator Blueprint

**Version:** 0.2.0-dev
**Author:** Mikeal Clark / WCNEGENTROPY HOLDINGS LLC
**License:** MIT
**Status:** v1 implemented & validated; v2 thermodynamic modules implemented

---

## 1. Core Thesis

Magnetic field configurations can be characterized by a dimensionless **rotational parameter** ℛ(x) that encodes the local tendency toward collapse, expansion, or equilibrium. This parameter is derived from the geometry of field lines — their curvature, tension, and pressure balance — rather than from raw field magnitudes or current densities alone.

| ℛ Value | Physical Regime | Description |
|---------|----------------|-------------|
| ℛ < 1 | **Contracting** | Inward magnetic tension dominates; field lines are "over-curved" relative to equilibrium. Structure collapses (Z-pinch, magnetic crush). |
| ℛ = 1 | **Self-contained** | Force balance achieved; curvature-driven tension exactly balances pressure gradients. Stable equilibrium (Bennett pinch, solitons). |
| ℛ > 1 | **Expanding** | Outward pressure dominates; field-line curvature is insufficient for self-containment. Structure dissipates (radiation, diffraction, wave propagation). |

The underlying physics is entirely classical electromagnetism (Maxwell + Lorentz). The novelty is the **reparameterization**: a geometric lens that unifies collapse, stability, and expansion phenomena under a single scalar diagnostic computed from field-line geometry.

---

## 2. Mathematical Framework

### 2.1 Foundational Quantities

Given a magnetic field B(x), define:

**Unit tangent vector** along field lines:
$$\mathbf{b} = \frac{\mathbf{B}}{|\mathbf{B}|}$$

**Curvature vector** of field lines (rate of change of direction along the line):
$$\boldsymbol{\kappa} = (\mathbf{b} \cdot \nabla)\,\mathbf{b}$$

with scalar curvature κ = |κ| and curvature normal n̂ = κ/κ.

**Magnetic tension force density** (force per unit volume acting to straighten curved field lines):
$$\mathbf{f}_{\text{tension}} = \frac{B^2}{\mu_0}\,\boldsymbol{\kappa}$$

**Magnetic pressure** and its gradient:
$$p_B = \frac{B^2}{2\mu_0}, \qquad \nabla p_B = \frac{1}{2\mu_0}\nabla B^2$$

**Total magnetic force density** (equivalent to divergence of the Maxwell stress tensor):
$$\mathbf{f}_{\text{mag}} = \frac{1}{\mu_0}\left[(\mathbf{B} \cdot \nabla)\mathbf{B} - \frac{1}{2}\nabla B^2\right] = \mathbf{f}_{\text{tension}} - \nabla p_B$$

These are standard MHD quantities. The framework's contribution is what we build from them.

### 2.2 Candidate ℛ Definitions

The framework defines **three candidate metrics**, each capturing a different aspect of the rotational geometry. The MVP implements all three so empirical testing determines which is most informative for which regime.

#### Metric A: Curvature Ratio (ℛ_κ)

$$\mathcal{R}_\kappa(\mathbf{x}) = \frac{\kappa(\mathbf{x})}{\kappa_{\text{eq}}(\mathbf{x})}$$

where κ_eq is the curvature that would produce force balance at point x given the local B and material pressure profile.

**Interpretation:** Ratio of actual field-line curvature to equilibrium curvature. Direct geometric measure.

**Strengths:** Purely geometric; independent of field strength normalization.
**Weaknesses:** Requires defining κ_eq per configuration. See §2.3 for normalization strategy.

**Analytic check:** For a Z-pinch with B_θ(r) = μ₀I/(2πr), field lines are circles of radius r, so κ = 1/r. In Bennett equilibrium, the equilibrium curvature at radius a is κ_eq = 1/a. Therefore ℛ_κ(r) = a/r. At the equilibrium boundary r = a: ℛ_κ = 1. Inside (r < a): ℛ_κ > 1 (contracting). Outside (r > a): ℛ_κ < 1 (expanding). ✓

#### Metric B: Collapse Indicator (ℛ_C)

$$\mathcal{R}_C(\mathbf{x}) = 1 + C(\mathbf{x})$$

where the collapse indicator C is the net force projected along the curvature normal, normalized:

$$C(\mathbf{x}) = \frac{\left(\mathbf{f}_{\text{mag}} - \nabla p_{\text{mat}}\right) \cdot \hat{\mathbf{n}}}{f_0}$$

with f₀ = B²/(2μ₀L₀) as the normalization scale and L₀ a characteristic length.

**Interpretation:** Signed measure of net inward/outward force along the direction field lines "want to straighten."

**Strengths:** Directly encodes force balance; includes material pressure. Most physically grounded metric.
**Weaknesses:** Requires choosing L₀ and including material pressure profile (which may not be available for pure EM configurations).

**For vacuum EM (no material pressure):** p_mat = 0 and the metric simplifies to measuring net magnetic force imbalance:
$$C_{\text{vac}}(\mathbf{x}) = \frac{\mathbf{f}_{\text{mag}} \cdot \hat{\mathbf{n}}}{f_0}$$

#### Metric C: Current-Field Misalignment (ℛ_J)

$$\mathbf{R}(\mathbf{x}) = \frac{\mathbf{B} \times (\nabla \times \mathbf{B})}{|\mathbf{B}|^2}$$

$$\mathcal{R}_J(\mathbf{x}) = \frac{|\mathbf{R}(\mathbf{x})|}{R_\star}$$

where R⋆ is calibrated so ℛ_J = 1 at known equilibria.

**Interpretation:** Measures how far the field is from force-free (∇ × B ∥ B). When J and B are aligned, |R| = 0 (force-free). When perpendicular, |R| is maximal (maximum Lorentz force).

**Connection to existing work:** In solar/coronal physics, the force-free parameter α (from ∇ × B = αB) is widely used. The misalignment metric R captures the *departure from* force-free conditions. This framework extends beyond α by also measuring the direction of departure and normalizing it against equilibrium. The solar physics literature on α-based coronal loop stability provides an existing validation corpus.

**Strengths:** Computable from B alone (no material pressure needed). Natural for vacuum EM. Connects to well-studied force-free field literature.
**Weaknesses:** Vanishes for force-free fields regardless of stability; less informative in low-β (magnetically dominated) plasmas where force-free is a good approximation.

### 2.3 The Normalization Problem

**This is the central theoretical challenge of the framework.**

Each metric requires a reference scale to set ℛ = 1 at equilibrium. If every geometry needs its own normalization, the framework is a family of local diagnostics rather than a universal parameter.

**Strategy: Normalization via local force balance ratio.**

The most promising universal approach is to define ℛ as a ratio that is self-normalizing:

$$\mathcal{R}_{\text{univ}}(\mathbf{x}) = \frac{|\mathbf{f}_{\text{inward}}|}{|\mathbf{f}_{\text{outward}}|}$$

where f_inward is the component of magnetic force directed toward the center of curvature (tension) and f_outward is the component directed away (pressure gradient + material pressure).

This ratio is:
- Dimensionless by construction
- Equal to 1 at force balance regardless of geometry
- Greater than 1 when inward forces dominate (collapse)
- Less than 1 when outward forces dominate (expansion)
- Computable at every grid point from local field quantities

**This inverts the convention** relative to the original formulation (ℛ > 1 now means collapse rather than expansion). We adopt this convention for the simulator because it maps directly to the physical intuition: ℛ > 1 means "more inward than outward" → crush. ℛ < 1 means "more outward than inward" → expand.

**Convention choice for implementation:**

| Convention | ℛ < 1 | ℛ = 1 | ℛ > 1 |
|-----------|-------|-------|-------|
| **Original (R&D doc)** | Collapse | Equilibrium | Expansion |
| **Force-ratio (simulator)** | Expansion | Equilibrium | Collapse |

The simulator will implement the force-ratio convention internally and provide a flag to display in either convention. The framework documents should settle on one convention before publication. For the MVP, we use the force-ratio convention because it avoids the normalization problem entirely.

**Fallback:** If the universal ratio proves degenerate in some geometries (e.g., where both inward and outward forces are near zero), use the per-metric definitions from §2.2 with documented, geometry-specific normalization constants.

### 2.4 Extension to Time-Dependent Fields (Full EM)

The original framework is defined for magnetostatic/MHD contexts where B alone determines the field configuration. For time-dependent EM fields (waves, pulses), the electric field E participates in the dynamics.

**Key identity:** In vacuum, ∇ × B = μ₀ε₀ ∂E/∂t. Substituting into Metric C:

$$\mathbf{R}_{\text{dyn}} = \frac{\mathbf{B} \times (\mu_0\epsilon_0\,\partial\mathbf{E}/\partial t)}{|\mathbf{B}|^2}$$

For a plane wave with E ⊥ B ⊥ k and |E| = c|B|, this yields R_dyn ∝ k̂, aligned with the **Poynting vector** S = E × B/μ₀. This is a derived result, not an assumption.

**Physical meaning:** For propagating EM waves, the rotational parameter points along the energy flow direction and its magnitude measures the "openness" of the field-line rotation — how far the curl-into-curl cascade departs from closing on itself.

**Implementation requirement:** The dynamic extension requires storing both B(x, t) and E(x, t) or equivalently B and ∂B/∂t. The simulator must support both static (B-only) and dynamic (full EM) modes.

**Long-term direction:** Geometric Algebra unifies E and B into the Faraday bivector F = E + IcB, where "rotation" is naturally defined in spacetime. The GA formulation would make ℛ a spacetime invariant rather than requiring separate static/dynamic definitions. This is a Phase 2+ goal, not MVP.

### 2.5 Global Constraints: Magnetic Virial Theorem

The magnetic virial theorem provides a global constraint on self-contained magnetic configurations: in a finite volume with no external confinement, the integral of magnetic pressure exceeds the integral of magnetic tension, meaning **no purely magnetic configuration can self-confine in vacuum without external forces or boundary conditions.**

This means:
- A volume-averaged ℛ = 1 is **impossible** for a free-standing magnetic structure in vacuum (the virial theorem forbids it).
- ℛ = 1 requires either material pressure (plasma), boundary conditions (conductors), or nonlinear medium effects (solitons).
- Free EM radiation in vacuum is fundamentally ℛ > 1 (expanding) when volume-averaged, consistent with the framework.

The virial theorem doesn't prevent **local** ℛ = 1 surfaces or regions — it constrains the volume integral. The simulator should compute both local ℛ(x) maps and volume-integrated ⟨ℛ⟩ to check consistency with virial constraints.

### 2.6 Thermodynamic Identity of ℛ (v2)

v1 asserted that "ℛ = 1 represents the negentropic state: maximum organization where inward and outward forces achieve local balance." v2 treats this as a hypothesis to be tested empirically via Tests A–D (see `MAGROT_v2_Plan.md` §3).

**Provisional formulation (pending Test A–D results):**

> ℛ = 1 is the **conditional entropy maximum** on the constrained MHD equilibrium manifold — the state toward which dissipative relaxation drives the system given conservation of magnetic helicity and total flux. The equilibrium manifold itself represents a negentropic configuration requiring external energy throughput to maintain. The framework therefore measures two distinct things: (1) how far the system is from optimal self-organization within its accessible state space (|ℛ - 1|), and (2) implicitly, the proximity to the boundary of the accessible state space beyond which the system undergoes irreversible reconfiguration (disruption).

**Key insight:** Replacing clock time *t* with entropy production σ as the evolution parameter transforms ℛ from a diagnostic into the **driving function** of the evolution itself, since |ℛ - 1| is proportional to the local free energy density gradient.

### 2.7 Free Energy Functional (v2)

$$F[B, p] = \int_V \left[ \frac{B^2}{2\mu_0} + \frac{p}{\gamma - 1} \right] dV - F_{\text{eq}}[K, \Phi, M]$$

where F_eq is the minimum-energy state satisfying the constraints:
- K = ∫ A·B dV (magnetic helicity, conserved in ideal MHD)
- Φ = ∫ B·dA (magnetic flux through boundary, conserved for ideal conductor)
- M = ∫ ρ dV (total mass, conserved)

F ≥ 0 always. F = 0 at equilibrium.

**ℛ–F connection:**

$$|\mathcal{R}(\mathbf{x}) - 1| \propto \frac{|\delta F / \delta B(\mathbf{x})|}{B^2 / 2\mu_0}$$

This makes ℛ the *dimensionless local gradient of the free energy functional*, measured in units of the local magnetic energy density. Where ℛ ≠ 1, there exists local free energy available for conversion to heat (entropy production).

**Evolution principle:**
- δF/δσ ≤ 0 (second law — guaranteed monotonic decrease)
- Near equilibrium: Onsager minimum entropy production → smooth relaxation toward ℛ = 1
- Far from equilibrium: maximum entropy production → rapid reconfiguration (disruption analog)

---

## 3. Simulator Architecture

### 3.1 Package: `magrot`

```
magrot/
├── __init__.py
├── numerics.py              # 4th-order finite difference utilities
├── fields/
│   ├── __init__.py
│   ├── analytic.py          # Analytic field generators (wire, Z-pinch, θ-pinch, dipole)
│   ├── grid.py              # CylindricalGrid and toroidal grid classes
│   └── io.py                # Import/export field data
├── geometry/
│   ├── __init__.py
│   ├── fieldlines.py        # Field-line tracing (adaptive RK45)
│   └── curvature.py         # κ = (b · ∇)b (local curvature)
├── stress/
│   ├── __init__.py
│   ├── maxwell.py           # Conservative J × B force computation
│   └── decompose.py         # Tension/pressure separation
├── rotation/
│   ├── __init__.py
│   ├── metrics.py           # All ℛ definitions (A, B, C, universal)
│   └── normalize.py         # Normalization schemes + convention toggle
├── thermodynamics/          # v2: Entropy-based state flow
│   ├── __init__.py
│   ├── free_energy.py       # F[B, p] functional computation
│   ├── entropy.py           # Local entropy production rate ṡ(x)
│   ├── constraints.py       # Helicity K, flux Φ, mass M conservation
│   ├── diagnostics.py       # F(σ) tracking, Lyapunov verification
│   └── state_flow.py        # Variational relaxation engine
├── stability/               # v2: Entropic hypothesis tests
│   ├── __init__.py
│   ├── hessian.py           # Multi-axis perturbation → Hessian eigenvalues (Test A)
│   ├── entropy_audit.py     # Entropy production accounting at equilibrium (Test B)
│   ├── manifold.py          # Constraint boundary mapping (Test C)
│   └── attractors.py        # Basin-of-attraction characterization
├── dynamics/                # v1 legacy (time-based, kept for comparison)
│   ├── __init__.py
│   ├── mhd_1d.py            # 1D radial MHD toy model
│   ├── em_wave.py           # Time-dependent EM wave mode
│   └── evolve.py            # Time-stepper interface
├── viz/
│   ├── __init__.py
│   ├── fields_2d.py         # 2D cross-section plots
│   ├── fields_3d.py         # 3D field-line rendering (PyVista)
│   ├── rotation_map.py      # ℛ(x) heatmaps and profiles
│   └── time_series.py       # ℛ(t) evolution plots
├── validation/
│   ├── __init__.py
│   ├── zpinch.py            # Z-pinch analytic benchmarks
│   ├── theta_pinch.py       # θ-pinch benchmarks
│   ├── bennett.py           # Bennett equilibrium checks
│   └── wave.py              # Plane wave ℛ verification
└── tests/
    ├── test_fields.py
    ├── test_geometry.py
    ├── test_stress.py
    ├── test_rotation.py
    ├── test_validation.py
    ├── test_thermodynamics.py   # v2: Entropy & free energy tests
    └── test_stability.py        # v2: Stability analysis tests
```

### 3.2 Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Core numerics** | JAX | Autodiff for ∇·σ and curvature tensors; JIT compilation; GPU-ready without refactor |
| **Field-line tracing** | `diffrax` (JAX-native ODE solver) | Adaptive stepping, JIT-compatible, handles stiff regions near X-points |
| **Fallback numerics** | NumPy + SciPy | For initial prototyping before JAX port; `solve_ivp` with RK45 for field lines |
| **Visualization** | Matplotlib (2D), PyVista (3D) | Standard, well-documented, publication-quality output |
| **Interactive** | Jupyter notebooks | For exploration, validation, and demo |
| **Testing** | pytest | Standard Python testing |

**Implementation order:** Start with NumPy/SciPy for rapid prototyping. Port performance-critical kernels (curvature calculation, stress tensor divergence) to JAX once the math is validated. The API should be backend-agnostic from the start (accept and return arrays, don't hardcode JAX ops in the interface layer).

### 3.3 Grid and Data Structures

**Primary grid:** Cylindrical (r, θ, z) for pinch configurations; Cartesian (x, y, z) for general use.

```python
@dataclass
class FieldGrid:
    """Discretized field configuration on a structured grid."""
    coords: str                    # 'cartesian' or 'cylindrical'
    shape: tuple[int, ...]         # (Nr, Nθ, Nz) or (Nx, Ny, Nz)
    bounds: tuple[tuple, ...]      # ((r_min, r_max), (θ_min, θ_max), (z_min, z_max))
    B: np.ndarray                  # Shape: (*shape, 3) — magnetic field components
    E: np.ndarray | None = None    # Shape: (*shape, 3) — electric field (dynamic mode)
    J: np.ndarray | None = None    # Shape: (*shape, 3) — current density
    p_mat: np.ndarray | None = None  # Shape: (*shape,) — material pressure

@dataclass
class RotationData:
    """Computed rotational metrics on a grid."""
    R_kappa: np.ndarray | None     # Metric A: curvature ratio
    R_collapse: np.ndarray | None  # Metric B: collapse indicator
    R_misalign: np.ndarray | None  # Metric C: current-field misalignment
    R_universal: np.ndarray | None # Universal force ratio
    kappa: np.ndarray              # Scalar curvature field
    kappa_vec: np.ndarray          # Curvature vector field
    tension: np.ndarray            # Tension force density
    pressure_grad: np.ndarray      # Pressure gradient force density
```

### 3.4 Numerical Considerations

**Finite differences:** Use **4th-order central differences** for all spatial derivatives from the start. Field-line curvature requires second derivatives of B (via derivatives of b = B/|B|), which amplifies grid noise. 2nd-order differences will produce noisy garbage for curvature on realistic grids.

**Stencil:**
```
∂f/∂x ≈ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12Δx)
```

**Boundary handling:** Ghost cells or periodic boundaries. For cylindrical grids, handle the r = 0 axis singularity with L'Hôpital-type limits (κ_θ → 0 as r → 0 for axisymmetric fields).

**Degeneracy handling:** ℛ_universal = |f_inward|/|f_outward| is undefined where f_outward = 0. Implement a floor: if |f_outward| < ε·f_ref, report ℛ = ∞ (pure collapse) or flag as degenerate. Similarly, if both forces are below ε·f_ref, report ℛ = 1 (trivial equilibrium, e.g., field-free region).

---

## 4. Test Cases and Validation Plan

### 4.1 Tier 1: Analytic Verification (Must Pass)

These tests have known analytic solutions. If ℛ doesn't produce correct values here, the implementation is wrong.

#### Test 1.1: Infinite Straight Wire

**Setup:** B_θ(r) = μ₀I/(2πr), B_r = B_z = 0.

**Expected ℛ behavior:**
- Field lines are concentric circles: κ = 1/r everywhere.
- No material pressure, no axial field.
- Tension force is inward; pressure gradient is outward (∝ ∂B²/∂r < 0 since B decreases with r, so pressure gradient is inward too — both forces point inward).
- For a wire in vacuum with no plasma, there's no outward force → ℛ_universal → ∞ everywhere. This correctly indicates the field would crush any confined plasma (if present) at all radii.
- **Key check:** ℛ_κ should scale as 1/r (or a/r for chosen reference).

#### Test 1.2: Ideal Z-Pinch (Bennett Equilibrium)

**Setup:** Cylindrical plasma column with axial current I, azimuthal field B_θ(r), and material pressure p(r) satisfying Bennett relation: 2μ₀NkT = μ₀I²/(8π), where N is line density.

**Expected ℛ behavior:**
- At the equilibrium radius a: ℛ_C = 1, ℛ_universal = 1 (by construction of Bennett balance).
- For r < a: ℛ > 1 (net inward force exceeds outward; this doesn't mean infinite compression because we're looking at a static equilibrium — the plasma pressure is what creates the balance at r = a).
- For r > a: ℛ < 1 (field still present but expanding/decaying with distance).
- **Key check:** The surface ℛ = 1 should coincide with the Bennett equilibrium radius.

#### Test 1.3: θ-Pinch

**Setup:** Axial field B_z(r), no azimuthal field. Field lines are straight (or very slightly curved).

**Expected ℛ behavior:**
- κ ≈ 0 everywhere (straight field lines).
- Tension force ≈ 0. Magnetic pressure gradient provides the only magnetic force.
- ℛ_κ → 0 (no curvature). ℛ_C depends only on pressure gradient vs material pressure.
- **Key check:** Dramatically different ℛ profile from Z-pinch, consistent with θ-pinch's known superior stability. Curvature-based metrics should clearly distinguish the two configurations.

#### Test 1.4: Vacuum Plane Wave

**Setup:** Monochromatic plane wave: B = B₀ ŷ cos(kz - ωt), E = E₀ x̂ cos(kz - ωt), with E₀ = cB₀.

**Expected ℛ behavior:**
- No material pressure, no static curvature in the usual sense.
- Metric C (dynamic extension): R_dyn ∝ k̂ (Poynting vector direction).
- ℛ_universal: no inward force component → ℛ < 1 (expanding), consistent with radiation propagating outward indefinitely.
- **Key check:** |R_dyn| should be constant and aligned with propagation direction. Time-averaged ℛ should be uniform in space.

### 4.2 Tier 2: Cross-Validation Against Known Results

These tests compare ℛ against published simulation or experimental data.

#### Test 2.1: Self-Pinching Electron Beam

**Source:** Halo formation and self-pinching models (modified Bennett equilibrium for relativistic beams).

**Procedure:**
1. Reconstruct approximate B and J profiles from published analytic model parameters.
2. Compute ℛ(r) using all metrics.
3. Compare radius where ℛ = 1 to the analytically predicted final filament radius.
4. Compare to reported PIC simulation results if available.

**Success criterion:** ℛ = 1 surface within 20% of predicted filament radius.

#### Test 2.2: Screw Pinch

**Setup:** Combined axial B_z and azimuthal B_θ fields (helical field lines).

**Procedure:**
1. Compute ℛ(r) for varying pitch angles (ratio of B_z to B_θ).
2. Compare stability predictions from ℛ profile to Suydam criterion and Mercier criterion.

**Success criterion:** ℛ-based stability boundary correlates with Suydam criterion predictions.

#### Test 2.3: Optical Soliton (Stretch Goal)

**Setup:** Nonlinear Schrödinger equation soliton solution in a 1D+ model: self-focusing (effective inward pressure from nonlinear refractive index) balances diffraction (outward pressure).

**Procedure:**
1. Construct the EM field profile of a fundamental soliton in a Kerr medium.
2. Compute ℛ in the transverse plane.
3. Verify ℛ → 1 for the stable soliton profile.

**Success criterion:** Stable soliton gives ℛ ≈ 1 in transverse plane; sub-critical power gives ℛ < 1 (diffracts); super-critical power gives ℛ > 1 (self-focuses/collapses).

### 4.3 Tier 3: Exploratory (Data Generation)

These don't have predetermined correct answers — they generate data for framework evaluation.

- **Time-evolving Z-pinch:** Track ℛ(r, t) during compression. Does ℛ predict instability onset earlier than energy-based diagnostics?
- **Dipole field:** Compute ℛ for a magnetic dipole. Where are the ℛ = 1 surfaces (if any)?
- **Helmholtz coil pair:** Compute ℛ in the uniform-field region between coils vs the diverging field outside.
- **Magnetic lens (quadrupole):** Compute ℛ map and correlate with known focusing/defocusing behavior.

---

## 5. Implementation Roadmap

### Phase 1: Static Framework (Weeks 1-3)

**Goal:** Compute and visualize ℛ(x) for static field configurations.

1. Implement `fields.analytic`: wire, Z-pinch, θ-pinch, dipole.
2. Implement `fields.grid`: cylindrical and Cartesian grids with 4th-order finite differences.
3. Implement `geometry.curvature`: κ(x) from discrete derivatives of b = B/|B|.
4. Implement `stress.maxwell`: full stress tensor and divergence.
5. Implement `stress.decompose`: tension/pressure separation.
6. Implement `rotation.metrics`: all four ℛ definitions.
7. Implement `viz.rotation_map`: 2D cross-section heatmaps of ℛ(r) and κ(r).
8. Run Tier 1 tests (1.1–1.3). Validate against analytic expectations.

**Deliverable:** Jupyter notebook showing ℛ profiles for Z-pinch, θ-pinch, and wire, with all four metrics compared side-by-side.

### Phase 2: Thermodynamic State Flow (replaces "Dynamic Extension")

**Goal:** Implement entropy-parameterized state evolution and resolve the thermodynamic identity of ℛ = 1. This replaces the v1 plan for time-dependent MHD with a thermodynamically grounded approach that uses entropy production (σ) rather than elapsed time (t) as the evolution parameter.

**Phase 2A — Thermodynamic Foundation (implemented):**
1. `thermodynamics/free_energy.py` — F[B, p] functional computation
2. `thermodynamics/entropy.py` — ṡ(x) = η|J|²/T with Spitzer resistivity
3. `thermodynamics/constraints.py` — Helicity K, flux Φ, mass M conservation
4. `thermodynamics/diagnostics.py` — F(σ) tracking, monotonicity verification, Lyapunov exponents

**Phase 2B — Entropic Identity Resolution (implemented):**
1. `stability/hessian.py` — Multi-axis perturbation → Hessian eigenvalues (Test A)
2. `stability/entropy_audit.py` — Entropy production at ℛ ≈ 1 (Test B)
3. `stability/manifold.py` — Constraint boundary mapping (Test C)
4. `stability/attractors.py` — Basin-of-attraction characterization

**Phase 2C — State Flow Engine (implemented):**
1. `thermodynamics/state_flow.py` — Variational relaxation engine with steepest descent, L-BFGS, and Onsager linear response schemes
2. Constraint enforcement via Lagrange multipliers

**Phase 2D — Tokamak Applications (pending):**
1. Tokamak refinements: q₀ tuning, X-point geometry (Cerfon-Freidberg), axis regularization
2. Disruption as manifold boundary crossing
3. β limit as thermodynamic transition (F(β) sign change)

**Phase 2E — Inherited v1 Priorities (pending, independent):**
1. Suydam criterion overlay on screw-pinch sweep
2. Radiation belt boundary comparison for dipole ℛ = 1 surface
3. R★ normalization for misalignment metric
4. Dynamic EM extension — R_dyn plane wave verification

See `MAGROT_v2_Plan.md` for the full specification, test protocols, and dependency graph.

### Phase 3: Cross-Validation

**Goal:** Compare ℛ against external benchmarks.

**Status:** Partially complete (v1 validation campaign covered Bennett, tokamak, earth dipole).

Remaining:
1. Suydam criterion correlation with screw-pinch sweep
2. Optical soliton ℛ = 1 verification (stretch goal)
3. Experimental data comparison (ITER, JET, DIII-D — deferred to v3+)

### Phase 4: Package and Release

**Goal:** Clean up for open-source release.

1. API documentation (docstrings + generated docs).
2. README with motivation, installation, quickstart.
3. Tutorial notebooks (at least 3).
4. CI/CD with pytest validation suite.
5. Push to GitHub (private initially, public when ready).

---

## 6. Open Questions

### 6.1 Resolved by v1

| # | Question | Resolution |
|---|----------|------------|
| 1 | Does ℛ_universal work across all geometries without tuning? | **YES** — validated across 8 geometries, though noisiest at edges |
| 2 | Do all four metrics agree? | **PARTIALLY** — they rank consistently (ℛ_κ > ℛ_C > \|R\| > ℛ_u) but measure different physics |
| 3 | Does ℛ provide early warning of instability? | **YES** — edge ℛ rises 3× before core moves during tokamak current ramp |
| 4 | Does ℛ = 1 correspond to known stable equilibria? | **YES** — Bennett, tokamak core, vacuum wire, θ-pinch |

### 6.2 Open Questions for v2

| # | Question | Test | Phase |
|---|----------|------|-------|
| 5 | Is ℛ = 1 a local minimum (attractor) or saddle point of F? | Test A: Hessian eigenvalues | 2B |
| 6 | Is entropy being produced at ℛ = 1? (ṡ > 0 or ṡ = 0?) | Test B: Entropy audit | 2A/2B |
| 7 | What happens when constraints defining ℛ = 1 manifold are violated? | Test C: Unconstrained relaxation | 2B |
| 8 | Does entropy-parameterized evolution reproduce time-based results? | Test D: State flow convergence | 2C |
| 9 | Is |ℛ - 1| proportional to local free energy density? | Compute F(x), correlate with ℛ(x) | 2A |
| 10 | Can ℛ detect the mesa edge (constraint boundary) before disruption? | Phase 2D disruption analysis | 2D |

### 6.3 Deferred to v3+

- Full 3D toroidal mode structure (n ≠ 0 perturbations)
- Coupling to transport codes for real-time ℛ feedback control
- Application to experimental data (ITER, JET, DIII-D)
- Geometric Algebra formulation (Faraday bivector → spacetime ℛ invariant)
- Optical soliton ℛ = 1 verification (§4.2 Test 2.3)
- Virial consistency verification (volume-integrated ⟨ℛ⟩)
- Force-free limit behavior (solar corona analog)

---

## 7. Dependencies and Environment

```
# Core
numpy>=1.24
scipy>=1.10
matplotlib>=3.7

# Optional (Phase 1+)
jax>=0.4.20
jaxlib>=0.4.20
diffrax>=0.5

# Optional (3D viz)
pyvista>=0.43

# Dev
pytest>=7.0
jupyter>=1.0
```

---

## 8. Connections and Future Directions

### 8.1 Geometric Algebra Path (Phase 2+)

The Faraday bivector F = E + IcB unifies E and B as components of a single spacetime rotation generator. Expressing ℛ in GA terms would:
- Unify the static and dynamic definitions naturally.
- Make ℛ a Lorentz scalar (frame-independent).
- Connect "magnetic rotation" to the deeper structure of EM as spacetime rotation.

### 8.2 Fusion Control Application

If ℛ proves to be a reliable stability diagnostic, real-time ℛ computation could serve as a feedback signal for Z-pinch or dense plasma focus experiments: drive electrode geometry/timing to maintain ℛ ≈ 1 in the compression region.

### 8.3 Topological Extension

Magnetic winding (Prior et al., 2020) provides a nonlocal, purely geometric measure of field-line entanglement. Combining local ℛ with global winding metrics could capture both local force balance and large-scale topological constraints. This is relevant for solar physics (CME prediction) and tokamak stability.

### 8.4 Negentropy Frame

v1 asserted ℛ = 1 as "the negentropic state." v2 refines this with thermodynamic rigor:

ℛ = 1 is the **conditional entropy maximum** on the constrained MHD equilibrium manifold. Dissipative relaxation (entropy production) drives the system toward ℛ = 1 given conservation of helicity, flux, and mass. The equilibrium manifold itself is negentropic — it requires external energy throughput to maintain against unconstrained relaxation. This "valley on a mesa" model means ℛ = 1 is simultaneously the bottom of a valley (stable within constraints) sitting atop a mesa (negentropic relative to unconstrained state space).

The framework measures: (1) |ℛ - 1| = how far the system is from optimal self-organization within its accessible state space, and (2) implicitly, proximity to the constraint boundary beyond which the system undergoes irreversible reconfiguration (disruption).

*Note: This formulation is provisional pending empirical results from Tests A–D (see §6.2 and `MAGROT_v2_Plan.md` §3). The framework's utility as a diagnostic is independent of the entropic/negentropic conclusion.*

---

## References

1. Maxwell stress tensor — Wikipedia
2. The force density in electrical machines modeled as tension and pressure gradients of magnetic field lines (2023) — AIP Advances 13(2):025363
3. Theory and applications of the Maxwell stress tensor — Field Precision
4. Magnetic tension — Wikipedia
5. Z-pinch, θ-pinch, and screw pinch equilibria — SNU OCW lecture notes
6. MHD Stability of Axisymmetric Plasmas in Closed Line Magnetic Fields — EPS 2002
7. Magnetic winding: a key to unlocking topological complexity — MacTaggart et al. (2020)
8. Halo formation and self-pinching of an electron beam — Kaganovich et al.
9. Special Relativity with Geometric Algebra: Electromagnetism — geometricalgebratutorial.com
10. Prior & MacTaggart, Magnetic winding (2020) — arXiv:2009.11712
