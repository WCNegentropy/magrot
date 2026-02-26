# MagRot Forge: JAX-Accelerated Coil Geometry Generator

**Date:** 2026-02-26
**Author:** WCNEGENTROPY HOLDINGS LLC
**Status:** Plan — pending implementation
**Hardware target:** A100 80GB (Hugging Face Spaces)
**Development environment:** GitHub Codespace (CPU, pre-port + testing)

---

## 1. Executive Summary

MagRot Forge turns the MAGROT diagnostic framework into a **generative design
engine**. The core insight: the validated R=1 Lyapunov functional
(integral |R-1|^2 dV) is a natural, physics-grounded, differentiable loss
function for optimizing magnetic confinement geometries. By porting the MAGROT
physics pipeline to JAX, we gain automatic differentiation through the entire
chain: coil parameters -> Biot-Savart B-field -> R metrics -> loss. JAX's
`jax.grad` computes how moving each coil control point changes global stability,
and an optimizer drives toward R=1 everywhere — i.e., perfect force balance.

**What this produces:** Novel stellarator and tokamak coil geometries that are
optimized for MHD stability from first principles. These geometries are the
protectable IP.

**What already exists:** The MAGROT 3D physics pipeline (curl, curvature, forces,
R metrics, free energy, Lyapunov functional) is validated on grids up to 64K
points with 53 passing unit tests and 37 regression checks. The mathematical
machinery is ready — it just needs a JAX backend and a Biot-Savart front end.

---

## 2. Architecture Overview

```
                    CODESPACE (CPU)                    │     HF SPACE (A100)
                                                       │
  ┌─────────────────────────────────────┐              │
  │  magrot.jax_core                    │              │
  │  ├── numerics_jax.py    (diff_4th)  │              │
  │  ├── curvature_jax.py   (kappa)     │              │
  │  ├── maxwell_jax.py     (curl, J×B) │              │
  │  ├── metrics_jax.py     (R_univ)    │              │
  │  ├── free_energy_jax.py (F, L_R)    │              │
  │  └── biot_savart.py     (coils→B)   │  ──deploy──> │  magrot_generative.py
  │                                     │              │  ├── Loss function
  │  tests/test_jax_parity.py           │              │  ├── jax.grad(Loss)
  │  (NumPy == JAX to 1e-6)             │              │  ├── optax optimizer
  └─────────────────────────────────────┘              │  ├── jax.vmap (batch)
                                                       │  └── Checkpoint + export
```

---

## 3. Assessment of Original Plan

### Strengths

1. **Core idea is sound.** R=1 Lyapunov is already the validated loss function.
   The 3D suite proved it has the right mathematical properties: differentiable,
   zero at equilibrium, positive-definite elsewhere, physically meaningful.

2. **JAX is the right choice.** The pipeline is pure array math (no branching on
   data values in the hot path, no Python control flow that depends on array
   contents). This is ideal for XLA compilation.

3. **Biot-Savart is well-established in JAX.** The stellarator optimization
   community (SIMSOPT, DESC) has proven that differentiable coil-to-field
   pipelines work. We're adding a novel loss function (R-based), not reinventing
   the field solver.

4. **vmap parallelization** is architecturally correct for population-based
   topology search.

### Issues Identified and Resolutions

| # | Issue | Resolution |
|---|-------|------------|
| 1 | `diff_4th` uses in-place mutation (`result[sl] = ...`) | Use `jnp.zeros_like` + `jax.lax.dynamic_update_slice` or rewrite as pure functional with `jnp.where` masking. See Section 5.1. |
| 2 | `metrics.py` uses `np.errstate` context manager | Replace with `jnp.where` guards — JAX handles NaN/inf via explicit masking, not context managers. |
| 3 | `scipy.constants.mu_0` import | Fine — it's just a float constant. No scipy computation needed at runtime. |
| 4 | `decompose.py` uses `np.maximum`/`np.zeros_like` | Direct 1:1 replacement with `jnp.maximum`/`jnp.zeros_like`. |
| 5 | `CylindricalGrid` dataclass with `__post_init__` | For the optimizer, we don't need the grid object — just pre-compute `r`, `dr`, `dtheta`, `dz` as static arrays and pass them through. The grid is not a parameter being optimized. |
| 6 | R_universal has conditional branches (`is_quiet`, `active`, etc.) | Replace with nested `jnp.where` chains. All branches are element-wise, no shape changes — fully JIT-compatible. |
| 7 | `vmap` over 10,000 topologies on A100 | **Too aggressive.** Each evaluation needs ~64K-point grid × ~15 arrays × 4 bytes = ~4MB. With optimizer state and gradients, ~20MB per candidate. 10K × 20MB = 200GB — exceeds A100 VRAM. Realistic batch: **128-512 candidates**, with sequential outer loop for larger populations. See Section 7. |
| 8 | Coil parameterization not specified | Use Fourier descriptors for closed curves (standard in stellarator optimization). Each coil = N_fourier × 3 (x,y,z) cosine/sine coefficients. ~20-40 params per coil, 10-20 coils = 200-800 total parameters. Very tractable for gradient-based optimization. |
| 9 | Loss function missing engineering constraints | Add coil curvature penalty, minimum coil-coil distance, coil-plasma distance, and total coil length regularization. Without these, the optimizer will produce physically unbuildable coils. |
| 10 | No mention of magnetic surfaces / flux surfaces | For stellarator optimization, the target isn't just R=1 everywhere — it's R=1 on closed flux surfaces with specific rotational transform (iota) profile. Add iota target to loss. |

---

## 4. Refined Plan: What to Build in Codespace (CPU)

Everything below runs on CPU with `jax[cpu]`. The A100 provides speed, not
different functionality. We validate correctness here, then deploy for scale.

### Phase F1: JAX Core Module (~750 lines to port)

**Goal:** Create `magrot/jax_core/` with JAX-native versions of the 7 core
computation modules, plus parity tests proving NumPy and JAX produce identical
results.

```
magrot/jax_core/
├── __init__.py
├── numerics_jax.py      # diff_4th_jax (functional, no mutation)
├── curvature_jax.py     # compute_curvature_cylindrical_jax
├── maxwell_jax.py       # compute_forces_conservative_cyl_jax
├── decompose_jax.py     # decompose_inward_outward_jax
├── metrics_jax.py       # compute_R_universal_jax (the loss-critical metric)
├── free_energy_jax.py   # lyapunov_R_jax (the actual loss function)
└── grid_jax.py          # Static grid arrays (not a dataclass)
```

**Key porting decisions:**

#### 4.1 `diff_4th_jax` — The Critical Function

Every derivative in the pipeline flows through `diff_4th`. The NumPy version
uses 5 in-place slice assignments. JAX requires a functional approach:

```python
# Strategy: Build each region separately, then sum masked contributions
def diff_4th_jax(arr, dx, axis=0):
    N = arr.shape[axis]
    # Interior: 4th-order central (padded to full size with zeros)
    interior = jnp.roll(arr, -2, axis) - 8*jnp.roll(arr, -1, axis) \
             + 8*jnp.roll(arr, 1, axis) - jnp.roll(arr, 2, axis)
    interior = interior / (12 * dx)
    # Mask out boundary points (first 2, last 2)
    mask = _build_interior_mask(arr.shape, axis, start=2, end=2)
    result = interior * mask
    # Add boundary stencils via jnp.where or lax.dynamic_update_slice
    # ...
    return result
```

Alternative (cleaner): use `jax.lax.conv_general_dilated` with a 1D convolution
kernel `[-1, 8, 0, -8, 1] / (12*dx)` along each axis. This is a single XLA op,
naturally handles boundaries via padding mode, and is extremely fast on GPU.

**Recommendation:** Implement both approaches, benchmark on CPU, deploy the
faster one to GPU.

#### 4.2 `metrics_jax.py` — R_universal Without Branches

The current R_universal computation uses 4 boolean masks (`is_quiet`, `active`,
`inward_dom`, `outward_dom`) with conditional assignment. In JAX:

```python
# All jnp.where chains are element-wise — fully JIT-compatible
R_universal = jnp.where(is_quiet, 1.0,
              jnp.where(active, total_inward / jnp.maximum(total_outward, abs_floor),
              jnp.where(inward_dom, R_MAX,
              jnp.where(outward_dom, R_MIN, 1.0))))
R_universal = jnp.clip(R_universal, R_MIN, R_MAX)
```

#### 4.3 Parity Tests

For every JAX function, a test that:
1. Creates identical inputs (NumPy array → `jnp.array`)
2. Runs both NumPy and JAX versions
3. Asserts `jnp.allclose(result_jax, result_numpy, atol=1e-6)`

This is critical — the loss function must produce the same physics as the
validated NumPy pipeline. Any discrepancy means the optimizer is chasing
numerical artifacts, not physics.

### Phase F2: Biot-Savart Integrator

**Goal:** Differentiable coil-to-B-field computation.

```python
def biot_savart_field(coil_points, eval_points):
    """Compute B at eval_points from a set of current-carrying coils.

    Parameters
    ----------
    coil_points : list of arrays, each (N_segments, 3)
        Discretized coil centerlines in (x, y, z).
    eval_points : array (M, 3)
        Points where B is evaluated.

    Returns
    -------
    B : array (M, 3)
        Magnetic field vector at each evaluation point.
    """
```

**Coil parameterization:** Each coil is a closed 3D curve defined by Fourier
descriptors:

    x(t) = sum_n [xc_n cos(nt) + xs_n sin(nt)]
    y(t) = sum_n [yc_n cos(nt) + ys_n sin(nt)]
    z(t) = sum_n [zc_n cos(nt) + zs_n sin(nt)]

With N_fourier = 10 harmonics per coil, each coil has 60 free parameters
(10 harmonics × 3 coordinates × 2 sin/cos). For a stellarator with 5 unique
coils (replicated by symmetry), that's 300 total optimization parameters.

**Biot-Savart law (discretized):**

    B(r) = (mu_0 I / 4pi) sum_segments [ dl × r_hat / |r - r'|^2 ]

This is a double loop (over segments and eval points) that maps perfectly to
JAX's vectorized operations. The inner product is fully differentiable w.r.t.
coil control points.

**Grid mapping:** The Biot-Savart output is in Cartesian (x,y,z). For
cylindrical R metrics, we need a coordinate transform B_cart → B_cyl. This is
a rotation matrix at each grid point — fully differentiable.

**Softening:** Add a small softening length to |r - r'| to prevent
singularities when eval points are near coils: |r - r'|^2 → |r - r'|^2 + eps^2.
This also makes the gradient well-behaved near coils.

### Phase F3: Loss Function

**Goal:** Physics-grounded, differentiable loss that produces buildable coils.

```python
def forge_loss(coil_params, grid_points, grid_info, weights):
    """Total loss function for coil optimization.

    Parameters
    ----------
    coil_params : array (N_coils, N_fourier, 3, 2)
        Fourier coefficients for all coils.
    grid_points : array (M, 3)
        Evaluation grid in Cartesian coordinates.
    grid_info : dict
        Static grid metadata (r, dr, dtheta, dz, R, etc.)
    weights : dict
        Loss term weights {w_R, w_F, w_H, w_eng}.
    """
    # 1. Coil params → physical coil curves
    coils = fourier_to_coil_points(coil_params, N_segments=128)

    # 2. Biot-Savart: coils → B field on grid
    B_cart = biot_savart_field(coils, grid_points)
    B_cyl = cartesian_to_cylindrical(B_cart, grid_points)

    # 3. MagRot JAX: B field → R metrics
    R_universal = compute_R_universal_jax(B_cyl, grid_info)

    # 4. Physics loss terms
    L_R = lyapunov_R_jax(R_universal, grid_info)    # ∫|R-1|² dV
    F_total = free_energy_jax(B_cyl, grid_info)      # Total stored energy

    # 5. Stability loss (lightweight Hessian proxy)
    #    Instead of full Hessian (expensive), penalize regions where
    #    R deviates sharply — these are potential instability sites
    R_grad_mag = jnp.sum(grad_R_magnitude(R_universal, grid_info))

    # 6. Engineering constraints (critical for buildability)
    L_curvature = coil_curvature_penalty(coils)       # Min bend radius
    L_distance = coil_distance_penalty(coils)          # Min coil-coil gap
    L_length = coil_length_penalty(coils)              # Total wire length

    # 7. Weighted sum
    loss = (weights['w_R'] * L_R
          + weights['w_F'] * F_total
          + weights['w_grad'] * R_grad_mag
          + weights['w_curv'] * L_curvature
          + weights['w_dist'] * L_distance
          + weights['w_len'] * L_length)

    return loss
```

**Loss term rationale:**

| Term | Purpose | Without It |
|------|---------|-----------|
| L_R = ∫\|R-1\|² dV | Drive toward global force balance | Optimizer ignores stability |
| F_total | Prefer lower-energy equilibria | May converge to high-energy saddle |
| R_grad_mag | Smooth R field (no sharp instability sites) | May have hidden instability modes |
| L_curvature | Coils must be physically bendable | Produces infinitely sharp bends |
| L_distance | Coils can't intersect or get too close | Produces overlapping coils |
| L_length | Prefer simpler, shorter coil sets | Produces unnecessarily complex coils |

### Phase F4: Optimization Loop Scaffolding

**Goal:** Build the optimization loop structure that runs on CPU (small scale)
and deploys to GPU (full scale) without code changes.

```python
# magrot_generative.py (simplified)

import jax
import optax

loss_and_grad = jax.value_and_grad(forge_loss)

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(coil_params)

for step in range(N_steps):
    loss, grads = loss_and_grad(coil_params, grid_points, grid_info, weights)
    updates, opt_state = optimizer.update(grads, opt_state)
    coil_params = optax.apply_updates(coil_params, updates)

    if step % 100 == 0:
        checkpoint(coil_params, step, loss)
```

**On CPU (Codespace):** Run with coarse grid (20^3 = 8K points), 2-3 coils,
N_fourier=5, for 1000 steps. Goal: verify gradients flow, loss decreases, coils
move in physically reasonable directions. Not trying to find good designs — just
validating the pipeline.

**On A100 (HF Space):** Full grid (64^3+ = 262K points), 10-20 coils,
N_fourier=10-15, for 10K+ steps with learning rate scheduling.

---

## 5. What CANNOT Be Done in Codespace

| Task | Why Not | When (HF Space) |
|------|---------|------------------|
| Full-scale optimization (64^3 grid, 20 coils) | Needs GPU VRAM + FLOPS | Phase F5 |
| `jax.vmap` batch over 128+ topologies | Needs ~10GB+ VRAM | Phase F5 |
| Production runs generating patentable geometries | Needs hours of A100 time | Phase F6 |
| Performance benchmarking (JAX vs NumPy speedup) | Meaningless on CPU | Phase F5 |
| Mixed-precision (float16/bfloat16) tuning | GPU-specific optimization | Phase F5 |

---

## 6. What CAN Be Done in Codespace (Full Scope)

| Phase | Task | Deliverable | Est. Effort |
|-------|------|-------------|-------------|
| F1a | Port `numerics.py` → `numerics_jax.py` | Functional `diff_4th_jax` | 1 session |
| F1b | Port `curvature.py` + `maxwell.py` → JAX | Full 3D curl + kappa in JAX | 1 session |
| F1c | Port `decompose.py` + `metrics.py` → JAX | R_universal in JAX | 1 session |
| F1d | Port `free_energy.py` → JAX | Lyapunov loss in JAX | 0.5 session |
| F1e | Write parity tests (NumPy == JAX) | `test_jax_parity.py` | 1 session |
| F2 | Write Biot-Savart integrator | `biot_savart.py` + tests | 1-2 sessions |
| F3 | Define loss function + engineering penalties | `forge_loss()` + tests | 1 session |
| F4 | Build optimization loop scaffolding | `magrot_generative.py` | 1 session |
| F4b | Small-scale CPU validation run | Verify gradients + loss decrease | 0.5 session |

**Total Codespace work: ~7-9 sessions**

---

## 7. GPU Deployment Plan (HF Space — Post-Codespace)

### Phase F5: A100 Deployment

1. Create HF Space with `A100` hardware, JAX + CUDA runtime
2. Copy `magrot/` package + `magrot_generative.py`
3. Run parity tests on GPU (verify CPU results reproduce)
4. Benchmark: measure grid points/second, gradients/second
5. Tune batch size for `jax.vmap` (target: 128-512 candidates per batch)
6. Implement mixed-precision where safe (Biot-Savart in float32, loss in float64)

### Phase F6: Production Geometry Generation

1. **Seed generation:** Initialize 1000+ random coil topologies using
   physically motivated priors (e.g., stellarator symmetry constraints,
   reasonable aspect ratios, known good starting shapes from literature)

2. **Multi-stage optimization:**
   - Stage 1: Coarse grid (32^3), fast optimizer (Adam, high LR), 5K steps
     → rapid topology screening, discard bottom 80%
   - Stage 2: Medium grid (48^3), fine optimizer (L-BFGS), 10K steps
     → refine surviving topologies
   - Stage 3: Fine grid (64^3+), final polish, 5K steps
     → production-quality geometries

3. **Geometry export:** Save optimized coil Fourier coefficients + reconstructed
   coil curves as standard formats (JSON, VTK, STEP for CAD import)

4. **Validation:** Run each final geometry back through the NumPy MAGROT pipeline
   (the validated version) to confirm R metrics match JAX results

### Memory Budget (A100 80GB)

| Component | Per-Candidate | 128 Candidates | 512 Candidates |
|-----------|--------------|----------------|----------------|
| B-field grid (64^3 × 3 × 4B) | 3 MB | 384 MB | 1.5 GB |
| R metrics + intermediates (~15 arrays) | 15 MB | 1.9 GB | 7.7 GB |
| Coil params + gradients | 0.1 MB | 13 MB | 51 MB |
| Optimizer state (Adam: 2× params) | 0.2 MB | 26 MB | 102 MB |
| XLA compilation overhead | ~2 GB (shared) | 2 GB | 2 GB |
| **Total** | **~18 MB** | **~4.3 GB** | **~11.4 GB** |

512 candidates at 64^3 resolution fits comfortably in 80GB. At 48^3 resolution,
we could push to ~2000 candidates per batch.

---

## 8. Differentiation from Existing Tools

The stellarator optimization community has mature tools (SIMSOPT, DESC, ROSE).
MagRot Forge's differentiator:

| Feature | SIMSOPT/DESC | MagRot Forge |
|---------|-------------|--------------|
| Loss function | Quasi-symmetry + iota profile | **R=1 Lyapunov** (force balance everywhere) |
| Stability metric | External MHD code (VMEC+COBRAVMEC) | **Built-in** (R is the stability diagnostic) |
| Differentiable? | SIMSOPT: finite-diff; DESC: autodiff | **Full autodiff** (JAX end-to-end) |
| Free energy tracking | Not primary | **Core feature** (F monotonicity) |
| Entropy production | Not available | **Built-in** (identifies driven state) |
| Hessian analysis | Separate computation | **Integrated** (loss term) |

The novel contribution is using **R-metric-based physics** as the optimization
target rather than quasi-symmetry or neoclassical transport. This is a genuinely
different approach to the stellarator design problem — instead of "make the field
quasi-symmetric," the objective is "make force balance perfect everywhere." These
are related but not identical goals, and R-based optimization may find geometries
that QS-based methods miss.

---

## 9. IP Strategy

The protectable assets from MagRot Forge fall into three categories:

1. **Generated coil geometries** — The primary IP. Each optimized coil set is a
   novel physical design that can be protected as a utility patent (method of
   magnetic confinement using specific coil geometry).

2. **Optimization methodology** — The use of R-metric Lyapunov functional as a
   differentiable loss for coil design is novel. This is a process patent.

3. **Software implementation** — The JAX pipeline itself. Protected as trade
   secret (private repo) or published for academic credit, depending on strategy.

**Key point:** The generative code (`magrot_generative.py`, `jax_core/`) should
stay in the private repo. The open-core (`magrot-core`) should NOT include JAX
modules or the Biot-Savart integrator. The `build_open_core.py` manifest needs
updating to explicitly exclude `jax_core/`.

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| JAX port introduces numerical discrepancy | Medium | High (wrong physics) | Parity tests with 1e-6 tolerance before any optimization |
| Loss landscape has many local minima | High | Medium (suboptimal designs) | Population-based search via vmap + simulated annealing schedule |
| Coils converge to unbuildable shapes | High | High (useless output) | Engineering penalty terms from day 1; validate with min bend radius |
| A100 OOM on large batches | Low | Low (just reduce batch) | Memory budget computed above; 512 candidates fits |
| Biot-Savart is the bottleneck | Medium | Medium (slow) | Use segment-parallel implementation; can also precompute matrix for fixed grid |
| R_universal has flat regions (gradient vanishes) | Medium | Medium (stuck) | Use L_R + F_total + R_grad together; F_total provides gradient signal even when R=1 |

---

## 11. Execution Order

```
CODESPACE (this repo)                          HF SPACE (A100)
═══════════════════                            ═══════════════
F1a: numerics_jax.py ──┐
F1b: curvature + maxwell_jax ──┤
F1c: metrics_jax.py ──┤
F1d: free_energy_jax.py ──┤
                       ├──> F1e: parity tests
F2: biot_savart.py ────┤
F3: forge_loss() ──────┤
F4: optimization loop ─┘
F4b: CPU validation run ──────────────────────> F5: A100 deployment + benchmark
                                                F6: Production geometry generation
                                                F7: Export + patent filing
```

Phases F1-F4 are fully achievable in this Codespace. Phase F5+ requires the
A100 HF Space.
