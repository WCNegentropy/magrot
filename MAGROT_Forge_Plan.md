# MagRot Forge: JAX-Accelerated Coil Geometry Generator

**Date:** 2026-02-26
**Author:** WCNEGENTROPY HOLDINGS LLC
**Status:** Plan — pending implementation
**Development environment:** GitHub Codespace (CPU, pre-port + testing)
**Production environment:** Private HuggingFace Space — A100 Large JupyterLab

---

## 0. Hardware Specification

| Resource | Codespace (Dev) | HF Space A100 Large (Prod) |
|----------|----------------|---------------------------|
| GPU | None (CPU only) | NVIDIA A100 80GB SXM |
| VRAM | — | 80 GB HBM2e (2 TB/s bandwidth) |
| System RAM | 16 GB | 142 GB |
| CPU | 4 vCPU | 12 vCPU |
| Cost | Included / $0.18/hr | $2.50/hr |
| Runtime | JupyterLab (Codespace) | JupyterLab (HF Docker template) |
| JAX backend | `jax[cpu]` | `jax[cuda12]` (CUDA 12.x, cuDNN 9) |
| Persistent storage | Git repo | HF Space persistent storage (50GB) |

**Key insight:** The HF Space uses the **JupyterLab Docker template**, not a
blind Gradio/Streamlit app. This means we have a full interactive notebook
environment with live 3D visualization, widget-based parameter steering, and
real-time monitoring — transforming the optimization from a batch job into an
**interactive command center**.

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
R metrics, free energy, Lyapunov functional) is validated on grids up to 218K
points (Nr=80, Ntheta=32, Nz=20) with 53 passing unit tests and 37 regression
checks (26 pass, 11 known findings). The mathematical machinery is ready — it
just needs a JAX backend and a Biot-Savart front end.

---

## 2. Architecture Overview

```
                    CODESPACE (CPU)                    │     HF SPACE (A100 Large)
                                                       │     JupyterLab Environment
  ┌─────────────────────────────────────┐              │
  │  magrot.jax_core                    │              │  ┌─────────────────────────────────┐
  │  ├── numerics_jax.py    (diff_4th)  │              │  │  Forge_Dashboard.ipynb           │
  │  ├── curvature_jax.py   (kappa)     │              │  │  ├── Live 3D coil viz (PyVista)  │
  │  ├── maxwell_jax.py     (curl, J×B) │              │  │  ├── R-field heatmaps (real-time)│
  │  ├── metrics_jax.py     (R_univ)    │              │  │  ├── Loss curve dashboard         │
  │  ├── free_energy_jax.py (F, L_R)    │  ──deploy──> │  │  ├── Weight sliders (live tune)  │
  │  └── biot_savart.py     (coils→B)   │              │  │  └── Tournament leaderboard      │
  │                                     │              │  └─────────────────────────────────┘
  │  magrot_generative.py               │              │
  │  ├── forge_loss()                   │              │  magrot_generative.py (GPU mode)
  │  ├── optax optimizer                │              │  ├── jax.vmap (4,096 candidates)
  │  └── CPU validation (20³, 3 coils)  │              │  ├── jax.checkpoint (rematerialization)
  │                                     │              │  ├── Async host RAM callbacks
  │  tests/test_jax_parity.py           │              │  ├── Tournament selection
  │  (NumPy == JAX to 1e-6)             │              │  └── Multi-epoch refinement
  └─────────────────────────────────────┘              │
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
| 7 | Original plan assumed 10K vmap batch on A100 | Revised. See Section 8 for full VRAM budget — with `jax.checkpoint`, 4,096 candidates fit comfortably. Without checkpointing, 2,000-3,000. |
| 8 | Coil parameterization not specified | Use Fourier descriptors for closed curves (standard in stellarator optimization). Each coil = N_fourier × 3 (x,y,z) cosine/sine coefficients. ~20-40 params per coil, 10-20 coils = 200-800 total parameters. Very tractable for gradient-based optimization. |
| 9 | Loss function missing engineering constraints | Add coil curvature penalty, minimum coil-coil distance, coil-plasma distance, and total coil length regularization. Without these, the optimizer will produce physically unbuildable coils. |
| 10 | No mention of magnetic surfaces / flux surfaces | For stellarator optimization, the target isn't just R=1 everywhere — it's R=1 on closed flux surfaces with specific rotational transform (iota) profile. Add iota target to loss. |
| 11 | **Original plan ignored JupyterLab environment** | The HF Space runs JupyterLab, not a blind script. We build an interactive dashboard with live 3D visualization, real-time loss monitoring, and dynamic weight sliders. See Section 9. |
| 12 | **Original plan wasted 142GB system RAM** | Use asynchronous host callbacks (`jax.experimental.io_callback`) to stream candidate states from GPU VRAM to host RAM continuously. The GPU never pauses for I/O — 100% utilization. See Section 8.3. |
| 13 | **No tournament / evolutionary strategy** | A single optimizer pass misses the landscape. Use multi-epoch tournament selection: massive seed → coarse filter → fine polish. See Section 10. |

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

#### 5.1 `diff_4th_jax` — The Critical Function

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

#### 5.2 `metrics_jax.py` — R_universal Without Branches

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

#### 5.3 Parity Tests

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
# magrot_generative.py (simplified — CPU validation mode)

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
N_fourier=10-15, with tournament selection and multi-epoch refinement. See
Sections 8-10.

---

## 6. Codespace Phase Summary

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

## 7. What Can vs. Cannot Be Done in Codespace

**Can do (CPU):**

| Task | Why |
|------|-----|
| All JAX porting (F1a-F1d) | Pure code translation, no GPU needed |
| Parity tests (F1e) | Small arrays, NumPy == JAX on CPU |
| Biot-Savart integrator (F2) | Math-only, no GPU dependency |
| Loss function definition (F3) | Function composition |
| Optimization loop structure (F4) | optax runs on CPU |
| Small-scale validation (F4b) | 20^3 grid, 3 coils, 1K steps |

**Cannot do (needs A100):**

| Task | Why | When |
|------|-----|------|
| Full-scale optimization (64^3, 20 coils) | Needs GPU VRAM + FLOPS | Phase F5 |
| `jax.vmap` batch over 2,000+ topologies | Needs 40GB+ VRAM | Phase F5 |
| `jax.checkpoint` memory tuning | GPU-specific rematerialization | Phase F5 |
| Live 3D dashboard (PyVista + trame) | Needs GPU rendering + JupyterLab | Phase F5b |
| Tournament selection at scale | Needs batch throughput | Phase F6 |
| Production runs generating patentable geometries | Needs hours of A100 time | Phase F6 |
| Mixed-precision (bfloat16) tuning | GPU-specific optimization | Phase F5 |
| Async host RAM callbacks | Needs GPU→host DMA path | Phase F5 |

---

## 8. VRAM Reality Check: Uncapping the Batch Size

### 8.1 Per-Candidate Memory Footprint (64^3 Grid)

| Component | Size | Notes |
|-----------|------|-------|
| B-field grid (64^3 × 3 × float32) | 3.1 MB | 3 cylindrical components |
| R metrics + 15 intermediate arrays | 15.5 MB | Forces, curvature, decomposition |
| Coil params (300 floats × float32) | 1.2 KB | Negligible |
| Optimizer state (Adam: 2× params) | 2.4 KB | Negligible |
| Gradient tape (backward pass) | ~18 MB | XLA stores activations for backprop |
| **Total without checkpointing** | **~37 MB** | Forward + backward combined |
| **Total with `jax.checkpoint`** | **~8 MB** | Recompute forward during backward |

### 8.2 Batch Size Scaling on A100 80GB

| Strategy | Per-Candidate | Batch Size | VRAM Used | VRAM Free |
|----------|--------------|------------|-----------|-----------|
| No checkpointing | 37 MB | 512 | 19 GB | 61 GB |
| No checkpointing | 37 MB | 2,000 | 74 GB | 6 GB |
| **`jax.checkpoint`** | **8 MB** | **4,096** | **33 GB** | **47 GB** |
| `jax.checkpoint` | 8 MB | 8,192 | 65 GB | 15 GB |
| XLA compilation overhead | — | — | ~2 GB | (shared) |

**Recommendation:** Use `jax.checkpoint` from the start. It wraps the loss
function with gradient rematerialization — trading ~30% more compute for ~4.5×
less memory. On the A100's 312 TFLOPS (bfloat16), the extra compute is
negligible. The memory savings unlock **4,096 parallel candidates** per vmap
call.

```python
# Wrap the loss function for gradient rematerialization
@jax.checkpoint
def forge_loss_checkpointed(coil_params, grid_points, grid_info, weights):
    return forge_loss(coil_params, grid_points, grid_info, weights)

# vmap over 4,096 candidates in a single call
batched_loss_and_grad = jax.vmap(
    jax.value_and_grad(forge_loss_checkpointed),
    in_axes=(0, None, None, None)  # Only coil_params vary per candidate
)
```

### 8.3 The 142GB System RAM Advantage

With 142 GB of host RAM, we never need to write checkpoints to disk during the
optimization loop. Instead, use **asynchronous host callbacks** to stream
candidate states from GPU VRAM → host RAM without pausing the GPU:

```python
import jax

def async_log_to_host(step, losses, best_coils):
    """Called asynchronously — GPU doesn't wait for this to complete."""
    # Store in host-side ring buffer (pre-allocated numpy array in RAM)
    history_buffer[step % BUFFER_SIZE] = {
        'losses': np.asarray(losses),          # GPU→CPU transfer
        'best_coils': np.asarray(best_coils),  # Only top-K, not all 4096
        'timestamp': time.time()
    }

# In the training loop:
jax.debug.callback(async_log_to_host, step, losses, top_k_coils)
```

**Memory budget (host RAM):**

| Component | Size | Notes |
|-----------|------|-------|
| Python + JAX runtime | ~4 GB | JupyterLab + JAX + libraries |
| History buffer (10K steps × top-50 candidates) | ~6 GB | Coil params + losses per step |
| PyVista rendering buffers | ~2 GB | 3D visualization mesh data |
| NumPy validation arrays | ~8 GB | For cross-checking against NumPy pipeline |
| OS + headroom | ~10 GB | |
| **Total used** | **~30 GB** | |
| **Available for data / working memory** | **~112 GB** | |

We can hold the **entire optimization history** (all 10K+ steps, all loss
components, top candidates at each step) in RAM without ever touching disk.
The GPU runs at 100% utilization — zero I/O stalls.

---

## 9. The JupyterLab Command Center (Phase F5b)

The HF Space JupyterLab environment transforms the optimization from a blind
batch job into an interactive, visual, human-steerable system.

### 9.1 Environment Setup

```bash
# In HF Space terminal (first boot):
pip install jax[cuda12] optax equinox
pip install pyvista[jupyter] trame trame-vuetify trame-vtk ipywidgets
pip install plotly  # For 2D dashboards
```

### 9.2 Live Dashboard: `Forge_Dashboard.ipynb`

The notebook is structured as a command center with four panels:

**Panel 1: Live 3D Coil + Field Visualization (PyVista + trame)**

```python
import pyvista as pv

plotter = pv.Plotter(notebook=True)

def update_viz(step, coil_coords, R_field):
    """Called every 50 optimization steps."""
    plotter.clear()
    # Draw coil curves
    for coil in coil_coords:
        spline = pv.Spline(coil, n_points=200)
        plotter.add_mesh(spline.tube(radius=0.02), color='copper')
    # Draw R=1 isosurface (the target equilibrium surface)
    grid = pv.ImageData(dimensions=(64, 64, 64))
    grid['R'] = R_field.flatten(order='F')
    iso = grid.contour([1.0], scalars='R')
    plotter.add_mesh(iso, opacity=0.3, color='blue')
    plotter.update()
```

**Panel 2: Real-Time Loss Dashboard (Plotly)**

Live-updating line plots for each loss component (L_R, F_total, R_grad,
L_curvature, L_distance, L_length) and the total loss. Enables immediate
diagnosis: if L_curvature is rising while L_R drops, the optimizer is trading
buildability for physics — time to adjust weights.

**Panel 3: Weight Sliders (ipywidgets) — Human-in-the-Loop Steering**

```python
import ipywidgets as widgets

w_R_slider = widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1,
                                  description='w_R (physics)')
w_curv_slider = widgets.FloatSlider(value=0.1, min=0.0, max=5.0, step=0.05,
                                     description='w_curv (bend)')
w_dist_slider = widgets.FloatSlider(value=0.5, min=0.0, max=5.0, step=0.05,
                                     description='w_dist (gap)')

# The optimizer reads current slider values at each step:
def get_live_weights():
    return {
        'w_R': w_R_slider.value,
        'w_curv': w_curv_slider.value,
        'w_dist': w_dist_slider.value,
        # ...
    }
```

This enables **dynamic weight adjustment while the optimizer runs**. If you see
coils getting too twisted in the 3D view, slide up `w_curv` and watch them
smooth out in real-time. If R isn't converging fast enough, crank up `w_R`. This
is impossible with a batch script — it's a direct consequence of the JupyterLab
environment.

**Panel 4: Tournament Leaderboard**

A live-updating table showing the top-K candidates by loss, with per-component
breakdown. Highlights candidates that are Pareto-optimal across different
loss terms (e.g., one candidate with the best L_R, another with the best
L_curvature). Enables manual selection of promising candidates for
high-resolution refinement.

### 9.3 Why This Matters for IP

The interactive dashboard isn't just convenience — it's a **competitive
advantage**. Existing stellarator optimization tools (SIMSOPT, DESC) run as
batch jobs. The researcher submits a configuration, waits hours, inspects
results, adjusts, resubmits. MagRot Forge's live dashboard enables:

- Real-time insight into loss landscape topology
- Human intuition guiding the optimizer (weight steering)
- Immediate identification of promising vs. dead-end candidates
- 10× faster iteration cycles (seconds to adjust, not hours to resubmit)

---

## 10. Tournament Selection: The Great Filter (Phase F6)

### 10.1 Strategy Overview

Rather than optimizing a single candidate to convergence, we run a **tournament**
that evolves a massive population through increasingly fine resolution. This is
how you find global optima in a landscape with many local minima.

```
Epoch 1: THE SEED (4,096 candidates)
  Grid: 32³ (32K points) — coarse but fast
  Optimizer: Adam (lr=3e-3, aggressive)
  Steps: 1,000
  Goal: Rapid topology screening
  ↓ Kill bottom 75% by total loss
  ↓

Epoch 2: THE FILTER (1,024 survivors)
  Grid: 48³ (110K points) — medium resolution
  Optimizer: Adam (lr=1e-3, moderate) → L-BFGS (last 500 steps)
  Steps: 3,000
  Goal: Refine promising topologies, identify clusters
  ↓ Kill bottom 75%, keep top 256
  ↓

Epoch 3: THE POLISH (256 elite candidates)
  Grid: 64³ (262K points) — full resolution
  Optimizer: L-BFGS (precise)
  Steps: 5,000
  Goal: Converge to true local optima
  ↓ Select top 10 by Pareto ranking
  ↓

EXPORT: Top 10 novel coil geometries
  → Fourier coefficients (JSON)
  → 3D coil curves (VTK)
  → CAD-ready surfaces (STEP)
  → Full MagRot diagnostic report (NumPy validation)
```

### 10.2 Epoch 1: The Seed

```python
# Initialize 4,096 random but physically bounded stellarator coil sets
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 4096)

def init_candidate(key):
    """Create one random coil set with physical priors."""
    coil_params = jax.random.normal(key, shape=(N_coils, N_fourier, 3, 2))
    # Scale to reasonable stellarator aspect ratios
    coil_params = coil_params * 0.1  # Small perturbations around circular
    # Add circular baseline (mode 1 = circle)
    coil_params = coil_params.at[:, 0, :, 0].add(
        stellarator_baseline_coils  # Pre-computed circular coil positions
    )
    return coil_params

all_candidates = jax.vmap(init_candidate)(keys)  # Shape: (4096, N_coils, ...)
```

At 32^3 resolution with `jax.checkpoint`, each candidate needs ~2 MB. 4,096
candidates = ~8 GB. This leaves 70+ GB free for XLA working memory.

### 10.3 Epoch 2: The Filter

After Epoch 1, sort all 4,096 by total loss. The top 1,024 are promoted.
But instead of just taking the top by scalar loss, apply **Pareto filtering**:
keep candidates that are best in *any* loss component, not just total. This
preserves diversity — a candidate with excellent L_R but mediocre L_curvature
might be one weight adjustment away from being the global best.

The resolution increases to 48^3, so per-candidate memory rises. 1,024
candidates at 48^3 with checkpointing: ~5 MB each = ~5 GB. Still comfortable.

### 10.4 Epoch 3: The Polish

The surviving 256 candidates get full 64^3 resolution. This is where L-BFGS
shines — it uses curvature information (approximate Hessian) to find the exact
bottom of each loss basin, not just the general direction.

256 candidates at 64^3 with checkpointing: ~8 MB each = ~2 GB. Trivial.

The optimizer can afford to run 5,000 steps with L-BFGS because the batch is
small and the resolution is high — each step is more expensive but more
informative.

### 10.5 Time Budget

| Epoch | Candidates | Grid | Steps | Est. Time/Step | Total Time |
|-------|-----------|------|-------|----------------|------------|
| 1 | 4,096 | 32³ | 1,000 | ~0.5s (vmap) | ~8 min |
| 2 | 1,024 | 48³ | 3,000 | ~0.8s (vmap) | ~40 min |
| 3 | 256 | 64³ | 5,000 | ~1.2s (vmap) | ~100 min |
| **Total** | | | | | **~2.5 hours** |

At $2.50/hr, a full tournament costs **~$6.25**. Running 4 tournaments with
different random seeds and weight configurations: ~$25 for a comprehensive
exploration of the design space.

---

## 11. Differentiation from Existing Tools

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
| Batch search | Sequential | **4,096 parallel** (vmap + tournament) |
| Interactive steering | No (batch jobs) | **Live dashboard** (JupyterLab + widgets) |
| Hardware | CPU clusters | **Single A100** (JAX XLA compilation) |

The novel contribution is using **R-metric-based physics** as the optimization
target rather than quasi-symmetry or neoclassical transport. This is a genuinely
different approach to the stellarator design problem — instead of "make the field
quasi-symmetric," the objective is "make force balance perfect everywhere." These
are related but not identical goals, and R-based optimization may find geometries
that QS-based methods miss.

---

## 12. IP Strategy

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

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| JAX port introduces numerical discrepancy | Medium | High (wrong physics) | Parity tests with 1e-6 tolerance before any optimization |
| Loss landscape has many local minima | High | Medium (suboptimal designs) | Tournament selection (4,096 seeds), Pareto filtering, weight steering |
| Coils converge to unbuildable shapes | High | High (useless output) | Engineering penalty terms from day 1; validate with min bend radius |
| A100 OOM on large batches | Low | Low (just reduce batch) | `jax.checkpoint` + memory budget computed above; 4,096 fits with headroom |
| Biot-Savart is the bottleneck | Medium | Medium (slow) | Segment-parallel implementation; precompute matrix for fixed grid |
| R_universal has flat regions (gradient vanishes) | Medium | Medium (stuck) | Use L_R + F_total + R_grad together; F_total provides gradient signal even when R=1 |
| HF Space disconnects mid-run | Medium | Medium (lost progress) | Auto-checkpoint every 100 steps to host RAM; periodic save to persistent storage |
| JupyterLab kernel OOM (host RAM) | Low | Low | History ring buffer capped; only store top-K candidates, not all 4,096 |
| XLA compilation time on first step | Certain | Low (one-time) | First `jit` call with 4,096 batch takes ~2-5 min to compile. Subsequent steps are instant. Budget for this. |

---

## 14. Execution Order

```
CODESPACE (this repo, CPU)                     HF SPACE (A100 Large, JupyterLab)
═══════════════════════════                    ═════════════════════════════════
F1a: numerics_jax.py ──┐
F1b: curvature + maxwell_jax ──┤
F1c: metrics_jax.py ──┤
F1d: free_energy_jax.py ──┤
                       ├──> F1e: parity tests
F2: biot_savart.py ────┤
F3: forge_loss() ──────┤
F4: optimization loop ─┘
F4b: CPU validation run
         │
         └──── deploy ──────────────────────> F5a: Environment setup (JAX CUDA)
                                              F5b: Forge_Dashboard.ipynb
                                              F5c: VRAM limit testing
                                              F5d: Parity tests on GPU
                                              F5e: Benchmark (pts/sec, grads/sec)
                                              F5f: Mixed-precision tuning
                                                │
                                              F6a: Epoch 1 — Seed 4,096 candidates
                                              F6b: Tournament → 1,024 survivors
                                              F6c: Epoch 2 — Filter at 48³
                                              F6d: Tournament → 256 elite
                                              F6e: Epoch 3 — Polish at 64³
                                              F6f: Select top 10 + export
                                                │
                                              F7: NumPy cross-validation
                                              F8: CAD export + IP filing
```

Phases F1-F4b are fully achievable in this Codespace (~7-9 sessions).
Phases F5-F8 require the A100 HF Space (~2-4 sessions at $2.50/hr).

---

## 15. Quick Reference: Key Numbers

| Metric | Value |
|--------|-------|
| JAX core lines to port | ~750 |
| Parity test tolerance | 1e-6 |
| Fourier params per coil | 60 (10 harmonics × 3 coords × 2 sin/cos) |
| Total optimization params (5 coils) | 300 |
| CPU validation grid | 20³ = 8K points |
| GPU coarse grid (Epoch 1) | 32³ = 32K points |
| GPU medium grid (Epoch 2) | 48³ = 110K points |
| GPU full grid (Epoch 3) | 64³ = 262K points |
| Max vmap batch (with checkpoint) | 4,096 candidates |
| Max vmap batch (without checkpoint) | ~2,000 candidates |
| Tournament survival rate | 25% per epoch |
| Full tournament cost | ~$6.25 (2.5 hrs × $2.50/hr) |
| XLA first-compile time | ~2-5 min (one-time) |
| Codespace sessions needed | 7-9 |
| A100 sessions needed | 2-4 |
