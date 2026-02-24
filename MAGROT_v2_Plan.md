# MAGROT v2 Development Plan

## Thermodynamic State Flow & Entropic Hypothesis Testing

**Version:** 2.0-plan-draft
**Author:** Mikeal Clark / WCNEGENTROPY HOLDINGS LLC
**Date:** February 2026
**Status:** Pre-implementation specification

---

## 0. Preamble: What Changed

MAGROT v1 (the combined v3 engineering fixes + Earth dipole + tokamak validation campaign) proved the framework works: ℛ correctly identifies equilibrium, discriminates configurations, tracks dynamics, self-diagnoses errors, and extends to toroidal fusion geometry. 33 regression checks passed across 8 test geometries.

Two foundational insights emerged during the tokamak analysis that reshape the v2 roadmap:

**Insight 1 — Time is the wrong evolution parameter.** The v1 "dynamics" test and the tokamak current ramp both parameterized state evolution by clock time *t*. But what we actually computed was a sequence of states driven by energy dissipation. The *t* parameter was just an index. Recent physics research (Wheeler-DeWitt equation, Rovelli's thermal time hypothesis, Boltzmann's entropy-arrow equivalence) converges on the same conclusion: what we call "time" may be an emergent artifact of entropy production, not a fundamental parameter. For MAGROT, this means we should parameterize evolution by entropy produced (σ), not by elapsed time — yielding exact thermodynamic state evolution rather than approximate time-dependent dynamics.

**Insight 2 — The ℛ = 1 thermodynamic identity is unproven.** We've asserted throughout v1 that ℛ = 1 is "the negentropic state: maximum magnetic self-organization." That's a hypothesis. The evidence is consistent with it, but equally consistent with ℛ = 1 being the *entropic* endpoint (maximum entropy given constraints) toward which dissipative systems relax. The distinction has major implications for the framework's interpretation, the negentropy thesis, and downstream applications. v2 must test this empirically rather than assume it.

These two insights are deeply compatible: replacing time with entropy production as the evolution parameter *directly enables* testing whether ℛ = 1 is entropic or negentropic, because the evolution itself is parameterized in thermodynamic terms.

---

## 1. Current State: What v1 Delivered

### 1.1 Validated Codebase

| Asset | Location | Status |
|-------|----------|--------|
| Framework specification | `MAGROT_Framework_Spec.md` | v0.1.0-draft, needs v2 update |
| v3 simulation (6 tests) | `magrot_v3__3_.py` | 22/22 regression checks |
| Earth dipole extension | `earth_dipole_v2__3_.py` | Novel ℛ(λ) universality finding |
| Tokamak simulation (6 tests) | `magrot_tokamak__1_.py` | 8/11 regression checks (3 threshold issues) |
| Package scaffold | `magrot/` | Stubs only — `dynamics/evolve.py` placeholder |
| v3 validation report | `MagRot_Validation_Report_v3.docx` | Complete |
| Tokamak validation report | `MagRot_Tokamak_Validation_Report_v1.docx` | Complete |

### 1.2 Proven Physics

1. **Equilibrium identification** — ℛ = 1 at Bennett boundary (0.16% error), vacuum wire (exact), θ-pinch (uniform), tokamak core (1.6%)
2. **Configuration discrimination** — Z-pinch vs θ-pinch separated by 11 OOM in curvature
3. **Dynamic tracking** — ℛ oscillates in phase with compression/expansion, converges under damping
4. **Self-diagnosis** — Bennett pressure error reported as ℛ = 2.0; corrected → ℛ = 1.0
5. **Toroidal geometry** — Inboard/outboard asymmetry (ΔR = 0.67) encodes ballooning physics
6. **Metric ranking** — ℛ_κ (precision, 3.7% std) > ℛ_C (force balance) > |R| (aligned-field) > ℛ_universal (sensitive but noisy)

### 1.3 Open Questions Inherited from v1

From the v3 validation report §6.2:
- Suydam criterion correlation with screw-pinch sweep (untested)
- R★ normalization for misalignment metric
- ℛ_C boundary spike interpretation
- Dynamic EM extension (R_dyn formalism exists, untested)

From the tokamak report:
- q₀ too high (3.5–4.0) — rational surface effects untested
- β sweep interpretation inverted from hypothesis
- Axis regularization needed (ρ < 0.15 spike)
- X-point/divertor geometry absent

---

## 2. Paradigm Shift: Thermodynamic State Flow

### 2.1 The Problem with Time

The v1 dynamics module (`dynamics/evolve.py`, currently a placeholder) was designed around time-stepping: ∂B/∂t, dt, CFL conditions, temporal integration. This approach has three fundamental problems:

**Problem 1 — Time may not be fundamental.** The Wheeler-DeWitt equation (canonical quantum gravity) contains no time variable. Boltzmann showed that the arrow of time IS entropy increase — they're identical, not correlated. If time is emergent from thermodynamics rather than fundamental, then parameterizing physics by *t* introduces an unnecessary and potentially misleading abstraction layer.

**Problem 2 — Time-stepping approximates.** Any finite-difference time integrator (FDTD, Runge-Kutta, symplectic) introduces truncation error proportional to Δt. The CFL condition couples spatial resolution to temporal resolution, creating computational constraints that have nothing to do with the physics. Step-size selection is an engineering choice, not a physical one.

**Problem 3 — Time doesn't tell you where you are thermodynamically.** Knowing that "t = 5 ms" tells you nothing about the system's thermodynamic state. Knowing that "σ = 0.3 J/K of entropy has been produced" tells you exactly how far the system has moved toward equilibrium. The evolution parameter should carry physical meaning.

### 2.2 The Solution: Entropy as Evolution Parameter

Replace *t* with σ (total irreversible entropy produced) as the evolution parameter:

```
σ = ∫∫ ṡ(x) dV dτ
```

where ṡ(x) is the local entropy production rate density. For resistive MHD:

```
ṡ_resistive = η|J|² / T
```

For general dissipative MHD:

```
ṡ_total = η|J|²/T + (viscous terms)/T + (thermal conduction terms)/T
```

**State space:** {B(x), p(x)} at every spatial point — the full magnetic and thermodynamic configuration.

**Free energy functional:**

```
F[B, p] = ∫ (B²/2μ₀ + p/(γ-1)) dV - F_eq
```

where F_eq is the minimum-energy state satisfying the constraints (flux conservation, helicity conservation, mass conservation). F ≥ 0 always. F = 0 at equilibrium.

**ℛ as the local gradient of F:** Where ℛ ≠ 1, there exists a local force imbalance. Force imbalance = free energy available for conversion to heat (entropy). The spatial map of |ℛ - 1| IS the map of where entropy production will (and must) occur.

**Evolution principle:**
- δF/δσ ≤ 0 (second law — guaranteed monotonic)
- Near equilibrium: Onsager's minimum entropy production → smooth relaxation toward ℛ = 1
- Far from equilibrium: maximum entropy production → rapid reconfiguration (disruption analog)

### 2.3 What This Buys Us

| Property | Time-based (v1) | Entropy-based (v2) |
|----------|----------------|-------------------|
| Evolution parameter | t (emergent, possibly non-fundamental) | σ (fundamental, second-law guaranteed) |
| Convergence guarantee | Depends on scheme stability | Guaranteed (F monotonically decreases) |
| Step size selection | CFL condition (numerical artifact) | dσ increment (physical meaning) |
| Relative evolution rates | Require explicit time scales | Self-consistent (more free energy → more entropy production → faster evolution) |
| Endpoint | "After enough time passes" | Minimum of free energy functional (variational) |
| Connection to ℛ | ℛ tracks state at each t | ℛ = ∂F/∂(local state) — ℛ IS the gradient |
| Coordinate dependence | Frame-dependent | Frame-independent |

### 2.4 Connection to ℛ

The critical insight: **ℛ - 1 is proportional to the local free energy density gradient.** This means:

- |ℛ - 1| maps where the system has available free energy
- The sign of (ℛ - 1) indicates the direction of relaxation (over-confined → expand, under-confined → contract)
- ∫|ℛ - 1|² dV is a global Lyapunov functional that must decrease along the entropy flow
- ℛ = 1 everywhere is the endpoint where F = F_eq and no more entropy can be produced (within constraints)

This transforms ℛ from a diagnostic (what it was in v1) into the **driving function** of the evolution itself.

---

## 3. Entropic Hypothesis Testing

### 3.1 The Two Questions

**Q1: Is ℛ = 1 negentropic or entropic?**

| If negentropic... | If entropic... |
|---|---|
| ℛ = 1 is a low-entropy, high-organization state | ℛ = 1 is the maximum-entropy state given constraints |
| Requires continuous energy input to maintain | System relaxes toward it by producing entropy |
| Disruption = entropy wins, ℛ = 1 collapses | Disruption = constraint violation, system leaves manifold |
| Analogous to: living organism, hurricane | Analogous to: Taylor state, thermal equilibrium |

**Q2: Is ℛ = 1 a stable attractor or an unstable extremum?**

| If stable attractor... | If unstable extremum... |
|---|---|
| Perturbations away from ℛ = 1 create restoring forces | Perturbations accelerate departure from ℛ = 1 |
| F[ℛ] has a local minimum at ℛ = 1 | F[ℛ] has a local maximum or saddle at ℛ = 1 |
| The Hessian ∂²F/∂ℛ² > 0 | The Hessian ∂²F/∂ℛ² ≤ 0 in some directions |

### 3.2 Preliminary Evidence from v1

**Evidence suggesting ℛ = 1 is an attractor (strong):**
- Damped Z-pinch converges TO ℛ = 1 under resistive dissipation (v3 Test 4)
- β sweep: perturbing pressure away from equilibrium in either direction increases |ℛ - 1| (tokamak Test 4)
- Self-consistent equilibrium rebuilds ℛ ≈ 1 across 5× current ramp at core (tokamak Test 5)

**Evidence suggesting it's entropic (the constrained-maximum interpretation):**
- Resistive damping IS entropy production, and it drives toward ℛ = 1
- Taylor relaxation theory predicts constrained entropy maximization → force-free equilibrium
- At ℛ = 1 in a tokamak, J ≠ 0 and η > 0, so ṡ = η|J|²/T > 0 — entropy is STILL being produced even at force balance. This means ℛ = 1 is NOT global thermodynamic equilibrium.

**The likely resolution (hypothesis to test):**

> ℛ = 1 is a **conditional entropy maximum** on the constrained MHD equilibrium manifold — the state toward which dissipative relaxation drives the system given conservation of magnetic helicity and total flux. The equilibrium manifold itself represents a negentropic configuration requiring external energy throughput to maintain. ℛ = 1 is simultaneously the bottom of a valley (stable within constraints) sitting on top of a mesa (negentropic relative to unconstrained state space).

### 3.3 Formal Test Protocol

#### Test A — Multi-Axis Perturbation Response (Hessian Map)

**Goal:** Determine if ℛ = 1 is a local minimum, maximum, or saddle point of the free energy functional.

**Method:**
1. Start from self-consistent tokamak equilibrium (ℛ_core ≈ 1.016)
2. Define perturbation axes:
   - Axis 1: Pressure scaling (already done in v1, extend range)
   - Axis 2: Current density profile perturbation (peaked → flat)
   - Axis 3: Boundary shape perturbation (elongation κ sweep)
   - Axis 4: Field geometry perturbation (toroidal field strength)
   - Axis 5: Non-axisymmetric perturbation (n=1 mode structure, if feasible in 2D via effective averaging)
3. For each axis, sweep perturbation amplitude ε from -δ to +δ
4. Compute F(ε) = ∫|ℛ - 1|² dV at each perturbation level
5. Fit parabola: F(ε) ≈ F₀ + ½ H ε²
6. Extract Hessian eigenvalue H for each axis

**Success criteria:**
- H > 0 for all axes → ℛ = 1 is a true minimum (stable attractor in all directions)
- H > 0 for some, H < 0 for others → saddle point (stable in some directions, unstable in others — the unstable directions are the MHD instability modes)
- H < 0 for all → maximum (unstable — would contradict v1 evidence)

**Deliverable:** Hessian eigenvalue spectrum. If saddle, identify which perturbation directions are unstable and compare to known MHD instability classification.

#### Test B — Entropy Production Accounting at Equilibrium

**Goal:** Determine if ℛ = 1 is a global thermodynamic equilibrium (ṡ = 0) or a driven steady state (ṡ > 0).

**Method:**
1. Compute ṡ(x) = η|J(x)|²/T(x) across the ℛ ≈ 1 tokamak equilibrium
2. Use Spitzer resistivity η(T) = η₀(T/T₀)^(-3/2) for realistic temperature dependence
3. Integrate total entropy production rate: Ṡ_total = ∫ ṡ dV
4. Convert to power: P_dissipated = T_avg × Ṡ_total
5. Compare with known tokamak power balance (ITER: ~50 MW auxiliary heating)

**Expected result:** Ṡ_total > 0 everywhere inside plasma (J ≠ 0, η > 0). This would definitively establish that ℛ = 1 is NOT a global thermodynamic equilibrium — it's a force-balance state within a driven, dissipating system.

**Deliverable:** Spatial map of ṡ(x) overlaid on ℛ(x). Quantitative comparison of integrated dissipation power with ITER operating parameters.

#### Test C — Unconstrained Relaxation (Constraint Removal)

**Goal:** Determine what happens when the constraints that define the ℛ = 1 manifold are violated.

**Method:**
1. Start from self-consistent tokamak equilibrium
2. Simulate constraint removal scenarios:
   - Remove external toroidal field (set B_tor decay with characteristic time τ_R)
   - Allow magnetic flux to diffuse through boundary (resistive wall)
   - Remove pressure source (radiative cooling without heating)
3. Track ℛ(x, σ) as the system evolves through entropy production
4. Record whether ℛ departs smoothly or catastrophically
5. Map the "mesa edge" — the constraint boundary beyond which ℛ = 1 ceases to exist

**Success criteria:**
- Smooth departure + gradual ℛ drift → ℛ = 1 manifold has soft boundaries
- Catastrophic departure + rapid ℛ explosion → ℛ = 1 manifold has sharp boundaries (disruption-like)
- Different scenarios hit different boundaries → the constraint landscape has structure that ℛ can map

**Deliverable:** Phase portrait showing system trajectory in (ℛ_core, ℛ_edge, F) space as constraints are relaxed. Identification of the "mesa edge" for each constraint type.

#### Test D — Thermodynamic State Flow Convergence

**Goal:** Validate that entropy-parameterized evolution produces the same endpoint as time-based evolution (v1 dynamics test), but with guaranteed convergence and physical step-size interpretation.

**Method:**
1. Reproduce the v1 damped Z-pinch scenario (a₀ = 0.5a, compressed start)
2. Implement the entropy-based evolution:
   - State: {a(σ), v(σ)} (shell radius and velocity as functions of entropy produced)
   - Entropy production: dσ = η|J|²/T × dV × dτ (computed from current state)
   - Evolution: da/dσ = v × (dτ/dσ), dv/dσ = F_net/m × (dτ/dσ)
   - Where dτ/dσ = 1/Ṡ is the inverse of instantaneous entropy production rate
3. Track both ℛ(σ) and F(σ)
4. Verify: F(σ) is monotonically decreasing
5. Verify: endpoint ℛ → 1.0 is identical to v1 time-based result
6. Compare: number of "steps" required, stability of evolution

**Success criteria:**
- F(σ) monotonically decreasing (second law) ✓/✗
- ℛ → 1.0 at endpoint (matches v1) ✓/✗
- Convergence independent of step size selection ✓/✗
- Physical interpretation of σ_total at convergence (total entropy produced to reach equilibrium)

**Deliverable:** Side-by-side comparison of ℛ(t) vs ℛ(σ) for the damped Z-pinch. Proof that F(σ) ≤ 0 for all σ.

---

## 4. Revised Framework Specification Updates

### 4.1 New §2.6: Thermodynamic Identity of ℛ

Replace the v1 assertion in the framework spec and v3 report (§6.3):

**Old (v1):**
> "ℛ = 1 represents the negentropic state: maximum organization where inward and outward forces achieve local balance."

**New (v2, pending Test A–D results):**
> ℛ = 1 is the **conditional entropy maximum** on the constrained MHD equilibrium manifold — the state toward which dissipative relaxation drives the system given conservation of magnetic helicity and total flux. The equilibrium manifold itself represents a negentropic configuration requiring external energy throughput to maintain. The framework therefore measures two distinct things: (1) how far the system is from optimal self-organization within its accessible state space (|ℛ - 1|), and (2) implicitly, the proximity to the boundary of the accessible state space beyond which the system undergoes irreversible reconfiguration (disruption).

*Note: This formulation is provisional and will be finalized based on empirical results from Tests A–D. If Test A shows all Hessian eigenvalues positive, the "conditional minimum" interpretation is confirmed. If Test B shows ṡ > 0 at ℛ = 1, the "driven steady state" interpretation is confirmed. The two together establish the "valley on a mesa" model.*

### 4.2 New §2.7: Free Energy Functional

Add to the mathematical framework:

```
F[B, p] = ∫_V [ B²/(2μ₀) + p/(γ-1) ] dV - F_eq[K, Φ, M]
```

where the equilibrium free energy F_eq is the minimum of F subject to constraints:
- K = ∫ A·B dV (magnetic helicity, conserved in ideal MHD)
- Φ = ∫ B·dA (magnetic flux through boundary, conserved for ideal conductor)
- M = ∫ ρ dV (total mass, conserved)

**ℛ-F connection:**

```
|ℛ(x) - 1| ∝ |δF/δB(x)| / (B²/2μ₀)
```

This makes ℛ the *dimensionless local gradient of the free energy functional*, measured in units of the local magnetic energy density.

### 4.3 Revised §5: Implementation Roadmap (Phase 2)

Replace the v1 Phase 2 ("Dynamic Extension: add time dependence") with:

> **Phase 2: Thermodynamic State Flow (replaces "Dynamic Extension")**
>
> **Goal:** Implement entropy-parameterized state evolution and resolve the thermodynamic identity of ℛ = 1.
>
> This phase replaces the v1 plan for time-dependent MHD with a thermodynamically grounded approach that uses entropy production (σ) rather than elapsed time (t) as the evolution parameter. The result is exact state flow along the free energy gradient rather than approximate time integration.

### 4.4 Revised dynamics/ Module

Replace the v1 placeholder structure:

```
magrot/
├── dynamics/          # v1: time-based (DEPRECATED)
│   ├── mhd_1d.py
│   ├── em_wave.py
│   └── evolve.py
```

With:

```
magrot/
├── thermodynamics/        # v2: entropy-based state flow
│   ├── __init__.py
│   ├── free_energy.py     # F[B, p] functional + constraints
│   ├── entropy.py         # ṡ(x) production rate density
│   ├── state_flow.py      # Variational relaxation along ∇F
│   ├── constraints.py     # Helicity, flux, mass conservation
│   └── diagnostics.py     # F(σ), Ṡ(σ), Lyapunov verification
├── stability/             # v2: entropic hypothesis tests
│   ├── __init__.py
│   ├── hessian.py         # Multi-axis perturbation → Hessian eigenvalues
│   ├── entropy_audit.py   # ṡ(x) at equilibrium, power balance
│   ├── manifold.py        # Constraint boundary mapping
│   └── attractors.py      # Basin of attraction characterization
├── dynamics/              # v1 legacy (kept for comparison)
│   ├── mhd_1d.py
│   ├── em_wave.py
│   └── evolve.py
```

---

## 5. Implementation Phases

### Phase 2A: Thermodynamic Foundation (Priority 1)

**Goal:** Implement the free energy functional and entropy production computation. No evolution yet — just the thermodynamic quantities that will drive everything else.

**Tasks:**
1. `free_energy.py` — Compute F[B, p] = ∫(B²/2μ₀ + p/(γ-1))dV for any field configuration
2. `entropy.py` — Compute ṡ(x) = η|J|²/T with Spitzer resistivity model
3. `constraints.py` — Compute magnetic helicity K = ∫A·B dV, total flux Φ, total mass M
4. `diagnostics.py` — Track F(σ), verify monotonic decrease, compute Lyapunov exponents

**Test on:** Existing v1 configurations (Z-pinch, tokamak equilibrium). No new physics needed — just compute F and ṡ for states we already have.

**Deliverable:** F and ṡ maps for Z-pinch Bennett equilibrium and ITER-scale tokamak. Quantitative entropy production rate at ℛ ≈ 1 (Test B deliverable).

**Regression:** F_zpinch_eq should be at or near minimum. ṡ should be spatially correlated with |ℛ - 1|.

### Phase 2B: Entropic Identity Resolution (Priority 1, parallel with 2A)

**Goal:** Execute Tests A–C to determine the thermodynamic character of ℛ = 1.

**Tasks:**
1. `hessian.py` — Multi-axis perturbation sweep on tokamak equilibrium (Test A)
2. `entropy_audit.py` — Spitzer entropy production at ℛ ≈ 1 (Test B — largely done in 2A)
3. `manifold.py` — Constraint relaxation simulations (Test C)
4. `attractors.py` — Basin-of-attraction mapping: how far can you perturb before the system doesn't return?

**Test on:** ITER-scale tokamak (primary), Z-pinch (simpler, faster iteration).

**Deliverable:** Hessian eigenvalue spectrum. Entropy production maps. Constraint boundary characterization. **Formal determination: is ℛ = 1 a conditional entropy maximum (valley-on-mesa) or something else?**

**This is the most scientifically important deliverable of v2.** It either confirms or revises the negentropy thesis as applied to MAGROT.

### Phase 2C: State Flow Engine (Priority 2, after 2A proves concept)

**Goal:** Implement entropy-parameterized evolution and validate against v1 dynamics.

**Tasks:**
1. `state_flow.py` — Variational relaxation engine:
   - Input: Initial state {B(x), p(x)} with ℛ ≠ 1
   - Evolution: Adjust B, p to reduce F by dF, producing entropy dσ
   - Output: Trajectory {B(x; σ), p(x; σ)} parameterized by σ
2. Relaxation scheme options:
   - Steepest descent: δB ∝ -δF/δB (simple, guaranteed convergent)
   - L-BFGS on F[B, p] (faster convergence for smooth landscapes)
   - Onsager linear response near equilibrium (physically motivated)
3. Constraint enforcement via Lagrange multipliers (helicity, flux, mass)

**Test on:** Damped Z-pinch (Test D — direct comparison with v1 time-based result).

**Deliverable:** ℛ(σ) evolution curves. Proof that F(σ) is monotonically decreasing. Side-by-side with v1 ℛ(t).

### Phase 2D: Tokamak Applications (Priority 3, after 2C works)

**Goal:** Apply the thermodynamic framework to tokamak-specific problems.

**Tasks:**
1. **Tokamak refinements from v1:**
   - Tune q₀ ≈ 1.0 for rational surface testing
   - Reframe β limit as outboard ℛ < 1 crossover
   - Add X-point geometry (Cerfon-Freidberg)
   - Axis regularization (L'Hôpital limits)
2. **Disruption as manifold boundary crossing:**
   - Use Phase 2B manifold mapping to identify which constraint violations produce disruption-like behavior
   - Compute ℛ signatures at the mesa edge
   - Proposal: ℛ disruption precursor = expanding region where |ℛ - 1| > threshold AND ṡ > ṡ_critical
3. **β limit as thermodynamic transition:**
   - Track where outboard ℛ crosses below 1.0 as β increases
   - At that point, thermal free energy exceeds magnetic free energy locally → confinement loss
   - Compute F(β) and identify the β_critical where ∂²F/∂β² changes sign

### Phase 2E: Inherited v1 Priorities (Priority 3, independent)

These items from the v3 report and tokamak v1 roadmap remain valid and can proceed in parallel:

1. **Suydam criterion overlay** on screw-pinch sweep (v3 report §6.2)
2. **Radiation belt boundary comparison** for dipole ℛ = 1 surface (v3 report §7)
3. **R★ normalization** for misalignment metric (v3 report §6.2)
4. **Dynamic EM extension** — R_dyn plane wave verification (framework spec §2.4)

These are important but don't depend on the thermodynamic paradigm shift and can be executed whenever bandwidth allows.

---

## 6. Updated Framework Spec: Open Questions

### 6.1 Resolved by v1

| # | Question (from spec §6) | Status |
|---|---|---|
| 1 | Does ℛ_universal work across all geometries without tuning? | **YES** — validated across 8 geometries, though noisiest at edges |
| 2 | Do all four metrics agree? | **PARTIALLY** — they rank consistently (ℛ_κ > ℛ_C > \|R\| > ℛ_u) but measure different physics |
| 3 | Does ℛ provide early warning of instability? | **YES** — edge ℛ rises 3× before core moves during current ramp |
| 4 | Does ℛ = 1 correspond to known stable equilibria? | **YES** — Bennett, tokamak core, vacuum wire, θ-pinch |

### 6.2 New Questions for v2

| # | Question | Test | Phase |
|---|---|---|---|
| 5 | Is ℛ = 1 a local minimum (attractor) or saddle point of F? | Test A: Hessian eigenvalues | 2B |
| 6 | Is entropy being produced at ℛ = 1? (ṡ > 0 or ṡ = 0?) | Test B: Entropy audit | 2A/2B |
| 7 | What happens when constraints defining ℛ = 1 manifold are violated? | Test C: Unconstrained relaxation | 2B |
| 8 | Does entropy-parameterized evolution reproduce time-based results? | Test D: State flow convergence | 2C |
| 9 | Is |ℛ - 1| proportional to local free energy density? | Compute F(x) and correlate with ℛ(x) | 2A |
| 10 | Can ℛ detect the mesa edge (constraint boundary) before disruption? | Phase 2D disruption analysis | 2D |

### 6.3 Questions Deferred to v3+

- Full 3D toroidal mode structure (n ≠ 0 perturbations)
- Coupling to transport codes for real-time ℛ feedback control
- Application to experimental data (ITER, JET, DIII-D)
- Geometric Algebra formulation (Faraday bivector → spacetime ℛ invariant)
- Optical soliton ℛ = 1 verification (framework spec §4.2 Test 2.3)

---

## 7. Risk Assessment

### 7.1 What Could Go Wrong

**Risk 1: Hessian shows ℛ = 1 is a saddle point, not a minimum.**
Impact: High — would mean ℛ = 1 is unstable in certain perturbation directions.
Mitigation: This is actually *physically expected* for tokamaks (they ARE unstable to certain modes). The key question becomes: do the unstable Hessian directions correspond to known MHD instabilities? If yes, the framework gains predictive power. If no, the formulation needs revision.
Likelihood: Medium. The β sweep already hints at this — ℛ responds asymmetrically to over-pressure vs under-pressure.

**Risk 2: Entropy production at ℛ = 1 is negligible (ṡ ≈ 0).**
Impact: Would challenge the "driven steady state" interpretation and support the "true equilibrium" interpretation.
Mitigation: This is unlikely for finite-resistivity plasma (J ≠ 0 → ṡ > 0 always), but may be true in the ideal MHD limit. We should compute both Spitzer (realistic) and ideal (η → 0) cases.
Likelihood: Low for realistic plasma, moderate for ideal MHD.

**Risk 3: Entropy-parameterized evolution fails to converge.**
Impact: Would undermine the thermodynamic state flow approach.
Mitigation: The variational formulation with constraint enforcement should guarantee convergence by construction (F bounded below, σ monotonic). If the relaxation scheme oscillates, switch from steepest descent to damped Newton or L-BFGS.
Likelihood: Low.

**Risk 4: The "valley on mesa" model is wrong — it's valleys all the way down.**
Impact: If ℛ = 1 turns out to be the global entropy maximum (not just conditional), the negentropy thesis doesn't apply in the magnetic domain. MAGROT still works as a diagnostic, but the philosophical framing changes significantly.
Mitigation: Test C (unconstrained relaxation) directly probes this. If removing constraints causes ℛ to depart from 1.0 toward a different attractor, the mesa model holds. If ℛ stays at 1.0 regardless, it's valleys all the way down.
Likelihood: Low (tokamaks clearly require energy input to maintain ℛ ≈ 1).

### 7.2 What Changes Based on Outcomes

| Outcome | Implication for MAGROT | Implication for Negentropy Thesis |
|---------|----------------------|----------------------------------|
| Valley-on-mesa confirmed | ℛ = 1 is conditional attractor + negentropic manifold | Strongest form: ℛ = 1 requires external order maintenance |
| Pure attractor (entropic) | ℛ = 1 is simply the relaxation endpoint | Weaker: self-organization is entropy-maximizing, not entropy-reversing |
| Saddle point | ℛ = 1 is unstable in specific directions = instability modes | Nuanced: negentropy only in certain directions of state space |
| Mesa, no valley | ℛ = 1 is maintained only by constraints, not preferred even within manifold | Reframe: the constraints themselves are negentropic, not the equilibrium |

**Crucially:** In ALL four outcomes, MAGROT remains valid as a diagnostic framework. ℛ still measures force balance, still discriminates configurations, still tracks dynamics. What changes is the *interpretation* — specifically, whether "ℛ = 1 is negentropic" is the right framing, or whether a more nuanced thermodynamic statement is needed. The framework's utility is independent of the philosophical conclusion.

---

## 8. Dependency Graph

```
Phase 2A: Thermodynamic Foundation
  ├── free_energy.py (F functional)
  ├── entropy.py (ṡ computation)
  ├── constraints.py (K, Φ, M)
  └── diagnostics.py (F tracking)
       │
       ├──→ Phase 2B: Entropic Identity (Tests A, B, C)
       │     ├── hessian.py (Test A — needs F from 2A)
       │     ├── entropy_audit.py (Test B — needs ṡ from 2A)
       │     ├── manifold.py (Test C — needs constraints from 2A)
       │     └── attractors.py (synthesis of A+B+C)
       │          │
       │          └──→ Framework Spec §2.6 update (entropic identity resolved)
       │
       └──→ Phase 2C: State Flow Engine (Test D)
             ├── state_flow.py (needs F, ṡ, constraints from 2A)
             └── v1 dynamics comparison
                  │
                  └──→ Phase 2D: Tokamak Applications
                        ├── Disruption = manifold boundary (needs 2B + 2C)
                        ├── β limit as thermodynamic transition (needs 2A + 2C)
                        └── Tokamak refinements (q₀ tuning, X-point — independent)

Phase 2E: v1 Inherited (independent, parallel)
  ├── Suydam criterion overlay
  ├── Radiation belt comparison
  ├── R★ normalization
  └── Dynamic EM extension
```

---

## 9. Success Criteria for v2

### Minimum Viable v2
- [ ] F[B, p] computed for Z-pinch and tokamak configurations
- [ ] ṡ(x) maps generated for equilibrium states (Test B complete)
- [ ] Hessian eigenvalue spectrum for at least 3 perturbation axes (Test A partial)
- [ ] Formal statement on thermodynamic identity of ℛ = 1 (entropic, negentropic, or conditional)
- [ ] Framework spec §2.6 updated with empirical basis

### Full v2
- [ ] All minimum viable criteria
- [ ] All 5 perturbation axes tested (Test A complete)
- [ ] Constraint relaxation simulations (Test C complete)
- [ ] Entropy-parameterized state flow engine working (Test D complete)
- [ ] ℛ(σ) reproduces v1 ℛ(t) for damped Z-pinch
- [ ] F(σ) proven monotonically decreasing
- [ ] Tokamak refinements: q₀ ≈ 1, X-point geometry, β crossover reframed
- [ ] v2 validation report documenting all findings
- [ ] Framework spec updated to v0.2.0

### Stretch Goals
- [ ] Disruption precursor identification via ℛ + ṡ combined diagnostic
- [ ] Suydam criterion correlation confirmed
- [ ] 3D mode structure (n ≠ 0) feasibility study
- [ ] Publication-ready two-figure paper (v3 + tokamak + thermodynamic identity)

---

## 10. Publication Strategy

The v2 results naturally organize into a compelling paper structure:

**Title concept:** "A Thermodynamic Geometric Diagnostic for Magnetic Confinement: The Rotational Parameter ℛ"

**Core narrative:**
1. ℛ measures force balance from geometry alone (v1 result)
2. ℛ = 1 is the conditional entropy maximum on constrained MHD manifolds (v2 result)
3. |ℛ - 1| maps free energy available for entropy production (v2 connection)
4. Entropy-parameterized evolution along ℛ gradient reproduces known dynamics without time integration (v2 result)
5. Application to ITER-scale tokamak: ballooning physics, β limits, disruption precursors (v1 + v2 combined)

**Key figures:**
1. Z-pinch/θ-pinch discrimination (v1 — establishes the metric)
2. Tokamak ℛ map with inboard/outboard asymmetry (v1 — establishes geometry)
3. Hessian eigenvalue spectrum (v2 — establishes thermodynamic character)
4. ℛ(σ) vs ℛ(t) comparison (v2 — establishes the paradigm shift)
5. ṡ(x) overlaid on ℛ(x) at tokamak equilibrium (v2 — the punchline)

---

*End of Plan*
