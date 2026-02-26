"""
MagRot Forge — Generative coil geometry optimizer.

This is the main optimization loop that drives coil parameters toward
R=1 equilibrium.  Designed to run on CPU (small-scale validation) and
GPU (full-scale production) without code changes.

Usage (CPU validation)
----------------------
    python -m magrot.jax_core.magrot_generative

Usage (GPU production — in HF Space JupyterLab)
-------------------------------------------------
    Import and call run_forge() with larger grid/batch params.
"""

import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .grid_jax import make_grid_info
from .biot_savart import (
    init_circular_coils,
    grid_to_eval_points,
    fourier_to_all_coils,
)
from .forge_loss import forge_loss, forge_loss_scalar, DEFAULT_WEIGHTS


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class ForgeConfig:
    """All tunable parameters for a Forge optimization run."""
    # Grid
    Nr: int = 20
    Ntheta: int = 16
    Nz: int = 10
    r_range: tuple = (0.2, 1.5)
    z_range: tuple = (-0.5, 0.5)

    # Coils
    N_coils: int = 3
    N_fourier: int = 5
    N_segments: int = 64
    major_radius: float = 0.7
    minor_radius: float = 0.4

    # Optimizer
    learning_rate: float = 1e-3
    N_steps: int = 1000
    lr_schedule: str = 'cosine'  # 'constant' or 'cosine'

    # Loss weights
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # Logging
    log_every: int = 50
    checkpoint_every: int = 200
    output_dir: str = 'forge_output'

    # Random seed
    seed: int = 42


# ── Optimization loop ────────────────────────────────────────────────────

def run_forge(config: Optional[ForgeConfig] = None):
    """Run the Forge optimization loop.

    Parameters
    ----------
    config : ForgeConfig or None
        If None, uses defaults (small CPU validation run).

    Returns
    -------
    results : dict
        Final coil_params, loss history, and metadata.
    """
    if config is None:
        config = ForgeConfig()

    print("=" * 60)
    print("  MagRot Forge — Coil Geometry Optimizer")
    print("=" * 60)
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  Devices:    {jax.devices()}")
    print(f"  Grid:       {config.Nr} x {config.Ntheta} x {config.Nz}"
          f" = {config.Nr * config.Ntheta * config.Nz:,} points")
    print(f"  Coils:      {config.N_coils} coils, {config.N_fourier} harmonics"
          f" ({config.N_coils * config.N_fourier * 6} params)")
    print(f"  Steps:      {config.N_steps}")
    print(f"  LR:         {config.learning_rate} ({config.lr_schedule})")
    print("=" * 60)

    # 1. Build static grid
    grid_info = make_grid_info(
        Nr=config.Nr, Ntheta=config.Ntheta, Nz=config.Nz,
        r_range=config.r_range, z_range=config.z_range,
    )
    eval_cart = grid_to_eval_points(grid_info)

    # 2. Initialize coils
    key = jax.random.PRNGKey(config.seed)
    coil_params = init_circular_coils(
        N_coils=config.N_coils,
        major_radius=config.major_radius,
        minor_radius=config.minor_radius,
        N_fourier=config.N_fourier,
        key=key,
    )

    # 3. Build optimizer with LR schedule
    if config.lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.N_steps,
        )
    else:
        schedule = config.learning_rate

    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(coil_params)

    weights = config.weights

    # 4. JIT-compile the loss + grad function.
    #    Close over grid_info and eval_cart so they're captured as
    #    constants (not traced) — this makes the if Ntheta>1 checks work.
    def loss_fn(params):
        return forge_loss_scalar(params, grid_info, eval_cart, weights,
                                  config.N_segments)

    def loss_and_aux_fn(params):
        return forge_loss(params, grid_info, eval_cart, weights,
                          config.N_segments)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # 5. Training loop
    history = {
        'step': [], 'loss': [], 'L_R': [], 'F_total': [],
        'R_grad': [], 'L_curv': [], 'L_dist': [], 'L_len': [],
        'R_mean': [], 'R_std': [], 'time': [],
    }

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\n{'Step':>6} | {'Loss':>12} | {'L_R':>10} | {'R_mean':>8}"
          f" | {'R_std':>8} | {'L_curv':>8} | {'dt(s)':>6}")
    print("-" * 75)

    t_start = time.perf_counter()
    t_prev = t_start

    for step in range(config.N_steps):
        # Compute loss and gradient
        loss_val, grads = loss_and_grad(coil_params)

        # Apply gradient update
        updates, opt_state = optimizer.update(grads, opt_state, coil_params)
        coil_params = optax.apply_updates(coil_params, updates)

        # Logging
        if step % config.log_every == 0 or step == config.N_steps - 1:
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now

            # Get detailed loss components (one extra forward pass)
            _, aux = loss_and_aux_fn(coil_params)

            history['step'].append(step)
            history['loss'].append(float(aux['loss']))
            history['L_R'].append(float(aux['L_R']))
            history['F_total'].append(float(aux['F_total']))
            history['R_grad'].append(float(aux['R_grad']))
            history['L_curv'].append(float(aux['L_curv']))
            history['L_dist'].append(float(aux['L_dist']))
            history['L_len'].append(float(aux['L_len']))
            history['R_mean'].append(float(aux['R_mean']))
            history['R_std'].append(float(aux['R_std']))
            history['time'].append(t_now - t_start)

            print(f"{step:6d} | {float(aux['loss']):12.4f} | "
                  f"{float(aux['L_R']):10.4f} | {float(aux['R_mean']):8.4f}"
                  f" | {float(aux['R_std']):8.4f} | "
                  f"{float(aux['L_curv']):8.2f} | {dt:6.2f}")

        # Checkpoint
        if step > 0 and step % config.checkpoint_every == 0:
            _save_checkpoint(config.output_dir, step, coil_params, history)

    t_total = time.perf_counter() - t_start
    print("-" * 75)
    print(f"Completed {config.N_steps} steps in {t_total:.1f}s "
          f"({config.N_steps / t_total:.1f} steps/sec)")

    # Final checkpoint
    _save_checkpoint(config.output_dir, config.N_steps, coil_params, history)

    # Final diagnostics
    _, final_aux = loss_and_aux_fn(coil_params)
    print(f"\nFinal loss:   {float(final_aux['loss']):.6f}")
    print(f"Final L_R:    {float(final_aux['L_R']):.6f}")
    print(f"Final R_mean: {float(final_aux['R_mean']):.6f}")
    print(f"Final R_std:  {float(final_aux['R_std']):.6f}")

    return {
        'coil_params': coil_params,
        'history': history,
        'config': asdict(config),
        'final_aux': {k: float(v) for k, v in final_aux.items()},
    }


# ── Checkpoint I/O ───────────────────────────────────────────────────────

def _save_checkpoint(output_dir, step, coil_params, history):
    """Save coil params and history to disk."""
    np.save(os.path.join(output_dir, f'coil_params_step{step}.npy'),
            np.asarray(coil_params))
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)


def load_checkpoint(output_dir, step=None):
    """Load coil params from a checkpoint.

    Parameters
    ----------
    output_dir : str
    step : int or None
        If None, load the latest checkpoint.
    """
    if step is None:
        # Find latest
        import glob
        files = glob.glob(os.path.join(output_dir, 'coil_params_step*.npy'))
        if not files:
            raise FileNotFoundError(f"No checkpoints in {output_dir}")
        steps = [int(f.split('step')[1].split('.')[0]) for f in files]
        step = max(steps)

    path = os.path.join(output_dir, f'coil_params_step{step}.npy')
    coil_params = jnp.array(np.load(path))

    history = None
    hist_path = os.path.join(output_dir, 'history.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

    return coil_params, history, step


# ── CLI entry point ──────────────────────────────────────────────────────

def main():
    """Run a small CPU validation (Phase F4b)."""
    jax.config.update("jax_enable_x64", True)

    config = ForgeConfig(
        Nr=12, Ntheta=8, Nz=6,
        r_range=(0.3, 1.2),
        z_range=(-0.3, 0.3),
        N_coils=3,
        N_fourier=5,
        N_segments=32,
        N_steps=500,
        learning_rate=3e-3,
        lr_schedule='cosine',
        log_every=25,
        checkpoint_every=100,
        output_dir='forge_output_cpu_test',
    )

    results = run_forge(config)

    # Validate: loss should decrease
    losses = results['history']['loss']
    if len(losses) >= 2 and losses[-1] < losses[0]:
        print("\n[PASS] Loss decreased during optimization")
    else:
        print("\n[WARN] Loss did not decrease — investigate!")

    # Validate: R_mean should approach 1.0
    R_mean_final = results['history']['R_mean'][-1]
    R_mean_init = results['history']['R_mean'][0]
    if abs(R_mean_final - 1.0) < abs(R_mean_init - 1.0):
        print("[PASS] R_mean moved toward 1.0")
    else:
        print("[WARN] R_mean did not improve — investigate!")

    # Validate: gradients flowed (loss changed)
    if losses[0] != losses[-1]:
        print("[PASS] Gradients are flowing (loss changed)")
    else:
        print("[FAIL] Loss unchanged — gradients may be zero!")

    return results


if __name__ == '__main__':
    main()
