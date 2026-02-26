"""3D field visualization using matplotlib (no PyVista dependency).

Provides cross-section slicing, polar heatmaps, 3D quiver plots,
and 1D-vs-3D radial profile comparison for cylindrical and Cartesian grids.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_polar_slice(scalar_field, grid, z_index=None, title='', filename=None,
                     cmap='RdBu_r', vmin=None, vmax=None):
    """Plot an r-theta polar heatmap at a given z-slice.

    Parameters
    ----------
    scalar_field : ndarray, shape (Nr, Ntheta, Nz)
    grid : CylindricalGrid
    z_index : int or None
        Z-slice index.  Defaults to midplane.
    title : str
    filename : str or None
    cmap : str
    vmin, vmax : float or None
    """
    if z_index is None:
        z_index = scalar_field.shape[2] // 2

    data = scalar_field[:, :, z_index]  # (Nr, Ntheta)

    # Close the polar loop for smooth plotting
    theta = np.append(grid.theta, grid.theta[0] + 2 * np.pi)
    data_closed = np.concatenate([data, data[:, :1]], axis=1)

    r_mesh, theta_mesh = np.meshgrid(grid.r, theta, indexing='ij')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 6))
    pcm = ax.pcolormesh(theta_mesh, r_mesh, data_closed, cmap=cmap,
                        vmin=vmin, vmax=vmax, shading='auto')
    fig.colorbar(pcm, ax=ax, pad=0.1)
    ax.set_title(title, pad=20)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig, ax


def plot_rz_slice(scalar_field, grid, theta_index=0, title='', filename=None,
                  cmap='RdBu_r', vmin=None, vmax=None):
    """Plot an r-z cross-section at a given theta index.

    Parameters
    ----------
    scalar_field : ndarray, shape (Nr, Ntheta, Nz)
    grid : CylindricalGrid
    theta_index : int
    title : str
    filename : str or None
    """
    data = scalar_field[:, theta_index, :]  # (Nr, Nz)

    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(grid.z, grid.r, data, cmap=cmap,
                        vmin=vmin, vmax=vmax, shading='auto')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('r (m)')
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig, ax


def plot_three_slice_cartesian(scalar_field, grid, title='', filename=None,
                               cmap='viridis', log_scale=False):
    """Show three orthogonal midplane slices of a 3D Cartesian scalar field.

    Produces XY (z=0), XZ (y=0), and YZ (x=0) cross-sections.

    Parameters
    ----------
    scalar_field : ndarray, shape (Nx, Ny, Nz)
    grid : CartesianGrid
    title : str
    filename : str or None
    cmap : str
    log_scale : bool
        If True, plot log10(|field|).
    """
    nx, ny, nz = scalar_field.shape
    data = np.log10(np.maximum(np.abs(scalar_field), 1e-30)) if log_scale else scalar_field

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # XZ at y=0
    ax = axes[0]
    d = data[:, ny // 2, :]
    pcm = ax.pcolormesh(grid.z, grid.x, d, cmap=cmap, shading='auto')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('x (m)')
    ax.set_title(f'XZ slice (y=0)')
    fig.colorbar(pcm, ax=ax)

    # XY at z=0
    ax = axes[1]
    d = data[:, :, nz // 2]
    pcm = ax.pcolormesh(grid.y, grid.x, d, cmap=cmap, shading='auto')
    ax.set_xlabel('y (m)')
    ax.set_ylabel('x (m)')
    ax.set_title(f'XY slice (z=0)')
    fig.colorbar(pcm, ax=ax)

    # YZ at x=0
    ax = axes[2]
    d = data[nx // 2, :, :]
    pcm = ax.pcolormesh(grid.z, grid.y, d, cmap=cmap, shading='auto')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'YZ slice (x=0)')
    fig.colorbar(pcm, ax=ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig, axes


def plot_3d_quiver_cylindrical(B, grid, title='', filename=None, stride=4):
    """3D quiver plot of a vector field on a cylindrical grid.

    Converts cylindrical (r, theta, z) to Cartesian (x, y, z) for display.
    Subsamples by stride for readability.

    Parameters
    ----------
    B : ndarray, shape (Nr, Ntheta, Nz, 3)
    grid : CylindricalGrid
    title : str
    filename : str or None
    stride : int
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Subsample
    s = (slice(None, None, stride), slice(None, None, stride), slice(None, None, stride))
    R_s = grid.R[s]
    T_s = grid.THETA[s]
    Z_s = grid.Z[s]
    B_s = B[s]

    # Cylindrical to Cartesian positions
    X_pos = R_s * np.cos(T_s)
    Y_pos = R_s * np.sin(T_s)

    # Cylindrical to Cartesian field components
    Bx = B_s[..., 0] * np.cos(T_s) - B_s[..., 1] * np.sin(T_s)
    By = B_s[..., 0] * np.sin(T_s) + B_s[..., 1] * np.cos(T_s)
    Bz = B_s[..., 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X_pos.ravel(), Y_pos.ravel(), Z_s.ravel(),
              Bx.ravel(), By.ravel(), Bz.ravel(),
              length=0.3 * grid.dr * stride, normalize=True, alpha=0.6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(title)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig, ax


def plot_radial_comparison(r, profile_1d, profile_3d_avg, profile_3d_std,
                           ylabel='', title='', filename=None):
    """Compare a 1D radial profile against the 3D volume-averaged profile.

    Parameters
    ----------
    r : 1-d array
        Radial coordinate.
    profile_1d : 1-d array
        Reference 1D result.
    profile_3d_avg : 1-d array
        Mean over theta and z from the 3D computation.
    profile_3d_std : 1-d array
        Std over theta and z.
    ylabel : str
    title : str
    filename : str or None
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, profile_1d, 'k-', lw=2, label='1D reference')
    ax.plot(r, profile_3d_avg, 'r--', lw=1.5, label='3D mean')
    ax.fill_between(r, profile_3d_avg - profile_3d_std,
                    profile_3d_avg + profile_3d_std,
                    alpha=0.2, color='red', label='3D ± std')
    ax.set_xlabel('r (m)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig, ax
