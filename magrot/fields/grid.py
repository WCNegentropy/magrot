"""
Grid representations for structured magnetic field data.

Supports cylindrical (r, theta, z) and Cartesian (x, y, z) grids
with automatic coordinate setup and singularity avoidance.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CylindricalGrid:
    """Cylindrical grid (r, theta, z) for axisymmetric configurations.

    Automatically avoids the r=0 axis singularity by clamping r_min.
    """
    Nr: int
    Ntheta: int
    Nz: int
    r_range: tuple
    theta_range: tuple = (0, 2 * np.pi)
    z_range: tuple = (-1.0, 1.0)

    def __post_init__(self):
        self.r = np.linspace(max(self.r_range[0], 1e-6), self.r_range[1], self.Nr)
        self.theta = np.linspace(
            self.theta_range[0], self.theta_range[1], self.Ntheta, endpoint=False
        )
        self.z = np.linspace(self.z_range[0], self.z_range[1], self.Nz)
        self.dr = self.r[1] - self.r[0] if self.Nr > 1 else 1.0
        self.R, self.THETA, self.Z = np.meshgrid(
            self.r, self.theta, self.z, indexing='ij'
        )


@dataclass
class CartesianGrid:
    """Cartesian grid (x, y, z)."""
    Nx: int
    Ny: int
    Nz: int
    x_range: tuple
    y_range: tuple
    z_range: tuple

    def __post_init__(self):
        self.x = np.linspace(self.x_range[0], self.x_range[1], self.Nx)
        self.y = np.linspace(self.y_range[0], self.y_range[1], self.Ny)
        self.z = np.linspace(self.z_range[0], self.z_range[1], self.Nz)
        self.dx = self.x[1] - self.x[0] if self.Nx > 1 else 1.0
        self.dy = self.y[1] - self.y[0] if self.Ny > 1 else 1.0
        self.dz = self.z[1] - self.z[0] if self.Nz > 1 else 1.0
        self.X, self.Y, self.Z = np.meshgrid(
            self.x, self.y, self.z, indexing='ij'
        )
