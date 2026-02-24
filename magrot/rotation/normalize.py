"""
Normalization schemes and convention toggle.

The framework supports two conventions:

    Force-ratio (default / simulator):
        R > 1 -> inward dominates (contracting / collapsing)
        R = 1 -> equilibrium
        R < 1 -> outward dominates (expanding)

    Original (R&D document):
        R < 1 -> contracting
        R = 1 -> equilibrium
        R > 1 -> expanding

Use ``invert_convention`` to switch between them.
"""

import numpy as np


def invert_convention(R):
    """Convert between force-ratio and original convention.

    Mapping: R_original = 1 / R_force_ratio.
    """
    return 1.0 / np.maximum(R, 1e-30)
