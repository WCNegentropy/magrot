"""2D cross-section field and R heatmap plots."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_radial_profiles(r, profiles, title, filename, ylims=None):
    """Plot multiple radial-profile quantities vs radius.

    Parameters
    ----------
    r : 1-d array
        Radial coordinate.
    profiles : list of (label, data_or_list, kwargs)
        Each entry is (ylabel, data, plot_kwargs) or
        (ylabel, [(data, legend_label, kw), ...], {}).
    title : str
    filename : str
    ylims : dict or None
        Mapping label -> (ymin, ymax).
    """
    fig, axes = plt.subplots(len(profiles), 1,
                             figsize=(12, 4 * len(profiles)), sharex=True)
    if len(profiles) == 1:
        axes = [axes]

    for ax, (label, data, kwargs) in zip(axes, profiles):
        if isinstance(data, list):
            for d, l, kw in data:
                ax.plot(r, d, label=l, **kw)
            ax.legend(fontsize=10)
        else:
            ax.plot(r, data, **kwargs)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        if ylims and label in ylims:
            ax.set_ylim(ylims[label])

    axes[-1].set_xlabel('Radius r (m)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
