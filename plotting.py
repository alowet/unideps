import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stats import get_significance_stars


def hide_spines(axs):
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
    return axs



def plot_correlations(weights, nrows=4, ncols=7, save_path=None, n_layers=26, ndeps=26, frequent_deps=None):
    fig, axs = plt.subplots(nrows, ncols, figsize=(21, 12))

    # Create a diverging colormap that is white at zero
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)

    for row in range(nrows):
        for col in range(ncols):
            layer = row * ncols + col
            ax = axs[row, col]
            if layer >= n_layers:
                ax.axis("off")
                continue
            weights_layer = weights[layer]
            corrs = np.corrcoef(weights_layer)
            im = ax.pcolormesh(corrs, cmap=cmap, norm=norm)
            ax.set_title(f"Layer {layer}")
            if row == nrows - 1 and col == 0:
                ax.set_xticks(range(ndeps))
                ax.set_yticks(range(ndeps))
                ax.set_xticklabels(list(frequent_deps.keys()), rotation=45, ha="right")
                ax.set_yticklabels(list(frequent_deps.keys()))
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    # plot a single color bar out to the side
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()



def plot_stars(g, subset, xvar, yvar):
    """Plot significance stars and effect sizes on the graph."""
    # Add significance stars
    for ax in g.axes.flat:
        if not ax.get_title():  # Skip empty subplots
            continue

        layer_str, rel_str = ax.get_title().split(" | ")
        relation = rel_str.split(" = ")[1]
        layer = int(layer_str.split(" = ")[1])

        # Get data for this subplot
        subplot_data = subset[
            (subset["relation"] == relation) &
            (subset["layer"] == layer)
        ]

        # Add significance stars and effect sizes at appropriate positions
        for _, row in subplot_data.iterrows():
            stars = get_significance_stars(row["pvalue"], row["cramer_v"])
            if stars:
                x = row[xvar]
                y = row[yvar]

                # Add stars
                ax.text(x, y, stars,
                       ha='center', va='bottom',
                       color='black' if stars != "°" else 'gray')

                # Optionally add effect size for large effects
                if row["cramer_v"] > 0.3:  # Medium or large effect
                    effect_size_text = f"V={row['cramer_v']:.2f}"
                    ax.text(x, y, effect_size_text,
                           ha='left', va='bottom',
                           fontsize='x-small',
                           color='darkgray')

    for axvar, scale in [(xvar, plt.xscale), (yvar, plt.yscale)]:
        if 'frac' in axvar:
            scale("log")
