import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


# Compute chi2_contingency)
def compute_contingency_test(row):
    """Compute chi-square test and Cramer's V effect size for contingency table."""
    # Create contingency table
    joint_freq = row["joint_freq"]
    active_only = row["active_freq"] - joint_freq
    sim_only = row["sim_freq"] - joint_freq
    neither = row["d_sae"] - (active_only + sim_only + joint_freq)

    # Create contingency table
    table = np.array([[joint_freq, active_only], [sim_only, neither]])

    # If any expected frequencies might be too small, use Fisher's exact test
    # Also use Fisher's if we have any negative entries (which shouldn't happen but let's be safe)
    try:
        if (table < 0).any() or table.sum() < 20:

            odds_ratio, pvalue = fisher_exact(table)
            # For consistency with chi-square test, compute a test statistic
            chi2_stat = -2 * np.log(pvalue) if pvalue > 0 else np.inf
            expected_min = 0  # Not applicable for Fisher's test
        else:
            # Use chi-square test for larger samples
            chi2_stat, pvalue, dof, expected = chi2_contingency(table)
            expected_min = expected.min()

    except ValueError as e:
        # print(f"Error computing chi-square test for table: {table}")
        # print(f"Error message: {str(e)}")
        return pd.Series({
            "chi2_stat": np.nan,
            "pvalue": 1.0,  # Conservative: assume no significance
            "cramer_v": 0.0,
            "expected_min": 0.0,
            "test_used": "error"
        })

    # Compute Cramer's V effect size
    n = table.sum()
    min_dim = min(table.shape) - 1
    if n > 0 and chi2_stat != np.inf:
        cramer_v = np.sqrt(chi2_stat / (n * min_dim))
    else:
        cramer_v = 0.0

    return pd.Series({
        "chi2_stat": chi2_stat,
        "pvalue": pvalue,
        "cramer_v": cramer_v,
        "expected_min": expected_min,
        "test_used": "fisher" if table.sum() < 20 else "chi2"
    })


# Add significance stars
def get_significance_stars(pvalue, cramer_v=None):
    """Get significance stars, optionally considering effect size."""
    if pvalue >= 0.05:
        return ""

    # Only show stars if effect size is meaningful
    if cramer_v is not None and cramer_v < 0.05:  # Very small effect size
        return "°"  # Technically significant but tiny effect

    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    else:
        return "*"


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


def interpret_effect_size(cramer_v):
    """Interpret Cramer's V effect size."""
    if cramer_v < 0.05:
        return "negligible"
    elif cramer_v < 0.10:
        return "weak"
    elif cramer_v < 0.30:
        return "moderate"
    elif cramer_v < 0.50:
        return "strong"
    else:
        return "very strong"
