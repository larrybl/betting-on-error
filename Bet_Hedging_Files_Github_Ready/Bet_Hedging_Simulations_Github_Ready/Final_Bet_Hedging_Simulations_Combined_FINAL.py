#!/usr/bin/env python3
"""Reproduce the Chapter 3 simulation figures used in the submitted thesis.

This script preserves the mathematical model, parameters, random seeds, and
simulation logic of the original combined Chapter 3 pipeline. It generates only
the simulation figures retained in the submitted thesis: Figures 3.2-3.9.

No external data files are required. Cached numerical results are stored in the
script-specific ``Additional Files`` directory and figures are written as
300-dpi PNG files to the script-specific ``Figures`` directory.

Run with:
    python Final_Bet_Hedging_Simulations_Combined_FINAL.py

Use ``--force`` to recompute cached numerical analyses.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import numpy as np


# --------------------------------------------------------------------------- #
# Repository-safe paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem

ADDITIONAL_DIR = HERE / f"{SCRIPT_STEM} Additional Files"
FIGURE_DIR = HERE / f"{SCRIPT_STEM} Figures"

ADDITIONAL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_SURFACE_CACHE = ADDITIONAL_DIR / "four_ploidy_metric_surfaces_v2.npz"
CONTINUOUS_RESILIENCE_CACHE = ADDITIONAL_DIR / "extinction_resilience_v3.npz"
EVOLVING_RATE_CACHE = ADDITIONAL_DIR / "evolving_missegregation_rate_v1.npz"
FITNESS_COST_TRACKING_CACHE = ADDITIONAL_DIR / "fitness_cost_step_tracking_v1.npz"


# --------------------------------------------------------------------------- #
# Shared figure style
# --------------------------------------------------------------------------- #

STYLE = {
    "dpi": 300,
    "base_fs": 29,
    "axis_fs": 29,
    "tick_fs": 29,
    "legend_fs": 25,
    "annotation_fs": 23,
    "panel_fs": 29,
}

LATER_FIGURE_AXIS_FS = 25
LATER_FIGURE_TICK_FS = 25

SINGLE_PANEL_FIGSIZE = (16.0, 11.0)
TWO_PANEL_FIGSIZE = (22.0, 11.0)


def apply_style():
    """Apply the shared thesis figure typography."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": STYLE["base_fs"],
        "axes.labelsize": STYLE["axis_fs"],
        "xtick.labelsize": STYLE["tick_fs"],
        "ytick.labelsize": STYLE["tick_fs"],
        "legend.fontsize": STYLE["legend_fs"],
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": STYLE["dpi"],
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def save_figure(fig, filename):
    """Write one 300-dpi PNG atomically."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    target = FIGURE_DIR / filename
    temporary = FIGURE_DIR / f".{target.stem}.tmp.png"

    try:
        fig.savefig(
            temporary,
            format="png",
            dpi=STYLE["dpi"],
            bbox_inches="tight",
            facecolor="white",
        )

        if temporary.stat().st_size == 0:
            raise OSError(f"Empty figure output: {temporary}")

        os.replace(temporary, target)

    finally:
        temporary.unlink(missing_ok=True)

    plt.close(fig)
    print(f"Figure written: {target}")


def add_panel_label(
    ax,
    label,
    x=0.0,
    y=1.03,
    *,
    ha="left",
    va="bottom",
    bbox=None,
):
    """Add a consistent bold panel letter."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=STYLE["panel_fs"],
        fontweight="bold",
        clip_on=False,
        bbox=bbox,
    )


def floating_axes(ax):
    """Use open, outward axes and no background grid."""
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))


def set_later_figure_axis_text(ax):
    """Use slightly smaller axis and tick text for Figures 3.4 onwards."""
    ax.xaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    ax.yaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    ax.tick_params(
        axis="both",
        labelsize=LATER_FIGURE_TICK_FS,
    )


# --------------------------------------------------------------------------- #
# Model constants
# --------------------------------------------------------------------------- #

POPULATION_SIZE = 50_000
FITNESS_COST = 0.70

PLOIDY_NAMES = [
    "Haploid (1n)",
    "Diploid (2n)",
    "Triploid (3n)",
    "Tetraploid (4n)",
]

PLOIDY_COLOURS = [
    "#CEB49D",
    "#F2A58B",
    "#8DBDE3",
    "#94C8AA",
]

LOW_ENV_COLOUR = "#F3D6E5"
HIGH_ENV_COLOUR = "#DDEBCF"

ENV_CMAP = matplotlib.colors.ListedColormap(
    [LOW_ENV_COLOUR, HIGH_ENV_COLOUR]
)

PERFORMANCE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "pastel_performance",
    ["#9CC9E2", "#F7F2E8", "#E7A3B1"],
)

RATE_DIVERGING_CMAP_AII = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rate_purple_cream_orange",
    ["#6042A6", "#F4EBDD", "#D36B1F"],
)

ILLUSTRATIVE_MISSEGREGATION_RATE = 0.08
ILLUSTRATIVE_SWITCH_RATE = 0.20
LONG_GENERATIONS = 1_000
ILLUSTRATIVE_RANDOM_SEED = 0

GENERATIONS = int(os.environ.get("BH_GENERATIONS", "1000"))
NUM_RUNS = int(os.environ.get("BH_NUM_RUNS", "30"))

LEGACY_GRID_POINTS = os.environ.get("BH_GRID_POINTS")

MISSEGREGATION_ROWS = int(
    os.environ.get(
        "BH_MISSEGREGATION_ROWS",
        LEGACY_GRID_POINTS or "50",
    )
)

SWITCH_RATE_COLUMNS = int(
    os.environ.get(
        "BH_SWITCH_RATE_COLUMNS",
        LEGACY_GRID_POINTS or "25",
    )
)

RANDOM_SEED = 20260811

MISSEGREGATION_RATES = np.logspace(
    -7,
    -1,
    MISSEGREGATION_ROWS,
)

SWITCH_RATES = np.concatenate(
    (
        [0.0],
        np.logspace(
            -3,
            0,
            SWITCH_RATE_COLUMNS - 1,
        ),
    )
)

EVOLVING_RATE_GRID = np.logspace(-7, 0, 49)
RATE_MUTATION_PROBABILITY = 0.02

FITNESS_COST_VALUES = np.asarray(
    [0.30, 0.50, 0.70, 0.90]
)

FITNESS_COST_COLOURS = [
    "#8DB7D5",
    "#A99ACB",
    "#E2A6B4",
    "#E7B77B",
]

FORCE_RERUN = False


# --------------------------------------------------------------------------- #
# Unchanged four-ploidy model
# --------------------------------------------------------------------------- #

def _binom(rng, n, p):
    n = int(n)

    if n <= 0 or p <= 0:
        return 0

    if p >= 1:
        return n

    if n <= 2_000_000_000:
        return int(rng.binomial(n, p))

    mean = float(n) * p

    if mean < 1e8:
        return min(n, int(rng.poisson(mean)))

    draw = rng.normal(
        mean,
        np.sqrt(mean * (1.0 - p)),
    )

    return int(
        min(
            float(n),
            max(0.0, draw),
        )
    )


def switch_environment(rng, current_state, switch_rate):
    if rng.random() < switch_rate:
        return -0.5 if current_state == 0.5 else 0.5

    return current_state


def mis_segregate(rng, h, d, tr, te, rate):
    """The exact transition rules from the original four-ploidy model."""
    mis_h = _binom(rng, h, rate)
    mis_d = _binom(rng, d, rate)
    mis_tr = _binom(rng, tr, rate)
    mis_te = _binom(rng, te, rate)

    new_h = 2 * (h - mis_h)
    new_d = 2 * (d - mis_d)
    new_tr = 2 * (tr - mis_tr)
    new_te = 2 * (te - mis_te)

    new_d += mis_h
    new_h += mis_d
    new_tr += mis_d
    new_d += mis_tr
    new_te += mis_tr
    new_tr += mis_te

    return new_h, new_d, new_tr, new_te


def simulate(
    generations,
    missegregation_rate,
    switch_rate,
    seed,
):
    """Run the original illustrative model and include generation zero."""
    rng = np.random.default_rng(seed)

    h = 0
    d = POPULATION_SIZE // 2
    tr = POPULATION_SIZE // 2
    te = 0

    environment = 0.5

    history = [(h, d, tr, te)]
    environments = [environment]

    for _ in range(generations):
        environment = switch_environment(
            rng,
            environment,
            switch_rate,
        )

        if environment == -0.5:
            tr = int(
                float(tr)
                * (1.0 - FITNESS_COST)
            )
            te = int(
                float(te)
                * (1.0 - FITNESS_COST)
            )

        else:
            h = int(
                float(h)
                * (1.0 - FITNESS_COST)
            )
            d = int(
                float(d)
                * (1.0 - FITNESS_COST)
            )

        h, d, tr, te = mis_segregate(
            rng,
            h,
            d,
            tr,
            te,
            missegregation_rate,
        )

        history.append((h, d, tr, te))
        environments.append(environment)

    return (
        np.asarray(history, dtype=object),
        np.asarray(environments),
    )


def proportions(history):
    numeric = np.asarray(
        [
            [float(value) for value in row]
            for row in history
        ]
    )

    totals = numeric.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        numeric,
        totals,
        out=np.zeros_like(numeric),
        where=totals > 0,
    )


def numeric_totals(history):
    return np.asarray(
        [
            sum(float(value) for value in row)
            for row in history
        ]
    )


def smooth_series(values, window=15):
    values = np.asarray(values, dtype=float)

    window = max(
        1,
        min(
            int(window),
            len(values),
        ),
    )

    if window == 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    pad = window // 2

    padded = np.pad(
        values,
        pad,
        mode="edge",
    )

    return np.convolve(
        padded,
        np.ones(window) / window,
        mode="valid",
    )


def shannon_entropy(counts):
    total = float(sum(counts))

    if total <= 0:
        return 0.0

    fractions = np.asarray(
        [
            float(value) / total
            for value in counts
            if value > 0
        ]
    )

    return float(
        -np.sum(
            fractions
            * np.log(fractions)
        )
    )


# --------------------------------------------------------------------------- #
# Figure 3.2
# --------------------------------------------------------------------------- #

def plate_points(n, rng):
    angle = rng.uniform(
        0,
        2 * np.pi,
        n,
    )

    radius = (
        0.88
        * np.sqrt(
            rng.uniform(
                0,
                1,
                n,
            )
        )
    )

    return np.column_stack(
        (
            radius * np.cos(angle),
            radius * np.sin(angle),
        )
    )


def draw_plate(
    ax,
    fractions,
    generation,
    environment,
    points,
    rng,
):
    env_name = (
        "Favours low ploidy"
        if environment == -0.5
        else "Favours high ploidy"
    )

    env_colour = (
        LOW_ENV_COLOUR
        if environment == -0.5
        else HIGH_ENV_COLOUR
    )

    ax.add_patch(
        Circle(
            (0, 0),
            1.0,
            facecolor=env_colour,
            edgecolor="0.22",
            lw=2.0,
        )
    )

    ax.add_patch(
        Circle(
            (0, 0),
            0.93,
            facecolor="none",
            edgecolor="white",
            lw=1.2,
            alpha=0.9,
        )
    )

    classes = rng.choice(
        4,
        size=len(points),
        p=fractions,
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=[
            PLOIDY_COLOURS[i]
            for i in classes
        ],
        s=45,
        edgecolors="#2F2F2F",
        linewidths=0.45,
        alpha=1.0,
    )

    ax.text(
        0.5,
        1.02,
        (
            f"Generation {generation}\n"
            f"{env_name}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=STYLE["annotation_fs"],
        fontweight="bold",
    )

    ax.text(
        0,
        -1.16,
        (
            f"1n {fractions[0]:.0%}   "
            f"2n {fractions[1]:.0%}\n"
            f"3n {fractions[2]:.0%}   "
            f"4n {fractions[3]:.0%}"
        ),
        ha="center",
        va="top",
        fontsize=STYLE["annotation_fs"],
        color="0.25",
    )

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.42, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_environment_strip(
    ax,
    environments,
    generations,
):
    ax.imshow(
        environments[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=ENV_CMAP,
        extent=[
            -0.5,
            generations + 0.5,
            0,
            1,
        ],
        vmin=-0.5,
        vmax=0.5,
    )

    ax.set_yticks([])

    ax.set_xlim(
        -0.5,
        generations + 0.5,
    )

    ax.set_ylabel(
        "Environment",
        rotation=0,
        ha="right",
        va="center",
        labelpad=18,
    )

    ax.spines[
        ["top", "right", "left", "bottom"]
    ].set_visible(False)


def make_figure_3_2():
    history, environments = simulate(
        120,
        ILLUSTRATIVE_MISSEGREGATION_RATE,
        ILLUSTRATIVE_SWITCH_RATE,
        ILLUSTRATIVE_RANDOM_SEED,
    )

    props = proportions(history)

    shown = [
        *range(0, 101, 10),
        120,
    ]

    x = np.arange(121)

    plate_rows = int(
        np.ceil(
            len(shown) / 6
        )
    )

    fig = plt.figure(
        figsize=(
            26.0,
            11.2 + 5.4 * plate_rows,
        )
    )

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            2.00,
            2.85 * plate_rows,
        ],
        hspace=0.32,
    )

    top_grid = outer[0].subgridspec(
        2,
        1,
        height_ratios=[
            0.30,
            1.76,
        ],
        hspace=0.34,
    )

    env_ax = fig.add_subplot(
        top_grid[0]
    )

    fraction_ax = fig.add_subplot(
        top_grid[1],
        sharex=env_ax,
    )

    plate_grid = outer[1].subgridspec(
        plate_rows,
        6,
        hspace=0.48,
        wspace=0.24,
    )

    plate_axes = [
        fig.add_subplot(
            plate_grid[
                i // 6,
                i % 6,
            ]
        )
        for i in range(
            plate_rows * 6
        )
    ]

    draw_environment_strip(
        env_ax,
        environments,
        120,
    )

    env_ax.tick_params(
        axis="x",
        labelbottom=False,
        bottom=False,
    )

    add_panel_label(
        env_ax,
        "A",
        x=-0.085,
        y=1.02,
        ha="left",
        va="bottom",
    )

    fraction_ax.stackplot(
        x,
        props.T,
        colors=PLOIDY_COLOURS,
        labels=PLOIDY_NAMES,
        alpha=1.0,
        linewidth=0,
    )

    fraction_ax.set_xlim(
        0,
        120,
    )

    fraction_ax.set_ylim(
        0,
        1,
    )

    fraction_ax.set_xticks(
        np.arange(
            0,
            121,
            10,
        )
    )

    fraction_ax.set_yticks(
        [
            0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
        [
            "0%",
            "25%",
            "50%",
            "75%",
            "100%",
        ],
    )

    fraction_ax.set_xlabel(
        "Generation"
    )

    fraction_ax.set_ylabel(
        "Fraction of population"
    )

    floating_axes(
        fraction_ax
    )

    add_panel_label(
        fraction_ax,
        "B",
        x=-0.085,
        y=1.02,
        ha="left",
        va="bottom",
    )

    point_rng = np.random.default_rng(
        80
    )

    colour_rng = np.random.default_rng(
        81
    )

    points = plate_points(
        130,
        point_rng,
    )

    for ax, generation in zip(
        plate_axes[:len(shown)],
        shown,
    ):
        draw_plate(
            ax,
            props[generation],
            generation,
            environments[generation],
            points,
            colour_rng,
        )

    for ax in plate_axes[
        len(shown):
    ]:
        ax.axis("off")

    add_panel_label(
        plate_axes[0],
        "C",
        x=-0.612,
        y=1.10,
        ha="left",
        va="bottom",
    )

    handles = [
        Patch(
            facecolor=colour,
            label=name,
        )
        for colour, name in zip(
            PLOIDY_COLOURS,
            PLOIDY_NAMES,
        )
    ]

    handles.extend(
        [
            Patch(
                facecolor=LOW_ENV_COLOUR,
                edgecolor="0.6",
                label="Environment favours low ploidy",
            ),
            Patch(
                facecolor=HIGH_ENV_COLOUR,
                edgecolor="0.6",
                label="Environment favours high ploidy",
            ),
        ]
    )

    fig.text(
        0.985,
        0.103,
        (
            f"Illustrative m = "
            f"{ILLUSTRATIVE_MISSEGREGATION_RATE:g}; "
            f"environmental switching probability p = "
            f"{ILLUSTRATIVE_SWITCH_RATE:g} per generation."
        ),
        ha="right",
        va="center",
        fontsize=STYLE["annotation_fs"] - 1,
        color="0.25",
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
        fontsize=STYLE["legend_fs"],
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.972,
        bottom=0.145,
    )

    save_figure(
        fig,
        (
            "Figure_3.2_Stochastic_Ploidy_Diversification_"
            "During_Environmental_Fluctuation.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 3.3
# --------------------------------------------------------------------------- #

def simulate_shared_environment(
    generations,
    rates,
    switch_rate,
    seed,
):
    """Preserve the original shared-environment illustrative simulation."""
    rng = np.random.default_rng(seed)

    states = [
        [
            0,
            POPULATION_SIZE // 2,
            POPULATION_SIZE // 2,
            0,
        ]
        for _ in rates
    ]

    histories = [
        [tuple(state)]
        for state in states
    ]

    environment = 0.5
    environments = [environment]

    for _ in range(generations):
        environment = switch_environment(
            rng,
            environment,
            switch_rate,
        )

        environments.append(
            environment
        )

        for index, rate in enumerate(rates):
            h, d, tr, te = states[index]

            if environment == -0.5:
                tr = int(
                    float(tr)
                    * (1.0 - FITNESS_COST)
                )

                te = int(
                    float(te)
                    * (1.0 - FITNESS_COST)
                )

            else:
                h = int(
                    float(h)
                    * (1.0 - FITNESS_COST)
                )

                d = int(
                    float(d)
                    * (1.0 - FITNESS_COST)
                )

            child_rng = np.random.default_rng(
                rng.integers(
                    0,
                    2**63 - 1,
                )
            )

            states[index] = list(
                mis_segregate(
                    child_rng,
                    h,
                    d,
                    tr,
                    te,
                    rate,
                )
            )

            histories[index].append(
                tuple(
                    states[index]
                )
            )

    return (
        [
            np.asarray(
                history,
                dtype=object,
            )
            for history in histories
        ],
        np.asarray(
            environments
        ),
    )


def make_figure_3_3():
    histories, environments = simulate_shared_environment(
        LONG_GENERATIONS,
        [0.005, 0.08],
        0.08,
        98,
    )

    lineage_labels = [
        "Low mis-segregation rate lineage (m = 0.005)",
        "Bet-hedging lineage (mis-segregation rate m = 0.08)",
    ]

    lineage_colours = [
        "#5B6770",
        "#D1495B",
    ]

    generations = len(environments) - 1
    x = np.arange(generations + 1)

    totals = [
        numeric_totals(history)
        for history in histories
    ]

    composition = [
        proportions(history)
        for history in histories
    ]

    diversity = []

    for values in composition:
        terms = np.zeros_like(
            values,
            dtype=float,
        )

        positive = values > 0

        terms[positive] = (
            values[positive]
            * np.log(values[positive])
        )

        diversity.append(
            -np.sum(
                terms,
                axis=1,
            )
        )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(
            20.0,
            27.0,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                0.30,
                2.35,
                0.95,
                1.55,
                1.55,
            ],
            "hspace": 0.36,
        },
    )

    (
        env_ax,
        total_ax,
        diversity_ax,
        first_ax,
        second_ax,
    ) = axes

    draw_environment_strip(
        env_ax,
        environments,
        generations,
    )

    for label, axis in zip(
        "ABCDE",
        axes,
    ):
        add_panel_label(
            axis,
            label,
            x=-0.085,
            y=1.16 if label == "C" else 1.06 if label == "E" else 1.02,
            ha="left",
            va="bottom",
        )

    for values, label, colour in zip(
        totals,
        lineage_labels,
        lineage_colours,
    ):
        total_ax.plot(
            x,
            np.log10(values),
            color=colour,
            lw=3.0,
            label=label,
        )

    total_ax.set_ylabel(
        "Accumulated descendants\n"
        "log₁₀(cells)"
    )

    # Raise the legend slightly so it no longer overlaps the red trajectory.
    total_ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.015, 1.01),
        ncol=2,
        fontsize=24
    )

    floating_axes(
        total_ax
    )

    for entropy, label, colour in zip(
        diversity,
        lineage_labels,
        lineage_colours,
    ):
        diversity_ax.plot(
            x,
            smooth_series(
                entropy,
                window=15,
            ),
            color=colour,
            lw=2.5,
            label=label,
        )

    diversity_ax.set_ylim(
        0,
        np.log(4) * 1.04,
    )

    diversity_ax.set_yticks(
        [
            0,
            np.log(2),
            np.log(4),
        ],
        [
            "0",
            "ln 2",
            "ln 4",
        ],
    )

    diversity_ax.set_ylabel(
        "Shannon diversity"
    )

    floating_axes(
        diversity_ax
    )

    diversity_ax.text(
        0.0,
        1.10,
        "Ploidy diversity",
        transform=diversity_ax.transAxes,
        fontsize=STYLE["annotation_fs"],
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )

    for ax, values in zip(
        (
            first_ax,
            second_ax,
        ),
        composition,
    ):
        ax.stackplot(
            x,
            values.T,
            colors=PLOIDY_COLOURS,
            alpha=1.0,
            linewidth=0,
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.set_yticks(
            [
                0,
                0.5,
                1,
            ],
            [
                "0%",
                "50%",
                "100%",
            ],
        )

        ax.set_ylabel(
            "Ploidy fractions"
        )

        floating_axes(
            ax
        )

    first_ax.text(
        0.0,
        1.10,
        "Low mis-segregation rate lineage (m = 0.005)",
        transform=first_ax.transAxes,
        fontsize=STYLE["annotation_fs"],
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )

    second_ax.text(
        0.0,
        1.10,
        "Bet-hedging lineage (mis-segregation rate m = 0.08)",
        transform=second_ax.transAxes,
        fontsize=STYLE["annotation_fs"],
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )

    second_ax.set_xlabel(
        "Generation"
    )

    handles = [
        Patch(
            facecolor=colour,
            label=name,
        )
        for colour, name in zip(
            PLOIDY_COLOURS,
            PLOIDY_NAMES,
        )
    ]

    second_ax.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.25,
        ),
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.975,
        bottom=0.09,
    )

    # Reduce only the white space between Panels A and B.
    pos_b = total_ax.get_position()

    total_ax.set_position([
        pos_b.x0,
        pos_b.y0,
        pos_b.width,
        pos_b.height + 0.006,
    ])

    save_figure(
        fig,
        (
            "Figure_3.3_Higher_Karyotypic_Randomisation_"
            "Increases_Long-Term_Growth_Under_"
            "Environmental_Fluctuation.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Main replicated parameter sweep for Figures 3.4-3.6
# --------------------------------------------------------------------------- #

def run_once(
    mis_rate,
    switch_rate,
    rng,
    *,
    population_size=POPULATION_SIZE,
    fitness_cost=FITNESS_COST,
    generations=GENERATIONS,
    initial_high_fraction=0.5,
    early_window=25,
    crash_ratio=0.70,
):
    """Run one population using the original theory-model rules."""
    tr = int(
        round(
            population_size
            * initial_high_fraction
        )
    )

    d = int(
        population_size - tr
    )

    h = 0
    te = 0
    environment = 0.5

    log_growth = []
    entropies = []
    early_entropies = []
    crashed = False

    for generation in range(
        generations
    ):
        before = (
            h
            + d
            + tr
            + te
        )

        if before <= 0:
            return (
                0.0,
                np.nan,
                0.0,
                0.0,
                True,
                True,
            )

        environment = switch_environment(
            rng,
            environment,
            switch_rate,
        )

        if environment == -0.5:
            h = int(
                float(h)
                * (1.0 - fitness_cost)
            )

            d = int(
                float(d)
                * (1.0 - fitness_cost)
            )

        else:
            tr = int(
                float(tr)
                * (1.0 - fitness_cost)
            )

            te = int(
                float(te)
                * (1.0 - fitness_cost)
            )

        h, d, tr, te = mis_segregate(
            rng,
            h,
            d,
            tr,
            te,
            mis_rate,
        )

        after = (
            h
            + d
            + tr
            + te
        )

        if after <= 0:
            return (
                0.0,
                np.nan,
                0.0,
                float(
                    np.mean(
                        early_entropies
                        or [0.0]
                    )
                ),
                True,
                True,
            )

        growth_ratio = (
            float(after)
            / float(before)
        )

        crashed = (
            crashed
            or growth_ratio
            <= crash_ratio
        )

        log_growth.append(
            np.log(
                growth_ratio
            )
        )

        entropy = shannon_entropy(
            (
                h,
                d,
                tr,
                te,
            )
        )

        entropies.append(
            entropy
        )

        if generation < early_window:
            early_entropies.append(
                entropy
            )

    geometric_mean = float(
        np.exp(
            np.mean(
                log_growth
            )
        )
    )

    temporal_variance = float(
        np.var(
            log_growth
        )
    )

    mean_entropy = float(
        np.mean(
            entropies
        )
    )

    early_entropy = float(
        np.mean(
            early_entropies
        )
    )

    return (
        geometric_mean,
        temporal_variance,
        mean_entropy,
        early_entropy,
        False,
        crashed,
    )


def calculate_surfaces():
    shape = (
        len(
            MISSEGREGATION_RATES
        ),
        len(
            SWITCH_RATES
        ),
    )

    growth = np.zeros(shape)
    variance = np.zeros(shape)
    entropy = np.zeros(shape)
    extinction = np.zeros(shape)

    seeds = np.random.SeedSequence(
        RANDOM_SEED
    ).spawn(
        np.prod(shape)
        * NUM_RUNS
    )

    seed_index = 0

    for row, mis_rate in enumerate(
        MISSEGREGATION_RATES
    ):
        print(
            f"Calculating main sweep row "
            f"{row + 1}/"
            f"{len(MISSEGREGATION_RATES)}"
        )

        for col, switch_rate in enumerate(
            SWITCH_RATES
        ):
            cell = []

            for _ in range(
                NUM_RUNS
            ):
                rng = np.random.default_rng(
                    seeds[
                        seed_index
                    ]
                )

                seed_index += 1

                cell.append(
                    run_once(
                        mis_rate,
                        switch_rate,
                        rng,
                    )
                )

            growth[
                row,
                col,
            ] = np.mean(
                [
                    value[0]
                    for value
                    in cell
                ]
            )

            valid_variance = [
                value[1]
                for value
                in cell
                if np.isfinite(
                    value[1]
                )
            ]

            variance[
                row,
                col,
            ] = (
                np.mean(
                    valid_variance
                )
                if valid_variance
                else np.nan
            )

            entropy[
                row,
                col,
            ] = np.mean(
                [
                    value[2]
                    for value
                    in cell
                ]
            )

            extinction[
                row,
                col,
            ] = np.mean(
                [
                    value[4]
                    for value
                    in cell
                ]
            )

    return (
        growth,
        variance,
        entropy,
        extinction,
    )


def load_or_calculate_surfaces():
    if (
        MAIN_SURFACE_CACHE.exists()
        and not FORCE_RERUN
    ):
        z = np.load(
            MAIN_SURFACE_CACHE
        )

        valid = (
            int(z["generations"]) == GENERATIONS
            and int(z["num_runs"]) == NUM_RUNS
            and np.array_equal(
                z["mis_rates"],
                MISSEGREGATION_RATES,
            )
            and np.array_equal(
                z["switch_rates"],
                SWITCH_RATES,
            )
        )

        if valid:
            print(
                f"Using cached analysis: "
                f"{MAIN_SURFACE_CACHE}"
            )

            return (
                z["growth"],
                z["variance"],
                z["entropy"],
                z["extinction"],
            )

    (
        growth,
        variance,
        entropy,
        extinction,
    ) = calculate_surfaces()

    np.savez(
        MAIN_SURFACE_CACHE,
        growth=growth,
        variance=variance,
        entropy=entropy,
        extinction=extinction,
        mis_rates=MISSEGREGATION_RATES,
        switch_rates=SWITCH_RATES,
        generations=GENERATIONS,
        num_runs=NUM_RUNS,
    )

    return (
        growth,
        variance,
        entropy,
        extinction,
    )


def rate_ticks(values):
    targets = np.logspace(
        -7,
        -1,
        7,
    )

    locations = [
        int(
            np.argmin(
                np.abs(
                    values
                    - value
                )
            )
        )
        for value in targets
    ]

    labels = [
        f"$10^{{{power}}}$"
        for power in range(
            -7,
            0,
        )
    ]

    return (
        locations,
        labels,
    )


def switch_ticks(values):
    locations = [0]

    labels = [
        "Constant\n($p=0$)"
    ]

    for target in (
        1e-2,
        1e-1,
        1.0,
    ):
        index = int(
            np.argmin(
                np.abs(
                    values
                    - target
                )
            )
        )

        locations.append(
            index
        )

        labels.append(
            f"$10^{{"
            f"{int(np.round(np.log10(values[index])))}"
            f"}}$"
        )

    return (
        locations,
        labels,
    )


def mismatch_values(
    surface,
    normalise=True,
):
    mismatch = []
    values = []

    for col in range(
        1,
        len(SWITCH_RATES) - 1,
    ):
        p = SWITCH_RATES[col]

        column = surface[
            :,
            col,
        ]

        if normalise:
            column = (
                column
                / np.nanmax(
                    column
                )
            )

        mismatch.extend(
            np.log10(
                MISSEGREGATION_RATES
                / p
            )
        )

        values.extend(
            column
        )

    return (
        np.asarray(
            mismatch
        ),
        np.asarray(
            values
        ),
    )


def binned_summary_band(
    x,
    y,
    edges,
    mode="sem",
):
    """Return the original binned mean and summary band."""
    mode = mode.lower()

    if mode not in {
        "sem",
        "sd",
        "iqr",
    }:
        raise ValueError(
            "mode must be "
            "'sem', 'sd', or 'iqr'"
        )

    centres = (
        edges[:-1]
        + edges[1:]
    ) / 2

    means = np.full(
        len(centres),
        np.nan,
    )

    lower = np.full(
        len(centres),
        np.nan,
    )

    upper = np.full(
        len(centres),
        np.nan,
    )

    counts = np.zeros(
        len(centres),
        dtype=int,
    )

    for index in range(
        len(centres)
    ):
        chosen = (
            (
                x >= edges[index]
            )
            & (
                x < edges[
                    index + 1
                ]
            )
        )

        values = np.asarray(
            y[chosen],
            dtype=float,
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        counts[index] = len(
            values
        )

        if not len(values):
            continue

        means[index] = np.mean(
            values
        )

        if mode == "iqr":
            (
                lower[index],
                upper[index],
            ) = np.percentile(
                values,
                [
                    25,
                    75,
                ],
            )

        else:
            spread = (
                np.std(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else 0.0
            )

            if (
                mode == "sem"
                and len(values) > 1
            ):
                spread /= np.sqrt(
                    len(values)
                )

            lower[index] = (
                means[index]
                - spread
            )

            upper[index] = (
                means[index]
                + spread
            )

    return (
        centres,
        means,
        lower,
        upper,
        counts,
    )


# --------------------------------------------------------------------------- #
# Figure 3.4
# --------------------------------------------------------------------------- #

def make_figure_3_4(growth):
    baseline = growth[
        0:1,
        :,
    ]

    advantage = np.log(
        growth
        / baseline
    )

    limit = float(
        np.nanmax(
            np.abs(
                advantage
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE
    )

    image = ax.imshow(
        advantage,
        origin="lower",
        aspect="auto",
        cmap=PERFORMANCE_CMAP,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        alpha=1.0,
    )

    x, xl = switch_ticks(
        SWITCH_RATES
    )

    y, yl = rate_ticks(
        MISSEGREGATION_RATES
    )

    ax.set_xticks(
        x,
        xl,
    )

    ax.set_yticks(
        y,
        yl,
    )

    ax.set_xlabel(
        "Environmental switching "
        "probability per generation"
    )

    ax.set_ylabel(
        "Mis-segregation probability "
        "per cell division"
    )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.025,
    )

    colourbar.set_label(
        "ln(geometric mean fitness / "
        "lowest-rate reference)"
    )

    floating_axes(
        ax
    )

    set_later_figure_axis_text(ax)
    colourbar.ax.yaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    colourbar.ax.tick_params(labelsize=LATER_FIGURE_TICK_FS)

    fig.subplots_adjust(
        left=0.16,
        right=0.87,
        bottom=0.16,
        top=0.97,
    )

    save_figure(
        fig,
        (
            "Figure_3.4_Long-Term_Advantage_of_"
            "Karyotypic_Randomisation_Across_"
            "Environmental_Switching_and_"
            "Mis-Segregation_Rates.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 3.5
# --------------------------------------------------------------------------- #

def make_figure_3_5(growth):
    ratios = (
        growth
        / growth[
            0:1,
            :,
        ]
    )

    (
        mismatch,
        relative,
    ) = mismatch_values(
        ratios,
        normalise=False,
    )

    (
        _,
        geometric_mean,
    ) = mismatch_values(
        growth,
        normalise=False,
    )

    log_advantage = np.log(
        relative
    )

    edges = np.arange(
        -6.25,
        2.76,
        0.5,
    )

    (
        centres,
        means,
        lower,
        upper,
        _,
    ) = binned_summary_band(
        mismatch,
        log_advantage,
        edges,
        mode="sem",
    )

    valid = np.isfinite(
        means
    )

    fig, ax = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE
    )

    points = ax.scatter(
        mismatch,
        log_advantage,
        c=geometric_mean,
        cmap=PERFORMANCE_CMAP,
        s=38,
        alpha=1.0,
        edgecolors="none",
        rasterized=True,
        label=(
            "Individual parameter "
            "combinations"
        ),
    )

    colourbar = fig.colorbar(
        points,
        ax=ax,
        pad=0.025,
    )

    colourbar.set_label(
        "Geometric mean fitness"
    )

    ax.fill_between(
        centres[valid],
        lower[valid],
        upper[valid],
        color="0.55",
        alpha=0.28,
        linewidth=0,
        label=(
            "SEM within each interval"
        ),
    )

    ax.plot(
        centres[valid],
        means[valid],
        color="black",
        lw=3.0,
        marker="o",
        ms=7,
        label=(
            "Mean within each interval"
        ),
    )

    ax.axhline(
        0,
        color="0.30",
        lw=1.3,
        ls="--",
    )

    ax.set_xlabel(
        r"Difference between rates, "
        r"$\log_{10}(m/p)$"
    )

    ax.set_ylabel(
        "Log geometric-mean fitness advantage\n"
        "over the lowest-m population"
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    floating_axes(
        ax
    )

    set_later_figure_axis_text(ax)
    colourbar.ax.yaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    colourbar.ax.tick_params(labelsize=LATER_FIGURE_TICK_FS)

    fig.tight_layout()

    save_figure(
        fig,
        (
            "Figure_3.5_The_Benefit_of_Randomisation_"
            "Depends_on_the_Relationship_Between_"
            "Mis-Segregation_and_Environmental_"
            "Switching_Rates.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 3.6
# --------------------------------------------------------------------------- #

def make_figure_3_6(growth):
    (
        mismatch,
        relative_growth,
    ) = mismatch_values(
        growth,
        normalise=True,
    )

    (
        _,
        geometric_mean,
    ) = mismatch_values(
        growth,
        normalise=False,
    )

    edges = np.arange(
        -6.25,
        2.76,
        0.5,
    )

    (
        centres,
        means,
        lower,
        upper,
        _,
    ) = binned_summary_band(
        mismatch,
        relative_growth,
        edges,
        mode="sem",
    )

    valid = np.isfinite(
        means
    )

    fig, ax = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE
    )

    points = ax.scatter(
        mismatch,
        relative_growth,
        c=geometric_mean,
        cmap=PERFORMANCE_CMAP,
        s=38,
        alpha=1.0,
        edgecolors="none",
        rasterized=True,
        label=(
            "Individual parameter "
            "combinations"
        ),
    )

    colourbar = fig.colorbar(
        points,
        ax=ax,
        pad=0.025,
    )

    colourbar.set_label(
        "Geometric mean fitness"
    )

    ax.fill_between(
        centres[valid],
        lower[valid],
        upper[valid],
        color="0.55",
        alpha=0.28,
        linewidth=0,
        label=(
            "SEM within each interval"
        ),
    )

    ax.plot(
        centres[valid],
        means[valid],
        color="black",
        lw=3.0,
        marker="o",
        ms=7,
        label=(
            "Mean within each interval"
        ),
    )

    ax.set_xlabel(
        r"Difference between rates, "
        r"$\log_{10}(m/p)$"
    )

    ax.set_ylabel(
        "Geometric mean fitness\n"
        "(fraction of the maximum at the same p)"
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    floating_axes(
        ax
    )

    set_later_figure_axis_text(ax)
    colourbar.ax.yaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    colourbar.ax.tick_params(labelsize=LATER_FIGURE_TICK_FS)

    fig.tight_layout()

    save_figure(
        fig,
        (
            "Figure_3.6_A_Broad_Optimum_in_"
            "Mis-Segregation_Rate_Across_"
            "Environmental_Switching_Regimes.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 3.7
# --------------------------------------------------------------------------- #

def load_or_calculate_continuous_resilience(
    repeats=500,
):
    """Preserve the original Figure 3.7 stress test exactly."""
    rates = np.logspace(
        -7,
        -0.5,
        18,
    )

    generations = 1_000
    population_size = 20
    fitness_cost = 0.99
    switch_rate = 0.05

    if (
        CONTINUOUS_RESILIENCE_CACHE.exists()
        and not FORCE_RERUN
    ):
        z = np.load(
            CONTINUOUS_RESILIENCE_CACHE
        )

        valid = (
            int(
                z["repeats"]
            )
            == repeats
            and int(
                z["generations"]
            )
            == generations
            and int(
                z["population_size"]
            )
            == population_size
            and float(
                z["fitness_cost"]
            )
            == fitness_cost
            and float(
                z["switch_rate"]
            )
            == switch_rate
            and np.array_equal(
                z["rates"],
                rates,
            )
        )

        if valid:
            print(
                f"Using cached resilience analysis: "
                f"{CONTINUOUS_RESILIENCE_CACHE}"
            )

            return (
                z["diversity"],
                z["survived"],
                rates,
            )

    diversity = np.zeros(
        (
            len(rates),
            repeats,
        )
    )

    survived = np.zeros_like(
        diversity
    )

    master = np.random.default_rng(
        RANDOM_SEED
        + 909
    )

    for row, rate in enumerate(
        rates
    ):
        print(
            f"Calculating resilience rate "
            f"{row + 1}/{len(rates)}"
        )

        for col in range(
            repeats
        ):
            result = run_once(
                rate,
                switch_rate,
                np.random.default_rng(
                    master.integers(
                        0,
                        2**63 - 1,
                    )
                ),
                population_size=population_size,
                fitness_cost=fitness_cost,
                generations=generations,
            )

            diversity[
                row,
                col,
            ] = result[2]

            survived[
                row,
                col,
            ] = (
                0.0
                if result[4]
                else 1.0
            )

    np.savez(
        CONTINUOUS_RESILIENCE_CACHE,
        diversity=diversity,
        survived=survived,
        rates=rates,
        repeats=repeats,
        generations=generations,
        population_size=population_size,
        fitness_cost=fitness_cost,
        switch_rate=switch_rate,
    )

    return (
        diversity,
        survived,
        rates,
    )


def make_figure_3_7():
    (
        diversity,
        survived,
        rates,
    ) = (
        load_or_calculate_continuous_resilience()
    )

    mean_diversity = diversity.mean(
        axis=1
    )

    mean_survival = survived.mean(
        axis=1
    )

    survival_sem = (
        survived.std(
            axis=1,
            ddof=1,
        )
        / np.sqrt(
            survived.shape[1]
        )
    )

    batch_size = 5

    batch_diversity = (
        diversity.reshape(
            len(rates),
            -1,
            batch_size,
        )
        .mean(
            axis=2
        )
    )

    batch_survival = (
        survived.reshape(
            len(rates),
            -1,
            batch_size,
        )
        .mean(
            axis=2
        )
    )

    fig, ax = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE
    )

    points = ax.scatter(
        batch_diversity.ravel(),
        batch_survival.ravel(),
        c=np.repeat(
            np.log10(
                rates
            ),
            batch_diversity.shape[1],
        ),
        cmap=RATE_DIVERGING_CMAP_AII,
        vmin=-7,
        vmax=0,
        s=34,
        alpha=1.0,
        edgecolors="none",
        rasterized=True,
        label=(
            "Five-run batch estimates"
        ),
    )

    order = np.argsort(
        mean_diversity
    )

    ax.fill_between(
        mean_diversity[order],
        np.maximum(
            mean_survival[order]
            - survival_sem[order],
            0,
        ),
        np.minimum(
            mean_survival[order]
            + survival_sem[order],
            1,
        ),
        color="0.55",
        alpha=0.28,
        linewidth=0,
        label=(
            "SEM across simulations"
        ),
    )

    ax.plot(
        mean_diversity[order],
        mean_survival[order],
        color="black",
        lw=3.0,
        marker="o",
        ms=7,
        label=(
            "Mean at each "
            "mis-segregation rate"
        ),
    )

    colourbar = fig.colorbar(
        points,
        ax=ax,
        pad=0.025,
    )

    colourbar.set_label(
        "Mis-segregation rate, "
        "log₁₀(m)"
    )

    ax.set_xlabel(
        "Mean ploidy diversity "
        "(Shannon entropy)"
    )

    ax.set_ylabel(
        "Extinction resilience\n"
        "(probability of survival)"
    )

    ax.set_ylim(
        -0.04,
        1.04,
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    floating_axes(
        ax
    )

    set_later_figure_axis_text(ax)
    colourbar.ax.yaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    colourbar.ax.tick_params(labelsize=LATER_FIGURE_TICK_FS)

    fig.tight_layout()

    save_figure(
        fig,
        (
            "Figure_3.7_Ploidy_Diversity_Increases_"
            "Population_Resilience_to_Extinction.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 3.8
# --------------------------------------------------------------------------- #

def evolve_rate_once(
    generations,
    switch_probabilities,
    seed,
    initial_rate,
    record_every=10,
    fitness_cost=FITNESS_COST,
):
    """Preserve the original heritable-rate extension exactly."""
    rng = np.random.default_rng(
        seed
    )

    state = np.zeros(
        (
            len(
                EVOLVING_RATE_GRID
            ),
            4,
        ),
        dtype=float,
    )

    start = int(
        np.argmin(
            np.abs(
                np.log(
                    EVOLVING_RATE_GRID
                )
                - np.log(
                    initial_rate
                )
            )
        )
    )

    state[
        start,
        1:3,
    ] = 0.5

    environment = 0.5
    recorded = []
    recorded_generations = []

    log_rates = np.log(
        EVOLVING_RATE_GRID
    )

    probabilities = (
        np.full(
            generations,
            float(
                switch_probabilities
            ),
        )
        if np.isscalar(
            switch_probabilities
        )
        else np.asarray(
            switch_probabilities,
            dtype=float,
        )
    )

    if (
        len(probabilities)
        != generations
    ):
        raise ValueError(
            "One environmental switching "
            "probability is required "
            "per generation"
        )

    for generation, p in enumerate(
        probabilities,
        start=1,
    ):
        environment = switch_environment(
            rng,
            environment,
            float(p),
        )

        selected = state.copy()

        if environment == -0.5:
            selected[
                :,
                0:2,
            ] *= (
                1.0
                - fitness_cost
            )

        else:
            selected[
                :,
                2:4,
            ] *= (
                1.0
                - fitness_cost
            )

        m = EVOLVING_RATE_GRID[
            :,
            None,
        ]

        h, d, tr, te = selected.T

        offspring = np.column_stack(
            (
                (
                    2.0
                    * (
                        1.0
                        - m[:, 0]
                    )
                    * h
                    + m[:, 0]
                    * d
                ),
                (
                    2.0
                    * (
                        1.0
                        - m[:, 0]
                    )
                    * d
                    + m[:, 0]
                    * h
                    + m[:, 0]
                    * tr
                ),
                (
                    2.0
                    * (
                        1.0
                        - m[:, 0]
                    )
                    * tr
                    + m[:, 0]
                    * d
                    + m[:, 0]
                    * te
                ),
                (
                    2.0
                    * (
                        1.0
                        - m[:, 0]
                    )
                    * te
                    + m[:, 0]
                    * tr
                ),
            )
        )

        eta = (
            RATE_MUTATION_PROBABILITY
        )

        mutated = (
            offspring
            * (
                1.0
                - eta
            )
        )

        mutated[
            1:
        ] += (
            offspring[
                :-1
            ]
            * (
                eta
                / 2.0
            )
        )

        mutated[
            :-1
        ] += (
            offspring[
                1:
            ]
            * (
                eta
                / 2.0
            )
        )

        mutated[0] += (
            offspring[0]
            * (
                eta
                / 2.0
            )
        )

        mutated[-1] += (
            offspring[-1]
            * (
                eta
                / 2.0
            )
        )

        total = mutated.sum()

        state = (
            mutated
            / total
            if total > 0
            else mutated
        )

        if (
            generation == 1
            or generation
            % record_every
            == 0
            or generation
            == generations
        ):
            weights = state.sum(
                axis=1
            )

            recorded.append(
                float(
                    np.exp(
                        np.sum(
                            weights
                            * log_rates
                        )
                    )
                )
            )

            recorded_generations.append(
                generation
            )

    return (
        np.asarray(
            recorded_generations
        ),
        np.asarray(
            recorded
        ),
    )


def fixed_environment_evolution(
    repeats=24,
    generations=1_000,
):
    probabilities = np.concatenate(
        (
            [0.0],
            np.logspace(
                -3,
                0,
                20,
            ),
        )
    )

    low_start = 1e-6
    high_start = 0.5
    focal_p = 0.05

    if (
        EVOLVING_RATE_CACHE.exists()
        and not FORCE_RERUN
    ):
        z = np.load(
            EVOLVING_RATE_CACHE
        )

        if (
            int(
                z["repeats"]
            )
            == repeats
            and int(
                z["generations"]
            )
            == generations
        ):
            print(
                f"Using cached evolving-rate analysis: "
                f"{EVOLVING_RATE_CACHE}"
            )

            return (
                z["times"],
                z["probabilities"],
                z["surface"],
                z["low"],
                z["high"],
                float(
                    z["focal_p"]
                ),
            )

    master = np.random.default_rng(
        RANDOM_SEED
        + 1201
    )

    surface_runs = []
    low_runs = []
    high_runs = []
    times = None

    for p in probabilities:
        runs = []

        for _ in range(
            repeats
        ):
            times, values = evolve_rate_once(
                generations,
                p,
                master.integers(
                    0,
                    2**63 - 1,
                ),
                1e-4,
            )

            runs.append(
                np.log10(
                    values
                )
            )

        surface_runs.append(
            np.mean(
                runs,
                axis=0,
            )
        )

    for initial, destination in (
        (
            low_start,
            low_runs,
        ),
        (
            high_start,
            high_runs,
        ),
    ):
        for _ in range(
            repeats
        ):
            _, values = evolve_rate_once(
                generations,
                focal_p,
                master.integers(
                    0,
                    2**63 - 1,
                ),
                initial,
            )

            destination.append(
                np.log10(
                    values
                )
            )

    surface = np.asarray(
        surface_runs
    ).T

    low = np.asarray(
        low_runs
    )

    high = np.asarray(
        high_runs
    )

    np.savez(
        EVOLVING_RATE_CACHE,
        times=times,
        probabilities=probabilities,
        surface=surface,
        low=low,
        high=high,
        focal_p=focal_p,
        repeats=repeats,
        generations=generations,
    )

    return (
        times,
        probabilities,
        surface,
        low,
        high,
        focal_p,
    )


def make_figure_3_8():
    (
        times,
        probabilities,
        surface,
        low,
        high,
        focal_p,
    ) = (
        fixed_environment_evolution()
    )

    plotted_mask = probabilities < 1.0
    plotted_probabilities = probabilities[plotted_mask]
    plotted_surface = surface[:, plotted_mask]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=TWO_PANEL_FIGSIZE,
        gridspec_kw={
            "width_ratios": [
                1.15,
                1.0,
            ],
            "wspace": 0.34,
        },
    )

    ax_a, ax_b = axes

    rate_norm = (
        matplotlib.colors.Normalize(
            vmin=-7,
            vmax=0,
        )
    )

    image = ax_a.imshow(
        plotted_surface,
        origin="lower",
        aspect="auto",
        extent=[
            0,
            len(plotted_probabilities) - 1,
            times[0],
            times[-1],
        ],
        cmap=RATE_DIVERGING_CMAP_AII,
        norm=rate_norm,
        interpolation="nearest",
        alpha=1.0,
    )

    tick_indices = [
        0,
        7,
        13,
        18,
    ]

    ax_a.set_xticks(
        tick_indices,
        [
            "Constant",
            "0.01",
            "0.1",
            "0.5",
        ],
    )

    ax_a.set_xlabel(
        "Environmental switching probability, p"
    )

    ax_a.set_ylabel(
        "Generation"
    )

    add_panel_label(
        ax_a,
        "A",
    )

    colourbar = fig.colorbar(
        image,
        ax=ax_a,
        pad=0.025,
    )

    colourbar.set_label(
        "Mean evolved rate, log₁₀(m)"
    )

    ax_a.axhline(
        times[0],
        color="white",
        ls="--",
        lw=1.3,
    )

    ax_a.text(
        0.03,
        0.04,
        "All populations start at m = 10⁻⁴",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE[
            "annotation_fs"
        ],
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            pad=2.0,
        ),
    )

    floating_axes(
        ax_a
    )

    low_colour = "#4C78A8"
    high_colour = "#D1495B"

    for values, colour, label in (
        (
            low,
            low_colour,
            "Starts at low m",
        ),
        (
            high,
            high_colour,
            "Starts at high m",
        ),
    ):
        mean = values.mean(
            axis=0
        )

        sem = (
            values.std(
                axis=0,
                ddof=1,
            )
            / np.sqrt(
                values.shape[0]
            )
        )

        ax_b.fill_between(
            times,
            mean - sem,
            mean + sem,
            color=colour,
            alpha=0.18,
            linewidth=0,
        )

        ax_b.plot(
            times,
            mean,
            color=colour,
            lw=2.8,
            label=label,
        )

    ax_b.axhline(
        np.log10(
            focal_p
        ),
        color="0.25",
        ls="--",
        lw=1.6,
        label="Environmental rate",
    )

    ax_b.set_xlabel(
        "Generation"
    )

    ax_b.set_ylabel(
        "Mean heritable rate, log₁₀(m)"
    )

    add_panel_label(
        ax_b,
        "B",
    )

    ax_b.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(
            1.0,
            0.03,
        ),
        fontsize=23,
        alignment="right",
    )

    floating_axes(
        ax_b
    )

    set_later_figure_axis_text(ax_a)
    set_later_figure_axis_text(ax_b)
    colourbar.ax.yaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    colourbar.ax.tick_params(labelsize=LATER_FIGURE_TICK_FS)

    fig.subplots_adjust(
        top=0.96,
        bottom=0.17,
        left=0.09,
        right=0.96,
    )

    save_figure(
        fig,
        (
            "Figure_3.8_Environmental_Fluctuation_"
            "Selects_a_Heritable_Rate_of_"
            "Karyotypic_Randomisation.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Figure 3.9
# --------------------------------------------------------------------------- #

def fitness_cost_step_tracking_analysis(
    repeats=40,
):
    """Preserve the original six-block tracking analysis exactly."""
    step_values = np.asarray(
        [
            0.005,
            0.02,
            0.08,
            0.20,
            0.01,
            0.05,
        ]
    )

    step_length = 2_000

    schedule = np.repeat(
        step_values,
        step_length,
    )

    if (
        FITNESS_COST_TRACKING_CACHE.exists()
        and not FORCE_RERUN
    ):
        z = np.load(
            FITNESS_COST_TRACKING_CACHE
        )

        valid = (
            int(
                z["repeats"]
            )
            == repeats
            and np.array_equal(
                z["fitness_costs"],
                FITNESS_COST_VALUES,
            )
            and np.array_equal(
                z["schedule"],
                schedule,
            )
        )

        if valid:
            print(
                f"Using cached fitness-cost tracking analysis: "
                f"{FITNESS_COST_TRACKING_CACHE}"
            )

            return {
                name: z[name]
                for name
                in z.files
                if name
                != "repeats"
            }

    master = np.random.default_rng(
        RANDOM_SEED
        + 4301
    )

    runs_by_cost = []
    times = None

    for cost in FITNESS_COST_VALUES:
        runs = []

        for _ in range(
            repeats
        ):
            times, rates = evolve_rate_once(
                len(
                    schedule
                ),
                schedule,
                master.integers(
                    0,
                    2**63 - 1,
                ),
                1e-4,
                record_every=20,
                fitness_cost=float(
                    cost
                ),
            )

            runs.append(
                rates
            )

        runs_by_cost.append(
            runs
        )

    runs_by_cost = np.asarray(
        runs_by_cost
    )

    sampled_p = schedule[
        times - 1
    ]

    np.savez(
        FITNESS_COST_TRACKING_CACHE,
        times=times,
        runs=runs_by_cost,
        sampled_p=sampled_p,
        schedule=schedule,
        fitness_costs=FITNESS_COST_VALUES,
        repeats=repeats,
    )

    return {
        "times": times,
        "runs": runs_by_cost,
        "sampled_p": sampled_p,
        "schedule": schedule,
        "fitness_costs": FITNESS_COST_VALUES,
    }


def make_figure_3_9():
    data = (
        fitness_cost_step_tracking_analysis()
    )

    fig, rate_ax = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE
    )

    for row, (
        cost,
        colour,
    ) in enumerate(
        zip(
            data[
                "fitness_costs"
            ],
            FITNESS_COST_COLOURS,
        )
    ):
        log_runs = np.log10(
            data[
                "runs"
            ][row]
        )

        mean_log = log_runs.mean(
            axis=0
        )

        sd_log = log_runs.std(
            axis=0,
            ddof=1,
        )

        mean = (
            10
            ** mean_log
        )

        lower = (
            10
            ** (
                mean_log
                - sd_log
            )
        )

        upper = (
            10
            ** (
                mean_log
                + sd_log
            )
        )

        rate_ax.fill_between(
            data[
                "times"
            ],
            lower,
            upper,
            color=colour,
            alpha=0.16,
            linewidth=0,
        )

        rate_ax.plot(
            data[
                "times"
            ],
            mean,
            color=colour,
            lw=2.5,
            label=(
                f"Fitness cost = "
                f"{100 * cost:.0f}%"
            ),
        )

    rate_ax.set_yscale(
        "log"
    )

    rate_ax.set_xlabel(
        "Generation"
    )

    rate_ax.set_ylabel(
        "Mean evolved mis-segregation rate"
    )

    floating_axes(
        rate_ax
    )

    environment_ax = (
        rate_ax.twinx()
    )

    environment_ax.step(
        data[
            "times"
        ],
        data[
            "sampled_p"
        ],
        where="post",
        color="0.30",
        lw=1.8,
        ls="--",
        label=(
            "Environmental switching "
            "probability, p"
        ),
    )

    environment_ax.set_yscale(
        "log"
    )

    environment_ax.set_ylabel(
        "Environmental switching probability, p",
        color="0.30",
    )

    environment_ax.tick_params(
        axis="y",
        colors="0.30",
        labelsize=STYLE[
            "tick_fs"
        ],
    )

    environment_ax.grid(
        False
    )

    environment_ax.spines[
        "top"
    ].set_visible(
        False
    )

    environment_ax.spines[
        "bottom"
    ].set_visible(
        False
    )

    environment_ax.spines[
        "left"
    ].set_visible(
        False
    )

    environment_ax.spines[
        "right"
    ].set_visible(
        True
    )

    environment_ax.spines[
        "right"
    ].set_position(
        (
            "outward",
            6,
        )
    )

    environment_ax.spines[
        "right"
    ].set_color(
        "0.30"
    )

    environment_ax.yaxis.set_label_position(
        "right"
    )

    environment_ax.yaxis.tick_right()

    (
        rate_handles,
        rate_labels,
    ) = rate_ax.get_legend_handles_labels()

    (
        env_handles,
        env_labels,
    ) = environment_ax.get_legend_handles_labels()

    rate_ax.legend(
        rate_handles
        + env_handles,
        rate_labels
        + env_labels,
        frameon=False,
        ncol=2,
        loc="lower right",
        fontsize=19.4,
    )

    set_later_figure_axis_text(rate_ax)
    environment_ax.yaxis.label.set_fontsize(LATER_FIGURE_AXIS_FS)
    environment_ax.tick_params(
        axis="y",
        labelsize=LATER_FIGURE_TICK_FS,
    )

    fig.subplots_adjust(
        top=0.97,
        bottom=0.14,
        left=0.13,
        right=0.85,
    )

    save_figure(
        fig,
        (
            "Figure_3.9_Selection_Strength_Determines_"
            "How_Closely_the_Evolved_Mis-Segregation_"
            "Rate_Tracks_Environmental_Change.png"
        ),
    )


# --------------------------------------------------------------------------- #
# Source-data exports
# --------------------------------------------------------------------------- #

def write_main_surface_csv(
    growth,
    variance,
    entropy,
    extinction,
):
    path = (
        ADDITIONAL_DIR
        / "Figures_3.4_to_3.6_Main_Parameter_Sweep.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "mis-segregation rate",
                "environmental switching rate",
                "geometric mean fitness",
                "temporal variance",
                "Shannon entropy",
                "extinction probability",
            ]
        )

        for row, m in enumerate(
            MISSEGREGATION_RATES
        ):
            for col, p in enumerate(
                SWITCH_RATES
            ):
                writer.writerow(
                    [
                        m,
                        p,
                        growth[
                            row,
                            col,
                        ],
                        variance[
                            row,
                            col,
                        ],
                        entropy[
                            row,
                            col,
                        ],
                        extinction[
                            row,
                            col,
                        ],
                    ]
                )

    print(
        f"Source data written: "
        f"{path}"
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv=None):
    global FORCE_RERUN

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "recompute cached "
            "numerical analyses"
        ),
    )

    args = parser.parse_args(
        argv
    )

    FORCE_RERUN = (
        args.force
    )

    apply_style()

    make_figure_3_2()
    make_figure_3_3()

    (
        growth,
        variance,
        entropy,
        extinction,
    ) = load_or_calculate_surfaces()

    write_main_surface_csv(
        growth,
        variance,
        entropy,
        extinction,
    )

    make_figure_3_4(
        growth
    )

    make_figure_3_5(
        growth
    )

    make_figure_3_6(
        growth
    )

    make_figure_3_7()
    make_figure_3_8()
    make_figure_3_9()

    print(
        f"Figures: "
        f"{FIGURE_DIR}"
    )

    print(
        f"Caches and source data: "
        f"{ADDITIONAL_DIR}"
    )


if __name__ == "__main__":
    main()