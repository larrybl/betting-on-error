"""Reproduce the Chapter 3 MAD2 competition figures and statistics.

Outputs
-------
Figure 3.10. Genotypes used in the preliminary mad2Δ competition
Figure 3.11. Environmental regime determines the competitive outcome of mad2Δ

The script also prints the summary statistics and hypothesis tests reported for
this preliminary competition experiment. Figure files are written to ``MAD2_GitHubReady_FINAL Figures``
as 300-dpi PNG files.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.stats import kruskal, mannwhitneyu


# -----------------------------------------------------------------------------
# Output and plotting configuration
# -----------------------------------------------------------------------------
FIGDIR = Path("MAD2_GitHubReady_FINAL Figures")
SAVE_FORMATS = ("png",)

# Two-point increase relative to the previous thesis plotting settings.
plt.rcParams.update({
    "font.size": 29,
    "axes.titlesize": 31,
    "axes.labelsize": 29,
    "xtick.labelsize": 29,
    "ytick.labelsize": 29,
    "legend.fontsize": 25,
})

HOT_COLOR = "#FF1744"
COLD_COLOR = "#2196F3"
BASE_COLOR = "#757575"


# -----------------------------------------------------------------------------
# Experimental data
# -----------------------------------------------------------------------------
CONSTANT_30 = "Constant 30 °C"
CONSTANT_39 = "Constant 39 °C"
FIVE_DAY_CYCLE = "Five-day cycle"

DAYS = np.array([0, 6, 12, 18, 24], dtype=int)
ALL_DAYS = np.arange(0, 25)

RAW_DATA = {
    CONSTANT_30: {
        0: [50, 50, 50],
        6: [29, 33, 26],
        12: [16, 19, 19],
        18: [7, 9, 12],
        24: [0, 0, 0],
    },
    CONSTANT_39: {
        0: [50, 50, 50],
        6: [43, 39, 45],
        12: [32, 29, 43],
        18: [11, 17, 18],
        24: [0, 0, 1],
    },
    FIVE_DAY_CYCLE: {
        0: [50, 50, 50],
        6: [46, 66, 71],
        12: [29, 74, 76],
        18: [9, 89, 88],
        24: [0, 100, 100],
    },
}

REGIME_ORDER = [CONSTANT_30, CONSTANT_39, FIVE_DAY_CYCLE]


def make_periodic_regime(period_days: int, n_days: int = 24) -> list[str]:
    """Return day-0 baseline followed by one hot day every ``period_days``."""
    return ["B"] + ["H" if day % period_days == 0 else "C" for day in range(n_days)]


THERMAL_MAP = {
    CONSTANT_30: ["B"] + ["C"] * 24,
    CONSTANT_39: ["B"] + ["H"] * 24,
    FIVE_DAY_CYCLE: make_periodic_regime(5),  # one 39 °C day followed by four 30 °C days
}


def state_color(state: str) -> str:
    if state == "H":
        return HOT_COLOR
    if state == "C":
        return COLD_COLOR
    return BASE_COLOR


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure atomically in the requested formats."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in SAVE_FORMATS:
        target = FIGDIR / f"{stem}.{ext}"
        temporary = FIGDIR / f".{stem}.{ext}.tmp"
        try:
            fig.savefig(
                temporary,
                format=ext,
                dpi=300 if ext == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
            if temporary.stat().st_size == 0:
                raise OSError(f"Empty figure output: {temporary}")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"saved {target}")


def get_stats(regime_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, sds, ns = [], [], []
    for day in DAYS:
        values = np.asarray(RAW_DATA[regime_name][int(day)], dtype=float)
        means.append(values.mean())
        sds.append(values.std(ddof=1))
        ns.append(values.size)
    return np.asarray(means), np.asarray(sds), np.asarray(ns)


# -----------------------------------------------------------------------------
# Figure 3.11: competition trajectories
# -----------------------------------------------------------------------------
def plot_competition_trajectories(ax: plt.Axes) -> None:
    """Plot mean mad2Δ frequency with ±1 SD and temperature-coloured segments."""
    endpoint_labels = []

    for regime_name in REGIME_ORDER:
        means, sds, ns = get_stats(regime_name)
        thermal_regime = THERMAL_MAP[regime_name]

        mean_daily = interp1d(DAYS, means, kind="linear")(ALL_DAYS)
        sd_daily = interp1d(DAYS, sds, kind="linear")(ALL_DAYS)

        segments, segment_colours = [], []
        for i in range(len(ALL_DAYS) - 1):
            x_segment = ALL_DAYS[i:i + 2]
            y_segment = mean_daily[i:i + 2]
            segment_colour = state_color(thermal_regime[i + 1])

            segments.append([(x_segment[0], y_segment[0]), (x_segment[1], y_segment[1])])
            segment_colours.append(segment_colour)
            ax.fill_between(
                x_segment,
                np.clip(y_segment - sd_daily[i:i + 2], 0, 100),
                np.clip(y_segment + sd_daily[i:i + 2], 0, 100),
                color=segment_colour,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )

        ax.add_collection(LineCollection(segments, colors=segment_colours, linewidths=3.4))

        for i, day in enumerate(DAYS):
            marker_colour = state_color(thermal_regime[int(day)])
            ax.plot(
                day,
                means[i],
                "o",
                color=marker_colour,
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=3,
            )

        if len(set(thermal_regime[1:])) > 1:
            for k in range(2, len(thermal_regime)):
                if thermal_regime[k] != thermal_regime[k - 1]:
                    transition_day = k - 1
                    ax.plot(
                        transition_day,
                        mean_daily[transition_day],
                        "o",
                        color=state_color(thermal_regime[k]),
                        markersize=9,
                        markeredgecolor="white",
                        markeredgewidth=1.4,
                        zorder=4,
                    )

        endpoint_labels.append((means[-1], int(ns[-1])))

    # Offset overlapping n labels while keeping them close to their endpoints.
    used_positions = []
    for endpoint, n in sorted(endpoint_labels, key=lambda item: item[0]):
        y = endpoint
        while any(abs(y - used) < 8 for used in used_positions):
            y += 8
        used_positions.append(y)
        ax.text(24.35, y, f"n={n}", fontsize=23, va="center", ha="left")

    ax.set_xlabel("Day")
    ax.set_ylabel(r"$\mathit{mad2\Delta}$ strain frequency (%)")
    ax.set_xticks(DAYS)
    ax.set_ylim(-5, 110)
    ax.set_xlim(-0.5, 27)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    legend = [
        Line2D([0], [0], color=COLD_COLOR, lw=3.4, marker="o", markersize=9,
               label="30 °C"),
        Line2D([0], [0], color=HOT_COLOR, lw=3.4, marker="o", markersize=9,
               label="39 °C"),
    ]
    ax.legend(handles=legend, frameon=False, loc="best")


def make_figure_3_11() -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_competition_trajectories(ax)
    fig.tight_layout()
    save_figure(
        fig,
        "Figure_3.11_Environmental_regime_determines_the_competitive_outcome_of_mad2del",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3.10: genotype table
# -----------------------------------------------------------------------------
HEADER_BG = "#D9E7F2"
HEADER_FG = "#263746"
ROW_MUTANT_BG = "#F8E1DA"
ROW_WT_BG = "#DDEEDF"
DIFF_BORDER = "#C98578"
SAME_BORDER = "#B7C3CC"
GRID_LW = 0.8


def draw_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    background: str,
    *,
    foreground: str = "black",
    fontsize: int = 14,
    border_color: str = SAME_BORDER,
    border_width: float = GRID_LW,
    bold: bool = False,
) -> None:
    ax.add_patch(
        plt.Rectangle(
            (x, y), width, height,
            facecolor=background,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=1,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=foreground,
        weight="bold" if bold else "normal",
        zorder=2,
        multialignment="center",
        linespacing=1.3,
    )


def make_figure_3_10() -> None:
    columns = [
        "Strain",
        "Background",
        "Mating type",
        r"$\mathbfit{URA3}$ locus",
        r"$\mathbfit{HIS3}$ locus",
        r"$\mathbfit{LEU2}$ locus",
        r"$\mathbfit{LYS2}$ locus",
        r"$\mathbfit{MAD2}$ locus",
    ]

    # BY4742 genotype as stated in the thesis Methods: MATα ura3-0 his3-0 leu2-0 lys2-0.
    mutant_row = [
        r"$\mathit{mad2\Delta}$",
        "BY4742",
        "MATα",
        r"$\mathit{ura3-0}$",
        r"$\mathit{his3-0}$",
        r"$\mathit{leu2-0}$",
        r"$\mathit{lys2-0}$",
        r"$\mathit{mad2\Delta::kanMX}$",
    ]
    wt_row = [
        "WT BY4742",
        "BY4742",
        "MATα",
        r"$\mathit{ura3-0}$",
        r"$\mathit{his3-0}$",
        r"$\mathit{leu2-0}$",
        r"$\mathit{lys2-0}$",
        r"WT $\mathit{MAD2}$",
    ]

    data = [mutant_row, wt_row]
    col_widths = [2.5, 1.5, 1.5, 1.6, 1.6, 1.6, 1.6, 2.8]
    differing = [i for i in range(len(columns)) if data[0][i] != data[1][i]]

    row_height = 1.3
    total_width = sum(col_widths) + 0.4
    fig_height = row_height * 3.25 + 0.9
    fig, ax = plt.subplots(figsize=(total_width, fig_height))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    x_positions = [0.2]
    for width in col_widths[:-1]:
        x_positions.append(x_positions[-1] + width)

    header_y = fig_height - row_height - 0.18
    for ci, (label, x, width) in enumerate(zip(columns, x_positions, col_widths)):
        is_different = ci in differing
        draw_cell(
            ax, x, header_y, width, row_height, label, HEADER_BG,
            foreground=HEADER_FG,
            fontsize=15,
            border_color=DIFF_BORDER if is_different else SAME_BORDER,
            border_width=2.0 if is_different else GRID_LW,
            bold=True,
        )

    row_backgrounds = [ROW_MUTANT_BG, ROW_WT_BG]
    for ri, row in enumerate(data):
        y = header_y - (ri + 1) * row_height
        for ci, (value, x, width) in enumerate(zip(row, x_positions, col_widths)):
            is_different = ci in differing
            draw_cell(
                ax, x, y, width, row_height, value, row_backgrounds[ri],
                fontsize=14,
                border_color=DIFF_BORDER if is_different else SAME_BORDER,
                border_width=2.2 if is_different else GRID_LW,
            )

    ax.text(
        total_width / 2,
        0.10,
        "Red borders indicate loci that differ between strains.",
        ha="center",
        va="bottom",
        fontsize=14,
        color="#757575",
        style="italic",
    )

    fig.tight_layout(pad=0.25)
    save_figure(fig, "Figure_3.10_Genotypes_used_in_the_preliminary_mad2del_competition")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Statistical summaries reported in Chapter 3
# -----------------------------------------------------------------------------
def print_statistics() -> None:
    print("\nMAD2 competition summary")
    print("=" * 72)

    for regime_name in REGIME_ORDER:
        means, sds, ns = get_stats(regime_name)
        print(f"\n{regime_name}")
        print(f"{'day':>5}{'mean':>10}{'SD':>10}{'n':>5}   replicates")
        for i, day in enumerate(DAYS):
            reps = RAW_DATA[regime_name][int(day)]
            print(f"{day:>5}{means[i]:>10.1f}{sds[i]:>10.2f}{ns[i]:>5}   {reps}")

    print("\nHierarchical tests")
    print("=" * 72)
    for day in [6, 12, 18, 24]:
        groups = [np.asarray(RAW_DATA[name][day], dtype=float) for name in REGIME_ORDER]
        all_values = np.concatenate(groups)
        print(f"\nDay {day}")

        if len(np.unique(all_values)) > 1:
            h_stat, p_kw = kruskal(*groups)
            print(f"Kruskal-Wallis: H={h_stat:.4f}, p={p_kw:.6f}")
            if p_kw < 0.05:
                for i in range(len(REGIME_ORDER)):
                    for j in range(i + 1, len(REGIME_ORDER)):
                        u_pair, p_pair = mannwhitneyu(
                            groups[i], groups[j], alternative="two-sided"
                        )
                        print(
                            f"Pairwise {REGIME_ORDER[i]} vs {REGIME_ORDER[j]}: "
                            f"U={u_pair:.1f}, p={p_pair:.6f}"
                        )
        else:
            print("Kruskal-Wallis: not defined because all values are identical")

        cycling = np.asarray(RAW_DATA[FIVE_DAY_CYCLE][day], dtype=float)
        constants = np.concatenate([
            np.asarray(RAW_DATA[CONSTANT_30][day], dtype=float),
            np.asarray(RAW_DATA[CONSTANT_39][day], dtype=float),
        ])
        u_stat, p_planned = mannwhitneyu(cycling, constants, alternative="two-sided")
        print(
            "Planned comparison, five-day cycle vs pooled constant regimes: "
            f"U={u_stat:.1f}, p={p_planned:.6f}"
        )

    day0 = np.asarray(RAW_DATA[FIVE_DAY_CYCLE][0], dtype=float)
    day24 = np.asarray(RAW_DATA[FIVE_DAY_CYCLE][24], dtype=float)
    u_stat, p_value = mannwhitneyu(day24, day0, alternative="two-sided")
    print(
        "\nFive-day cycle day 24 vs day 0, two-sided Mann-Whitney: "
        f"U={u_stat:.1f}, p={p_value:.6f}"
    )


if __name__ == "__main__":
    make_figure_3_10()
    make_figure_3_11()
    print_statistics()
