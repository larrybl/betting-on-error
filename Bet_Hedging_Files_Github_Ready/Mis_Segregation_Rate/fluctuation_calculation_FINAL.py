#!/usr/bin/env python3
"""Reproduce thesis Figure 4.5 and export its source data.

The script performs the Ma-Sandri-Sarkar maximum-likelihood fluctuation
analysis with the Stewart partial-plating correction used in the thesis.
It generates only Figure 4.5, the current thesis figure showing the three
engineered CEN3 alleles and chromosome III loss rates at 30 °C and 39 °C.

No external input files are required. Source-data CSV files are written to the
script-specific ``Additional Files`` directory and the figure is written as a
300-dpi PNG to the script-specific ``Figures`` directory.

Run with:
    python fluctuation_calculation_FINAL.py

Use ``--check`` to run the estimator validation instead of generating outputs.
"""

import sys
import csv
from pathlib import Path
import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
ADDITIONAL_DIR = HERE / f"{SCRIPT_STEM} Additional Files"
FIGURE_DIR = HERE / f"{SCRIPT_STEM} Figures"

STYLE = {
    "dpi": 300,
    "base_fs": 29,
    "axis_fs": 29,
    "tick_fs": 29,
    "legend_fs": 25,
    "annotation_fs": 23,
    "panel_fs": 29,
}


# =============================================================================
# INPUT
# =============================================================================
PLATE_VOL_ML   = 0.100      # volume plated on 5-FOA
CULTURE_VOL_ML = 5.0        # volume of the culture it came from
YEPD_DILUTION  = 1e5        # dilution for the viable count
YEPD_VOL_ML    = 0.100      # volume of that dilution plated on YEPD

EPSILON = PLATE_VOL_ML / CULTURE_VOL_ML          # plating efficiency = 0.02

# strain -> dict(yepd = colonies on the 1e5 dilution, foa = 5-FOA counts)
DATA = {
    "30C": {
        "1649 WT CEN": dict(
            yepd=120,
            foa=[60, 42, 65, 39, 48, 59, 45, 54, 39, 22, 38, 41, 43, 53, 44],
        ),
        "1647 CDEII-del": dict(
            yepd=112,
            foa=[294, 311, 257, 264, 389, 399, 381, 423, 212, 342, 204, 157,
                 206, 260, 376],
        ),
        "1648 CDEI&II-del": dict(
            yepd=83,
            foa=[2156, 1967, 2678, 3021, 1999, 1568, 999, 2567, 2853, 3402,
                 3653, 3683, 2011, 2431, 3012],
        ),
    },
    "39C": {
        "1649 WT CEN": dict(
            yepd=35,
            foa=[324, 525, 138],
        ),
        "1647 CDEII-del": dict(
            yepd=14,
            foa=[461, 294, 624],
        ),
        "1648 CDEI&II-del": dict(
            yepd=11,
            foa=[2100, 3087, 2876],
        ),
    },
}

TEMPERATURES = ["30C", "39C"]
REFERENCE = "1649 WT CEN"            # within-temperature reference strain
BASE_TEMP = "30C"                    # reference temperature for the T comparison

# Fraction of 5-FOA resistant colonies that were also nourseothricin sensitive,
# i.e. genuine whole-chromosome loss rather than recombination / ura3 point
# mutation. 1.0 means the counts above are already nat-adjusted.
# Keyed (temperature, strain) -- 39 C plates were not separately scored, so the
# 30 C fraction is carried over by default; edit here if they were.
# The 39 °C 5-FOA-resistant colonies were replica plated to nourseothricin
# and the replica plates contained no colonies. The complete 39 °C 5-FOA
# counts are therefore treated as putative chromosome-loss events. Values for
# 30 °C must be replaced here if its replica plates require an adjustment.
NAT_SENSITIVE_FRACTION = {(t, s): 1.0 for t in DATA for s in DATA[t]}

MIN_CULTURES_WARN = 5   # below this the profile CI is wide and asymmetric


def total_cells(yepd_count):
    return yepd_count * YEPD_DILUTION / YEPD_VOL_ML * CULTURE_VOL_ML


# =============================================================================
# LIKELIHOOD
# =============================================================================
def psi_coefficients(e, nmax):
    """Taylor coefficients of the partial-plating pgf exponent."""
    if not 0 < e <= 1:
        raise ValueError("plating efficiency must be in (0, 1]")
    if e == 1.0:
        c = np.empty(nmax + 1)
        c[0] = -1.0
        k = np.arange(1, nmax + 1)
        c[1:] = 1.0 / (k * (k + 1))
        return c

    a = np.empty(nmax + 1)
    a[0] = e * np.log(e)
    if nmax >= 1:
        a[1] = -e * (np.log(e) + 1.0)
    if nmax >= 2:
        k = np.arange(2, nmax + 1)
        a[2:] = e / (k * (k - 1))

    c = np.empty(nmax + 1)
    c[0] = a[0] / (1.0 - e)
    for n in range(1, nmax + 1):
        c[n] = (a[n] - e * c[n - 1]) / (1.0 - e)
    return c


def log_pmf(m, e, nmax, c=None):
    """log P(r = 0..nmax), convolution run in log space to avoid underflow."""
    if c is None:
        c = psi_coefficients(e, nmax)
    if m <= 0:
        out = np.full(nmax + 1, -np.inf)
        out[0] = 0.0
        return out

    k = np.arange(1, nmax + 1)
    with np.errstate(divide="ignore"):
        log_kc = np.log(k * c[1:])

    log_q = np.empty(nmax + 1)
    log_q[0] = 0.0
    log_m = np.log(m)
    for n in range(1, nmax + 1):
        t = log_kc[:n] + log_q[n - 1::-1][:n]
        mx = t.max()
        log_q[n] = (-np.inf if mx == -np.inf else
                    mx + np.log(np.exp(t - mx).sum()) + log_m - np.log(n))
    return m * c[0] + log_q


def loglik(m, counts, e, c=None, nmax=None):
    nmax = int(max(counts)) if nmax is None else nmax
    lp = log_pmf(m, e, nmax, c)
    return float(np.sum(lp[np.asarray(counts, dtype=int)]))


def mle(counts, e, tol=1e-5):
    """MSS maximum likelihood estimate of m, golden-section search on log m."""
    nmax = int(max(counts))
    c = psi_coefficients(e, nmax)
    f = lambda lm: -loglik(np.exp(lm), counts, e, c, nmax)

    lm0 = np.log(max(np.mean(counts), 1.0) / e)
    lo, hi = lm0 - 4.0, lm0 + 4.0
    while f(lo) < f(lo + 0.1) and lo > lm0 - 12:
        lo -= 1.0
    while f(hi) < f(hi - 0.1) and hi < lm0 + 12:
        hi += 1.0

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)
    while hi - lo > tol:
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo); f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo); f2 = f(x2)
    lm = 0.5 * (lo + hi)
    return float(np.exp(lm)), -f(lm), c, nmax


def profile_ci(counts, e, m_hat, ll_hat, c, nmax, level=0.95):
    """Likelihood-profile interval: 2*(ll_hat - ll(m)) = chi2_{1,level}."""
    crit = 0.5 * chi2.ppf(level, 1)
    g = lambda lm: ll_hat - loglik(np.exp(lm), counts, e, c, nmax) - crit
    lm_hat = np.log(m_hat)

    def hunt(direction):
        step, edge = 0.05 * direction, lm_hat
        for _ in range(200):
            edge += step
            if g(edge) > 0:
                a, b = edge - step, edge
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    if g(mid) > 0:
                        b = mid
                    else:
                        a = mid
                return float(np.exp(0.5 * (a + b)))
        return float("nan")

    return hunt(-1), hunt(+1)


def lrt_equal_rates(A, B):
    """LRT that two fits share one per-division rate (null: m_i = mu*N_i)."""
    ll_free = A["ll"] + B["ll"]

    def ll_null(mu):
        return (loglik(mu * A["N"] / A["f_nat"], A["foa"], EPSILON, A["c"], A["nmax"]) +
                loglik(mu * B["N"] / B["f_nat"], B["foa"], EPSILON, B["c"], B["nmax"]))

    lo = np.log(min(A["mu"], B["mu"])) - 1.0
    hi = np.log(max(A["mu"], B["mu"])) + 1.0
    grid = np.linspace(lo, hi, 25)
    vals = [ll_null(np.exp(x)) for x in grid]
    j = int(np.argmax(vals))
    fine = np.linspace(grid[max(j - 1, 0)], grid[min(j + 1, 24)], 25)
    best = max(ll_null(np.exp(x)) for x in fine)

    stat = 2.0 * (ll_free - best)
    return stat, float(chi2.sf(max(stat, 0.0), 1))


def fold_ci(ref, r, z=1.959963985):
    """95% CI on the ratio, propagating each profile interval on the log scale."""
    se = lambda d: (np.log(d["mu_hi"]) - np.log(d["mu_lo"])) / (2 * z)
    half = z * np.hypot(se(ref), se(r))
    fold = r["mu"] / ref["mu"]
    return fold * np.exp(-half), fold * np.exp(half)


# =============================================================================
# VALIDATION (only with --check)
# =============================================================================
def simulate(m, e, ncult, rng):
    out = []
    for _ in range(ncult):
        k = rng.poisson(m)
        if k == 0:
            out.append(0); continue
        sizes = np.ceil(1.0 / rng.random(k)).astype(np.int64) - 1
        out.append(int(rng.binomial(sizes, e).sum()))
    return out


def check(seed=7, reps=6):
    """Recovery of a known m, at both culture counts used in the experiment."""
    rng = np.random.default_rng(seed)
    print(f"{'cultures':>9s} {'true m':>10s} {'recovered':>10s} {'error':>8s}")
    for ncult in (15, 3):
        for m_true in (340.0, 1600.0, 8300.0):
            est = np.mean([mle(simulate(m_true, EPSILON, ncult, rng), EPSILON)[0]
                           for _ in range(reps)])
            print(f"{ncult:9d} {m_true:10.0f} {est:10.0f} {est/m_true - 1:+7.1%}")
        print()


# =============================================================================
# REPORT
# =============================================================================
def fit_temperature(temp):
    out = {}
    for name, d in DATA[temp].items():
        N = total_cells(d["yepd"])
        f_nat = NAT_SENSITIVE_FRACTION[(temp, name)]
        m_hat, ll_hat, c, nmax = mle(d["foa"], EPSILON)
        lo, hi = profile_ci(d["foa"], EPSILON, m_hat, ll_hat, c, nmax)
        out[name] = dict(temp=temp, strain=name, foa=d["foa"], n=len(d["foa"]),
                         N=N, m=m_hat, ll=ll_hat, c=c, nmax=nmax, f_nat=f_nat,
                         mu=f_nat * m_hat / N,
                         mu_lo=f_nat * lo / N, mu_hi=f_nat * hi / N)
    return out


HEADER = (f"{'strain':<20s} {'n':>3s} {'cells/culture':>14s} {'rate/division':>14s} "
          f"{'95% CI':>25s} {'fold':>7s} {'fold 95% CI':>19s} {'p':>9s}")


def print_temperature(temp, fit):
    ref = fit[REFERENCE]
    print(f"--- {temp}   (reference strain: {REFERENCE}) ---")
    print(HEADER)
    for name, r in fit.items():
        ci = f"[{r['mu_lo']:.2e}, {r['mu_hi']:.2e}]"
        if name == REFERENCE:
            print(f"{name:<20s} {r['n']:3d} {r['N']:14.2e} {r['mu']:14.3e} {ci:>25s} "
                  f"{1.0:7.2f} {'-':>19s} {'-':>9s}")
            continue
        _, p = lrt_equal_rates(ref, r)
        f_lo, f_hi = fold_ci(ref, r)
        print(f"{name:<20s} {r['n']:3d} {r['N']:14.2e} {r['mu']:14.3e} {ci:>25s} "
              f"{r['mu']/ref['mu']:7.2f} {f'[{f_lo:.2f}, {f_hi:.2f}]':>19s} {p:9.1e}")
    print()


def print_temperature_effect(fits):
    others = [t for t in TEMPERATURES if t != BASE_TEMP]
    if not others:
        return
    for temp in others:
        print(f"--- {temp} vs {BASE_TEMP}, within strain ---")
        print(f"{'strain':<20s} {BASE_TEMP + ' rate':>14s} {temp + ' rate':>14s} "
              f"{'fold':>7s} {'fold 95% CI':>19s} {'p':>9s}")
        for name in DATA[BASE_TEMP]:
            if name not in fits[temp]:
                continue
            a, b = fits[BASE_TEMP][name], fits[temp][name]
            _, p = lrt_equal_rates(a, b)
            f_lo, f_hi = fold_ci(a, b)
            print(f"{name:<20s} {a['mu']:14.3e} {b['mu']:14.3e} "
                  f"{b['mu']/a['mu']:7.2f} {f'[{f_lo:.2f}, {f_hi:.2f}]':>19s} {p:9.1e}")
        print()


HOT_COLOUR = "#FF1744"
COLD_COLOUR = "#2196F3"
CDE_COLOURS = {"CDEI": "#3d5a80", "CDEII": "#98c1d9", "CDEIII": "#ee6c4d"}

CEN3_SEQUENCE = (
    "GTCACATG"
    "ATGATATTTGATTTTATTATATTTTTAAAAAAAGTAAAAAATAAAAAGTAGTTTATTTTTAAAAAATAAAATTTAAAATATTAG"
    "TGTATTTGATTTCCGAAAGTTAAAA"
)
CDE_SEQUENCES = {
    "CDEI": CEN3_SEQUENCE[:8],
    "CDEII": CEN3_SEQUENCE[8:92],
    "CDEIII": CEN3_SEQUENCE[92:],
}


def apply_style():
    import matplotlib.pyplot as plt
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
        "savefig.dpi": STYLE["dpi"],
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def add_panel_label(ax, label, x=-0.08, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=STYLE["panel_fs"], fontweight="bold", clip_on=False)


def _floating_axes(ax):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))


def _draw_allele_schematics(ax, strains):
    from matplotlib.patches import Rectangle

    display_widths = {
        "CDEI": 28,
        "CDEII": 48,
        "CDEIII": 40,
    }
    starts = {
        "CDEI": 0,
        "CDEII": display_widths["CDEI"],
        "CDEIII": display_widths["CDEI"] + display_widths["CDEII"],
    }
    box_height = 0.82

    deleted = {
        "1649 WT CEN": set(),
        "1647 CDEII-del": {"CDEII"},
        "1648 CDEI&II-del": {"CDEI", "CDEII"},
    }
    labels = {
        "1649 WT CEN": "CEN3-WT",
        "1647 CDEII-del": "CEN3-cdeIIΔ",
        "1648 CDEI&II-del": "CEN3-cdeI,IIΔ",
    }

    ax.text(
        -0.02,
        1.03,
        "Engineered CEN3 Alleles",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE["panel_fs"],
        clip_on=False,
    )

    for row, strain in enumerate(strains[::-1]):
        y = row

        for element in ("CDEI", "CDEII", "CDEIII"):
            x, width = starts[element], display_widths[element]

            if element in deleted[strain]:
                ax.add_patch(
                    Rectangle(
                        (x, y - box_height / 2),
                        width,
                        box_height,
                        facecolor="white",
                        edgecolor="0.55",
                        linewidth=1.1,
                        linestyle="--",
                    )
                )
                ax.text(
                    x + width / 2,
                    y,
                    f"Δ{element}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="0.42",
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (x, y - box_height / 2),
                        width,
                        box_height,
                        facecolor=CDE_COLOURS[element],
                        edgecolor="black",
                        linewidth=0.9,
                    )
                )
                sequence = CDE_SEQUENCES[element]
                shown = sequence if len(sequence) <= 25 else sequence[:18] + "…" + sequence[-12:]
                ax.text(
                    x + width / 2,
                    y,
                    shown,
                    ha="center",
                    va="center",
                    fontsize=18,
                    family="DejaVu Sans Mono",
                    color="white" if element != "CDEII" else "#17222B",
                )

        ax.text(
            -6,
            y,
            labels[strain],
            ha="right",
            va="center",
            fontsize=STYLE["annotation_fs"],
        )

    total_width = sum(display_widths.values())
    ax.set_xlim(-8, total_width + 2)
    ax.set_ylim(-0.65, 2.75)
    ax.axis("off")
    add_panel_label(ax, "A")

def _plot_rate_panel(ax, fits, strains):
    colours = {"30C": COLD_COLOUR, "39C": HOT_COLOUR}
    offsets = np.linspace(-0.14, 0.14, len(TEMPERATURES))
    x = np.arange(len(strains), dtype=float)

    ax.text(
        -0.02,
        1.03,
        "Measured Chromosome III Loss Rate",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=STYLE["panel_fs"],
        clip_on=False,
    )

    for offset, temp in zip(offsets, TEMPERATURES):
        estimates = np.asarray([fits[temp][strain]["mu"] for strain in strains])
        lower = np.asarray([fits[temp][strain]["mu_lo"] for strain in strains])
        upper = np.asarray([fits[temp][strain]["mu_hi"] for strain in strains])
        ax.errorbar(x + offset, estimates,
                    yerr=np.vstack((estimates - lower, upper - estimates)),
                    color=colours[temp], marker="o", markersize=11,
                    markeredgecolor="white", markeredgewidth=1.0,
                    linestyle="none", linewidth=2.2, capsize=5,
                    label=temp.replace("C", " °C"))
    ax.set_yscale("log")
    ax.set_xticks(x, ["CEN3-WT", "CEN3-cdeIIΔ", "CEN3-cdeI,IIΔ"])
    ax.set_ylabel("Chromosome III Loss Rate\n(Per Cell Division)")
    ax.legend(frameon=False, title="Growth temperature",
              fontsize=STYLE["legend_fs"], title_fontsize=STYLE["legend_fs"],
              loc="lower right")
    _floating_axes(ax)
    add_panel_label(ax, "B")


def plot_figure_4_5(fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strains = list(DATA[BASE_TEMP])
    fig = plt.figure(figsize=(16.0, 11.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 2.4], hspace=0.34)
    _draw_allele_schematics(fig.add_subplot(gs[0]), strains)
    _plot_rate_panel(fig.add_subplot(gs[1]), fits, strains)
    fig.subplots_adjust(top=0.97, bottom=0.13, left=0.14, right=0.98)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "Figure_4.5_Engineered_CEN3_Alleles_and_Chromosome_III_Loss_Rates_at_30C_and_39C.png"
    fig.savefig(path, format="png", dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure written: {path}")


def save_additional_files(fits):
    ADDITIONAL_DIR.mkdir(parents=True, exist_ok=True)
    with (ADDITIONAL_DIR / "Figure_4.5_chromosome_loss_rate_estimates.csv").open(
        "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Temperature", "Strain", "Cultures", "Cells per Culture",
                         "Rate per Cell Division", "95% CI Lower", "95% CI Upper"])
        for temp in TEMPERATURES:
            for strain, result in fits[temp].items():
                writer.writerow([temp.replace("C", " °C"), strain, result["n"], result["N"],
                                 result["mu"], result["mu_lo"], result["mu_hi"]])
    with (ADDITIONAL_DIR / "Figure_4.5_fluctuation_assay_source_data.csv").open(
        "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Temperature", "Strain", "Culture", "YEPD Colonies",
                         "5-FOA-Resistant Colonies"])
        for temp in TEMPERATURES:
            for strain, values in DATA[temp].items():
                for culture, count in enumerate(values["foa"], start=1):
                    writer.writerow([temp.replace("C", " °C"), strain, culture,
                                     values["yepd"], count])


def main():
    apply_style()
    fits = {temperature: fit_temperature(temperature) for temperature in TEMPERATURES}
    for temperature in TEMPERATURES:
        print_temperature(temperature, fits[temperature])
    print_temperature_effect(fits)
    thin = [(temperature, strain) for temperature in TEMPERATURES
            for strain, result in fits[temperature].items()
            if result["n"] < MIN_CULTURES_WARN]
    if thin:
        n = fits[thin[0][0]][thin[0][1]]["n"]
        print(f"note: {thin[0][0]} has {n} cultures per strain; the profile intervals there are wide and the fold-change CIs are approximate.")
    plot_figure_4_5(fits)
    save_additional_files(fits)
    print(f"Figure directory: {FIGURE_DIR}")
    print(f"Source-data directory: {ADDITIONAL_DIR}")
    return fits


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
