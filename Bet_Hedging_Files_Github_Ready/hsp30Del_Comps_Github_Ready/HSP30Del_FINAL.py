"""Reproduce thesis Figure 6.4: HSP30 deletion and CEN3-edited-strain persistence.

The default workflow generates only the current thesis Figure 6.4. Historical
control and exploratory figure functions are retained for provenance but remain
disabled unless explicitly enabled in the source.

Outputs are written to:
    HSP30Del_FINAL Figures/Thesis Figures/

Run with:
    python HSP30Del_FINAL.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import mannwhitneyu, kruskal
from scipy.interpolate import interp1d
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
ADDITIONAL_FILES_DIR = SCRIPT_DIR / f"{SCRIPT_STEM} Additional Files"
FIGURE_DIR = SCRIPT_DIR / f"{SCRIPT_STEM} Figures"
THESIS_FIGURE_DIR = FIGURE_DIR / "Thesis Figures"
ADDITIONAL_FILES_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

THESIS_STYLE = {
    "base_fs": 29, "axis_fs": 29, "tick_fs": 29, "legend_fs": 25,
    "annotation_fs": 23, "dense_axis_fs": 25, "dense_tick_fs": 23,
    "dense_legend_fs": 20,
}

# ===========================================================================
# CURRENT THESIS FIGURE FROM THIS SCRIPT
#   6.4   HSP30 deletion reduces the competitive persistence associated with
#         CEN3-cdeI,IIΔ under fluctuating heat stress.
# Legacy control/supplementary variants remain in the source for provenance but
# are not written unless MAKE_NONTHESIS_FIGURES is deliberately enabled.
# ===========================================================================

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Helvetica','Arial','DejaVu Sans'],
                     'font.size': THESIS_STYLE['base_fs'],
                     'axes.labelsize': THESIS_STYLE['axis_fs'],
                     'axes.titlesize': THESIS_STYLE['axis_fs'],
                     'xtick.labelsize': THESIS_STYLE['tick_fs'],
                     'ytick.labelsize': THESIS_STYLE['tick_fs']})

# True to also open figures interactively, as the script used to.
SHOW = False
# Current thesis uses only Figure 6.4. Legacy/supplementary exploratory figures are disabled.
MAKE_NONTHESIS_FIGURES = False

HOT_COLOR  = '#FF1744'
COLD_COLOR = '#2196F3'
BASE_COLOR = '#757575'   # retained for non-temperature-specific annotations

def make_regime(period, n_rows=24):
    """'H' every `period` rows starting at row D0, 'C' otherwise, with a
    baseline placeholder ('B') prepended at index 0 for Day 0."""
    return ['B'] + ['H' if d % period == 0 else 'C' for d in range(n_rows)]

def state_color(state):
    if state == 'H': return HOT_COLOR
    if state == 'C': return COLD_COLOR
    return BASE_COLOR

thermal_regimes = {
    'CONST_30': ['B'] + ['C'] * 24,
    'CONST_39': ['B'] + ['H'] * 24,
    'CYCLE_5D': make_regime(5),   # 1H : 4C -> H at rows D0, D5, D10, D15, D20
}

# 6-day uniform sampling interval: Day 0 (baseline), Day 6, Day 12, Day 18, Day 24.
days     = [0, 6, 12, 18, 24]
all_days = np.arange(0, 25)

# =============================================================================
# SHARED HELPERS
# =============================================================================
_HEADER_BG = '#1A1A2E'; _HEADER_FG = 'white'
_ROW0_BG   = '#FFF3E0'; _ROW1_BG   = '#E8F5E9'
_DIFF_BDR  = '#E53935'; _SAME_BDR  = '#BDBDBD'; _GRID_LW = 0.8

def _draw_cell(ax, x, y, w, h, text, bg, fg='black', fontsize=10,
               border_color=_SAME_BDR, border_lw=_GRID_LW, bold=False):
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=bg,edgecolor=border_color,
                                linewidth=border_lw,zorder=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=fg,
            weight='bold' if bold else 'normal',zorder=2,multialignment='center',linespacing=1.4)

def _render_genotype_table(columns, data, col_widths, title, row_height=1.25):
    differing = [i for i in range(len(columns)) if data[0][i] != data[1][i]]
    total_w   = sum(col_widths) + 0.4
    fig_h     = row_height * 3.2 + 0.6
    fig, ax   = plt.subplots(figsize=(total_w, fig_h))
    ax.set_xlim(0, total_w); ax.set_ylim(0, fig_h); ax.axis('off')
    cum_x = [0.2]
    for w in col_widths[:-1]:
        cum_x.append(cum_x[-1] + w)
    header_y = fig_h - row_height - 0.3
    for ci,(label,x,w) in enumerate(zip(columns,cum_x,col_widths)):
        is_d = ci in differing
        _draw_cell(ax,x,header_y,w,row_height,label,_HEADER_BG,fg=_HEADER_FG,fontsize=10,
                   border_color=_DIFF_BDR if is_d else _SAME_BDR,
                   border_lw=2.0 if is_d else _GRID_LW,bold=True)
    for ri,row_vals in enumerate(data):
        y  = header_y-(ri+1)*row_height
        bg = _ROW0_BG if ri==0 else _ROW1_BG
        for ci,(val,x,w) in enumerate(zip(row_vals,cum_x,col_widths)):
            is_d    = ci in differing
            cell_bg = ('#FFCCBC' if (is_d and ri==0) else
                       '#C8E6C9' if (is_d and ri==1) else bg)
            _draw_cell(ax,x,y,w,row_height,val,cell_bg,fontsize=9,
                       border_color=_DIFF_BDR if is_d else _SAME_BDR,
                       border_lw=2.2 if is_d else _GRID_LW, bold=False)
    ax.text(total_w/2,fig_h-0.02,title,ha='center',va='top',
            fontsize=12,weight='normal',color='#1A1A2E')
    ax.text(total_w/2,0.12,
            "Red borders indicate loci that differ between strains. "
            "ho::HYG/ho reflects reciprocal marker design between runs.",
            ha='center',va='bottom',fontsize=8,color='#757575',style='italic')
    plt.tight_layout(pad=0.3)
    if SHOW:
        plt.show()
    plt.close(fig)

def _get_stats(raw_data, regime_name):
    means, sems, n_reps = [], [], []
    for d in days:
        reps = raw_data[regime_name].get(d, [])
        if not reps:
            means.append(np.nan); sems.append(np.nan); n_reps.append(0)
        else:
            arr = np.array(reps, dtype=float); n = len(arr)
            means.append(arr.mean()); sems.append(arr.std(ddof=1)/np.sqrt(n)); n_reps.append(n)
    return np.array(means), np.array(sems), n_reps

def _plot_with_interpolation(ax, raw_data, conditions, thermal_map):
    final_data = []
    for cond in conditions:
        means, sems, n_reps = _get_stats(raw_data, cond)
        regime       = thermal_map[cond]
        interp_func  = interp1d(days, means, kind='linear')
        daily_values = interp_func(all_days)

        # Segment (day i -> day i+1) colored by regime[i+1]: the row growing
        # / completing during that block. regime[0] is the baseline placeholder.
        segments, colors_list = [], []
        for i in range(len(all_days) - 1):
            segments.append([(all_days[i], daily_values[i]), (all_days[i+1], daily_values[i+1])])
            colors_list.append(state_color(regime[i+1]))
        lc = LineCollection(segments, colors=colors_list, linewidths=3)
        ax.add_collection(lc)

        for i, day_val in enumerate(days):
            # The first marker uses the first experimental growth temperature,
            # rather than a separate grey baseline colour.  Later markers use
            # the final growth interval before the sample was collected.
            marker_index = 1 if day_val == 0 else day_val
            marker_color = state_color(regime[marker_index])
            ax.errorbar(day_val, means[i], yerr=sems[i], fmt='none',
                        ecolor=marker_color, markersize=0, capsize=3, capthick=1,
                        elinewidth=2, alpha=0.3, zorder=2)

        is_constant = len(set(regime[1:])) == 1   # ignore the baseline placeholder
        if not is_constant:
            for k in range(2, len(regime)):
                if regime[k] != regime[k-1]:
                    day_pt            = k - 1
                    transition_y      = daily_values[day_pt]
                    transition_color  = state_color(regime[k])
                    ax.plot(all_days[day_pt], transition_y, 'o', color=transition_color,
                            markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=4)
        final_data.append({'cond': cond, 'final_mean': means[-1], 'n': n_reps[-1], 'regime': regime})
    return final_data

def _add_n_labels(ax, final_data):
    final_data_sorted = sorted(final_data, key=lambda x: x['final_mean'])
    used_positions = []
    for fd in final_data_sorted:
        y_pos = fd['final_mean']; n_val = fd['n']
        for used_y in used_positions:
            if abs(used_y - y_pos) < 8: y_pos = used_y + 8
        used_positions.append(y_pos)
        ax.text(24.3, y_pos, f'n={n_val}', fontsize=20, va='center', ha='left')

def _format_ax(ax, ylabel, title):
    ax.set_xlabel("Day", fontsize=26)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=24)
    ax.set_xticks(days); ax.set_ylim(-5, 110); ax.set_xlim(-0.5, 27)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

def _run_stats(raw_data, conditions, display_names, cycle_key, const_keys, label):
    def sig_stars(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "ns"
    print("\n" + "="*80)
    print(f"HIERARCHICAL STATISTICAL TESTING — {label}")
    print("="*80)
    print("  Level 1: Kruskal-Wallis omnibus | Level 2: Pairwise Mann-Whitney U")
    print("="*80)
    for d in [6, 12, 18, 24]:
        print(f"\n{'─'*70}\n  DAY {d}\n{'─'*70}")
        groups = {}
        for cond in conditions:
            reps = raw_data[cond].get(d, [])
            if reps: groups[cond] = np.array(reps, dtype=float)
        group_names  = list(groups.keys())
        group_arrays = [groups[n] for n in group_names]
        disp_names   = [display_names[n] for n in group_names]
        print(f"\n  {'Condition':<22s} {'n':>4s} {'Mean':>8s} {'SEM':>8s}")
        print(f"  {'─'*46}")
        for dn, arr in zip(disp_names, group_arrays):
            print(f"  {dn:<22s} {len(arr):>4d} {arr.mean():>8.2f} {arr.std(ddof=1)/np.sqrt(len(arr)):>8.2f}")
        all_vals = np.concatenate(group_arrays) if group_arrays else np.array([])
        if len(np.unique(all_vals)) <= 1:
            print(f"\n  Kruskal-Wallis: SKIPPED (all values identical)")
        elif len(group_arrays) >= 3:
            stat_kw, p_kw = kruskal(*group_arrays)
            print(f"\n  LEVEL 1 — Kruskal-Wallis: H={stat_kw:.4f}, p={p_kw:.6f} {sig_stars(p_kw)}")
            if p_kw < 0.05:
                print(f"\n  LEVEL 2 — Pairwise Mann-Whitney U:")
                print(f"  {'Comparison':<45s} {'U':>8s} {'p':>12s} {'Sig':>6s}")
                print(f"  {'─'*75}")
                for i,j in list(combinations(range(len(group_names)),2)):
                    stat_u,p_u = mannwhitneyu(group_arrays[i],group_arrays[j],alternative='two-sided')
                    print(f"  {disp_names[i]} vs {disp_names[j]:<25s} {stat_u:>8.1f} {p_u:>12.6f} {sig_stars(p_u):>6s}")
            else:
                print("  → Omnibus ns; skipping pairwise.")
        print(f"\n  PLANNED — Cycling vs Constant:")
        cyc  = groups.get(cycle_key, np.array([]))
        cons = np.concatenate([groups.get(k, np.array([])) for k in const_keys])
        if len(cyc) > 0 and len(cons) > 0:
            stat_u,p_u = mannwhitneyu(cyc, cons, alternative='two-sided')
            print(f"  Cycling (n={len(cyc)}) vs Constant (n={len(cons)}): U={stat_u:.1f}, p={p_u:.6f} {sig_stars(p_u)}")
            print(f"  Cycling mean={cyc.mean():.2f}, Constant mean={cons.mean():.2f}")
    print("\n" + "="*80)
    print(f"END — {label}")
    print("="*80)

_LEGEND = [
    Line2D([0],[0], color=COLD_COLOR, lw=3, marker='o', label='Cold (30°C)', markersize=8),
    Line2D([0],[0], color=HOT_COLOR,  lw=3, marker='o', label='Hot (39°C)',  markersize=8),
]

_COLS = [
    "Strain", "Background", "MAT", "CEN locus", "ho",
    "ura3", "Upstream periCEN", "cyh2", "Downstream periCEN", "hsp30",
]
_CW = [2.9, 1.2, 0.9, 2.1, 1.4, 1.1, 2.0, 1.0, 2.1, 2.0]


# =============================================================================
# FIGURE 1 — CEN3-cdeI,IIΔ hsp30Δ  vs  WT CEN3 hsp30Δ
# Tracking: CEN3-cdeI,IIΔ hsp30Δ frequency (both runs express this)
# Run 1: CEN3-edit ho::HYG tracked directly
# Run 2: WT CEN3 ho::HYG tracked → invert (100 - value) to get CEN3-edit frequency
# Plot shows single averaged line per condition (n=6 combined)
# =============================================================================

# Run 1: CEN3-edit hsp30Δ ho::HYG frequency measured directly
_r1_raw = {
    'C30': {0:[50,50,50], 6:[49,45,11], 12:[2,0,0],    18:[0,0,0],    24:[0,0,0]},
    'C39': {0:[50,50,50], 6:[43,39,34], 12:[40,27,28], 18:[15,11,9],  24:[0,0,0]},
    'C5D': {0:[50,50,50], 6:[36,26,33], 12:[30,16,28], 18:[24,11,19], 24:[11,7,12]},
}

# Run 2: WT CEN3 hsp30Δ ho::HYG measured → invert each value to get CEN3-edit frequency
_r2_wt_hyg = {
    'C30': {0:[50,50,50], 6:[76,89,85],  12:[95,97,86],   18:[100,100,100], 24:[100,100,100]},
    'C39': {0:[50,50,50], 6:[61,72,59],  12:[75,89,64],   18:[98,100,87],   24:[100,100,100]},
    'C5D': {0:[50,50,50], 6:[76,71,69],  12:[86,84,78],   18:[54,96,89],    24:[96,100,41]},
}
_r2_edit = {k: {d: [100 - v for v in vals] for d, vals in cond.items()}
            for k, cond in _r2_wt_hyg.items()}

# Combined n=6: Run 1 values + inverted Run 2 values
F1_C30 = r"$\boldsymbol{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$  Constant Cold"
F1_C39 = r"$\boldsymbol{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$  Constant Hot"
F1_C25 = r"$\boldsymbol{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$  Five-day cycle"

raw_data_f1 = {
    F1_C30: {d: _r1_raw['C30'][d] + _r2_edit['C30'][d] for d in days},
    F1_C39: {d: _r1_raw['C39'][d] + _r2_edit['C39'][d] for d in days},
    F1_C25: {d: _r1_raw['C5D'][d] + _r2_edit['C5D'][d] for d in days},
}
thermal_map_f1   = {F1_C30: thermal_regimes['CONST_30'], F1_C39: thermal_regimes['CONST_39'], F1_C25: thermal_regimes['CYCLE_5D']}
display_names_f1 = {F1_C30: "Constant Cold", F1_C39: "Constant Hot", F1_C25: "Five-day cycle"}
conditions_f1    = [F1_C30, F1_C39, F1_C25]

if MAKE_NONTHESIS_FIGURES:
    # --- Print timepoint averages ---
    print("\n" + "="*60)
    print("FIGURE 1 — CEN3-edit hsp30Δ FREQUENCY: TIMEPOINT AVERAGES")
    print("="*60)
    cond_labels = [("Constant Cold", 'C30'), ("Constant Hot", 'C39'), ("Five-day cycle", 'C5D')]
    cond_keys   = {F1_C30: 'C30', F1_C39: 'C39', F1_C25: 'C5D'}

    for label, ck in cond_labels:
        print(f"\n  {label}")
        print(f"  {'Day':>5s}  {'Run1 mean':>10s}  {'Run2 mean':>10s}  {'Combined mean':>14s}  {'Combined SEM':>13s}")
        print(f"  {'─'*60}")
        for d in days:
            r1  = np.array(_r1_raw[ck][d], dtype=float)
            r2  = np.array(_r2_edit[ck][d], dtype=float)
            cb  = np.array(raw_data_f1[[k for k,v in cond_keys.items() if v==ck][0]][d], dtype=float)
            sem = cb.std(ddof=1) / np.sqrt(len(cb))
            print(f"  {d:>5d}  {r1.mean():>10.1f}  {r2.mean():>10.1f}  {cb.mean():>14.1f}  {sem:>13.2f}")

    # --- Plot single averaged lines ---
    print("\n" + "="*60)
    print("GENERATING FIGURE 1 — CEN3-edit hsp30Δ vs WT CEN3 hsp30Δ  (n=6)")
    print("="*60)

    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 10))
    fd1 = _plot_with_interpolation(ax1, raw_data_f1, conditions_f1, thermal_map_f1)
    _add_n_labels(ax1, fd1)
    _format_ax(ax1,
        ylabel=r"$\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$ Frequency (%)",
        title=r"$\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$ Vs WT CEN3 $\mathit{hsp30\Delta}$")
    ax1.legend(handles=_LEGEND, frameon=False, loc='best', fontsize=22)
    plt.tight_layout()
    if MAKE_NONTHESIS_FIGURES:
        fig1.savefig(FIGURE_DIR / "figS_hsp30_edit_vs_wt.png", dpi=300, bbox_inches="tight", facecolor="white")
        print("\nFigure 1 complete! saved: figS_hsp30_edit_vs_wt.png")
    if SHOW:
        plt.show()
    plt.close(fig1)

    _render_genotype_table(
        _COLS,
        [
            [r"$\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$",
             "s288c", r"$\alpha$", r"$cen3\text{-}cdeI,II\Delta$", "ho::HYG\nor\nho",
             "ura3-d", "periCEN3::URA3+", r"$cyh2^r$", "periCEN3::NAT", r"$hsp30::KANMX$"],
            [r"WT CEN3 $\mathit{hsp30\Delta}$",
             "s288c", r"$\alpha$", "WT CEN3", "ho::HYG\nor\nho",
             "ura3-d", "periCEN3::URA3+", r"$cyh2^r$", "periCEN3::NAT", r"$hsp30::KANMX$"],
        ],
        _CW,
        r"Strain Genotypes — $\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$ vs WT CEN3 $\mathit{hsp30\Delta}$",
    )
    print("Genotype table 1 rendered!")

    _run_stats(raw_data_f1, conditions_f1, display_names_f1,
               cycle_key=F1_C25, const_keys=[F1_C30, F1_C39],
               label="CEN3-cdeI,IIΔ hsp30Δ vs WT CEN3 hsp30Δ  [n=6 combined]")



# =============================================================================
# FIGURE 2 — WT CEN3 hsp30Δ ho::HYG  vs  WT CEN3
# Tracking: WT CEN3 hsp30Δ ho::HYG frequency
# =============================================================================

# Run 1: WT CEN3 hsp30Δ ho::HYG frequency measured directly (triplicate)
_f2_r1_raw = {
    'C30': {0:[50,50,50], 6:[46,43,41], 12:[37,31,35], 18:[18,14,17],  24:[3,2,4]},
    'C39': {0:[50,50,50], 6:[26,23,31], 12:[0,0,0],    18:[0,0,0],    24:[0,0,0]},
    'C5D': {0:[50,50,50], 6:[37,29,39], 12:[29,15,21], 18:[6,0,3],    24:[0,0,0]},
}

# Run 2 (reciprocal marker): WT CEN3 ho::HYG (the non-hsp30Δ competitor, now
# carrying the marker instead) measured directly, triplicate -> invert
# (100 - value) to get WT CEN3 hsp30Δ frequency. Confirmed against Image 1:
# per-day triplicate means match to within rounding.
_f2_r2_wt_hyg = {
    'C30': {0:[50,50,50], 6:[61,65,55], 12:[74,73,69], 18:[89,89,79],  24:[100,100,100]},
    'C39': {0:[50,50,50], 6:[69,60,81], 12:[100,100,100], 18:[100,100,100], 24:[100,100,100]},
    'C5D': {0:[50,50,50], 6:[70,75,59], 12:[82,86,77], 18:[100,100,100], 24:[100,100,100]},
}
_f2_r2_hsp30 = {k: {d: [100 - v for v in vals] for d, vals in cond.items()}
                 for k, cond in _f2_r2_wt_hyg.items()}

F2_C30 = "WT CEN3 hsp30Δ ho::HYG — Constant Cold"
F2_C39 = "WT CEN3 hsp30Δ ho::HYG — Constant Hot"
F2_C25 = "WT CEN3 hsp30Δ ho::HYG — Five-day cycle"

# Combined: Run 1 + Run 2, both triplicate (n=6)
raw_data_f2 = {
    F2_C30: {d: _f2_r1_raw['C30'][d] + _f2_r2_hsp30['C30'][d] for d in days},
    F2_C39: {d: _f2_r1_raw['C39'][d] + _f2_r2_hsp30['C39'][d] for d in days},
    F2_C25: {d: _f2_r1_raw['C5D'][d] + _f2_r2_hsp30['C5D'][d] for d in days},
}
thermal_map_f2   = {F2_C30: thermal_regimes['CONST_30'], F2_C39: thermal_regimes['CONST_39'], F2_C25: thermal_regimes['CYCLE_5D']}
display_names_f2 = {F2_C30: "Constant Cold", F2_C39: "Constant Hot", F2_C25: "Five-day cycle"}
conditions_f2    = [F2_C30, F2_C39, F2_C25]

if MAKE_NONTHESIS_FIGURES:
    print("\n" + "="*60)
    print("GENERATING FIGURE 2 — WT CEN3 hsp30Δ vs WT CEN3")
    print("="*60)

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    fd2 = _plot_with_interpolation(ax2, raw_data_f2, conditions_f2, thermal_map_f2)
    _add_n_labels(ax2, fd2)
    _format_ax(ax2,
        ylabel=r"WT CEN3 $\mathit{hsp30\Delta}$ Frequency (%)",
        title=r"WT CEN3 $\mathit{hsp30\Delta}$ Vs WT CEN3")
    ax2.legend(handles=_LEGEND, frameon=False, loc='best', fontsize=22)
    plt.tight_layout()
    if MAKE_NONTHESIS_FIGURES:
        fig2.savefig(FIGURE_DIR / "figS_hsp30_wt_vs_wt.png", dpi=300, bbox_inches="tight", facecolor="white")
        print("\nFigure 2 complete! saved: figS_hsp30_wt_vs_wt.png")
    if SHOW:
        plt.show()
    plt.close(fig2)

    _render_genotype_table(
        _COLS,
        [
            [r"WT CEN3 $\mathit{hsp30\Delta}$",
             "s288c", r"$\alpha$", "WT CEN3", "ho::HYG\nor\nho",
             "ura3-d", "periCEN3::URA3+", r"$cyh2^r$", "periCEN3::NAT", r"$hsp30::KANMX$"],
            ["WT CEN3",
             "s288c", r"$\alpha$", "WT CEN3", "ho::HYG\nor\nho",
             "ura3-d", "periCEN3::URA3+", r"$cyh2^r$", "periCEN3::NAT", "WT HSP30"],
        ],
        _CW,
        r"Strain Genotypes — WT CEN3 $\mathit{hsp30\Delta}$ vs WT CEN3",
    )
    print("Genotype table 2 rendered!")

    _run_stats(raw_data_f2, conditions_f2, display_names_f2,
               cycle_key=F2_C25, const_keys=[F2_C30, F2_C39],
               label="WT CEN3 hsp30Δ vs WT CEN3")



# =============================================================================
# FIGURE 3 — CEN3-cdeI,IIΔ hsp30Δ ho::HYG  vs  CEN3-cdeI,IIΔ
# Tracking: CEN3-cdeI,IIΔ hsp30Δ ho::HYG frequency
# =============================================================================

# Run 1: CEN3-edit hsp30Δ ho::HYG frequency measured directly (triplicate)
_f3_r1_raw = {
    'C30': {0:[50,50,50], 6:[33,31,38], 12:[14,21,27], 18:[3,12,14], 24:[0,0,0]},
    'C39': {0:[50,50,50], 6:[13,12,18], 12:[0,0,0],  18:[0,0,0],    24:[0,0,0]},
    'C5D': {0:[50,50,50], 6:[29,22,36], 12:[13,10,3],  18:[1,0,0],    24:[0,0,0]},
}

# Run 2 (reciprocal marker): Edit CEN3 ho::HYG (the non-hsp30Δ competitor, now
# carrying the marker instead) measured directly, triplicate -> invert
# (100 - value) to get CEN3-edit hsp30Δ frequency. Confirmed against the
# source image: this triplicate's per-day means match Image 2 almost exactly,
# including resolving the earlier flagged Day12/Constant Hot value - the true
# triplicate there is [100,100,100], not the single 0.0 read previously.
_f3_r2_edit_hyg = {
    'C30': {0:[50,50,50], 6:[59,64,59], 12:[75,77,67], 18:[92,96,98],  24:[100,100,100]},
    'C39': {0:[50,50,50], 6:[80,76,88], 12:[100,100,100], 18:[100,100,100], 24:[100,100,100]},
    'C5D': {0:[50,50,50], 6:[59,71,63], 12:[97,100,97], 18:[100,100,100], 24:[100,100,100]},
}
_f3_r2_hsp30 = {k: {d: [100 - v for v in vals] for d, vals in cond.items()}
                 for k, cond in _f3_r2_edit_hyg.items()}

F3_C30 = "CEN3-edit hsp30Δ ho::HYG — Constant Cold"
F3_C39 = "CEN3-edit hsp30Δ ho::HYG — Constant Hot"
F3_C25 = "CEN3-edit hsp30Δ ho::HYG — Five-day cycle"

# Combined: Run 1 + Run 2, both triplicate (n=6)
raw_data_f3 = {
    F3_C30: {d: _f3_r1_raw['C30'][d] + _f3_r2_hsp30['C30'][d] for d in days},
    F3_C39: {d: _f3_r1_raw['C39'][d] + _f3_r2_hsp30['C39'][d] for d in days},
    F3_C25: {d: _f3_r1_raw['C5D'][d] + _f3_r2_hsp30['C5D'][d] for d in days},
}
thermal_map_f3   = {F3_C30: thermal_regimes['CONST_30'], F3_C39: thermal_regimes['CONST_39'], F3_C25: thermal_regimes['CYCLE_5D']}
display_names_f3 = {F3_C30: "Constant Cold", F3_C39: "Constant Hot", F3_C25: "Five-day cycle"}
conditions_f3    = [F3_C30, F3_C39, F3_C25]

if MAKE_NONTHESIS_FIGURES:
    print("\n" + "="*60)
    print("GENERATING FIGURE 3 — CEN3-edit hsp30Δ vs CEN3-edit")
    print("="*60)

    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 10))
    fd3 = _plot_with_interpolation(ax3, raw_data_f3, conditions_f3, thermal_map_f3)
    _add_n_labels(ax3, fd3)
    _format_ax(ax3,
        ylabel=r"$\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$ Frequency (%)",
        title=r"$\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$ Vs $\mathit{CEN3\text{-}cdeI,II\Delta}$")
    ax3.legend(handles=_LEGEND, frameon=False, loc='best', fontsize=22)
    plt.tight_layout()
    if MAKE_NONTHESIS_FIGURES:
        fig3.savefig(FIGURE_DIR / "figS_hsp30_edit_vs_edit.png", dpi=300, bbox_inches="tight", facecolor="white")
        print("\nFigure 3 complete! saved: figS_hsp30_edit_vs_edit.png")
    if SHOW:
        plt.show()
    plt.close(fig3)

    _render_genotype_table(
        _COLS,
        [
            [r"$\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$",
             "s288c", r"$\alpha$", r"$cen3\text{-}cdeI,II\Delta$", "ho::HYG\nor\nho",
             "ura3-d", "periCEN3::URA3+", r"$cyh2^r$", "periCEN3::NAT", r"$hsp30::KANMX$"],
            [r"$\mathit{CEN3\text{-}cdeI,II\Delta}$",
             "s288c", r"$\alpha$", r"$cen3\text{-}cdeI,II\Delta$", "ho::HYG\nor\nho",
             "ura3-d", "periCEN3::URA3+", r"$cyh2^r$", "periCEN3::NAT", "WT HSP30"],
        ],
        _CW,
        r"Strain Genotypes — $\mathit{CEN3\text{-}cdeI,II\Delta\ hsp30\Delta}$ vs $\mathit{CEN3\text{-}cdeI,II\Delta}$",
    )
    print("Genotype table 3 rendered!")

    _run_stats(raw_data_f3, conditions_f3, display_names_f3,
               cycle_key=F3_C25, const_keys=[F3_C30, F3_C39],
               label="CEN3-cdeI,IIΔ hsp30Δ vs CEN3-cdeI,IIΔ")


# ===========================================================================
# THESIS FIGURE 6.4 -- HSP30 epistasis
# ===========================================================================

# Reference competition: CEN3-cdeI,IIdel vs CEN3-WT, no hsp30 deletion.
# Percent edited strain, days 0/6/12/18/24, mean of six 5 mL replicates.
# Source: Chapter 5 tube competition data.
REFERENCE = {
    'C30': [50, 1.7,  0,    0,    0],
    'C39': [50, 63.8, 28.8, 18.3, 0],
    'C5D': [50, 37.8, 32.8, 33.0, 35.8],
}

# Standard deviations across the same six Chapter 5 tube cultures. Values are
# stored as percentages to match REFERENCE and the HSP30 competition data.
REFERENCE_SD = {
    'C30': [0, 2.1,  0,    0,    0],
    'C39': [0, 38.1, 23.8, 17.0, 0],
    'C5D': [0, 36.6, 45.2, 48.2, 38.5],
}

REGIME_KEYS = ['C30', 'C39', 'C5D']
REGIME_TITLES = {'C30': 'Constant 30 °C', 'C39': 'Constant 39 °C',
                 'C5D': 'Five-day cycle'}

# Trajectories 1 and 2 carry the comparison; 3 and 4 are the controls showing
# hsp30-delta is deleterious on its own, and are deliberately subordinate.
TRAJ_STYLE = {
    'reference':  dict(color='#1A1A2E', lw=2.6, ls='-',  marker='o', ms=6.5, zorder=6),
    'both_hsp30': dict(color='#D81B60', lw=2.6, ls='-',  marker='s', ms=6.5, zorder=6),
    'wt_track':   dict(color='#9E9E9E', lw=1.3, ls='--', marker='^', ms=4.0, zorder=3),
    'edit_track': dict(color='#9E9E9E', lw=1.3, ls=':',  marker='v', ms=4.0, zorder=3),
}
TRAJ_LABELS = {
    'reference':  'CEN3-cdeI,II\u0394 vs CEN3-WT',
    'both_hsp30': 'CEN3-cdeI,II\u0394 hsp30\u0394 vs CEN3-WT hsp30\u0394',
    'wt_track':   'CEN3-WT hsp30\u0394 vs CEN3-WT',
    'edit_track': 'CEN3-cdeI,II\u0394 hsp30\u0394 vs CEN3-cdeI,II\u0394',
}


def _mean_sd(reps):
    """Mean and SD across replicates (the chapter quotes spread, not SEM)."""
    a = np.asarray(reps, dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), 0
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0), a.size


def _pooled(run1, run2_inverted, regime):
    """Reciprocal marker orientations pooled, second already inverted."""
    return {d: run1[regime][d] + run2_inverted[regime][d] for d in days}


def collect_trajectories():
    """The three competitions measured in this script, per regime."""
    return {
        'both_hsp30': {k: _pooled(_r1_raw, _r2_edit, k) for k in REGIME_KEYS},
        'wt_track':   {k: _pooled(_f2_r1_raw, _f2_r2_hsp30, k) for k in REGIME_KEYS},
        'edit_track': {k: _pooled(_f3_r1_raw, _f3_r2_hsp30, k) for k in REGIME_KEYS},
    }


def _thermal_phase(regime, day):
    """Temperature applied during the interval beginning at ``day``."""
    if regime == 'C30':
        return 'C'
    if regime == 'C39':
        return 'H'
    return 'H' if int(day) % 5 == 0 else 'C'


def _fill_house_style_sd(ax, regime, x, means, sds, alpha=0.16, zorder=1.5):
    """Shade mean +/- SD using blue for cold and red for hot intervals."""
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    for day in range(int(np.floor(x.min())), int(np.ceil(x.max()))):
        xx = np.linspace(day, day + 1, 17)
        mean_i = np.interp(xx, x, means)
        sd_i = np.interp(xx, x, sds)
        colour = HOT_COLOR if _thermal_phase(regime, day) == 'H' else COLD_COLOR
        ax.fill_between(
            xx,
            np.clip(mean_i - sd_i, 0, 100),
            np.clip(mean_i + sd_i, 0, 100),
            color=colour,
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )


def print_day24_table(trajs):
    """Day-24 mean and SD for every competition in every regime, so the numbers
    quoted in the chapter can be checked against the figure."""
    print("\n" + "=" * 74)
    print("DAY-24 SUMMARY -- mean and SD across replicates")
    print("=" * 74)
    order = ['reference', 'both_hsp30', 'wt_track', 'edit_track']
    for reg in REGIME_KEYS:
        print(f"\n  {REGIME_TITLES[reg]}")
        print(f"  {'competition':<44}{'n':>4}{'mean':>9}{'SD':>9}")
        print("  " + "-" * 66)
        for t in order:
            if t == 'reference':
                m = REFERENCE[reg][-1]
                sd = REFERENCE_SD[reg][-1]
                print(f"  {TRAJ_LABELS[t]:<44}{6:>4}{m:>9.1f}{sd:>9.1f}")
            else:
                m, sd, n = _mean_sd(trajs[t][reg][24])
                print(f"  {TRAJ_LABELS[t]:<44}{n:>4}{m:>9.1f}{sd:>9.1f}")

    ref24 = REFERENCE['C5D'][-1]
    bh24, _, _ = _mean_sd(trajs['both_hsp30']['C5D'][24])
    decrease_pp = ref24 - bh24
    relative_reduction = 100.0 * decrease_pp / ref24 if ref24 else float('nan')
    print(f"\n  Five-day cycle, day 24: {ref24:.1f}% -> {bh24:.1f}%  "
          f"({decrease_pp:.1f} percentage-point decrease; "
          f"{relative_reduction:.1f}% relative reduction)")
    print("\n  All shaded regions show SD across six reciprocal-marker replicates.")


def make_hsp30_epistasis_figure(
        out_path=None,
        controls_out=None):
    """Draw the focal HSP30 test and the two deletion controls separately.

    Keeping the control comparisons in their own figure avoids superimposing
    four trajectories. Shaded regions show SD across six reciprocal-marker
    replicates for every trajectory.
    """
    if out_path is None:
        out_path = THESIS_FIGURE_DIR / "Figure_6.4_HSP30_Deletion_Reduces_Competitive_Persistence_Associated_with_CEN3-cdeI_II_Delta.png"
    if controls_out is None:
        controls_out = FIGURE_DIR / "Figure_6.4_controls_NOT_IN_THESIS.png"
    trajs = collect_trajectories()

    def phase_for_interval(reg, x0):
        if reg == 'C30':
            return 'C'
        if reg == 'C39':
            return 'H'
        return 'H' if int(x0) % 5 == 0 else 'C'

    def thermal_line(ax, reg, x, y, ls, marker, lw, label, zorder):
        # Draw one artist per uninterrupted thermal block.  The earlier
        # implementation drew hundreds of tiny segments, which restarted the
        # dash pattern on every segment and made dashed trajectories look solid.
        boundaries = np.arange(int(min(x)), int(max(x)) + 1, dtype=float)
        boundary_y = np.interp(boundaries, x, y)
        phases = [phase_for_interval(reg, d) for d in boundaries[:-1]]
        start = 0
        for i in range(1, len(phases) + 1):
            if i == len(phases) or phases[i] != phases[start]:
                xx = boundaries[start:i + 1]
                yy = boundary_y[start:i + 1]
                colour = HOT_COLOR if phases[start] == 'H' else COLD_COLOR
                ax.plot(xx, yy, color=colour, lw=lw, ls=ls,
                        solid_capstyle='round', dash_capstyle='round',
                        zorder=zorder)
                start = i
        marker_colours = [
            HOT_COLOR if phase_for_interval(reg, max(0, d - 1)) == 'H'
            else COLD_COLOR
            for d in x
        ]
        for d, value, colour in zip(x, y, marker_colours):
            ax.plot(d, value, marker=marker, ms=8.0, color=colour,
                    mec='white', mew=0.7, ls='none', zorder=zorder + 0.2)
        return Line2D([0], [0], color='#555555', lw=lw, ls=ls,
                      marker=marker, ms=6, label=label)

    styles = {
        'reference':  ('-', 'o', 3.2, '#6C6C6C'),
        'both_hsp30': ((0, (6, 3)), 's', 3.0, '#C995A7'),
        'wt_track':   ((0, (6, 2)), '^', 1.9, '#9FC4AB'),
        'edit_track': ((0, (2, 2)), 'v', 1.9, '#B6A2CA'),
    }

    def draw_row(axes, names, add_reduction=False):
        for col, reg in enumerate(REGIME_KEYS):
            ax = axes[col]
            ax.axhline(50, color='#BDBDBD', lw=0.8, ls=':', zorder=1)
            for name in names:
                if name == 'reference':
                    means = np.asarray(REFERENCE[reg], dtype=float)
                    sds = np.asarray(REFERENCE_SD[reg], dtype=float)
                else:
                    means, sds = [], []
                    for d in days:
                        mean, sd, _ = _mean_sd(trajs[name][reg][d])
                        means.append(mean); sds.append(sd)
                    means = np.asarray(means); sds = np.asarray(sds)

                linestyle, marker, linewidth, shade_colour = styles[name]
                thermal_line(ax, reg, days, means, linestyle, marker,
                             linewidth, TRAJ_LABELS[name], 5)
                _fill_house_style_sd(ax, reg, days, means, sds,
                                     alpha=.16, zorder=1.5)

            ax.text(0.5, 1.03, REGIME_TITLES[reg], transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=THESIS_STYLE['dense_axis_fs'])
            ax.set_xlabel('Day', fontsize=THESIS_STYLE['dense_axis_fs'])
            ax.set_xticks(days)
            ax.set_ylim(0, 100)
            ax.set_xlim(-1, 27)
            ax.tick_params(labelsize=THESIS_STYLE['dense_tick_fs'], direction='out')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_position(('outward', 6))
            ax.spines['bottom'].set_position(('outward', 6))

            if add_reduction and reg == 'C5D':
                ref24 = REFERENCE[reg][-1]
                bh24, _, _ = _mean_sd(trajs['both_hsp30'][reg][24])
                decrease_pp = ref24 - bh24
                ax.annotate('', xy=(25.2, ref24), xytext=(25.2, bh24),
                            arrowprops=dict(arrowstyle='<->', lw=1.45,
                                            color='#444444'))
                ax.text(25.95, (ref24 + bh24) / 2,
                        f"{decrease_pp:.1f} percentage\npoints lower",
                        fontsize=THESIS_STYLE['annotation_fs'],
                        ha='left', va='center', color='#444444')

    def handles_for(names):
        genotype_handles = [
            Line2D([0], [0], color='#555555', lw=styles[name][2],
                   ls=styles[name][0], marker=styles[name][1], ms=5.5,
                   label=TRAJ_LABELS[name])
            for name in names]
        genotype_handles += [
            Line2D([0], [0], color=COLD_COLOR, lw=2.5, label='30 °C interval'),
            Line2D([0], [0], color=HOT_COLOR, lw=2.5, label='39 °C interval')]
        return genotype_handles

    with plt.rc_context({'font.family': 'sans-serif', 'font.size': THESIS_STYLE['dense_tick_fs']}):
        # Main thesis figure: the matched edit-versus-WT comparison.
        fig, axes = plt.subplots(1, 3, figsize=(22, 8.5), sharex=True, sharey=True)
        draw_row(axes, ('reference', 'both_hsp30'), add_reduction=True)
        axes[0].set_ylabel('Frequency of first-listed strain (%)', fontsize=THESIS_STYLE['dense_axis_fs'])
        heat_handles = [
            Line2D([0], [0], color=COLD_COLOR, lw=3.0, marker='o', ms=7,
                   markeredgecolor='white', markeredgewidth=0.7,
                   label='Mean, 30 °C day'),
            Line2D([0], [0], color=HOT_COLOR, lw=3.0, marker='o', ms=7,
                   markeredgecolor='white', markeredgewidth=0.7,
                   label='Mean, 39 °C day'),
        ]
        genotype_handles = [
            Line2D([0], [0], color='#555555', lw=styles[name][2],
                   ls=styles[name][0], marker=styles[name][1], ms=5.5,
                   label=TRAJ_LABELS[name])
            for name in ('reference', 'both_hsp30')
        ]
        heat_leg = fig.legend(handles=heat_handles,
                              frameon=False, fontsize=THESIS_STYLE['dense_legend_fs'],
                              loc='lower center', ncol=2,
                              bbox_to_anchor=(0.5, 0.105),
                              handlelength=3.0, handletextpad=0.7,
                              columnspacing=1.4)
        genotype_leg = fig.legend(handles=genotype_handles,
                                  frameon=False, fontsize=THESIS_STYLE['dense_legend_fs'],
                                  loc='lower center', ncol=2,
                                  bbox_to_anchor=(0.5, 0.062),
                                  handlelength=3.0, handletextpad=0.7,
                                  columnspacing=1.4)
        fig.add_artist(heat_leg)
        fig.tight_layout(rect=(0.025, .22, 1, 1), w_pad=1.5)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Current thesis uses only Figure 6.4; stop before archival control variants.
        print(f"\nFigure saved: {out_path}")
        print_day24_table(trajs)
        return trajs



# ---------------------------------------------------------------------------
# ARCHIVAL ALTERNATIVES -- not generated by the current-thesis workflow
# ---------------------------------------------------------------------------

FIG63_STYLE = {
    'reference':  dict(color='#4A4A4A', ls='-',  marker='o', lw=2.2, ms=5.6),
    'both_hsp30': dict(color='#B66E86', ls='--', marker='s', lw=2.2, ms=5.4),
    'wt_track':   dict(color='#6E9277', ls='-.', marker='^', lw=1.8, ms=5.2),
    'edit_track': dict(color='#8D79A8', ls=':',  marker='v', lw=1.9, ms=5.2),
}

FIG63_SHORT_LABELS = {
    'reference':  'Edit vs WT',
    'both_hsp30': 'Edit hsp30Δ vs WT hsp30Δ',
    'wt_track':   'WT hsp30Δ vs WT',
    'edit_track': 'Edit hsp30Δ vs Edit',
}

_COLD_BG = '#DCECF6'
_HOT_BG = '#F7DFE0'


def _trajectory_summary(trajs, name, regime):
    """Return day-wise mean and SD for one competition."""
    if name == 'reference':
        return (np.asarray(REFERENCE[regime], dtype=float),
                np.asarray(REFERENCE_SD[regime], dtype=float))
    means, sds = [], []
    for day in days:
        mean, sd, _ = _mean_sd(trajs[name][regime][day])
        means.append(mean)
        sds.append(sd)
    return np.asarray(means), np.asarray(sds)


def _shade_thermal_history(ax, regime):
    """Encode temperature as background, leaving line style to identify strains."""
    if regime == 'C30':
        ax.axvspan(0, 24, color=_COLD_BG, alpha=0.62, lw=0, zorder=0)
    elif regime == 'C39':
        ax.axvspan(0, 24, color=_HOT_BG, alpha=0.62, lw=0, zorder=0)
    else:
        for day in range(24):
            colour = _HOT_BG if day % 5 == 0 else _COLD_BG
            ax.axvspan(day, day + 1, color=colour, alpha=0.62, lw=0, zorder=0)


def _tidy_trajectory_axis(ax, regime, show_xlabel):
    _shade_thermal_history(ax, regime)
    ax.axhline(50, color='#8A8A8A', lw=0.8, ls=':', zorder=1)
    ax.set_xlim(-0.5, 24.8)
    ax.set_ylim(0, 100)
    ax.set_xticks(days)
    ax.set_xlabel('Day' if show_xlabel else '')
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_figure_6_3c(out_path='Figure_6.3c_HSP30_grouped_trajectories.png'):
    """Recommended layout: focal comparison above, deletion controls below."""
    trajs = collect_trajectories()
    with plt.rc_context({'font.family': 'Helvetica', 'font.size': 10}):
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), sharex=True, sharey=True)
        rows = (('reference', 'both_hsp30'), ('wt_track', 'edit_track'))

        for col, regime in enumerate(REGIME_KEYS):
            for row, names in enumerate(rows):
                ax = axes[row, col]
                _tidy_trajectory_axis(ax, regime, show_xlabel=(row == 1))
                for name in names:
                    means, sds = _trajectory_summary(trajs, name, regime)
                    style = FIG63_STYLE[name]
                    ax.plot(days, means, color=style['color'], ls=style['ls'],
                            marker=style['marker'], lw=style['lw'], ms=style['ms'],
                            mec='white', mew=0.55, zorder=4)
                    if name != 'reference':
                        _fill_house_style_sd(ax, regime, days, means, sds,
                                             alpha=.13, zorder=2)

                if row == 0:
                    ax.set_title(REGIME_TITLES[regime], fontsize=10.5, pad=8)

            if regime == 'C5D':
                ref24 = REFERENCE[regime][-1]
                hsp24, _, _ = _mean_sd(trajs['both_hsp30'][regime][24])
                axes[0, col].annotate('', xy=(23.2, ref24), xytext=(23.2, hsp24),
                                      arrowprops=dict(arrowstyle='<->', color='#333333', lw=1))
                axes[0, col].text(22.5, (ref24 + hsp24) / 2,
                                  f'{ref24:.1f}% vs {hsp24:.1f}%', ha='right', va='center',
                                  fontsize=8.5, color='#333333')

        axes[0, 0].set_ylabel('Frequency of first-listed strain (%)')
        axes[1, 0].set_ylabel('Frequency of first-listed strain (%)')
        fig.text(0.008, 0.72, 'Primary comparison', rotation=90,
                 va='center', ha='left', fontsize=9, color='#555555')
        fig.text(0.008, 0.29, 'Deletion controls', rotation=90,
                 va='center', ha='left', fontsize=9, color='#555555')

        handles = [Line2D([0], [0], **FIG63_STYLE[name],
                          label=FIG63_SHORT_LABELS[name])
                   for name in ('reference', 'both_hsp30', 'wt_track', 'edit_track')]
        handles += [Patch(facecolor=_COLD_BG, edgecolor='none', label='30 °C interval'),
                    Patch(facecolor=_HOT_BG, edgecolor='none', label='39 °C interval')]
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
                   fontsize=8.6, bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout(rect=(0.035, 0.09, 1, 1), h_pad=2.0, w_pad=1.4)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f'Figure saved: {out_path}')


def make_figure_6_3d(out_path='Figure_6.3d_HSP30_endpoint_summary.png'):
    """Alternative layout: day-24 endpoint means and SDs as horizontal dots."""
    trajs = collect_trajectories()
    order = ('reference', 'both_hsp30', 'wt_track', 'edit_track')
    ypos = np.arange(len(order))[::-1]
    with plt.rc_context({'font.family': 'Helvetica', 'font.size': 10}):
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharex=True, sharey=True)
        for ax, regime in zip(axes, REGIME_KEYS):
            ax.axvline(50, color='#A0A0A0', lw=0.8, ls=':', zorder=1)
            for y, name in zip(ypos, order):
                means, sds = _trajectory_summary(trajs, name, regime)
                x = means[-1]
                sd = None if name == 'reference' else sds[-1]
                style = FIG63_STYLE[name]
                if sd is not None:
                    ax.errorbar(x, y, xerr=sd, fmt='none', ecolor=style['color'],
                                elinewidth=1.2, capsize=3, zorder=2)
                ax.scatter(x, y, s=54, marker=style['marker'], color=style['color'],
                           edgecolor='white', linewidth=0.6, zorder=3)
                ax.text(min(x + 3.0, 96), y, f'{x:.1f}', va='center', fontsize=8.5,
                        color=style['color'])
            ax.set_title(REGIME_TITLES[regime], fontsize=10.5, pad=8)
            ax.set_xlim(-5, 100)
            ax.set_xlabel('Day-24 frequency (%)')
            ax.set_yticks(ypos, [FIG63_SHORT_LABELS[name] for name in order])
            ax.tick_params(labelsize=8.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
        fig.tight_layout(w_pad=1.5)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f'Figure saved: {out_path}')


def make_figure_6_3e(out_path='Figure_6.3e_HSP30_trajectory_heatmap.png'):
    """Alternative layout: compact heatmaps of mean frequency through time."""
    trajs = collect_trajectories()
    order = ('reference', 'both_hsp30', 'wt_track', 'edit_track')
    cmap = LinearSegmentedColormap.from_list('frequency',
                                             ['#83B9D7', '#FAF7F2', '#BF7E92'])
    norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    with plt.rc_context({'font.family': 'Helvetica', 'font.size': 10}):
        fig = plt.figure(figsize=(13.2, 4.1), constrained_layout=True)
        grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06])
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        colourbar_axis = fig.add_subplot(grid[0, 3])
        image = None
        for ax, regime in zip(axes, REGIME_KEYS):
            matrix = np.vstack([_trajectory_summary(trajs, name, regime)[0]
                                for name in order])
            image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix[row, col]
                    text_colour = 'white' if value < 18 or value > 82 else '#333333'
                    ax.text(col, row, f'{value:.1f}', ha='center', va='center',
                            fontsize=8.2, color=text_colour)
            ax.set_title(REGIME_TITLES[regime], fontsize=10.5, pad=8)
            ax.set_xticks(range(len(days)), days)
            ax.set_xlabel('Day', fontsize=THESIS_STYLE['dense_axis_fs'])
            ax.set_yticks(range(len(order)), [FIG63_SHORT_LABELS[name] for name in order])
            ax.tick_params(length=0, labelsize=8.6)
            for spine in ax.spines.values():
                spine.set_visible(False)
        cbar = fig.colorbar(image, cax=colourbar_axis)
        cbar.set_label('Mean frequency of first-listed strain (%)', fontsize=9)
        cbar.ax.tick_params(labelsize=8.5)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f'Figure saved: {out_path}')


if __name__ == "__main__":
    # Current-thesis workflow: generate Figure 6.4 only.
    make_hsp30_epistasis_figure()
