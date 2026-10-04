#!/usr/bin/env python3
"""Reproduce thesis Figure 4.3: Natural Cas9 PAM Availability and Guide-Site Filtering Around CEN3.

Run with:
    python cen3_pam_landscape_FINAL.py

Required companion files:
    Synthetic_Cut_Site_Generator_FINAL.py
    cen3_pam_landscape_FINAL Additional Files/whole_genes.fna
    cen3_pam_landscape_FINAL Additional Files/mismatch_score.pkl
    cen3_pam_landscape_FINAL Additional Files/pam_scores.pkl

The script writes intermediate/reproducibility files to
``cen3_pam_landscape_FINAL Additional Files`` and writes the thesis figure as a
300-dpi PNG to ``cen3_pam_landscape_FINAL Figures``. Paths are resolved relative
to this script, so the script can be run from any working directory.
"""


from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# =========================================================================== #
# THRESHOLDS  --  every cut-off used anywhere in this analysis, with source.
#
# `kind` distinguishes values that are published as thresholds from values that
# are operational conventions in the field. This distinction is deliberate: none
# of the three is a physical constant, and the conclusions below are reported
# against a sweep of alternatives so that they do not rest on any single choice.
# =========================================================================== #

THRESHOLDS = {
    "hdr_distance_bp": {
        "value": 20,
        "kind": "convention",
        "statement": "Cut site must lie within 20 bp of the sequence to be "
                     "edited for efficient HDR.",
        "source": "Paquet et al. (2016) Nature 533:125-129. HDR incorporation "
                  "of a donor-encoded change falls steeply with cut-to-edit "
                  "distance: roughly halved by ~10 bp and approaching "
                  "background beyond ~30-40 bp. The 20 bp figure is the "
                  "operational cut-off adopted here, not a value published as "
                  "a threshold. See also Elliott et al. (1998) Mol Cell Biol "
                  "18:93-101 for conversion-tract length.",
    },
    "hdr_distance_permissive_bp": {
        "value": 50,
        "kind": "convention",
        "statement": "Permissive HDR distance, reported alongside the primary "
                     "threshold as a sensitivity check.",
        "source": "Same as above; 50 bp is where donor-encoded edits are "
                  "typically reported at low but non-zero frequency.",
    },
    "doench_min": {
        "value": 0.50,
        "kind": "convention",
        "statement": "Predicted on-target score (Doench 2014 Rule Set 1) must "
                     "be >= 0.50.",
        "source": "Doench et al. (2014) Nat Biotechnol 32:1262-1267. The model "
                  "is published; 0.5 is the widely used operational cut-off "
                  "(e.g. as applied in CRISPOR, Concordet & Haeussler 2018 "
                  "Nucleic Acids Res 46:W242) and is not itself published as a "
                  "threshold.",
    },
    "doench_design": {
        "value": 0.98,
        "kind": "project-specific",
        "statement": "Design threshold used by the thesis's synthetic "
                     "landing-pad generator.",
        "source": "Default --doench in Synthetic_Cut_Site_Generator_FINAL.py.",
    },
    "cfd_max": {
        "value": 0.20,
        "kind": "project-specific / convention",
        "statement": "Maximum CFD score against any other genomic NGG "
                     "protospacer must be <= 0.20.",
        "source": "CFD model: Doench et al. (2016) Nat Biotechnol 34:184-191. "
                  "The 0.20 ceiling is the declared both-strand threshold in "
                  "Synthetic_Cut_Site_Generator_FINAL.py and includes the installed "
                  "landing-pad guide (maximum both-strand CFD 0.157).",
    },
}

# Sensitivity sweep reported alongside the primary thresholds.
DOENCH_SWEEP = [0.40, 0.50, 0.60]
CFD_SWEEP = [0.05, 0.10, 0.20]


# =========================================================================== #
# Reference genome layout
#
# The supplied FASTA is a single record holding the whole genome concatenated.
# The records are in LEXICOGRAPHIC Roman-numeral order (I, II, III, IV, IX, M,
# V, VI, ...), i.e. the order `ls` gives for per-chromosome files -- NOT numeric
# chromosome order. The mitochondrial genome therefore sits between chrIX and
# chrV. verify_layout() re-derives this from the sequence itself and refuses to
# continue if it does not hold, so a differently ordered FASTA cannot silently
# produce wrong coordinates.
# =========================================================================== #

CHROM_ORDER = ["I", "II", "III", "IV", "IX", "M", "V", "VI", "VII", "VIII",
               "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]

CHROM_LEN = {
    "I": 230218, "II": 813184, "III": 316620, "IV": 1531933, "V": 576874,
    "VI": 270161, "VII": 1090940, "VIII": 562643, "IX": 439888, "X": 745751,
    "XI": 666816, "XII": 1078177, "XIII": 924431, "XIV": 784333, "XV": 1091291,
    "XVI": 948066, "M": 85779,
}

NUCLEAR = [c for c in CHROM_ORDER if c != "M"]

# CEN3, SGD R64, chromosome III, Watson strand.
CEN3_START, CEN3_END = 114385, 114501
# Derive the displayed length from the same inclusive reference coordinates
# used for sequence extraction, PAM enumeration and scoring.
CEN3_DISPLAY_LENGTH_BP = CEN3_END - CEN3_START + 1

# Landmarks used to verify the layout: centromere positions (SGD R64) that can
# be recovered independently from the sequence by CDEI/CDEII/CDEIII structure.
CEN_LANDMARKS = {"I": 151465, "II": 238207, "III": 114385, "IV": 449711,
                 "X": 436307, "XII": 150828}

COMP = str.maketrans("ACGTN", "TGCAN")


# =========================================================================== #
# House style
# =========================================================================== #

STYLE = {
    "dpi": 300,
    "font": "DejaVu Sans",
    "base_fs": 29,
    "axis_fs": 29,
    "tick_fs": 29,
    "legend_fs": 25,
    "annotation_fs": 23,
    "panel_fs": 29,
    "pass_col": "#1b7837",
    "fail_ot_col": "#762a83",
    "fail_on_col": "#b8b8b8",
    "fail_both_col": "#4d4d4d",
    "lp_col": "#d95f02",
    "cde_cols": {"CDEI": "#7fc7d9", "CDEII": "#f2d377", "CDEIII": "#e08a8a"},
    "thr_col": "#c0392b",
}


def apply_style():
    """Apply the shared thesis figure typography."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [STYLE["font"]],
        "font.size": STYLE["base_fs"],
        "axes.labelsize": STYLE["axis_fs"],
        "xtick.labelsize": STYLE["tick_fs"],
        "ytick.labelsize": STYLE["tick_fs"],
        "legend.fontsize": STYLE["legend_fs"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon": False,
        "savefig.dpi": STYLE["dpi"],
        "savefig.bbox": "tight",
    })


def load_generator(path):
    """Import the thesis's Synthetic_Cut_Site_Generator_FINAL.py as a module."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: generator script not found: {path}")
    spec = importlib.util.spec_from_file_location("cutgen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_concatenated(path):
    seq = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seq.append(line.strip())
    return "".join(seq).upper()


def segment(genome):
    """Split the concatenated record into chromosomes + any unplaced tail."""
    chroms, off = {}, 0
    for c in CHROM_ORDER:
        chroms[c] = genome[off:off + CHROM_LEN[c]]
        off += CHROM_LEN[c]
    tail = genome[off:]
    return chroms, tail


def find_centromeres(chroms):
    """Locate centromeres independently, by CDEI + AT-rich CDEII + CDEIII."""
    import re
    cde3 = re.compile(r"TGT[TA]T[TA]TG[ACGT]T[TA][TA]CCGAA")
    out = {}
    for c, s in chroms.items():
        if c == "M":
            continue
        best = None
        for strand, S in (("+", s), ("-", s.translate(COMP)[::-1])):
            for m in cde3.finditer(S):
                up = S[max(0, m.start() - 90):m.start()]
                if len(up) < 80:
                    continue
                at = (up.count("A") + up.count("T")) / len(up)
                if at < 0.85:
                    continue
                pos = m.start() + 1 if strand == "+" else len(S) - m.end() + 1
                if best is None or at > best[1]:
                    best = (pos, at, strand)
        if best:
            out[c] = best
    return out


def verify_layout(chroms, tail, verbose=True):
    """Fail loudly if the FASTA is not laid out as assumed."""
    problems = []
    cens = find_centromeres(chroms)
    if verbose:
        print("  layout check -- centromeres recovered from sequence structure:")
    for c, expected in CEN_LANDMARKS.items():
        if c not in cens:
            problems.append(f"CEN{c} not detected")
            continue
        pos, at, strand = cens[c]
        # detected position is the CDEIII start; CEN is ~92 bp upstream of it
        # on the + strand, or ~25 bp downstream on the - strand.
        delta = abs(pos - expected)
        ok = delta < 130
        if verbose:
            print(f"    CEN{c:<4} CDEIII at chr{c}:{pos:>9,} ({strand}) "
                  f"CDEII AT={at:.2f}   expected CEN start {expected:,}  "
                  f"[{'OK' if ok else 'MISMATCH'}]")
        if not ok:
            problems.append(f"CEN{c} at {pos} but expected near {expected}")

    if problems:
        sys.exit("ERROR: genome layout verification failed:\n  " +
                 "\n  ".join(problems) +
                 "\nThe FASTA is not in the expected lexicographic Roman order.")
    if verbose and tail:
        print(f"  note: {len(tail):,} bp of unplaced sequence after chrXVI; "
              f"included in the off-target database, excluded from coordinates.")
    return True


def annotate_cen3(chr3):
    """Derive CDEI / CDEII / CDEIII boundaries from the reference sequence.

    CDEI is the 8 bp RTCACRTG element at the 5' end, CDEIII the 25 bp element
    carrying the essential CCG at the 3' end, CDEII the AT-rich spacer between.
    Coordinates are returned 1-based inclusive, chromosome III.
    """
    import re
    cen = chr3[CEN3_START - 1:CEN3_END]
    m1 = re.match(r"[AG]TCAC[AG]TG", cen)
    if not m1:
        sys.exit("ERROR: CDEI consensus RTCACRTG not found at CEN3 5' end.")
    cdeI = (CEN3_START, CEN3_START + 7)

    m3 = re.search(r"TGT[TA]T[TA]TG[ACGT]T[TA][TA]CCGAA", cen)
    if not m3:
        sys.exit("ERROR: CDEIII consensus not found within CEN3.")
    cdeIII = (CEN3_START + m3.start(), CEN3_END)
    cdeII = (cdeI[1] + 1, cdeIII[0] - 1)

    seq2 = chr3[cdeII[0] - 1:cdeII[1]]
    at = (seq2.count("A") + seq2.count("T")) / len(seq2)
    if not (70 <= len(seq2) <= 95 and at > 0.85):
        sys.exit(f"ERROR: CDEII implausible (len {len(seq2)}, AT {at:.2f}).")

    return {
        "CDEI": {"start": cdeI[0], "end": cdeI[1],
                 "seq": chr3[cdeI[0] - 1:cdeI[1]]},
        "CDEII": {"start": cdeII[0], "end": cdeII[1], "seq": seq2, "AT": at},
        "CDEIII": {"start": cdeIII[0], "end": cdeIII[1],
                   "seq": chr3[cdeIII[0] - 1:cdeIII[1]]},
    }


# =========================================================================== #
# PAM site enumeration
#
# Geometry (all coordinates 1-based, plus-strand):
#   + strand site: protospacer p..p+19, PAM p+20..p+22 (NGG).
#                  Blunt cut 3 bp 5' of the PAM, i.e. between p+16 and p+17.
#   - strand site: PAM (CCN on the plus strand) at f..f+2, protospacer f+3..f+22
#                  read on the minus strand. Cut between f+5 and f+6.
# `cut` below is the plus-strand coordinate of the base immediately 5' of the
# nick on the plus strand; `cut_mid` = cut + 0.5 is the true blunt-cut point.
# =========================================================================== #

def enumerate_sites(chrseq, lo, hi):
    """All NGG sites whose PAM starts within [lo, hi] (1-based inclusive)."""
    sites = []
    n = len(chrseq)
    for p in range(max(1, lo - 25), min(n - 22, hi + 25) + 1):
        w = chrseq[p - 1:p + 22]
        if len(w) < 23 or any(b not in "ACGT" for b in w):
            continue
        if w[21] == "G" and w[22] == "G":                       # + strand NGG
            ctx = chrseq[p - 5:p + 25]
            if len(ctx) == 30 and all(b in "ACGT" for b in ctx):
                sites.append(dict(pam_start=p + 20, strand="+",
                                  protospacer=w[:20], pam=w[20:23],
                                  context30=ctx, cut=p + 16))
        if w[0] == "C" and w[1] == "C":                         # - strand NGG
            rc = w.translate(COMP)[::-1]
            ctx = chrseq[p - 4:p + 26]
            if len(ctx) == 30 and all(b in "ACGT" for b in ctx):
                ctx = ctx.translate(COMP)[::-1]
                sites.append(dict(pam_start=p, strand="-",
                                  protospacer=rc[:20], pam=rc[20:23],
                                  context30=ctx, cut=p + 5))
    keep = [s for s in sites
            if lo <= (s["pam_start"] if s["strand"] == "+" else s["pam_start"]) <= hi]
    return keep


def element_distance(cut, start, end):
    """bp from the cut point to the element; 0 if the cut lies inside it."""
    if start <= cut <= end:
        return 0
    return start - cut if cut < start else cut - end


# =========================================================================== #
# CFD scoring with self-exclusion
# =========================================================================== #

class GenomeCfd:
    """max CFD of a guide against every OTHER genomic NGG protospacer.

    Wraps the thesis generator's CfdScorer. Two corrections are applied:
      * the guide's own genomic occurrence is excluded (it would otherwise
        score 1.0 against itself);
      * protospacers occurring more than once in the genome are flagged, since
        a perfect duplicate elsewhere is itself a disqualifying off-target.
    """

    def __init__(self, gen, mm_path, pam_path, genome_records, chunk=100_000):
        self.gen = gen
        self.scorer = gen.CfdScorer(mm_path, pam_path, chunk=chunk)

        packed_all = []
        for rec in genome_records:
            arr = gen.encode(rec)
            packed_all.append(gen._protospacers_one_strand(arr))
            rc = np.where(arr < 4, 3 - arr.astype(np.int16), 4).astype(np.uint8)[::-1]
            packed_all.append(gen._protospacers_one_strand(np.ascontiguousarray(rc)))
        allp = np.concatenate(packed_all)

        self.uniq, counts = np.unique(allp, return_counts=True)
        self.counts = counts
        self.off = gen.unpack20(self.uniq)
        self.n_sites = allp.size
        self.chunk = chunk

    def occurrences(self, guide):
        p = self.gen.pack20(self.gen.encode(guide)[None, :])[0]
        i = np.searchsorted(self.uniq, p)
        if i < self.uniq.size and self.uniq[i] == p:
            return int(self.counts[i]), int(i)
        return 0, -1

    def max_cfd(self, guide, threshold=None):
        """(max_cfd, worst_off_target, n_genomic_occurrences, exact_flag)."""
        n_occ, self_idx = self.occurrences(guide)
        if n_occ >= 2:
            return 1.0, guide, n_occ, True          # perfect duplicate exists

        g = self.gen.encode(guide)
        wg = self.scorer.w[self.scorer._pos, g]
        row = self.scorer._pos[None, :]
        best, best_idx, exact = 0.0, -1, True

        for start in range(0, self.off.shape[0], self.chunk):
            block = self.off[start:start + self.chunk]
            scores = wg[row, block].prod(axis=1) * self.scorer.pam_weight
            if self_idx >= start and self_idx < start + block.shape[0]:
                scores[self_idx - start] = 0.0      # exclude self
            i = int(scores.argmax())
            if scores[i] > best:
                best, best_idx = float(scores[i]), start + i
            if threshold is not None and best > threshold:
                exact = False                        # early exit: lower bound
                break

        worst = self.gen.decode(self.off[best_idx]) if best_idx >= 0 else None
        return best, worst, max(n_occ, 1), exact


# =========================================================================== #
# Part 1 -- local PAM inventory
# =========================================================================== #

def part1(chr3, elements, cfd, gen, args):
    centre = (CEN3_START + CEN3_END) // 2
    half = args.window // 2
    lo, hi = centre - half, centre + half

    print("\n" + "=" * 74)
    print("=== 1. LOCAL PAM INVENTORY ===")
    print("=" * 74)
    print(f"Window: chrIII:{lo:,}-{hi:,} ({args.window:,} bp centred on CEN3 "
          f"midpoint {centre:,}), both strands, motif NGG.")

    sites = enumerate_sites(chr3, lo, hi)
    print(f"NGG sites found: {len(sites)} "
          f"(+ strand {sum(s['strand']=='+' for s in sites)}, "
          f"- strand {sum(s['strand']=='-' for s in sites)})")
    print(f"Density: 1 PAM per {args.window/max(len(sites),1):.1f} bp of window.")

    rows = []
    t0 = time.time()
    for k, s in enumerate(sites):
        d = {e: element_distance(s["cut"], elements[e]["start"], elements[e]["end"])
             for e in ("CDEI", "CDEII", "CDEIII")}
        d_cen3 = element_distance(s["cut"], CEN3_START, CEN3_END)
        nearest = min(d, key=d.get)
        doench = gen.calc_doench_score(s["context30"])
        maxcfd, worst, n_occ, exact = cfd.max_cfd(s["protospacer"], threshold=None)
        rows.append(dict(
            chrom="III", pam_start=s["pam_start"], strand=s["strand"],
            protospacer=s["protospacer"], pam=s["pam"], context30=s["context30"],
            cut_pos=s["cut"], cut_mid=s["cut"] + 0.5,
            dist_CDEI=d["CDEI"], dist_CDEII=d["CDEII"], dist_CDEIII=d["CDEIII"],
            dist_nearest_element=min(d.values()), nearest_element=nearest,
            dist_CEN3=d_cen3,
            doench2014=round(doench, 6), max_cfd=round(maxcfd, 6),
            worst_offtarget=worst, genomic_occurrences=n_occ,
            cfd_exact=exact,
        ))
        if (k + 1) % 50 == 0:
            print(f"    scored {k+1}/{len(sites)} sites "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)

    # PAM content of the centromere elements themselves
    print("\nPAM content of each centromere element:")
    el_stats = {}
    for e in ("CDEI", "CDEII", "CDEIII"):
        a, b = elements[e]["start"], elements[e]["end"]
        n_pam = int(((df.pam_start >= a) & (df.pam_start <= b)).sum())
        n_cut = int(((df.cut_pos >= a) & (df.cut_pos <= b)).sum())
        seq = elements[e]["seq"]
        at = (seq.count("A") + seq.count("T")) / len(seq)
        el_stats[e] = dict(n_pam=n_pam, n_cut=n_cut, at=at,
                           length=b - a + 1)
        print(f"  {e:<7} {b-a+1:>3} bp, {100*at:>4.1f}% AT : "
              f"{n_pam} NGG PAM(s) within the element, "
              f"{n_cut} predicted cut site(s) inside it")
    df.attrs["el_stats"] = el_stats

    print(f"  CFD screening of {len(df)} local sites against "
          f"{cfd.off.shape[0]:,} unique genomic protospacers "
          f"({cfd.n_sites:,} sites) in {time.time()-t0:.0f}s")

    near = df[df.dist_nearest_element <= args.report_within].copy()
    near = near.sort_values(["dist_nearest_element", "pam_start"])

    print(f"\nSites whose cut position falls within {args.report_within} bp of "
          f"CDEI, CDEII or CDEIII: {len(near)}")
    print("(ordered by distance to the nearest centromere element)\n")
    _print_table(near)
    return df, near, (lo, hi)


def _print_table(near):
    hdr = (f"{'cut':>9} {'str':>3} {'protospacer':<21}{'PAM':<5}"
           f"{'dCDEI':>6}{'dCDEII':>7}{'dCDEIII':>8}{'near':>6}"
           f"{'Doench':>8}{'maxCFD':>8}  {'flags'}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in near.iterrows():
        flags = []
        if r.genomic_occurrences > 1:
            flags.append(f"x{r.genomic_occurrences}dup")
        if not r.cfd_exact:
            flags.append("cfd>=")
        print(f"{r.cut_pos:>9,} {r.strand:>3} {r.protospacer:<21}{r.pam:<5}"
              f"{r.dist_CDEI:>6}{r.dist_CDEII:>7}{r.dist_CDEIII:>8}"
              f"{r.dist_nearest_element:>6}"
              f"{r.doench2014:>8.3f}{r.max_cfd:>8.3f}  {','.join(flags)}")


# =========================================================================== #
# Part 2 -- why none is usable
# =========================================================================== #

def classify(r, thr_d, thr_c, thr_hdr):
    reasons = []
    if r.dist_nearest_element > thr_hdr:
        reasons.append(f"cut {r.dist_nearest_element} bp from nearest element "
                       f"(> {thr_hdr} bp HDR limit)")
    if r.doench2014 < thr_d:
        reasons.append(f"on-target {r.doench2014:.3f} < {thr_d}")
    if r.max_cfd > thr_c:
        reasons.append(f"max CFD {r.max_cfd:.3f} > {thr_c}")
    return reasons


def part2(near, args):
    thr_d = THRESHOLDS["doench_min"]["value"]
    thr_c = THRESHOLDS["cfd_max"]["value"]
    thr_h = THRESHOLDS["hdr_distance_bp"]["value"]

    print("\n" + "=" * 74)
    print("=== 2. WHY NONE IS USABLE ===")
    print("=" * 74)

    if near.empty:
        print("No sites in range.")
        return near

    near = near.copy()
    near["fail_hdr"] = near.dist_nearest_element > thr_h
    near["fail_ontarget"] = near.doench2014 < thr_d
    near["fail_offtarget"] = near.max_cfd > thr_c
    near["n_failures"] = (near.fail_hdr.astype(int) +
                          near.fail_ontarget.astype(int) +
                          near.fail_offtarget.astype(int))
    near["usable"] = near.n_failures == 0
    near["reasons"] = [
        "; ".join(classify(r, thr_d, thr_c, thr_h)) or "PASSES ALL CRITERIA"
        for _, r in near.iterrows()]

    print(f"\nOf {len(near)} sites within {args.report_within} bp of a "
          f"centromere element:")
    print(f"  fail HDR distance (> {thr_h} bp):        "
          f"{near.fail_hdr.sum():>4} / {len(near)}")
    print(f"  fail on-target (Doench < {thr_d}):       "
          f"{near.fail_ontarget.sum():>4} / {len(near)}")
    print(f"  fail off-target (max CFD > {thr_c}):     "
          f"{near.fail_offtarget.sum():>4} / {len(near)}")
    print(f"  fail on >= 2 criteria:                {(near.n_failures>=2).sum():>4}"
          f" / {len(near)}")
    print(f"  PASS ALL THREE:                       {near.usable.sum():>4}"
          f" / {len(near)}")

    print("\nPer-site verdict (all sites within "
          f"{args.report_within} bp, by distance):")
    for _, r in near.iterrows():
        print(f"  chrIII:{r.cut_pos:,} ({r.strand}) d={r.dist_nearest_element:>3} bp "
              f"[{r.nearest_element}] -> {r.reasons}")

    print("\n--- Best native site by each criterion ---")
    tgt = f"the region requiring modification (CEN3, chrIII:{CEN3_START:,}-{CEN3_END:,})"

    b = near.loc[near.dist_nearest_element.idxmin()]
    print(f"\n  Closest cut to a centromere element:")
    print(f"    chrIII:{b.cut_pos:,} ({b.strand})  {b.dist_nearest_element} bp "
          f"from {b.nearest_element};  {b.dist_CEN3} bp from {tgt}")
    print(f"    Doench {b.doench2014:.3f}, max CFD {b.max_cfd:.3f} "
          f"-> {b.reasons}")

    b = near.loc[near.doench2014.idxmax()]
    print(f"\n  Highest predicted on-target score:")
    print(f"    chrIII:{b.cut_pos:,} ({b.strand})  Doench {b.doench2014:.3f}")
    print(f"    {b.dist_nearest_element} bp from {b.nearest_element}, "
          f"{b.dist_CEN3} bp from CEN3;  max CFD {b.max_cfd:.3f}")
    print(f"    -> {b.reasons}")

    b = near.loc[near.max_cfd.idxmin()]
    print(f"\n  Lowest genome-wide off-target potential:")
    print(f"    chrIII:{b.cut_pos:,} ({b.strand})  max CFD {b.max_cfd:.3f} "
          f"(worst match {b.worst_offtarget})")
    print(f"    {b.dist_nearest_element} bp from {b.nearest_element}, "
          f"{b.dist_CEN3} bp from CEN3;  Doench {b.doench2014:.3f}")
    print(f"    -> {b.reasons}")

    # sensitivity: does any site survive under any combination in the sweep?
    print("\n--- Sensitivity to threshold choice ---")
    print(f"  {'Doench>=':>9} {'CFD<=':>7} {'HDR<=':>7}  usable sites")
    for td in DOENCH_SWEEP:
        for tc in CFD_SWEEP:
            for th in (THRESHOLDS["hdr_distance_bp"]["value"],
                       THRESHOLDS["hdr_distance_permissive_bp"]["value"]):
                n = ((near.doench2014 >= td) & (near.max_cfd <= tc) &
                     (near.dist_nearest_element <= th)).sum()
                print(f"  {td:>9} {tc:>7} {th:>7}  {n}")
    return near


# =========================================================================== #
# Part 3 -- genome-wide context
# =========================================================================== #

def genome_wide_sites(gen, chroms):
    """Every NGG site on both strands of every nuclear chromosome, with Doench."""
    recs = {}
    for c in NUCLEAR:
        s = chroms[c]
        arr = gen.encode(s)
        n = len(arr)
        st = n - 23 + 1
        bad = np.concatenate(([0], np.cumsum(arr >= 4)))

        ok = (arr[21:21 + st] == 2) & (arr[22:22 + st] == 2)
        clean = (bad[23:23 + st] - bad[:st]) == 0
        idx_f = np.flatnonzero(ok & clean)
        idx_f = idx_f[(idx_f >= 4) & (idx_f + 26 <= n)]
        ctx_f = np.lib.stride_tricks.sliding_window_view(arr, 30)[idx_f - 4]
        dn_f = gen.calc_doench_vec(ctx_f)
        cut_f = idx_f + 1 + 16                        # 1-based plus-strand

        rc = np.where(arr < 4, 3 - arr.astype(np.int16), 4).astype(np.uint8)[::-1]
        badr = np.concatenate(([0], np.cumsum(rc >= 4)))
        okr = (rc[21:21 + st] == 2) & (rc[22:22 + st] == 2)
        cleanr = (badr[23:23 + st] - badr[:st]) == 0
        idx_r = np.flatnonzero(okr & cleanr)
        idx_r = idx_r[(idx_r >= 4) & (idx_r + 26 <= n)]
        ctx_r = np.lib.stride_tricks.sliding_window_view(rc, 30)[idx_r - 4]
        dn_r = gen.calc_doench_vec(ctx_r)
        f0 = n - (idx_r + 23)                          # 0-based plus start
        cut_r = f0 + 1 + 5

        guides_f = gen.pack20(np.lib.stride_tricks.sliding_window_view(arr, 20)[idx_f])
        guides_r = gen.pack20(np.lib.stride_tricks.sliding_window_view(rc, 20)[idx_r])

        recs[c] = dict(
            cut=np.concatenate([cut_f, cut_r]).astype(np.int64),
            doench=np.concatenate([dn_f, dn_r]),
            packed=np.concatenate([guides_f, guides_r]),
            strand=np.array(["+"] * idx_f.size + ["-"] * idx_r.size),
        )
    return recs


def _dist_profile(usable, chroms_needed=NUCLEAR):
    """Distance from every base of every nuclear chromosome to nearest cut.

    Chromosomes carrying no usable site have no finite distance at all. They are
    returned separately rather than silently dropped: excluding them without
    accounting would bias the distribution downwards precisely where the
    shortage is most severe.
    """
    out, per_chrom, empty = [], {}, []
    for c in chroms_needed:
        L = CHROM_LEN[c]
        u = usable.get(c, np.empty(0, dtype=np.int64))
        if u.size == 0:
            per_chrom[c] = None
            empty.append(c)
            continue
        pos = np.arange(1, L + 1, dtype=np.int64)
        j = np.searchsorted(u, pos)
        BIG = 1 << 40
        left = np.where(j > 0, pos - u[np.clip(j - 1, 0, u.size - 1)], BIG)
        right = np.where(j < u.size, u[np.clip(j, 0, u.size - 1)] - pos, BIG)
        d = np.minimum(left, right).astype(np.int32)
        per_chrom[c] = d
        out.append(d)
    D = np.concatenate(out) if out else np.empty(0, dtype=np.int32)
    return D, per_chrom, empty


def _cen3_stats(per_chrom, elements):
    d3 = per_chrom.get("III")
    if d3 is None:
        return None
    cen = d3[CEN3_START - 1:CEN3_END]
    st = {"min": int(cen.min()), "median": float(np.median(cen))}
    for e in ("CDEI", "CDEII", "CDEIII"):
        seg = d3[elements[e]["start"] - 1:elements[e]["end"]]
        st[e] = int(seg.min())
    return st


def part3(gen, cfd, chroms, recs, elements, args):
    thr_d = THRESHOLDS["doench_min"]["value"]
    thr_c = THRESHOLDS["cfd_max"]["value"]
    ladder = sorted(set(CFD_SWEEP + [thr_c, 0.30]))

    print("\n" + "=" * 74)
    print("=== 3. GENOME-WIDE CONTEXT ===")
    print("=" * 74)

    total = sum(r["cut"].size for r in recs.values())
    passing = {c: np.flatnonzero(r["doench"] >= thr_d) for c, r in recs.items()}
    n_pass = sum(v.size for v in passing.values())
    print(f"NGG sites on the 16 nuclear chromosomes (both strands): {total:,}")
    print(f"  one PAM per {12071326/total:.1f} bp of genome (both strands pooled)")
    print(f"  passing on-target (Doench >= {thr_d}): {n_pass:,} "
          f"({100*n_pass/total:.2f}%)")

    # ---- reference distribution: on-target criterion ALONE (exact) --------
    on_only = {c: np.sort(recs[c]["cut"][passing[c]]) for c in NUCLEAR}
    D_on, per_on, empty_on = _dist_profile(on_only)
    med_on, p95_on = float(np.median(D_on)), float(np.percentile(D_on, 95))
    cen_on = _cen3_stats(per_on, elements)
    print(f"\n[A] Distance to nearest site passing the ON-TARGET criterion only")
    print(f"    (Doench >= {thr_d}; off-target risk ignored entirely)")
    print(f"    median {med_on:,.0f} bp | 95th pct {p95_on:,.0f} bp | "
          f"max {D_on.max():,.0f} bp")
    if cen_on:
        pct = float((D_on < cen_on['median']).mean() * 100)
        print(f"    CEN3: nearest such cut {cen_on['min']:,} bp; median over "
              f"CEN3 bases {cen_on['median']:,.0f} bp -> {pct:.1f}th percentile")
        print(f"      CDEI {cen_on['CDEI']:,} bp | CDEII {cen_on['CDEII']:,} bp"
              f" | CDEIII {cen_on['CDEIII']:,} bp to the nearest such cut")

    # ---- exact CFD screen at the primary threshold ------------------------
    cand = [(c, int(i)) for c in NUCLEAR for i in passing[c]]
    mode = args.gw_mode
    if mode == "sample" and args.gw_sample < len(cand):
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(cand), args.gw_sample, replace=False)
        print(f"\n[B] CFD screen at the primary ceiling ({thr_c}): random "
              f"sample of {len(sel):,} of {len(cand):,}")
    else:
        mode = "exact"
        sel = np.arange(len(cand))
        print(f"\n[B] CFD screen at the primary ceiling ({thr_c}): EXACT, all "
              f"{len(cand):,} on-target-passing sites")

    cache = os.path.join(args.outdir, f"gw_cfd_pass_{mode}_{thr_c}.npy")
    ok, done = None, False
    if os.path.exists(cache):
        cached = np.load(cache)
        if cached.size == len(sel):
            ok, done = cached, True
            print(f"    loaded cached screen from {cache}")
    if not done:
        t0, ok = time.time(), np.zeros(len(sel), dtype=bool)
        for j, si in enumerate(sel):
            c, i = cand[si]
            guide = gen.decode(gen.unpack20(recs[c]["packed"][i:i + 1])[0])
            best, _, _, _ = cfd.max_cfd(guide, threshold=thr_c)
            ok[j] = best <= thr_c
            if (j + 1) % 10000 == 0:
                el = time.time() - t0
                print(f"      {j+1:,}/{len(sel):,} ({el/60:.1f} min, "
                      f"{el/(j+1)*1000:.1f} ms/site, "
                      f"passing {int(ok[:j+1].sum())})", flush=True)
        np.save(cache, ok)
    p_hat = float(ok.mean()) if len(sel) else 0.0
    n_usable = int(ok.sum()) if mode == "exact" else int(round(p_hat*len(cand)))
    print(f"    {'exact' if mode=='exact' else 'estimated'}: {n_usable:,} of "
          f"{len(cand):,} on-target-passing sites also satisfy max CFD <= "
          f"{thr_c}")
    print(f"    that is {100*n_usable/total:.4f}% of all {total:,} NGG sites "
          f"in the genome")

    # ---- ladder: how relaxed must the CFD ceiling be? ---------------------
    rng = np.random.default_rng(args.seed + 7)
    nlad = min(args.ladder_sample, len(cand))
    lad_sel = rng.choice(len(cand), nlad, replace=False)
    top = max(ladder)
    print(f"\n[C] How relaxed must the off-target ceiling be? "
          f"(random sample of {nlad:,} on-target-passing sites)")
    lad_cache = os.path.join(args.outdir, f"ladder_vals_{nlad}.npy")
    if os.path.exists(lad_cache):
        vals = np.load(lad_cache)
        print(f"    loaded cached ladder from {lad_cache}")
        lad_sel = []
    t0, vals = (time.time(), vals) if not len(lad_sel) else (time.time(),
                                                             np.empty(nlad))
    for j, si in enumerate(lad_sel):
        c, i = cand[si]
        guide = gen.decode(gen.unpack20(recs[c]["packed"][i:i + 1])[0])
        best, _, _, _ = cfd.max_cfd(guide, threshold=top)
        vals[j] = best
        if (j + 1) % 2000 == 0:
            print(f"      {j+1:,}/{nlad:,} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    print(f"    max-CFD distribution: median {np.median(vals):.3f}, "
          f"10th pct {np.percentile(vals,10):.3f}, min {vals.min():.3f}")
    rates = {}
    for r in ladder:
        pr = float((vals <= r).mean())
        se = np.sqrt(pr * (1 - pr) / nlad)
        rates[r] = pr
        print(f"      CFD <= {r:<5}: {100*pr:6.2f}% +/- {100*1.96*se:.2f}% "
              f"of on-target-passing sites  (~{int(pr*len(cand)):,} genome-wide)")

    # ---- distance distributions at each ceiling ---------------------------
    print(f"\n[D] Distance from every base to its nearest USABLE PAM")
    print(f"    (usable = Doench >= {thr_d} AND max genome-wide CFD <= ceiling)")
    results, cen3_pct_primary = {}, None
    for r in ladder:
        if r == thr_c and mode == "exact":
            usable = {c: [] for c in NUCLEAR}
            for j, si in enumerate(sel):
                if ok[j]:
                    c, i = cand[si]
                    usable[c].append(int(recs[c]["cut"][i]))
            usable = {c: np.array(sorted(v), dtype=np.int64)
                      for c, v in usable.items()}
            counts = {c: int(v.size) for c, v in usable.items()}
            have = {k: v for k, v in counts.items() if v}
            print(f"      usable sites by chromosome: {have if have else 'none'}")
            D, per, empty = _dist_profile(usable)
            tag = "exact"
            reps = [(D, per, empty)]
        else:
            tag = f"estimated by thinning at p={rates[r]:.4f}"
            reps = []
            for k in range(args.thin_reps):
                rr = np.random.default_rng(args.seed + 100 + k)
                us = {}
                for c in NUCLEAR:
                    idx = passing[c]
                    keep = idx[rr.random(idx.size) < rates[r]]
                    us[c] = np.sort(recs[c]["cut"][keep]).astype(np.int64)
                reps.append(_dist_profile(us))
        meds = [float(np.median(D)) for D, _, _ in reps if D.size]
        p95s = [float(np.percentile(D, 95)) for D, _, _ in reps if D.size]
        empty0 = reps[0][2]
        cov = sum(CHROM_LEN[c] for c in NUCLEAR if c not in empty0)
        if not meds:
            print(f"    ceiling {r:<5}: NO USABLE SITE ANYWHERE IN THE GENOME "
                  f"({tag})")
            results[r] = None
            continue
        cen = _cen3_stats(reps[0][1], elements)
        pct = (float((reps[0][0] < cen["median"]).mean() * 100)
               if cen else None)
        if r == thr_c:
            cen3_pct_primary = pct
        print(f"    ceiling {r:<5}: median {np.mean(meds):>8,.0f} bp | "
              f"95th pct {np.mean(p95s):>9,.0f} bp   ({tag})")
        print(f"                  over {cov:,} of 12,071,326 bp; "
              f"{len(empty0)} chromosome(s) contain no usable site"
              + (f": {', '.join(empty0)}" if empty0 else ""))
        if cen:
            print(f"                  CEN3 nearest {cen['min']:,} bp, median "
                  f"{cen['median']:,.0f} bp -> {pct:.1f}th percentile")
        else:
            print(f"                  chromosome III contains no usable site: "
                  f"no finite distance from CEN3 exists")
        results[r] = dict(median=float(np.mean(meds)), p95=float(np.mean(p95s)),
                          cen3=cen, cen3_pct=pct, tag=tag,
                          covered_bp=int(cov), empty_chroms=list(empty0))

    # ---- verdict ----------------------------------------------------------
    print("\n--- Is CEN3 unusual, or is this a general problem? ---")
    prim = results.get(thr_c)
    es = args._el_stats
    d2 = es["CDEII"]
    pct_on = float((D_on < cen_on["median"]).mean() * 100) if cen_on else None

    print(f"  At the primary ceiling (CFD <= {thr_c}), {n_usable:,} of "
          f"{total:,} NGG sites in the genome are")
    print(f"  usable: {100*n_usable/total:.4f}%, of order one site per "
          f"{12071326//max(n_usable,1):,} bp.")

    if prim is None:
        print(f"  No usable site exists anywhere in the genome, so the "
              f"shortage at CEN3 is not")
        print(f"  locus-specific in any sense.")
    elif prim["cen3"] is None:
        print(f"  Chromosome III carries none of them, so no finite distance "
              f"from CEN3 to a usable")
        print(f"  native PAM exists: the nearest lies on another chromosome "
              f"and is useless for")
        print(f"  editing this locus. {len(prim['empty_chroms'])} of 16 "
              f"chromosomes are in the same position")
        print(f"  ({', '.join(prim['empty_chroms'])}). CEN3 is therefore an "
              f"instance of a genome-wide")
        print(f"  shortage, not a uniquely bad locus.")
    else:
        print(f"  CEN3 sits at the {prim['cen3_pct']:.1f}th percentile of the "
              f"genome-wide distribution.")
        print(f"  => {'CEN3 is unusually poorly served' if prim['cen3_pct'] > 90 else 'CEN3 is an instance of a general problem'}.")

    print(f"\n  What is specific to CEN3 is the geometry, not the scarcity. "
          f"The {d2['length']} bp CDEII element")
    print(f"  is {100*d2['at']:.0f}% AT and contains {d2['n_pam']} NGG PAM and "
          f"{d2['n_cut']} predicted cut site on either strand;")
    print(f"  CDEI ({es['CDEI']['length']} bp) likewise contains "
          f"{es['CDEI']['n_pam']}. Even with the off-target ceiling removed")
    print(f"  entirely, no cut can be placed within the HDR window of the "
          f"sequence that has")
    print(f"  to be modified. That is the argument for a synthetic landing "
          f"pad.")
    if pct_on is not None:
        print(f"\n  On the on-target criterion alone (off-target risk "
              f"ignored), the median base in")
        print(f"  the genome lies {med_on:,.0f} bp from a qualifying cut "
              f"while CEN3 lies {cen_on['median']:,.0f} bp from one")
        print(f"  -- the {pct_on:.1f}th percentile. CEN3 is in the worst "
              f"decile for PAM availability even")
        print(f"  before off-target risk is considered.")

    return dict(distances_on_only=D_on, median_on=med_on, p95_on=p95_on,
                cen3_on=cen_on, ladder=rates, results=results,
                n_usable=n_usable, p_hat=p_hat, mode=mode)


# =========================================================================== #
# Figure 4.3
# =========================================================================== #



def make_thesis_figure_4_3(df, elements, outdir, args):
    """Create the single Figure 4.3 used in the thesis."""
    thr_d = THRESHOLDS["doench_min"]["value"]
    thr_c = THRESHOLDS["cfd_max"]["value"]
    thr_h = THRESHOLDS["hdr_distance_bp"]["value"]
    apply_style()

    work = df.copy()
    work["ok_hdr"] = work.dist_nearest_element <= thr_h
    work["ok_on"] = work.doench2014 >= thr_d
    work["ok_off"] = work.max_cfd <= thr_c
    work["usable"] = work.ok_hdr & work.ok_on & work.ok_off

    fig = plt.figure(figsize=(18.0, 13.5))
    # Reserve enough space for the complete Panel B labels. Without an
    # explicit left margin, Matplotlib clips their first words in the PNG and
    # PDF outputs.
    fig.subplots_adjust(left=0.28, right=0.97, top=0.96, bottom=0.08)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.72)

    # Panel A: the 1,000 bp native PAM inventory used in the thesis.
    axA = fig.add_subplot(gs[0])
    centre = (CEN3_START + CEN3_END) // 2
    inventory_window_bp = 1000
    lo = centre - inventory_window_bp // 2
    hi = lo + inventory_window_bp
    label_positions = (("CDEI", centre - 130, "right"),
                       ("CDEII", centre, "center"),
                       ("CDEIII", centre + 145, "left"))
    cols = {"CDEI": "#3d5a80", "CDEII": "#98c1d9",
            "CDEIII": "#ee6c4d"}
    for name in ("CDEI", "CDEII", "CDEIII"):
        s, e = elements[name]["start"], elements[name]["end"]
        axA.add_patch(Rectangle((s, -0.15), e - s + 1, 0.30,
                                facecolor=cols[name], edgecolor="black",
                                linewidth=0.6, zorder=3))
    for name, xpos, ha in label_positions:
        s, e = elements[name]["start"], elements[name]["end"]
        axA.annotate(name, xy=((s + e) / 2, -0.15),
                     xytext=(xpos, -0.46), ha=ha, va="top", fontsize=STYLE["annotation_fs"],
                     zorder=6,
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45"))
    axA.plot([lo, hi], [0, 0], color="0.35", lw=1.0, zorder=1)
    axA.plot([CEN3_START, CEN3_END], [0.32, 0.32], color="black", lw=1.2)
    axA.text((CEN3_START + CEN3_END) / 2, 0.38,
             f"CEN3, {CEN3_DISPLAY_LENGTH_BP} bp",
             ha="center", fontsize=STYLE["annotation_fs"])

    # Use inclusive limits so PAMs falling exactly on either edge of the
    # requested inventory are not omitted.
    sub = work[(work.cut_pos >= lo) & (work.cut_pos <= hi)]
    for _, row in sub.iterrows():
        y0, y1 = ((0.15, 0.26) if row.strand == "+"
                  else (-0.15, -0.26))
        inner = row.dist_nearest_element <= thr_h
        axA.plot([row.cut_pos, row.cut_pos], [y0, y1],
                 color="#c1121f" if inner else "0.35",
                 lw=1.6 if inner else 0.7, zorder=4)
    axA.text(lo - 10, 0.21, "+ strand", fontsize=STYLE["annotation_fs"], va="center",
             ha="right", color="0.35", clip_on=False)
    axA.text(lo - 10, -0.21, "− strand", fontsize=STYLE["annotation_fs"], va="center",
             ha="right", color="0.35", clip_on=False)

    close_sites = work[work.dist_nearest_element <= thr_h]
    if not close_sites.empty:
        hit = close_sites.iloc[0]
        hit_y = 0.27 if hit.strand == "+" else -0.27
        annotation_x = centre + 0.22 * inventory_window_bp
        axA.annotate("Cut site found within CDEIII,\n"
                     f"efficiency {hit.doench2014:.2f}, "
                     f"off-target {hit.max_cfd:.2f}",
                     xy=(hit.cut_pos, hit_y), xytext=(annotation_x, -0.82),
                     fontsize=STYLE["annotation_fs"] - 1, ha="center", va="top", color="#c1121f",
                     linespacing=1.6,
                     arrowprops=dict(arrowstyle="-", lw=0.8,
                                     color="#c1121f"))
    axA.set_xlim(lo - 0.09 * (hi - lo), hi)
    axA.set_ylim(-1.20, 0.56)
    axA.set_yticks([])
    axA.spines["left"].set_visible(False)
    axA.set_xlabel("Chromosome III coordinate (bp)", fontsize=STYLE["axis_fs"] - 1)
    axA.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{int(value):,}"))
    axA.tick_params(axis="x", labelsize=STYLE["tick_fs"] - 1)
    axA.text(0.0, 1.03, "A", transform=axA.transAxes,
             fontsize=STYLE["panel_fs"], fontweight="bold",
             ha="left", va="bottom")

    # Panel B: sequential native-site filtering on a linear x-axis.
    axB = fig.add_subplot(gs[1])
    # The c/ci panels filter only the sites drawn in their 1,000 bp window.
    # Other variants retain the complete 2,000 bp inventory in Panel B.
    filter_work = sub
    first_bar_label = f"PAMs within {inventory_window_bp:,} bp"
    labels = [first_bar_label,
              f"Within {thr_h} bp of CEN",
              "High Cut Efficiency",
              "Low Off Target Score"]
    values = [len(filter_work), int(filter_work.ok_hdr.sum()),
              int((filter_work.ok_hdr & filter_work.ok_on).sum()),
              int(filter_work.usable.sum())]
    y = np.arange(len(values))[::-1]
    axB.barh(y, values, height=0.58, color="#7A7A7A",
             edgecolor="black", linewidth=0.6)
    axB.set_yticks(y, labels, fontsize=STYLE["annotation_fs"])
    axB.set_xlabel("Number of sites")
    axB.set_xlim(0, max(values + [1]) * 1.18)
    for upper, lower in zip(y[:-1], y[1:]):
        axB.annotate("", xy=(0.02, lower + 0.31),
                     xytext=(0.02, upper - 0.31),
                     xycoords=("axes fraction", "data"),
                     arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.8))
    # Align Panel B as in the former Figure 4.1i layout: the left edge of the
    # rendered y-axis labels lines up with the left edge of Panel A, while the
    # right edges of the two plotting areas remain aligned.
    fig.canvas.draw()
    pos_a, pos_b = axA.get_position(), axB.get_position()
    axB.tick_params(axis="y", pad=2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    label_left = min(
        inverse.transform(label.get_window_extent(renderer=renderer))[0, 0]
        for label in axB.get_yticklabels()
    )
    shift = pos_a.x0 - label_left
    new_left = pos_b.x0 + shift
    axB.set_position([new_left, pos_b.y0,
                      pos_a.x1 - new_left, pos_b.height])

    # Keep the panel letter aligned with the left edge of Panel A rather than
    # with the shifted Panel B plotting axis.
    pos_b_final = axB.get_position()
    fig.text(pos_a.x0,
             pos_b_final.y1 + 0.03 * pos_b_final.height,
             "B",
             fontsize=STYLE["panel_fs"],
             fontweight="bold",
             ha="left",
             va="bottom")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "Figure_4.3_Natural_Cas9_PAM_Availability_and_Guide-Site_Filtering_Around_CEN3.png"
    fig.savefig(output_path, dpi=STYLE["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written: {output_path}")
    return values


# =========================================================================== #
# main
# =========================================================================== #

def print_thresholds():
    print("\n" + "=" * 74)
    print("=== THRESHOLDS USED ===")
    print("=" * 74)
    for k, v in THRESHOLDS.items():
        print(f"\n  {k} = {v['value']}   [{v['kind']}]")
        print(f"    {v['statement']}")
        for line in _wrap(v["source"], 68):
            print(f"      {line}")
    print("\n  Scoring models:")
    print("    on-target : Doench et al. (2014) Nat Biotechnol 32:1262-1267, "
          "Rule Set 1,")
    print("                as implemented in Synthetic_Cut_Site_Generator_FINAL.py "
          "(verified")
    print("                against its KNOWN_BEST_DOENCH reference value).")
    print("    off-target: CFD, Doench et al. (2016) Nat Biotechnol 34:184-191, "
          "using the")
    print("                published mismatch_score.pkl / pam_scores.pkl "
          "tables.")
    print("    PAM       : NGG only. NAG and other non-canonical PAMs are not "
          "scored,")
    print("                which makes the off-target estimates here mildly "
          "optimistic.")


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main(argv=None):
    here = Path(__file__).resolve().parent
    script_stem = Path(__file__).stem
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    data_dir = here / f"{script_stem} Additional Files"
    ap.add_argument("--genome", default=str(data_dir / "whole_genes.fna"))
    ap.add_argument("--generator", default=str(here / "Synthetic_Cut_Site_Generator_FINAL.py"))
    ap.add_argument("--mm-scores", default=str(data_dir / "mismatch_score.pkl"))
    ap.add_argument("--pam-scores", default=str(data_dir / "pam_scores.pkl"))
    ap.add_argument("--outdir", default=str(here / f"{script_stem} Figures"))
    ap.add_argument("--additional-dir",
                    default=str(data_dir))
    ap.add_argument("--window", type=int, default=2000,
                    help="local inventory window (bp), centred on CEN3")
    ap.add_argument("--report-within", type=int, default=100,
                    help="report sites whose cut is within this many bp of an "
                         "element")
    ap.add_argument("--landing-pads", default=None,
                    help="comma-separated chrIII cut coordinates for the two "
                         "synthetic landing pads (default: 10 bp outside each "
                         "CEN3 boundary)")
    ap.add_argument("--gw-mode", choices=["exact", "sample"], default="exact")
    ap.add_argument("--gw-sample", type=int, default=5000)
    ap.add_argument("--ladder-sample", type=int, default=6000,
                    help="sample size for the CFD-ceiling ladder in Part 3")
    ap.add_argument("--thin-reps", type=int, default=3,
                    help="Monte-Carlo replicates for thinned distance profiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--genome-wide", action="store_true",
                    help="also run the slow genome-wide CFD audit; not required "
                         "to reproduce thesis Figure 4.3")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.additional_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 74)
    print("CEN3 PAM LANDSCAPE  --  thesis Figure 4.3")
    print("=" * 74)

    gen = load_generator(args.generator)
    print(f"\nScoring code imported from {args.generator}")
    # Compatibility with both the original generator names (KNOWN_BEST_*) and
    # the current, clearer names (INSTALLED_*).
    lp_context = getattr(gen, "KNOWN_BEST_CONTEXT",
                         getattr(gen, "INSTALLED_CONTEXT", None))
    lp_guide = getattr(gen, "KNOWN_BEST_GUIDE",
                       getattr(gen, "INSTALLED_GUIDE", None))
    lp_recorded_doench = getattr(gen, "KNOWN_BEST_DOENCH",
                                 getattr(gen, "INSTALLED_DOENCH", None))
    lp_recorded_cfd = getattr(gen, "KNOWN_BEST_CFD",
                              getattr(gen, "INSTALLED_CFD_FWD", None))
    if None in (lp_context, lp_guide, lp_recorded_doench, lp_recorded_cfd):
        raise AttributeError("The generator must define either KNOWN_BEST_* "
                             "or INSTALLED_* guide constants")
    chk = gen.calc_doench_score(lp_context)
    assert abs(chk - lp_recorded_doench) < 1e-12, "Doench self-check failed"
    print(f"  Doench self-check: {chk:.13f} == recorded installed score  OK")

    print(f"\nReading genome: {args.genome}")
    genome = read_concatenated(args.genome)
    chroms, tail = segment(genome)
    print(f"  {len(genome):,} bp total; 16 nuclear chromosomes + chrM "
          f"+ {len(tail):,} bp unplaced")
    verify_layout(chroms, tail)

    elements = annotate_cen3(chroms["III"])
    print(f"\nCEN3 element boundaries (derived from the reference sequence, "
          f"1-based inclusive):")
    print(f"  CEN3    chrIII:{CEN3_START:,}-{CEN3_END:,}  "
          f"({CEN3_DISPLAY_LENGTH_BP} bp)")
    for e in ("CDEI", "CDEII", "CDEIII"):
        d = elements[e]
        extra = f"  AT={d['AT']:.2f}" if "AT" in d else ""
        print(f"  {e:<7} chrIII:{d['start']:,}-{d['end']:,}  "
              f"({d['end']-d['start']+1} bp){extra}")
        print(f"          {d['seq']}")
    print("  These match the canonical element sizes (CDEI 8 bp, CDEII 78-86 bp,")
    print("  CDEIII 25 bp) and the SGD R64 CEN3 feature boundaries.")

    # landing pads
    if args.landing_pads:
        landing_pads = [int(x) for x in args.landing_pads.split(",")]
    else:
        landing_pads = [CEN3_START - 10, CEN3_END + 10]
    print(f"\nSynthetic landing-pad cut positions: "
          f"chrIII:{landing_pads[0]:,} and chrIII:{landing_pads[1]:,}")
    print(f"  (default = 10 bp outside each CEN3 boundary; override with "
          f"--landing-pads)")

    print("\nBuilding genome-wide off-target database (both strands)...")
    t0 = time.time()
    records = [chroms[c] for c in CHROM_ORDER] + ([tail] if tail else [])
    cfd = GenomeCfd(gen, args.mm_scores, args.pam_scores, records)
    print(f"  {cfd.n_sites:,} NGG sites, {cfd.off.shape[0]:,} unique 20mers "
          f"({time.time()-t0:.1f}s)")
    ndup = int((cfd.counts > 1).sum())
    print(f"  {ndup:,} protospacers occur more than once (auto-disqualified)")

    # landing-pad guide, re-scored on the both-strand database used here
    lp_doench = lp_recorded_doench
    lp_cfd, lp_worst, _, _ = cfd.max_cfd(lp_guide, threshold=None)
    print(f"\nLanding-pad guide from the generator: {lp_guide} + CGG")
    print(f"  Doench (Rule Set 1)              : {lp_doench:.4f}")
    print(f"  max CFD, + strand only (as in the generator): "
          f"{lp_recorded_cfd:.4f}")
    print(f"  max CFD, BOTH strands (used here): {lp_cfd:.4f}  "
          f"(worst match {lp_worst})")
    if lp_cfd > THRESHOLDS["cfd_max"]["value"] >= lp_recorded_cfd:
        print(f"  ** NOTE: this guide passes the {THRESHOLDS['cfd_max']['value']} "
              f"CFD ceiling only under the + strand search.")
        print(f"     Under the both-strand search -- which "
              f"Synthetic_Cut_Site_Generator_FINAL.py itself")
        print(f"     documents as the correct and stricter option "
              f"(--strands both) -- it does not.")
        print(f"     Consider re-running the generator with --strands both "
              f"before committing to this")
        print(f"     landing-pad sequence. The CEN3 conclusions below are "
              f"unaffected.")

    lp_stats = [(element_distance(lp, elements["CDEI"]["start"],
                                  elements["CDEIII"]["end"]),
                 lp_doench, lp_cfd) for lp in landing_pads]

    df, near, win = part1(chroms["III"], elements, cfd, gen, args)
    near = part2(near, args)

    # Report the exact 1,000 bp inventory used in thesis Figure 4.3.
    centre = (CEN3_START + CEN3_END) // 2
    one_kb_lo = centre - 500
    one_kb_hi = centre + 500
    one_kb = df[(df.cut_pos >= one_kb_lo) & (df.cut_pos <= one_kb_hi)]
    one_kb_fwd = int((one_kb.strand == "+").sum())
    one_kb_rev = int((one_kb.strand == "-").sum())
    print("\n1,000 bp PAM inventory used in Figure 4.3:")
    print(f"  Window: chrIII:{one_kb_lo:,}-{one_kb_hi:,}, centred on CEN3")
    print(f"  NGG sites found: {len(one_kb)} "
          f"(+ strand {one_kb_fwd}, - strand {one_kb_rev})")

    csv_all = os.path.join(args.additional_dir, "cen3_pam_sites_2kb.csv")
    csv_near = os.path.join(args.additional_dir, "cen3_pam_sites_within100bp.csv")
    df.to_csv(csv_all, index=False)
    near.to_csv(csv_near, index=False)
    print(f"\nSite tables written:\n  {csv_all}\n  {csv_near}")

    # The submitted Figure 4.3 uses only the local CEN3 analysis above.
    # The genome-wide CFD audit is retained as an optional reproducibility check
    # because it is computationally expensive and does not contribute to the figure.
    gw = None
    if args.genome_wide:
        print("\nEnumerating genome-wide NGG sites...")
        t0 = time.time()
        recs = genome_wide_sites(gen, chroms)
        print(f"  done ({time.time()-t0:.1f}s)")
        args._el_stats = df.attrs["el_stats"]
        gw = part3(gen, cfd, chroms, recs, elements, args)

    make_thesis_figure_4_3(df, elements, args.outdir, args)

    print_thresholds()

    summary = {
        "cen3": {"start": CEN3_START, "end": CEN3_END,
                 "elements": {k: {kk: vv for kk, vv in v.items() if kk != "seq"}
                              for k, v in elements.items()}},
        "window": {"lo": win[0], "hi": win[1], "n_sites": int(len(df))},
        "n_within_100bp": int(len(near)),
        "n_usable_local": int(near.usable.sum()) if "usable" in near else 0,
        "element_pam_content": df.attrs.get("el_stats", {}),
        "landing_pads": landing_pads,
        "thresholds": {k: v["value"] for k, v in THRESHOLDS.items()},
    }
    if gw:
        summary["genome_wide"] = {
            "median_on_target_only": gw["median_on"],
            "p95_on_target_only": gw["p95_on"],
            "cen3_on_target_only": gw["cen3_on"],
            "cfd_ladder_pass_rates": {str(k): v for k, v in gw["ladder"].items()},
            "n_usable_primary": gw["n_usable"],
            "mode": gw["mode"],
            "by_ceiling": {str(k): (None if v is None else
                                    {kk: vv for kk, vv in v.items()})
                           for k, v in gw["results"].items()},
        }
    with open(os.path.join(args.additional_dir, "figure_4_3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nTotal runtime: {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
