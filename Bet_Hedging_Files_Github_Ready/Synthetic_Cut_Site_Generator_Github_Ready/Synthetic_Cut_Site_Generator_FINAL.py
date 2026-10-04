#!/usr/bin/env python3
"""Reproduce the synthetic Cas9 landing-pad design analysis and thesis Figure 4.4.

Run with:
    python Synthetic_Cut_Site_Generator_FINAL.py

Required input files are expected in:
    Synthetic_Cut_Site_Generator_FINAL Additional Files/

Required files:
    whole_genes.fna
    mismatch_score.pkl
    pam_scores.pkl

Intermediate results and caches are written to the same ``Additional Files``
directory. The thesis figure is written as a 300-dpi PNG to
``Synthetic_Cut_Site_Generator_FINAL Figures``. All paths are resolved relative to
this script so it can be run from any working directory.

The default ``all`` command performs the full five-run design analysis. The
search is computationally intensive because each run samples 2 × 10^8
candidate sequences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from itertools import product

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_GENOME = "whole_genes.fna"
DEFAULT_STATE_SUFFIX = " Additional Files"
DEFAULT_FIGURES_SUFFIX = " Figures"

# The guide installed in the landing-pad strains.
INSTALLED_GUIDE = "GTAGCAAAAGAGCTGCACCG"
INSTALLED_CONTEXT = "CATGGTAGCAAAAGAGCTGCACCGCGGTCG"
INSTALLED_DOENCH = 0.9902378339167913
INSTALLED_CFD_FWD = 0.09694230623063167
INSTALLED_CFD_BOTH = 0.15686274509803921

BASES = "ACGT"
ENC = bytes.maketrans(b"ACGTacgt", bytes([0, 1, 2, 3, 0, 1, 2, 3]))
COMP = str.maketrans("ACGT", "TGCA")
GUIDE_LEN = 20
WINDOW = 23          # 20nt protospacer + NGG
CONTEXT_LEN = 30     # 4nt + 20nt guide + 3nt PAM + 3nt

# Design criteria, §4.2 and §4.3
HDR_LIMIT = 20       # bp from cut to the element being edited
DOENCH_MIN_NATIVE = 0.5
DOENCH_MIN_DESIGN = 0.98
CFD_MAX = 0.20  # relaxed two-strand limit; installed guide scores 0.1569
PUTATIVE_HIT_CFD = 0.023

# CDEIII conserved core. Unique in S288C, so a safe anchor for locating CEN3
# by sequence rather than by coordinate (the genome file is concatenated).
CDEIII_ANCHOR = "TTTCCGAAAGTTAAAAAAGAAATAGT"
# Element offsets relative to the anchor start, matched to the published
# consensus sequences (CDEI RTCACRTG; CDEII AT-rich, 78-86 bp; CDEIII 25 bp
# with the anchor beginning at CDEIII position 10).
OFF_CDEI_START, OFF_CDEI_END = -101, -93
OFF_CDEII_END = -9
OFF_CDEIII_START, OFF_CDEIII_END = -9, 16

# chrIII coordinates of the elements, used only for plotting Figure 4.1.
# Verified two independent ways: by fitting the distance columns of the site
# CSVs, and from the sequence, which places the CDEIII CCG at positions 13-15
# and therefore at chrIII 114,489-114,491.
CDE_COORDS = {"CDEI": (114385, 114392),
              "CDEII": (114393, 114476),
              "CDEIII": (114477, 114501)}
CEN3_COORDS = (114385, 114501)

rc = lambda s: s.translate(COMP)[::-1]


# --------------------------------------------------------------------------- #
# Doench 2014 on-target model (Rule Set 1)
# --------------------------------------------------------------------------- #

DOENCH_PARAMS = [
    (1, 'G', -0.2753771), (2, 'A', -0.3238875), (2, 'C', 0.17212887), (3, 'C', -0.1006662),
    (4, 'C', -0.2018029), (4, 'G', 0.24595663), (5, 'A', 0.03644004), (5, 'C', 0.09837684),
    (6, 'C', -0.7411813), (6, 'G', -0.3932644), (11, 'A', -0.466099), (14, 'A', 0.08537695),
    (14, 'C', -0.013814), (15, 'A', 0.27262051), (15, 'C', -0.1190226), (15, 'T', -0.2859442),
    (16, 'A', 0.09745459), (16, 'G', -0.1755462), (17, 'C', -0.3457955), (17, 'G', -0.6780964),
    (18, 'A', 0.22508903), (18, 'C', -0.5077941), (19, 'G', -0.4173736), (19, 'T', -0.054307),
    (20, 'G', 0.37989937), (20, 'T', -0.0907126), (21, 'C', 0.05782332), (21, 'T', -0.5305673),
    (22, 'T', -0.8770074), (23, 'C', -0.8762358), (23, 'G', 0.27891626), (23, 'T', -0.4031022),
    (24, 'A', -0.0773007), (24, 'C', 0.28793562), (24, 'T', -0.2216372), (27, 'G', -0.6890167),
    (27, 'T', 0.11787758), (28, 'C', -0.1604453), (29, 'G', 0.38634258), (1, 'GT', -0.6257787),
    (4, 'GC', 0.30004332), (5, 'AA', -0.8348362), (5, 'TA', 0.76062777), (6, 'GG', -0.4908167),
    (11, 'GG', -1.5169074), (11, 'TA', 0.7092612), (11, 'TC', 0.49629861), (11, 'TT', -0.5868739),
    (12, 'GG', -0.3345637), (13, 'GA', 0.76384993), (13, 'GC', -0.5370252), (16, 'TG', -0.7981461),
    (18, 'GG', -0.6668087), (18, 'TC', 0.35318325), (19, 'CC', 0.74807209), (19, 'TG', -0.3672668),
    (20, 'AC', 0.56820913), (20, 'CG', 0.32907207), (20, 'GA', -0.8364568), (20, 'GG', -0.7822076),
    (21, 'TC', -1.029693), (22, 'CG', 0.85619782), (22, 'CT', -0.4632077), (23, 'AA', -0.5794924),
    (23, 'AG', 0.64907554), (24, 'AG', -0.0773007), (24, 'CG', 0.28793562), (24, 'TG', -0.2216372),
    (26, 'GT', 0.11787758), (28, 'GG', -0.69774),
]
INTERCEPT, GC_HIGH, GC_LOW = 0.59763615, -0.1665878, -0.2026259


def calc_doench_score(seq: str) -> float:
    """Reference scalar Doench 2014 score for a 30mer context.

    Kept as the ground truth the vectorised path is checked against.
    """
    seq = str(seq)
    score = INTERCEPT
    guide = seq[4:24]
    gc = guide.count("G") + guide.count("C")
    score += abs(10 - gc) * (GC_LOW if gc <= 10 else GC_HIGH)
    for pos, model_seq, weight in DOENCH_PARAMS:
        if seq[pos:pos + len(model_seq)] == model_seq:
            score += weight
    return 1.0 / (1.0 + math.exp(-score))


def _build_doench_terms():
    singles, doubles = [], []
    for pos, model_seq, weight in DOENCH_PARAMS:
        idx = [BASES.index(b) for b in model_seq]
        (singles if len(model_seq) == 1 else doubles).append(
            (pos, idx[0], weight) if len(model_seq) == 1
            else (pos, idx[0], idx[1], weight))
    return singles, doubles


_SINGLES, _DOUBLES = _build_doench_terms()


def calc_doench_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorised Doench 2014 over an (M, 30) uint8 array of encoded contexts.

    Numerically identical to calc_doench_score; this is what makes scanning
    hundreds of millions of candidates tractable.
    """
    score = np.full(arr.shape[0], INTERCEPT, dtype=np.float64)
    guide = arr[:, 4:24]
    gc = ((guide == 1) | (guide == 2)).sum(axis=1)
    score += np.abs(10 - gc) * np.where(gc <= 10, GC_LOW, GC_HIGH)
    for pos, b, w in _SINGLES:
        if pos < arr.shape[1]:
            score += w * (arr[:, pos] == b)
    for pos, b1, b2, w in _DOUBLES:
        if pos + 1 < arr.shape[1]:
            score += w * ((arr[:, pos] == b1) & (arr[:, pos + 1] == b2))
    return 1.0 / (1.0 + np.exp(-score))


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def encode(seq: str) -> np.ndarray:
    return np.frombuffer(seq.encode("ascii", "replace").translate(ENC), dtype=np.uint8)


def decode(arr: np.ndarray) -> str:
    return "".join(BASES[i] for i in arr)


def decode_many(arr: np.ndarray) -> list[str]:
    table = np.frombuffer(b"ACGT", dtype=np.uint8)
    return [row.tobytes().decode() for row in table[arr]]


def pack20(arr: np.ndarray) -> np.ndarray:
    packed = np.zeros(arr.shape[0], dtype=np.uint64)
    for i in range(arr.shape[1]):
        packed = (packed << np.uint64(2)) | arr[:, i].astype(np.uint64)
    return packed


def unpack20(packed: np.ndarray, length: int = GUIDE_LEN) -> np.ndarray:
    out = np.empty((packed.shape[0], length), dtype=np.uint8)
    for i in range(length):
        shift = np.uint64(2 * (length - 1 - i))
        out[:, i] = ((packed >> shift) & np.uint64(3)).astype(np.uint8)
    return out


def n_mismatch(a: str, b: str) -> list[int]:
    """1-based mismatch positions. Position 20 is PAM-proximal."""
    return [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]


# --------------------------------------------------------------------------- #
# Stage 0 — genome to protospacer sets
# --------------------------------------------------------------------------- #

def read_genome(path: str, mode: str = "fasta") -> list[str]:
    with open(path) as fh:
        text = fh.read()
    if mode == "legacy":
        parts = text.split("complete sequence")
        if len(parts) < 2:
            raise ValueError("legacy mode: 'complete sequence' not in header")
        return ["".join(parts[1].split()).upper()]
    records, current = [], []
    for line in text.splitlines():
        if line.startswith(">"):
            if current:
                records.append("".join(current).upper())
                current = []
        else:
            current.append(line.strip())
    if current:
        records.append("".join(current).upper())
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def _protospacers_one_strand(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    if n < WINDOW:
        return np.empty(0, dtype=np.uint64)
    starts = n - WINDOW + 1
    pam_ok = (arr[21:21 + starts] == 2) & (arr[22:22 + starts] == 2)
    bad = np.concatenate(([0], np.cumsum(arr >= 4)))
    clean = (bad[WINDOW:WINDOW + starts] - bad[:starts]) == 0
    idx = np.flatnonzero(pam_ok & clean)
    if idx.size == 0:
        return np.empty(0, dtype=np.uint64)
    guides = np.lib.stride_tricks.sliding_window_view(arr, GUIDE_LEN)[idx]
    return pack20(guides)


def build_protospacer_set(records, strands="fwd", verbose=True) -> np.ndarray:
    chunks = []
    for rec in records:
        arr = encode(rec)
        chunks.append(_protospacers_one_strand(arr))
        if strands == "both":
            rcarr = np.where(arr < 4, 3 - arr.astype(np.int16), 4).astype(np.uint8)[::-1]
            chunks.append(_protospacers_one_strand(np.ascontiguousarray(rcarr)))
    allp = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)
    uniq = np.unique(allp)
    if verbose:
        print(f"    {strands:<5}: {allp.size:,} sites, {uniq.size:,} unique 20mers")
    return uniq


def build_protospacer_occurrences(records, strands="both") -> np.ndarray:
    """Return every genomic protospacer occurrence, retaining duplicates."""
    chunks = []
    for rec in records:
        arr = encode(rec)
        chunks.append(_protospacers_one_strand(arr))
        if strands == "both":
            rcarr = np.where(arr < 4, 3 - arr.astype(np.int16), 4).astype(np.uint8)[::-1]
            chunks.append(_protospacers_one_strand(np.ascontiguousarray(rcarr)))
    packed = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)
    return unpack20(packed)


def get_protospacer_occurrences(args) -> np.ndarray:
    """Load or cache every both-strand genomic NGG protospacer occurrence."""
    cache = os.path.join(args.state, "protospacer_occurrences.both.npy")
    if os.path.exists(cache) and not args.force:
        return unpack20(np.load(cache))
    records = read_genome(args.genome, args.genome_mode)
    unpacked = build_protospacer_occurrences(records, "both")
    np.save(cache, pack20(unpacked))
    return unpacked


def get_protospacers(args) -> dict:
    """{'fwd': (N,20) uint8, 'both': (M,20) uint8}, cached under --state."""
    out = {}
    records = None
    for strands in ("fwd", "both"):
        cache = os.path.join(args.state, f"protospacers.{strands}.npy")
        if os.path.exists(cache) and not args.force:
            out[strands] = np.load(cache)
            continue
        if records is None:
            print(f"[stage 0] reading genome {args.genome}")
            records = read_genome(args.genome, args.genome_mode)
            total = sum(len(r) for r in records)
            print(f"    {len(records)} record(s), {total:,} bp")
            if len(records) == 1 and total > 2_000_000:
                print("    WARNING: single FASTA record. If this concatenates "
                      "chromosomes, a few protospacers span junctions and do "
                      "not exist in the genome. Conservative: this can only "
                      "raise a maximum CFD, never lower it.")
        out[strands] = build_protospacer_set(records, strands)
        np.save(cache, out[strands])
    return {k: unpack20(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# CFD off-target model (Doench 2016)
# --------------------------------------------------------------------------- #

def _load_pickle(path: str):
    with open(path, "rb") as fh:
        try:
            return pickle.load(fh)
        except UnicodeDecodeError:
            fh.seek(0)
            return pickle.load(fh, encoding="latin1")


class CfdScorer:
    """CFD scoring of one guide against an array of genomic protospacers.

    Only NGG sites are retained upstream, so the two PAM nucleotides CFD uses
    are always 'GG'. Hard-coding pam='GG' is correct here, not an approximation.
    """

    def __init__(self, mm_path: str, pam_path: str, chunk: int = 200_000):
        mm = _load_pickle(mm_path)
        pam = _load_pickle(pam_path)
        self.pam_weight = float(pam["GG"])
        self.chunk = chunk
        comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        w = np.ones((GUIDE_LEN, 4, 4), dtype=np.float64)
        missing = []
        for pos in range(GUIDE_LEN):
            for gi, g in enumerate(BASES):
                g_rna = 'U' if g == 'T' else g
                for oi, o in enumerate(BASES):
                    if g == o:
                        continue
                    key = f"r{g_rna}:d{comp[o]},{pos + 1}"
                    if key in mm:
                        w[pos, gi, oi] = mm[key]
                    else:
                        missing.append(key)
        if missing:
            print(f"    WARNING: {len(missing)} mismatch keys absent "
                  f"(e.g. {missing[:3]}); treated as weight 1.0")
        self.w = w
        self._pos = np.arange(GUIDE_LEN)

    def max_cfd(self, guide: str, off_targets: np.ndarray,
                threshold: float | None = None, exclude_self: bool = False):
        """(max_cfd, worst_off_target). threshold enables early exit.

        exclude_self zeroes perfect matches, required when the guide is itself
        a native genomic protospacer and would otherwise score 1.0 against its
        own locus.
        """
        g = encode(guide)
        if g.size != GUIDE_LEN or g.max() >= 4:
            raise ValueError(f"guide must be 20nt of ACGT, got {guide!r}")
        wg = self.w[self._pos, g]
        row = self._pos[None, :]
        best, best_idx = 0.0, -1
        for start in range(0, off_targets.shape[0], self.chunk):
            block = off_targets[start:start + self.chunk]
            scores = wg[row, block].prod(axis=1) * self.pam_weight
            if exclude_self:
                # Must test sequence identity, not score. Several CFD mismatch
                # weights are exactly 1.0, so a genuine off-target can score
                # the same as a perfect match and would be wrongly discarded.
                scores[(block == g).all(axis=1)] = 0.0
            i = int(scores.argmax())
            if scores[i] > best:
                best, best_idx = float(scores[i]), start + i
            if threshold is not None and best > threshold:
                break
        worst = decode(off_targets[best_idx]) if best_idx >= 0 else None
        return best, worst

    def both(self, guide: str, sets: dict, exclude_self: bool = False) -> dict:
        """True maxima on both strand sets, no early exit."""
        out = {}
        for k in ("fwd", "both"):
            v, w = self.max_cfd(guide, sets[k], None, exclude_self)
            out[f"max_cfd_{k}"] = v
            out[f"worst_{k}"] = w
        return out

    def count_hits(self, guide: str, off_targets: np.ndarray,
                   threshold: float = PUTATIVE_HIT_CFD,
                   exclude_self: bool = False) -> int:
        """Count genomic protospacers with CFD at or above ``threshold``."""
        g = encode(guide)
        if g.size != GUIDE_LEN or g.max() >= 4:
            raise ValueError(f"guide must be 20nt of ACGT, got {guide!r}")
        wg = self.w[self._pos, g]
        row = self._pos[None, :]
        total = 0
        self_removed = False
        for start in range(0, off_targets.shape[0], self.chunk):
            block = off_targets[start:start + self.chunk]
            scores = wg[row, block].prod(axis=1) * self.pam_weight
            count = int(np.count_nonzero(scores >= threshold))
            if exclude_self and not self_removed:
                # Remove only the intended occurrence. An identical sequence
                # elsewhere in the genome remains a genuine putative hit.
                if np.any((block == g).all(axis=1)):
                    count -= 1
                    self_removed = True
            total += max(count, 0)
        return total


def get_scorer(args) -> CfdScorer:
    for p in (args.mm_scores, args.pam_scores):
        if not os.path.exists(p):
            sys.exit(f"ERROR: cannot find {p}. The CFD stage needs the two "
                     f"Doench 2016 tables.")
    return CfdScorer(args.mm_scores, args.pam_scores)


def calc_cfd_reference(guide: str, off_target: str, mm_scores, pam_scores,
                       pam: str = "GG") -> float:
    """Original scalar CFD, retained so the vectorised path can be checked.

    Carried over from the pre-consolidation scripts. Not used in the pipeline;
    `selfcheck` exercises it against CfdScorer so it cannot rot unnoticed.
    """
    score = 1.0
    # The original helper's own complement map. It must include U, because the
    # substitution above has already turned T into U; the module-level rc()
    # translates ACGT only and would raise on a U.
    comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A'}
    off_target = off_target.replace('T', 'U')
    guide = guide.replace('T', 'U')
    for i, sl in enumerate(off_target):
        if guide[i] != sl:
            score *= mm_scores['r' + guide[i] + ':d' + comp[sl] + ',' + str(i + 1)]
    return score * pam_scores[pam]


def cmd_selfcheck(args):
    """Verify the vectorised paths against the scalar references."""
    rng = np.random.default_rng(0)
    ctx = rng.integers(0, 4, size=(2000, CONTEXT_LEN), dtype=np.uint8)
    ctx[:, 24], ctx[:, 25], ctx[:, 26] = 1, 2, 2
    fast = calc_doench_vec(ctx)
    slow = np.array([calc_doench_score(decode(r)) for r in ctx])
    d1 = np.abs(fast - slow).max()
    print(f"  Doench vectorised vs scalar, n=2000 : max |diff| {d1:.3e}  "
          f"{'OK' if d1 < 1e-12 else 'FAIL'}")

    mm = _load_pickle(args.mm_scores)
    pam = _load_pickle(args.pam_scores)
    scorer = CfdScorer(args.mm_scores, args.pam_scores)
    guides = decode_many(rng.integers(0, 4, size=(50, GUIDE_LEN), dtype=np.uint8))
    offs = rng.integers(0, 4, size=(200, GUIDE_LEN), dtype=np.uint8)
    worst = 0.0
    for g in guides:
        v, w = scorer.max_cfd(g, offs, None)
        worst = max(worst, abs(v - calc_cfd_reference(g, w, mm, pam)))
    print(f"  CFD vectorised vs scalar, 50 guides : max |diff| {worst:.3e}  "
          f"{'OK' if worst < 1e-12 else 'FAIL'}")

    ok = abs(calc_doench_score(INSTALLED_CONTEXT) - INSTALLED_DOENCH) < 1e-12
    print(f"  installed guide Doench reproduces   : {'OK' if ok else 'FAIL'}")
    if d1 >= 1e-12 or worst >= 1e-12 or not ok:
        sys.exit("SELFCHECK FAILED")
    print("  all checks passed")


# --------------------------------------------------------------------------- #
# Search stages
# --------------------------------------------------------------------------- #

def stage1_search(n_candidates, threshold, batch, rng, verbose=True):
    """Sample random 30mer contexts (4nt + guide + CGG + 3nt), keep the best.

    Streams in batches; never materialises the full candidate list.
    """
    hits, done, t0 = [], 0, time.time()
    while done < n_candidates:
        m = min(batch, n_candidates - done)
        arr = rng.integers(0, 4, size=(m, CONTEXT_LEN), dtype=np.uint8)
        arr[:, 24], arr[:, 25], arr[:, 26] = 1, 2, 2      # fixed CGG PAM
        scores = calc_doench_vec(arr)
        for i in np.flatnonzero(scores >= threshold):
            hits.append((decode(arr[i]), float(scores[i])))
        done += m
    hits.sort(key=lambda h: h[1], reverse=True)
    if verbose:
        print(f"    stage 1: {len(hits):,}/{n_candidates:,} "
              f"({len(hits)/n_candidates*100:.5f}%) in {time.time()-t0:.0f}s")
    return hits


def stage3_tune(guide: str, pam: str = "CGG", top_n: int = 5):
    """All 4^4 x 4^3 = 16,384 flanking contexts, ranked by Doench.

    Returns (top_results, n_tied_at_optimum, all_scores). The model does not
    weight every flanking position, so the optimum is normally a large tie and
    any tied context may be chosen on cloning grounds instead.
    """
    starts = ["".join(p) for p in product(BASES, repeat=4)]
    ends = ["".join(p) for p in product(BASES, repeat=3)]
    middle = guide + pam
    contexts = [s + middle + e for s in starts for e in ends]
    arr = np.frombuffer("".join(contexts).encode().translate(ENC),
                        dtype=np.uint8).reshape(len(contexts), CONTEXT_LEN)
    scores = calc_doench_vec(arr)
    order = np.argsort(-scores, kind="stable")
    n_tied = int(np.count_nonzero(scores == scores[order[0]]))
    return ([(contexts[i], float(scores[i])) for i in order[:top_n]],
            n_tied, scores)


# --------------------------------------------------------------------------- #
# §4.2 — composition
# --------------------------------------------------------------------------- #

def _gg_overlapping(s: str, lo: int, hi: int) -> int:
    a = np.frombuffer(s.encode(), dtype=np.uint8)
    if a.size < 2:
        return 0
    isg = a == ord("G")
    return int((isg[:-1] & isg[1:])[lo:hi].sum())


def count_ngg(s: str) -> tuple[int, int]:
    """NGG target sites on each strand of s: a GG beginning at index 1..len-2."""
    hi = max(len(s) - 1, 0)
    return _gg_overlapping(s, 1, hi), _gg_overlapping(rc(s), 1, hi)


def at_pct(s: str) -> float:
    return (s.count("A") + s.count("T")) / len(s) * 100


def cmd_composition(args) -> dict:
    g = "".join(read_genome(args.genome, args.genome_mode))
    n = g.count(CDEIII_ANCHOR)
    if n != 1:
        sys.exit(f"ERROR: CDEIII anchor found {n} times, expected 1.")
    a = g.find(CDEIII_ANCHOR)
    cdeI = g[a + OFF_CDEI_START:a + OFF_CDEI_END]
    cdeII = g[a + OFF_CDEI_END:a + OFF_CDEII_END]
    cdeIII = g[a + OFF_CDEIII_START:a + OFF_CDEIII_END]
    cen3 = g[a + OFF_CDEI_START:a + OFF_CDEIII_END]
    flank = (g[a + OFF_CDEI_START - args.flank:a + OFF_CDEI_START]
             + g[a + OFF_CDEIII_END:a + OFF_CDEIII_END + args.flank])

    print("=" * 74)
    print("§4.2  CEN3 ELEMENTS")
    print("=" * 74)
    print(f"  CDEI   ({len(cdeI):>3} bp)  {cdeI}     consensus RTCACRTG")
    print(f"  CDEII  ({len(cdeII):>3} bp)  {cdeII}")
    print(f"  CDEIII ({len(cdeIII):>3} bp)  {cdeIII}")
    if not (cdeI[1:6] == "TCACA" and cdeI[7] == "G"):
        print("  WARNING: CDEI does not match RTCACRTG; check the offsets.")

    regions = [("CDEI", cdeI), ("CDEII", cdeII), ("CDEIII", cdeIII),
               ("CEN3 (whole)", cen3), (f"flanks (2x{args.flank})", flank),
               ("genome", g)]
    print(f"\n  {'region':<18}{'length':>12}{'AT %':>9}{'GG|CC dinucs':>16}"
          f"{'NGG fwd':>10}{'NGG rev':>9}{'bp/site':>10}")
    print("  " + "-" * 84)
    res = {}
    for name, s in regions:
        f, r = count_ngg(s)
        dens = (len(s) * 2 / (f + r)) if (f + r) else float("inf")
        print(f"  {name:<18}{len(s):>12,}{at_pct(s):>9.1f}"
              f"{s.count('GG') + s.count('CC'):>16,}{f:>10,}{r:>9,}"
              f"{(f'{dens:,.1f}' if dens != float('inf') else 'none'):>10}")
        res[name] = {"len": len(s), "at": at_pct(s), "ngg_fwd": f, "ngg_rev": r,
                     "gg_cc": s.count("GG") + s.count("CC"),
                     "bp_per_site": None if dens == float("inf") else dens}

    i = cdeIII.find("CCG")
    print("\n" + "=" * 74)
    print("§4.2  THE ONLY PAM IN CEN3")
    print("=" * 74)
    print(f"  NGG motifs in the {len(cen3)} bp centromere: "
          f"{sum(count_ngg(cen3))}")
    print(f"  CDEIII  {cdeIII}")
    print(f"          {' ' * i}^^^  CCG at CDEIII positions {i+1}-{i+3}, "
          f"chrIII {CDE_COORDS['CDEIII'][0]+i:,}-{CDE_COORDS['CDEIII'][0]+i+2:,}")
    print("  Reverse complement CGG. This is the sole NGG motif in CEN3 and it")
    print("  is the conserved CBF3-binding core of CDEIII.")
    print()
    print("  CDEII contains no GG or CC dinucleotide on either strand, so it")
    print("  cannot contain an NGG motif at all. This follows from the AT-")
    print("  richness the Cse4 nucleosome requires: structural, not incidental.")
    print("  The one available PAM cannot be silently mutated to block recutting")
    print("  The available PAM lies within the conserved CDEIII core, and both")
    print("  edited alleles retain CDEIII, so a product edited via this site")
    print("  would retain the same target region.")

    res["_cdeIII_ccg_pos"] = i + 1
    res["_sequences"] = {"CDEI": cdeI, "CDEII": cdeII, "CDEIII": cdeIII}
    save_state(args, "composition.json", res)
    return res


# --------------------------------------------------------------------------- #
# §4.2 — native landscape
# --------------------------------------------------------------------------- #

def build_cen3_landscape(genome: str, flank: int = 1000) -> "pd.DataFrame":
    """Enumerate every NGG target site around CEN3, directly from the genome.

    This removes the dependency on pre-made cen3_pam_sites_*.csv files. Verified
    against them: all 104 sites in a 1 kb flank reproduce exactly, coordinates,
    protospacers and PAMs.

    Coordinate conventions, checked against the original CSVs:
      pam_start is the leftmost chromosome-III coordinate of the PAM, on BOTH
      strands. Cas9 cuts between protospacer positions 17 and 18, so the blunt
      cut midpoint is pam_start - 3.5 on the forward strand and pam_start + 5.5
      on the reverse.
    """
    import pandas as pd
    a = genome.find(CDEIII_ANCHOR)
    if a < 0:
        raise ValueError("CDEIII anchor not found; cannot locate CEN3")
    # chrIII coordinate = genome index + offset. The anchor begins at CDEIII
    # position 10, i.e. chrIII CDEIII_start + 9.
    offset = (CDE_COORDS["CDEIII"][0] + 9) - a

    lo = CEN3_COORDS[0] - flank - offset
    hi = CEN3_COORDS[1] + flank - offset
    rows = []
    for i in range(lo - WINDOW, hi + 1):
        w = genome[i:i + WINDOW]
        if len(w) < WINDOW or not set(w) <= set(BASES):
            continue
        if w[21:23] == "GG":
            rows.append((i + offset + 20, "+", w[:20], w[20:23],
                         genome[i - 4:i + WINDOW + 3]))
        v = rc(w)
        if v[21:23] == "GG":
            rows.append((i + offset, "-", v[:20], v[20:23],
                         rc(genome[i - 3:i + WINDOW + 4])))
    df = pd.DataFrame(rows, columns=["pam_start", "strand", "protospacer",
                                     "pam", "context30"])
    df["cut_mid"] = np.where(df.strand == "+", df.pam_start - 3.5,
                             df.pam_start + 5.5)
    # Window rule, recovered from the original cen3_pam_sites CSVs: a site is
    # in scope when its PAM start lies within `flank` of the CEN3 midpoint.
    # This reproduces those files exactly (104 sites at flank = 1000).
    mid = (CEN3_COORDS[0] + CEN3_COORDS[1]) / 2
    df = df[(df.pam_start - mid).abs() <= flank].copy()

    for name, (s, e) in CDE_COORDS.items():
        df[f"dist_{name}"] = np.where(
            (df.cut_mid >= s) & (df.cut_mid <= e), 0,
            np.minimum(np.abs(df.cut_mid - s),
                       np.abs(df.cut_mid - e)).astype(int))
    dists = df[[f"dist_{n}" for n in CDE_COORDS]]
    df["dist_nearest_element"] = dists.min(axis=1).astype(int)
    df["nearest_element"] = [list(CDE_COORDS)[i] for i in dists.values.argmin(1)]
    df["doench2014"] = [calc_doench_score(c) if len(c) == CONTEXT_LEN
                        else float("nan") for c in df.context30]
    return df.sort_values("pam_start").reset_index(drop=True)


def cmd_native(args) -> list[dict]:
    import pandas as pd
    sets = get_protospacers(args)
    scorer = get_scorer(args)
    occurrences = get_protospacer_occurrences(args)

    if args.native_csv:
        frames = [pd.read_csv(p) for p in args.native_csv]
        cols = frames[0].columns
        df = (pd.concat([f[cols] for f in frames])
              .drop_duplicates(subset=["pam_start", "strand"])
              .reset_index(drop=True))
        print(f"[native] {len(df)} unique sites from {len(frames)} CSV file(s)")
    else:
        genome = "".join(read_genome(args.genome, args.genome_mode))
        df = build_cen3_landscape(genome, args.flank)
        print(f"[native] {len(df)} sites enumerated from the genome "
              f"(CEN3 ± {args.flank:,} bp); no CSV needed")

    rows = []
    for _, r in df.iterrows():
        guide = str(r.protospacer).upper()
        d = scorer.both(guide, sets, exclude_self=True)
        d["n_putative_hits_both"] = scorer.count_hits(
            guide, occurrences, exclude_self=True)
        rows.append({"pam_start": int(r.pam_start), "strand": r.strand,
                     "protospacer": guide, "pam": r.pam,
                     "cut_mid": float(r.cut_mid),
                     "dist_nearest_element": int(r.dist_nearest_element),
                     "nearest_element": r.nearest_element,
                     "doench2014": float(r.doench2014), **d})
    out = pd.DataFrame(rows)
    out["ok_hdr"] = out.dist_nearest_element <= HDR_LIMIT
    out["ok_on"] = out.doench2014 >= DOENCH_MIN_NATIVE
    out["ok_off"] = out.max_cfd_both <= args.cfd
    out["usable"] = out.ok_hdr & out.ok_on & out.ok_off

    print(f"  fail position (>{HDR_LIMIT} bp)   : "
          f"{(~out.ok_hdr).sum()} of {len(out)}")
    print(f"  fail on-target (<{DOENCH_MIN_NATIVE})    : "
          f"{(~out.ok_on).sum()} of {len(out)}")
    print(f"  fail off-target (>{args.cfd}), both strands: "
          f"{(~out.ok_off).sum()} of {len(out)}")
    print(f"  USABLE                     : {int(out.usable.sum())}")
    print(f"  min max-CFD anywhere in the window: "
          f"fwd {out.max_cfd_fwd.min():.4f}  both {out.max_cfd_both.min():.4f}")
    best = out.loc[out.max_cfd_both.idxmin()]
    print(f"  most specific native site  : {best.protospacer} "
          f"(Doench {best.doench2014:.3f}, both {best.max_cfd_both:.4f})")
    print(f"  installed designed guide   : {INSTALLED_GUIDE} "
          f"(Doench {INSTALLED_DOENCH:.3f}, both {INSTALLED_CFD_BOTH:.4f})")

    inner = out[out.ok_hdr]
    for _, r in inner.iterrows():
        print(f"  site cutting within {HDR_LIMIT} bp: chrIII:{r.pam_start:,} "
              f"{r.strand}  {r.protospacer}+{r.pam}  in {r.nearest_element}  "
              f"Doench {r.doench2014:.3f}  CFD both {r.max_cfd_both:.3f}")

    path = os.path.join(args.state, "native_rescored.csv")
    out.to_csv(path, index=False)
    print(f"  -> {path}")
    return out


# --------------------------------------------------------------------------- #
# §4.3 — search, validate, tune, null
# --------------------------------------------------------------------------- #

def _matching_runs(runs, args):
    """Select only runs made with the current strand basis and CFD limit."""
    return [r for r in runs
            if r.get("screened_on", "fwd") == args.strands
            and r.get("cfd") is not None
            and np.isclose(float(r["cfd"]), float(args.cfd))
            and ((args.top <= 0 and r.get("screened") == r.get("stage1"))
                 or (args.top > 0 and r.get("screened") ==
                     min(args.top, r.get("stage1", args.top))))]


def cmd_search(args) -> dict:
    sets = get_protospacers(args)
    scorer = get_scorer(args)
    occurrences = get_protospacer_occurrences(args)
    store = os.path.join(args.state, "runs.json")
    runs = json.load(open(store)) if os.path.exists(store) and not args.force else []
    # Threshold is part of the cache identity. A 0.10 run must never be reused
    # after the declared two-strand limit is changed to 0.20.
    done = {r["seed"] for r in _matching_runs(runs, args)}

    print(f"[search] screening both DNA strands at CFD <= {args.cfd}.")

    for seed in range(args.seed0, args.seed0 + args.runs):
        if seed in done:
            print(f"[search] seed {seed} ({args.strands}, CFD {args.cfd}): "
                  "cached, skipping")
            continue
        t0 = time.time()

        # Stage 1 is independent of the screening basis, so cache it: changing
        # --strands then costs only the CFD screen, not another full scan.
        hpath = os.path.join(args.state, f"hits_seed{seed}.pkl")
        hits = None
        if os.path.exists(hpath) and not args.force:
            blob = pickle.load(open(hpath, "rb"))
            if (blob["n_candidates"] == args.num_sequences
                    and blob["doench"] == args.doench):
                hits = blob["hits"]
                print(f"[search] seed {seed}: reusing {len(hits):,} cached "
                      f"stage-1 hits")
        if hits is None:
            print(f"[search] seed {seed}: {args.num_sequences:,} candidates")
            hits = stage1_search(args.num_sequences, args.doench, args.batch,
                                 np.random.default_rng(seed))
            pickle.dump({"hits": hits, "n_candidates": args.num_sequences,
                         "doench": args.doench}, open(hpath, "wb"))

        survivors = []
        screen_set = sets[args.strands]
        screen_hits = hits if args.top <= 0 else hits[:args.top]
        for i, (context, doench) in enumerate(screen_hits):
            guide = context[4:24]
            v, _ = scorer.max_cfd(guide, screen_set, args.cfd)
            if v <= args.cfd:
                d = scorer.both(guide, sets)
                d["n_putative_hits_both"] = scorer.count_hits(
                    guide, occurrences)
                tuned, n_tied, _ = stage3_tune(guide, top_n=1)
                survivors.append({"rank": i, "guide": guide, "context": context,
                                  "doench": doench, **d,
                                  "tuned_context": tuned[0][0],
                                  "tuned_doench": tuned[0][1],
                                  "tuned_ties": n_tied})
        runs.append({"seed": seed, "n_candidates": args.num_sequences,
                     "stage1": len(hits), "screened": len(screen_hits),
                     "stage2": len(survivors), "best_doench": hits[0][1],
                     "screened_on": args.strands, "cfd": args.cfd,
                     "survivors": survivors, "secs": time.time() - t0})
        runs.sort(key=lambda r: (r.get("screened_on", "fwd"), r["seed"]))
        json.dump(runs, open(store, "w"), indent=1)
        print(f"    stage 2: {len(survivors)} survivors "
              f"[{time.time()-t0:.0f}s total]")

    matching = _matching_runs(runs, args)
    res = summarise_runs(matching, args)
    print_sequences(args)
    return res


def summarise_runs(runs, args) -> dict:
    s1 = np.array([r["stage1"] for r in runs], float)
    s2 = np.array([r["stage2"] for r in runs], float)
    N = runs[0]["n_candidates"]
    allg = [g for r in runs for g in r["survivors"]]
    cb = np.array([g["max_cfd_both"] for g in allg]) if allg else np.array([])

    print("\n" + "=" * 74)
    print(f"§4.3  SEARCH FUNNEL — {len(runs)} independent runs of {N:,}")
    print("=" * 74)
    print(f"  stage 1 (Doench >= {args.doench}): mean {s1.mean():,.0f}  "
          f"SD {s1.std(ddof=1):,.0f}  range {s1.min():,.0f}-{s1.max():,.0f}")
    print(f"          pass rate {s1.mean()/N*100:.5f}% = 1 in {N/s1.mean():,.0f}")
    print(f"  stage 2 (CFD <= {args.cfd}, {runs[0]['screened_on']} strand"
          f"{'s' if runs[0]['screened_on']=='both' else ''}): "
          f"mean {s2.mean():.1f}  SD {s2.std(ddof=1):.1f}  "
          f"range {s2.min():.0f}-{s2.max():.0f}  total {s2.sum():.0f}")
    if s2.mean() > 0:
        print(f"  overall yield: 1 usable guide per {N/s2.mean():,.0f} sampled")
    else:
        print("  overall yield: NO SURVIVORS in any run.")
        print(f"  Nothing passed the uniform both-strand CFD <= {args.cfd} filter.")
    if s2.mean() > 0:
        print("\n  The stage-2 count varies between runs and is quoted as a mean.")
    else:
        print("\n  The stage-2 result was consistently zero across all runs.")
    print("  The stage-1 count varies by about one per cent.")
    if allg:
        print(f"\n  {len(allg)} survivors pooled, two-strand CFD: "
              f"min {cb.min():.4f}  median {np.median(cb):.4f}  max {cb.max():.4f}")
        print(f"  clearing {args.cfd} on BOTH strands: {(cb <= args.cfd).sum()} "
              f"of {len(cb)}")
        print(f"  more specific than the installed guide "
              f"({INSTALLED_CFD_BOTH:.4f}): {(cb < INSTALLED_CFD_BOTH).sum()} "
              f"of {len(cb)}")
    path = os.path.join(args.state, "survivors_pooled.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "guide", "doench", "max_cfd_fwd", "max_cfd_both",
                    "tuned_doench", "tuned_ties"])
        for r in runs:
            for g in r["survivors"]:
                w.writerow([r["seed"], g["guide"], g["doench"],
                            g["max_cfd_fwd"], g["max_cfd_both"],
                            g["tuned_doench"], g["tuned_ties"]])
    print(f"  -> {path}")
    return {"n_runs": len(runs), "stage1_mean": float(s1.mean()),
            "stage1_min": float(s1.min()), "stage1_max": float(s1.max()),
            "stage2_mean": float(s2.mean()), "stage2_min": float(s2.min()),
            "stage2_max": float(s2.max()), "n_candidates": N}


def cmd_validate(args) -> dict:
    guide = args.target.upper()
    sets = get_protospacers(args)
    scorer = get_scorer(args)
    tuned, n_tied, scores = stage3_tune(guide, top_n=args.top_n)
    np.save(os.path.join(args.state, "context_scores.npy"), scores)

    print("=" * 74)
    print(f"§4.3  VALIDATE  {guide} + CGG")
    print("=" * 74)
    print("  best flanking contexts:")
    for ctx, sc in tuned:
        print(f"    {ctx}  {sc:.10f}")
    print(f"    ({n_tied} of 16,384 contexts tie at the optimum)")
    print(f"  across all 16,384: worst {scores.min():.6f}  "
          f"median {np.median(scores):.6f}  best {scores.max():.6f}")

    if guide == INSTALLED_GUIDE:
        got = calc_doench_score(INSTALLED_CONTEXT)
        ok = "OK" if abs(got - INSTALLED_DOENCH) < 1e-12 else "MISMATCH"
        pct = (scores < got).mean() * 100
        print(f"\n  installed context {INSTALLED_CONTEXT}")
        print(f"    recorded Doench {INSTALLED_DOENCH:.16f}")
        print(f"    recomputed      {got:.16f}   [{ok}]")
        print(f"    {pct:.2f}th percentile of the 16,384; "
              f"+{got-np.median(scores):.4f} vs median, "
              f"{scores.max()-got:.4f} below the optimum")

    d = scorer.both(guide, sets)
    print(f"\n  off-target maxima (no early exit):")
    for k in ("fwd", "both"):
        w = d[f"worst_{k}"]
        mm = n_mismatch(guide, w)
        seed = [i for i in mm if i >= 13]
        print(f"    {k:<5} ({sets[k].shape[0]:>7,} protospacers): "
              f"{d[f'max_cfd_{k}']:.10f} vs {w}")
        print(f"          {len(mm)}/20 mismatches at {mm}, "
              f"{len(seed)} in the PAM-proximal seed")
    print(f"  verdict at {args.cfd}: "
          f"fwd {'PASS' if d['max_cfd_fwd'] <= args.cfd else 'FAIL'}, "
          f"both {'PASS' if d['max_cfd_both'] <= args.cfd else 'FAIL'}")
    if guide == INSTALLED_GUIDE:
        print(f"    recorded fwd  {INSTALLED_CFD_FWD:.10f} "
              f"(delta {d['max_cfd_fwd']-INSTALLED_CFD_FWD:+.10f})")
        print(f"    recorded both {INSTALLED_CFD_BOTH:.10f} "
              f"(delta {d['max_cfd_both']-INSTALLED_CFD_BOTH:+.10f})")
    res = {"guide": guide, "n_tied": n_tied, **d,
           "context_min": float(scores.min()),
           "context_median": float(np.median(scores)),
           "context_max": float(scores.max())}
    save_state(args, "validate.json", res)
    return res


def cmd_tune(args):
    for guide in args.target.upper().split(","):
        guide = guide.strip()
        tuned, n_tied, scores = stage3_tune(guide, top_n=args.top_n)
        print(f"\nTUNE  {guide} + CGG")
        for ctx, sc in tuned:
            print(f"  {ctx}  {sc:.10f}")
        print(f"  ({n_tied} of 16,384 tie; worst {scores.min():.6f}, "
              f"median {np.median(scores):.6f})")


def cmd_null(args) -> dict:
    paths = {k: os.path.join(args.state, f"null_{k}.npy") for k in ("fwd", "both")}
    cached = all(os.path.exists(p) for p in paths.values())
    if cached and not args.force:
        v = {k: np.load(p) for k, p in paths.items()}
        if len(v["fwd"]) == args.null_n:
            print(f"[null] cached (n = {args.null_n:,}), skipping. "
                  f"--force to recompute.")
            return {k: {"median": float(np.median(a))} for k, a in v.items()}

    sets = get_protospacers(args)
    scorer = get_scorer(args)
    rng = np.random.default_rng(args.null_seed)
    arr = rng.integers(0, 4, size=(args.null_n, GUIDE_LEN), dtype=np.uint8)
    guides = decode_many(arr)
    print("=" * 74)
    print(f"§4.3  RANDOM-GUIDE NULL, n = {args.null_n:,}")
    print("=" * 74)
    res = {}
    for k in ("fwd", "both"):
        v = np.array([scorer.max_cfd(g, sets[k], None)[0] for g in guides])
        np.save(os.path.join(args.state, f"null_{k}.npy"), v)
        sel = INSTALLED_CFD_FWD if k == "fwd" else INSTALLED_CFD_BOTH
        pct = (v < sel).mean() * 100
        print(f"  {k:<5}: median {np.median(v):.4f}  "
              f"IQR {np.percentile(v,25):.4f}-{np.percentile(v,75):.4f}  "
              f"min {v.min():.4f}")
        print(f"         {(v <= args.cfd).mean()*100:.2f}% clear {args.cfd}; "
              f"installed guide {sel:.4f} is at the {pct:.1f}th percentile")
        res[k] = {"median": float(np.median(v)),
                  "pct_clearing": float((v <= args.cfd).mean() * 100),
                  "installed_percentile": float(pct)}
    save_state(args, "null.json", res)
    return res


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _style():
    """Apply the shared thesis figure typography."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 29,
        "axes.labelsize": 29,
        "xtick.labelsize": 29,
        "ytick.labelsize": 29,
        "legend.fontsize": 25,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def _floating_axes(ax):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def _figure_data(args):
    """Load the common native-site, survivor, and run tables."""
    import pandas as pd
    all_runs = json.load(open(os.path.join(args.state, "runs.json")))
    runs = _matching_runs(all_runs, args)
    if not runs:
        sys.exit(f"ERROR: runs.json holds no runs screened on '{args.strands}'.")
    survivors = pd.DataFrame([g for r in runs for g in r["survivors"]])
    native = pd.read_csv(os.path.join(args.state, "native_rescored.csv"))
    return runs, survivors, native


def _designed_or_installed(survivors):
    """Return designed survivors or the installed guide for comparison.

    A strict both-strand screen can validly produce zero survivors. Downstream
    descriptive figures must therefore not assume that the pooled dataframe
    has columns. The fallback is labelled as a comparison, never as a pass.
    """
    import pandas as pd
    if not survivors.empty:
        return survivors.copy(), False
    return pd.DataFrame([{
        "guide": INSTALLED_GUIDE,
        "doench": INSTALLED_DOENCH,
        "max_cfd_both": INSTALLED_CFD_BOTH,
    }]), True


def _ensure_hit_counts(args, survivors, native):
    """Backfill hit counts when figures are made from older cached runs."""
    occurrences = scorer = None
    if "n_putative_hits_both" not in native.columns:
        scorer = get_scorer(args)
        occurrences = get_protospacer_occurrences(args)
        native = native.copy()
        native["n_putative_hits_both"] = [
            scorer.count_hits(str(g), occurrences, exclude_self=True)
            for g in native.protospacer]
    if "n_putative_hits_both" not in survivors.columns:
        if occurrences is None:
            scorer = get_scorer(args)
            occurrences = get_protospacer_occurrences(args)
        survivors = survivors.copy()
        survivors["n_putative_hits_both"] = [
            scorer.count_hits(str(g), occurrences)
            for g in survivors.guide]
    return survivors, native


def _top_genome_wide_natural_sites(args, n=100):
    """Return the highest-Doench natural NGG sites across the whole genome.

    Both DNA strands are scanned. Candidate ranking uses predicted cutting
    efficiency only; whole-genome CFD specificity is then calculated for the
    selected guides and displayed on the second axis. Results are cached.
    """
    import pandas as pd
    cache = os.path.join(args.state, f"top_{n}_genome_wide_natural.csv")
    if os.path.exists(cache) and not args.force:
        cached = pd.read_csv(cache)
        if len(cached) == n:
            return cached

    candidates = []
    for record_index, sequence in enumerate(
            read_genome(args.genome, args.genome_mode)):
        for strand, oriented in (("+", sequence), ("-", rc(sequence))):
            arr = encode(oriented)
            if arr.size < CONTEXT_LEN:
                continue
            contexts = np.lib.stride_tricks.sliding_window_view(
                arr, CONTEXT_LEN)
            valid = ((contexts[:, 25] == 2) & (contexts[:, 26] == 2)
                     & np.all(contexts < 4, axis=1))
            selected = contexts[valid]
            if not len(selected):
                continue
            scores = calc_doench_vec(selected)
            keep = min(max(n * 3, n), len(scores))
            indices = np.argpartition(scores, -keep)[-keep:]
            for context, score in zip(selected[indices], scores[indices]):
                candidates.append({
                    "record": record_index,
                    "strand": strand,
                    "guide": decode(context[4:24]),
                    "doench2014": float(score),
                })

    ranked = (pd.DataFrame(candidates)
              .sort_values("doench2014", ascending=False)
              .drop_duplicates("guide")
              .head(n)
              .reset_index(drop=True))
    if len(ranked) < n:
        raise RuntimeError(f"Only {len(ranked)} unique natural NGG guides were found")

    scorer = get_scorer(args)
    sets = get_protospacers(args)
    cfd = [scorer.both(str(guide), sets, exclude_self=True)
           for guide in ranked.guide]
    ranked["max_cfd_both"] = [result["max_cfd_both"] for result in cfd]
    ranked.to_csv(cache, index=False)
    return ranked







FIGURE_42_PALETTES = {
    # Monotonic lightening through the sequential filters: the complete
    # generated pool is darkest and the final survivor bar is lightest.
    "i":  ("#9B536A", "#B8798C", "#D2A2AF"),
}


def figure_4_4_thesis(args, palette_name="i"):
    """Create thesis Figure 4.4, showing filtering and generated-versus-natural sites.

    Panel A: how many candidates survive each filter.
    Panel B: the 100 naturally occurring sites with the highest predicted cut
             efficiency and the 25 best generated guides on the same axes.
    """
    import pandas as pd
    plt = _style()
    all_runs = json.load(open(os.path.join(args.state, "runs.json")))
    runs = _matching_runs(all_runs, args)
    if not runs:
        sys.exit(f"ERROR: runs.json holds no runs screened on '{args.strands}'. "
                 f"Run `search --strands {args.strands} --cfd {args.cfd}` first.")
    if len(runs) != len(all_runs):
        print(f"[figures] using {len(runs)} of {len(all_runs)} runs in "
              f"runs.json (those screened on {args.strands} at CFD <= {args.cfd})")

    s1 = np.array([r["stage1"] for r in runs], float)
    s2 = np.array([r["stage2"] for r in runs], float)
    N = runs[0]["n_candidates"]
    surv = pd.DataFrame([g for r in runs for g in r["survivors"]])
    if surv.empty:
        surv = pd.DataFrame(columns=["guide", "doench", "max_cfd_both"])
    nat = _top_genome_wide_natural_sites(args, n=100)

    # The natural comparison asks about the upper tail of cutting efficiency
    # across all genomic PAM sites, so all 100 preselected sites are shown.
    # Generated guides have already passed both filters; their displayed rank
    # prioritises predicted cutting efficiency and then lower maximum CFD.
    nat_top = nat.sort_values(
        ["doench2014", "max_cfd_both"], ascending=[False, True]).head(100)
    surv_top = surv[surv.guide != INSTALLED_GUIDE].sort_values(
        ["doench", "max_cfd_both"], ascending=[False, True]).head(25)

    fig = plt.figure(figsize=(20.0, 9.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.46)

    # ---- A: filtering process -------------------------------------------- #
    axA = fig.add_subplot(gs[0])
    vals = [N, s1.mean(), max(s2.mean(), 0.3)]
    stage_labels = ["Generated", "High Cut Efficiency", "Low Off-Target"]
    y = np.arange(len(vals))[::-1]
    process_colours = FIGURE_42_PALETTES[palette_name]
    axA.barh(y, vals, height=0.58, color=process_colours,
             edgecolor="black", linewidth=0.6)
    axA.set_xscale("log")
    axA.set_yticks(y, stage_labels, fontsize=29)
    axA.set_xlabel("Number of sequences")
    axA.text(0.0, 1.03, "A", transform=axA.transAxes, fontsize=29,
             fontweight="bold", ha="left", va="bottom")
    # Arrows show that each row is the output passed into the next filter.
    for upper, lower in zip(y[:-1], y[1:]):
        axA.annotate("", xy=(0.02, lower + 0.32), xytext=(0.02, upper - 0.32),
                     xycoords=("axes fraction", "data"),
                     arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.8))
    _floating_axes(axA)
    # ---- B: designed vs natural ------------------------------------------- #
    axB = fig.add_subplot(gs[1])
    marker_size = 200
    natural_colour = "#7A7A7A"
    # Deliberately reuse the final bar colour for every "Top designed" mark.
    designed_colour = process_colours[-1]
    axB.scatter(nat_top.max_cfd_both, nat_top.doench2014, s=marker_size, c=natural_colour,
                edgecolor="0.25", linewidth=0.3, zorder=2,
                label=f"Top naturally occurring (n = {len(nat_top)})")
    axB.scatter(surv_top.max_cfd_both, surv_top.doench, s=marker_size, c=designed_colour,
                edgecolor="black", linewidth=0.4, zorder=3,
                label=f"Top generated (n = {len(surv_top)})")
    all_x = np.r_[nat_top.max_cfd_both.to_numpy(float),
                  surv_top.max_cfd_both.to_numpy(float)]
    all_y = np.r_[nat_top.doench2014.to_numpy(float),
                  surv_top.doench.to_numpy(float)]
    xpad = max(0.015, 0.07 * np.ptp(all_x))
    ypad = max(0.015, 0.12 * np.ptp(all_y))
    axB.set_xlim(max(0, all_x.min() - xpad), min(1, all_x.max() + xpad))
    axB.set_ylim(max(0, all_y.min() - ypad), 1.0)
    axB.set_xlabel("Off-target score (CFD - lower better)")
    axB.set_ylabel("Cut efficiency (Doench - higher better)")
    # Labels are anchored outside the densest point clouds and protected by a
    # white background so text cannot collide visually with the observations.
    designed_label = f"Top generated (n = {len(surv_top)})"
    axB.text(0.97, 0.97, designed_label,
             transform=axB.transAxes, fontsize=25, color=designed_colour,
             ha="right", va="top", fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5))
    axB.text(0.97, 0.05, f"Top naturally occurring (n = {len(nat_top)})",
             transform=axB.transAxes, fontsize=25, color="0.35",
             ha="right", va="bottom", fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5))
    axB.text(0.0, 1.03, "B", transform=axB.transAxes, fontsize=29,
             fontweight="bold", ha="left", va="bottom")
    _floating_axes(axB)

    output_path = os.path.join(
        args.outdir,
        "Figure_4.4_Computational_Generation_and_Filtering_of_Synthetic_Cas9_Landing-Pad_Targets.png",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Figure 4.4 — {len(runs)} runs, stage1 {s1.mean():,.0f}, "
          f"stage2 {s2.mean():.1f}, top {len(surv_top)} designed vs "
          f"top {len(nat_top)} natural; CFD scores search the whole genome")
















def print_sequences(args):
    """Final block: the actual sequences, with their scores, ready to order.

    `all` calls both search and figures, so guard against printing twice.
    """
    if getattr(args, "_sequences_printed", False):
        return
    args._sequences_printed = True
    import pandas as pd
    store = os.path.join(args.state, "runs.json")
    nat_path = os.path.join(args.state, "native_rescored.csv")
    if not os.path.exists(store):
        return
    runs = _matching_runs(json.load(open(store)), args)
    if not runs:
        return
    surv = pd.DataFrame([g for r in runs for g in r["survivors"]])

    print("\n" + "=" * 78)
    print("SEQUENCES")
    print("=" * 78)
    print("  Doench = predicted cutting efficiency, higher is better (max 1).")
    print("  CFD    = worst match anywhere else in the genome, lower is better.")
    print("           all filtering and ranking use the both-strand maximum.\n")

    print("  INSTALLED IN THE LANDING-PAD STRAINS")
    print(f"    protospacer   {INSTALLED_GUIDE} + CGG")
    print(f"    30mer context {INSTALLED_CONTEXT}")
    print(f"    Doench {INSTALLED_DOENCH:.4f}   "
          f"CFD both {INSTALLED_CFD_BOTH:.4f}")

    if not surv.empty:
        surv = surv.sort_values("max_cfd_both").reset_index(drop=True)
        print(f"\n  DESIGNED CANDIDATES FROM THIS RUN "
              f"({len(surv)} from {len(runs)} runs, most specific first)")
        print(f"    {'#':<3}{'protospacer':<24}{'Doench':>8}"
              f"{'CFD both':>10}   {'context to order (30mer)':<32}")
        print("    " + "-" * 87)
        for i, r in surv.iterrows():
            mark = " *" if r.max_cfd_both < INSTALLED_CFD_BOTH else "  "
            ctx = r.get("tuned_context", "")
            if not isinstance(ctx, str) or not ctx:
                # older cached runs did not store it; it is cheap to recover
                ctx = stage3_tune(r.guide, top_n=1)[0][0][0]
            print(f"    {i+1:<3}{r.guide + ' + CGG':<24}{r.doench:>8.4f}"
                  f"{r.max_cfd_both:>10.4f}{mark} "
                  f"{ctx:<32}")
        n_better = int((surv.max_cfd_both < INSTALLED_CFD_BOTH).sum())
        print(f"    * = more specific than the installed guide "
              f"({n_better} of {len(surv)})")

    if os.path.exists(nat_path):
        nat = pd.read_csv(nat_path).sort_values("max_cfd_both").head(3)
        print(f"\n  BEST NATURAL SITES NEAR CEN3, FOR COMPARISON "
              f"(none is usable)")
        print(f"    {'protospacer':<24}{'Doench':>8}{'CFD both':>10}"
              f"   {'why it fails':<40}")
        print("    " + "-" * 84)
        for _, r in nat.iterrows():
            why = []
            if not r.ok_hdr:
                why.append(f"cuts {r.dist_nearest_element} bp away")
            if not r.ok_on:
                why.append("too inefficient")
            if not r.ok_off:
                why.append("off-target too high")
            print(f"    {r.protospacer + ' + ' + str(r.pam):<24}"
                  f"{r.doench2014:>8.4f}{r.max_cfd_both:>10.4f}   "
                  f"{'; '.join(why):<40}")
    print("=" * 78)



# --------------------------------------------------------------------------- #
# Sequence-score maps
# --------------------------------------------------------------------------- #



















def cmd_figures(args):
    """Create only the figure used in the submitted thesis."""
    figure_4_4_thesis(args, palette_name="i")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def find_input(name: str, explicit: str | None, roots: list[str]) -> str | None:
    """Locate an input file without requiring the user to pass a path.

    An explicit --flag always wins. Otherwise look for `name` in each root and
    one level below it, so a layout like
        project/script.py
        project/Yeast_Chr_Seqs/whole_genes.fna
    resolves with no configuration. Returns None if not found; callers decide
    whether that is fatal.
    """
    if explicit:
        return explicit if os.path.exists(explicit) else None
    import glob
    for root in roots:
        direct = os.path.join(root, name)
        if os.path.exists(direct):
            return direct
        hits = sorted(glob.glob(os.path.join(root, "*", name)))
        if hits:
            return hits[0]
    return None


def resolve_inputs(args) -> None:
    """Resolve required inputs from this script's own Additional Files folder.

    By default the repository layout is:

        Synthetic_Cut_Site_Generator_FINAL.py
        Synthetic_Cut_Site_Generator_FINAL Additional Files/
            whole_genes.fna
            mismatch_score.pkl
            pam_scores.pkl

    Explicit command-line paths override these defaults. The script does not
    search unrelated sibling folders, so a cloned repository behaves
    predictably on another computer.
    """
    state_dir = os.path.abspath(args.state)

    defaults = {
        "genome": os.path.join(state_dir, DEFAULT_GENOME),
        "mm_scores": os.path.join(state_dir, "mismatch_score.pkl"),
        "pam_scores": os.path.join(state_dir, "pam_scores.pkl"),
    }

    requested = {
        "genome": args._genome,
        "mm_scores": args._mm_scores,
        "pam_scores": args._pam_scores,
    }

    missing = []
    found = []

    for attr in ("genome", "mm_scores", "pam_scores"):
        candidate = requested[attr] if requested[attr] else defaults[attr]
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            setattr(args, attr, candidate)
            found.append((attr, candidate))
        else:
            flag = {
                "genome": "--genome",
                "mm_scores": "--mm-scores",
                "pam_scores": "--pam-scores",
            }[attr]
            missing.append(f"{candidate} ({flag})")

    # Native CEN3 site tables are optional. The script reconstructs them
    # directly from whole_genes.fna when no CSVs are supplied.
    args.native_csv = []
    if args._native_csv:
        missing_csv = [os.path.abspath(p) for p in args._native_csv
                       if not os.path.exists(p)]
        if missing_csv:
            missing.extend(f"{p} (--native-csv)" for p in missing_csv)
        else:
            args.native_csv = [os.path.abspath(p) for p in args._native_csv]
            found.extend(("native CSV", p) for p in args.native_csv)

    print("Inputs")
    print("------")
    for label, path in found:
        print(f"  {label:<32} {path}")
    if not args.native_csv:
        print(f"  {'native CEN3 site table':<32} built directly from whole_genes.fna")
    for item in missing:
        print(f"  MISSING: {item}")
    print(f"  {'state (cache + results)':<32} {args.state}")
    print(f"  {'figures':<32} {args.outdir}")
    print()

    if missing and args.command != "tune":
        sys.exit(
            "Could not find the required input files. Put whole_genes.fna, "
            "mismatch_score.pkl and pam_scores.pkl in the script-specific "
            "Additional Files directory, or supply explicit command-line paths."
        )


def save_state(args, name, obj):
    with open(os.path.join(args.state, name), "w") as fh:
        json.dump(obj, fh, indent=1)


def cmd_all(args):
    print("\n########## 0/5  selfcheck ##########\n")
    cmd_selfcheck(args)
    print("\n########## 1/5  composition (§4.2) ##########\n")
    cmd_composition(args)
    print("\n########## 2/5  native landscape (§4.2) ##########\n")
    cmd_native(args)
    print("\n########## 3/5  validate installed guide (§4.3) ##########\n")
    cmd_validate(args)
    print("\n########## 4/5  search + null (§4.3) ##########\n")
    cmd_search(args)
    cmd_null(args)
    print("\n########## 5/5  figures ##########\n")
    cmd_figures(args)
    print_sequences(args)
    print("\nDone. Numbers in", args.state, "| figures in", args.outdir)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="all",
                   choices=["composition", "native", "search", "validate",
                            "tune", "null", "figures", "selfcheck", "all"],
                   help="what to run; defaults to `all`, so the script does "
                        "the complete reproduction with no arguments at all")
    p.add_argument("--genome", dest="_genome", default=None,
                   help="optional path to whole_genes.fna; by default the file is read from "
                        "the script-specific Additional Files directory")
    p.add_argument("--genome-mode", choices=["fasta", "legacy"], default="fasta")
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _STEM = os.path.splitext(os.path.basename(__file__))[0]
    _DEFAULT_STATE = os.path.join(_HERE, _STEM + DEFAULT_STATE_SUFFIX)
    _DEFAULT_FIGURES = os.path.join(_HERE, _STEM + DEFAULT_FIGURES_SUFFIX)
    p.add_argument("--state", default=_DEFAULT_STATE,
                   help="directory for required files, cached intermediates and results")
    p.add_argument("--outdir", default=_DEFAULT_FIGURES,
                   help="where the thesis figure is written")
    p.add_argument("--force", action="store_true",
                   help="recompute rather than reuse cached state")

    p.add_argument("--mm-scores", dest="_mm_scores", default=None)
    p.add_argument("--pam-scores", dest="_pam_scores", default=None)
    p.add_argument("--native-csv", dest="_native_csv", nargs="+", default=None)

    p.add_argument("--strands", choices=["both"], default="both",
                   help="off-target screening basis; both strands are required "
                        "for uniform comparison of natural and designed sites")
    p.add_argument("--num-sequences", type=int, default=200_000_000)
    p.add_argument("--batch", type=int, default=2_000_000)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--doench", type=float, default=DOENCH_MIN_DESIGN)
    p.add_argument("--cfd", type=float, default=CFD_MAX)
    p.add_argument("--top", type=int, default=0,
                   help="stage-1 hits to screen; 0 (default) screens all")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--target", default=INSTALLED_GUIDE)
    p.add_argument("--null-n", type=int, default=1000)
    p.add_argument("--null-seed", type=int, default=20260806)
    p.add_argument("--flank", type=int, default=1000)

    args = p.parse_args(argv)
    args.genome = args.mm_scores = args.pam_scores = None
    args.native_csv = []
    os.makedirs(args.state, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nch4_cut_site_pipeline — running: {args.command}\n")
    resolve_inputs(args)
    {"composition": cmd_composition, "native": cmd_native, "search": cmd_search,
     "validate": cmd_validate, "tune": cmd_tune, "null": cmd_null,
     "figures": cmd_figures,
     "selfcheck": cmd_selfcheck,
     "all": cmd_all}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
