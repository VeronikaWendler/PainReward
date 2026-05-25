"""
Step 1 of the EEG decoding Stage 1 plan — within-passive decoding.

Three contrasts (independently per subject, then group-level cluster perm):
  (a) pain vs. money              — all passive trials, label = (attribute=='pain')
  (b) high-vs-low pain            — pain trials, level ∈ {1,2,4,5}, label = (level>=4)
  (c) high-vs-low money           — money trials, level ∈ {1,2,4,5}, label = (level>=4)

Outputs under derivatives/statistics/eeg_decoding_stage1/step1_passive/.

Per CLAUDE.md, this script does NOT use graceful per-subject error handling:
any missing file or malformed metadata raises and aborts the run.

Authors: Michel-Pierre Coll
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from decoding_utils import (
    BASE_PATH,
    RANDOM_STATE,
    ClusterResult,
    concat_window_features,
    cluster_perm_1samp_vs_chance,
    cv_auc_timecourse,
    cv_pearson_r_timecourse,
    fdr_correct_clusters,
    find_peak_window,
    haufe_pattern_timecourse,
    haufe_pattern_timecourse_ridge,
    load_passive_epochs,
    load_subject_list,
    plot_decoding_timecourse,
    plot_topomap_peak,
    save_params,
)

OUT_ROOT = BASE_PATH / "derivatives" / "statistics" / "eeg_decoding_stage1" / "step1_passive"
FIG_DIR = OUT_ROOT / "figures"


@dataclass
class Contrast:
    name: str                            # file-safe id
    pretty: str                          # for figure titles
    kind: str                            # "classification" or "regression"
    trial_filter: Callable[[pd.DataFrame], np.ndarray]   # boolean mask over metadata
    target_fn: Callable[[pd.DataFrame], np.ndarray]      # label vector (0/1 for class, float for reg)


def define_contrasts() -> List[Contrast]:
    return [
        Contrast(
            name="pain_vs_money",
            pretty="Pain vs. money cue",
            kind="classification",
            trial_filter=lambda md: np.ones(len(md), dtype=bool),
            target_fn=lambda md: (md["attribute"].values == "pain").astype(int),
        ),
        Contrast(
            name="pain_level",
            pretty="Pain level regression (passive)",
            kind="regression",
            trial_filter=lambda md: (md["attribute"].values == "pain"),
            target_fn=lambda md: md["level"].values.astype(float),
        ),
        Contrast(
            name="money_level",
            pretty="Money level regression (passive)",
            kind="regression",
            trial_filter=lambda md: (md["attribute"].values == "money"),
            target_fn=lambda md: md["level"].values.astype(float),
        ),
    ]


def _chance_for(kind: str) -> float:
    return 0.5 if kind == "classification" else 0.0


def _ylabel_for(kind: str) -> str:
    return "ROC-AUC" if kind == "classification" else "Pearson r"


def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 1: within-passive decoding (pain-vs-money + high-vs-low level)."
    )
    p.add_argument("--quick", action="store_true",
                   help="Smoke mode: 3 subjects, 200 permutations.")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Optional explicit subject list (e.g. sub-004 sub-005).")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--resample-hz", type=float, default=250.0)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Currently used only for cluster-perm parallelism.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--window-ms", type=float, default=100.0,
                   help="Sliding-window width in ms for feature binning (default 100).")
    p.add_argument("--step-ms", type=float, default=50.0,
                   help="Sliding-window step in ms (default 50).")
    return p.parse_args(argv)


def main(argv: list) -> int:
    args = parse_args(argv)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_params(OUT_ROOT, argv, extra={
        "n_permutations": args.n_permutations,
        "resample_hz": args.resample_hz,
        "alpha": args.alpha,
        "window_ms": args.window_ms,
        "step_ms": args.step_ms,
    })

    subjects = args.subjects if args.subjects else load_subject_list()
    if args.quick:
        subjects = subjects[:3]
        args.n_permutations = min(args.n_permutations, 200)
    print(f"Running Step 1 on {len(subjects)} subjects.", flush=True)

    contrasts = define_contrasts()
    # ------------------------------------------------------------------
    # Per-subject decoding for every contrast.
    # ------------------------------------------------------------------
    # For each contrast we accumulate:
    #   - per_subject_auc[name] : list of (subject, times_ms, auc) tuples
    #   - per_subject_pattern[name] : list of (subject, ch_names, pattern[n_chan,n_times])
    per_subject_auc: Dict[str, list] = {c.name: [] for c in contrasts}
    per_subject_pattern: Dict[str, list] = {c.name: [] for c in contrasts}
    info_ref: dict = {}   # keeps one mne.Info per contrast for topomap plotting later

    for sub in subjects:
        print(f"  [{sub}] loading passive epochs", flush=True)
        epochs = load_passive_epochs(sub, BASE_PATH, resample_hz=args.resample_hz)
        md = epochs.metadata
        X_full = epochs.get_data(copy=False)        # (n_trials, n_chan, n_times)
        times_ms = (epochs.times * 1000.0).astype(float)
        # Slide a window over the epoch and concatenate within-window samples as
        # features (no averaging) — each bin's feature vector has shape
        # (n_channels * win_samples,) with channels-blocked / win-fastest layout.
        X_full, times_ms, win_samples = concat_window_features(
            X_full, times_ms, args.window_ms, args.step_ms)

        for c in contrasts:
            mask = c.trial_filter(md)
            X = X_full[mask]
            y = c.target_fn(md.loc[mask])
            if X.shape[0] < 10:
                raise RuntimeError(
                    f"{sub}/{c.name}: too few trials after filter (n={X.shape[0]})."
                )
            if c.kind == "classification" and len(np.unique(y)) < 2:
                raise RuntimeError(
                    f"{sub}/{c.name}: single-class y after filter; got {np.unique(y)}."
                )
            if c.kind == "regression" and np.std(y) == 0:
                raise RuntimeError(
                    f"{sub}/{c.name}: zero-variance y after filter; got {np.unique(y)}."
                )
            if c.kind == "classification":
                y_int = y.astype(int)
                print(f"    [{c.name}] n_trials={X.shape[0]} "
                      f"({(y_int==0).sum()}/{(y_int==1).sum()}) — fitting CV (LDA)",
                      flush=True)
                score = cv_auc_timecourse(X, y_int, n_splits=5, random_state=RANDOM_STATE)
                pattern = haufe_pattern_timecourse(X, y_int, win_samples=win_samples)
            else:
                print(f"    [{c.name}] n_trials={X.shape[0]} "
                      f"(target range {y.min():.0f}..{y.max():.0f}) — fitting CV (Ridge)",
                      flush=True)
                score = cv_pearson_r_timecourse(X, y, n_splits=5, random_state=RANDOM_STATE)
                pattern = haufe_pattern_timecourse_ridge(X, y, win_samples=win_samples)
            per_subject_auc[c.name].append((sub, times_ms, score))
            per_subject_pattern[c.name].append((sub, list(epochs.ch_names), pattern))
            info_ref.setdefault(c.name, epochs.info.copy())

    # Persist the per-subject scores to long-format CSV.
    # 'score' = ROC-AUC for classification contrasts, Pearson r for regression.
    for c in contrasts:
        rows = []
        for sub, times_ms, score_tc in per_subject_auc[c.name]:
            rows.extend({"subject": sub, "time_ms": float(t),
                         "score": float(s), "kind": c.kind}
                        for t, s in zip(times_ms, score_tc))
        df = pd.DataFrame(rows)
        out_csv = OUT_ROOT / f"score_per_subject_{c.name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv}", flush=True)

    # Persist Haufe patterns per contrast.
    for c in contrasts:
        subs_p, ch_names_p, patterns = [], [], []
        for sub, ch_names, pattern in per_subject_pattern[c.name]:
            subs_p.append(sub)
            ch_names_p.append(ch_names)
            patterns.append(pattern)
        if not all(ch == ch_names_p[0] for ch in ch_names_p):
            raise RuntimeError(
                f"{c.name}: channel ordering differs across subjects; cannot stack patterns."
            )
        patterns_arr = np.stack(patterns, axis=0)   # (n_subj, n_chan, n_times)
        out_npz = OUT_ROOT / f"haufe_patterns_{c.name}.npz"
        np.savez_compressed(
            out_npz,
            subjects=np.array(subs_p),
            ch_names=np.array(ch_names_p[0]),
            times_ms=per_subject_auc[c.name][0][1],
            patterns=patterns_arr.astype(np.float32),
        )
        print(f"  wrote {out_npz} shape={patterns_arr.shape}", flush=True)

    # ------------------------------------------------------------------
    # Group-level inference (cluster perm vs chance, FDR across contrasts).
    # ------------------------------------------------------------------
    group_scores: Dict[str, np.ndarray] = {}
    group_times: Dict[str, np.ndarray] = {}
    for c in contrasts:
        records = per_subject_auc[c.name]
        # All subjects share the same times_ms because epochs were resampled
        # to the same rate; verify.
        ref_times = records[0][1]
        for sub, t, _ in records:
            if not np.allclose(t, ref_times):
                raise RuntimeError(
                    f"{c.name}/{sub}: time vector differs from reference subject."
                )
        scores = np.stack([a for _, _, a in records], axis=0)   # (n_subj, n_times)
        group_scores[c.name] = scores
        group_times[c.name] = ref_times

    cluster_results: Dict[str, ClusterResult] = {}
    for c in contrasts:
        print(f"  cluster perm — {c.name} ({c.kind})", flush=True)
        cluster_results[c.name] = cluster_perm_1samp_vs_chance(
            group_scores[c.name],
            chance=_chance_for(c.kind),
            n_permutations=args.n_permutations,
            tail=1,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
        )

    # FDR across the three contrasts.
    fdr_correct_clusters([cluster_results[c.name] for c in contrasts])

    for c in contrasts:
        res = cluster_results[c.name]
        times_ms = group_times[c.name]
        rows = []
        for k, inds in enumerate(res.cluster_inds):
            if len(inds) == 0:
                continue
            t_sum = float(res.t_obs[inds].sum())
            rows.append({
                "cluster_id": k,
                "p": float(res.cluster_pvals[k]),
                "p_fdr": float(res.cluster_pvals_fdr[k]),
                "t_sum": t_sum,
                "tmin_ms": float(times_ms[inds.min()]),
                "tmax_ms": float(times_ms[inds.max()]),
                "n_timepoints": int(len(inds)),
            })
        df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True) \
             if rows else pd.DataFrame(columns=["cluster_id","p","p_fdr","t_sum","tmin_ms","tmax_ms","n_timepoints"])
        out_csv = OUT_ROOT / f"cluster_stats_{c.name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv} ({len(df)} clusters)", flush=True)

    # ------------------------------------------------------------------
    # Plotting: time-course (with shaded sig cluster) + Haufe topomap at peak.
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")   # headless — never opens a window during batch runs

    for c in contrasts:
        scores = group_scores[c.name]
        times_ms = group_times[c.name]
        res = cluster_results[c.name]

        sig_mask = np.zeros_like(times_ms, dtype=bool)
        for k, inds in enumerate(res.cluster_inds):
            if res.cluster_pvals_fdr[k] < args.alpha:
                sig_mask[inds] = True

        plot_decoding_timecourse(
            times_ms=times_ms,
            scores=scores,
            sig_mask=sig_mask,
            title=c.pretty,
            chance=_chance_for(c.kind),
            ylabel=_ylabel_for(c.kind),
            out_path=FIG_DIR / f"timecourse_{c.name}.png",
        )

        # Group-mean Haufe pattern averaged over the peak window of t_obs.
        pat_npz = np.load(OUT_ROOT / f"haufe_patterns_{c.name}.npz", allow_pickle=False)
        patterns = pat_npz["patterns"]                          # (n_subj, n_chan, n_times)
        ch_names = list(pat_npz["ch_names"].astype(str))
        if list(info_ref[c.name].ch_names) != ch_names:
            raise RuntimeError(
                f"{c.name}: channel order in npz does not match epochs.info."
            )
        tmin_ms, tmax_ms = find_peak_window(res.t_obs, times_ms, min_width_ms=50.0)
        in_window = (times_ms >= tmin_ms) & (times_ms <= tmax_ms)
        if not in_window.any():
            raise RuntimeError(f"{c.name}: peak window {tmin_ms:.0f}-{tmax_ms:.0f} ms empty.")
        pat_window = patterns[:, :, in_window].mean(axis=2)     # (n_subj, n_chan)
        group_pattern = pat_window.mean(axis=0)                 # (n_chan,)

        plot_topomap_peak(
            pattern=group_pattern,
            info=info_ref[c.name],
            title=f"{c.pretty}\nHaufe pattern {tmin_ms:.0f}-{tmax_ms:.0f} ms",
            out_path=FIG_DIR / f"topomap_{c.name}_peakwindow.png",
        )
        print(f"  wrote figures for {c.name}", flush=True)

    print("Step 1 complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
