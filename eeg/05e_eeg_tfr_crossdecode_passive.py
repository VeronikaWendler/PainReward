#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 23
BASE_PATH = Path(os.getenv('basepath', Path(__file__).parent.parent.parent))
PARTICIPANTS_TSV = BASE_PATH / 'participants.tsv'
OUT_ROOT = BASE_PATH / 'derivatives' / 'statistics' / 'tfr_crossdecode_passive'
SCORE_DIR = OUT_ROOT / 'participant_scores'
FIG_DIR = OUT_ROOT / 'figures'
DOMAIN_CODES = {'pain': 'p', 'money': 'm'}
DIRECTIONS = [('pain', 'money'), ('money', 'pain')]
RIDGE_ALPHA = 1.0
CLUSTER_FORMING_ALPHA_TWO_SIDED = 0.05
INFER_TMIN = 0.0
INFER_TMAX = 1.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--subjects', nargs='*', default=None)
    p.add_argument('--n-permutations', type=int, default=5000)
    p.add_argument('--n-jobs', type=int, default=max(1, int(os.getenv('SLURM_CPUS_PER_TASK', '1'))))
    p.add_argument('--overwrite-scores', action='store_true')
    p.add_argument('--group-only', action='store_true')
    return p.parse_args()


def tfr_paths(sub: str) -> Dict[str, Path]:
    d = BASE_PATH / 'derivatives' / sub / 'eeg' / 'tfr'
    stem = f'{sub}_passive_cue'
    return {
        'power': d / f'{stem}_power.npy',
        'times': d / f'{stem}_times.npy',
        'freqs': d / f'{stem}_freqs.npy',
        'metadata': d / f'{stem}_metadata.csv',
        'channels': d / f'{stem}_channels.json',
    }


def score_path(sub: str) -> Path:
    return SCORE_DIR / f'{sub}_passive_crossdecode.npz'


def load_subject_list(explicit: Sequence[str] | None) -> List[str]:
    if explicit:
        subs = sorted(dict.fromkeys(map(str, explicit)))
    else:
        if not PARTICIPANTS_TSV.exists():
            raise FileNotFoundError(PARTICIPANTS_TSV)
        df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
        if 'participant_id' not in df.columns:
            raise KeyError(f'participant_id missing from {PARTICIPANTS_TSV}')
        subs = sorted(df['participant_id'].dropna().astype(str).unique())
    if not subs:
        raise RuntimeError('No participants selected')
    for sub in subs:
        missing = [str(p) for p in tfr_paths(sub).values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f'{sub}: missing inputs:\n' + '\n'.join(missing))
    return list(subs)


def make_regressor():
    return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=RIDGE_ALPHA))])


def make_classifier():
    return Pipeline([('scaler', StandardScaler()), ('ridge', RidgeClassifier(alpha=RIDGE_ALPHA))])


def matching_blocks(gs, gt, label):
    bs, bt = np.unique(gs), np.unique(gt)
    if not np.array_equal(bs, bt):
        raise RuntimeError(f'{label}: source/target block sets differ: {bs.tolist()} vs {bt.tolist()}')
    if len(bs) < 2:
        raise RuntimeError(f'{label}: need >=2 blocks')
    return bs


def safe_corr(y, pred):
    if np.std(y) == 0 or np.std(pred) == 0:
        return 0.0
    r = float(np.corrcoef(y, pred)[0, 1])
    return r if np.isfinite(r) else 0.0


def cross_regression_timecourse(Xs, ys, gs, Xt, yt, gt):
    blocks = matching_blocks(gs, gt, 'ordinal')
    pred = np.full((len(yt), Xt.shape[2]), np.nan, dtype=np.float32)
    for b in blocks:
        tr, te = gs != b, gt == b
        if not np.any(tr) or not np.any(te):
            raise RuntimeError(f'block {b}: empty train/test')
        if np.std(ys[tr]) == 0:
            raise RuntimeError(f'block {b}: zero training target variance')
        for ti in range(Xs.shape[2]):
            model = make_regressor()
            model.fit(np.asarray(Xs[tr, :, ti], float), ys[tr])
            pred[te, ti] = model.predict(np.asarray(Xt[te, :, ti], float))
    if not np.isfinite(pred).all():
        raise RuntimeError('non-finite ordinal predictions')
    return np.asarray([safe_corr(yt, pred[:, ti]) for ti in range(Xt.shape[2])], dtype=np.float32)


def cross_classification_timecourse(Xs, ys, gs, Xt, yt, gt):
    if set(np.unique(ys)) != {0, 1} or set(np.unique(yt)) != {0, 1}:
        raise RuntimeError('binary labels must be 0/1 in both domains')
    blocks = matching_blocks(gs, gt, 'highlow')
    decision = np.full((len(yt), Xt.shape[2]), np.nan, dtype=np.float32)
    for b in blocks:
        tr, te = gs != b, gt == b
        if len(np.unique(ys[tr])) < 2:
            raise RuntimeError(f'block {b}: source training fold has one class')
        for ti in range(Xs.shape[2]):
            model = make_classifier()
            model.fit(np.asarray(Xs[tr, :, ti], float), ys[tr])
            decision[te, ti] = model.decision_function(np.asarray(Xt[te, :, ti], float))
    if not np.isfinite(decision).all():
        raise RuntimeError('non-finite binary predictions')
    return np.asarray([roc_auc_score(yt, decision[:, ti]) for ti in range(Xt.shape[2])], dtype=np.float32)


def decode_freq_ordinal(Xs, ys, gs, Xt, yt, gt, fi):
    return cross_regression_timecourse(Xs[:, :, fi, :], ys, gs, Xt[:, :, fi, :], yt, gt)


def decode_freq_highlow(Xs, ys, gs, Xt, yt, gt, fi):
    return cross_classification_timecourse(Xs[:, :, fi, :], ys, gs, Xt[:, :, fi, :], yt, gt)


def process_subject(sub: str, n_jobs: int, overwrite: bool):
    out = score_path(sub)
    if out.exists() and not overwrite:
        print(f'[{sub}] cached -> skip', flush=True)
        return

    p = tfr_paths(sub)
    power = np.load(p['power'], mmap_mode='r')
    times = np.load(p['times']).astype(float)
    freqs = np.load(p['freqs']).astype(float)
    meta = pd.read_csv(p['metadata'])

    if power.ndim != 4 or len(meta) != power.shape[0] or power.shape[2] != len(freqs) or power.shape[3] != len(times):
        raise RuntimeError(f'{sub}: TFR shape/metadata mismatch')
    required = ['condition', 'level_1_5', 'blocks.thisRepN']
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(f'{sub}: metadata missing {missing}')

    cond = meta['condition'].astype(str).str.lower().str.strip().to_numpy()
    level = pd.to_numeric(meta['level_1_5'], errors='coerce').to_numpy(float)
    groups = pd.to_numeric(meta['blocks.thisRepN'], errors='coerce').to_numpy(float)
    if not np.isfinite(level).all() or not np.isfinite(groups).all():
        raise RuntimeError(f'{sub}: non-finite level/block')
    groups = groups.astype(int)

    ordinal_r = np.empty((2, len(freqs), len(times)), np.float32)
    highlow_auc = np.empty_like(ordinal_r)
    counts = np.empty((2, 5), int)  # src ord, tgt ord, src hl, tgt hl, blocks

    print(f'[{sub}] trials={len(meta)}, channels={power.shape[1]}, freqs={len(freqs)}, times={len(times)}', flush=True)

    for di, (source_name, target_name) in enumerate(DIRECTIONS):
        sm = cond == DOMAIN_CODES[source_name]
        tm = cond == DOMAIN_CODES[target_name]
        Xs, Xt = np.asarray(power[sm], np.float32), np.asarray(power[tm], np.float32)
        ys, yt = level[sm], level[tm]
        gs, gt = groups[sm], groups[tm]
        if len(np.unique(ys)) < 5 or len(np.unique(yt)) < 5:
            raise RuntimeError(f'{sub}/{source_name}->{target_name}: not all levels 1..5 present')
        blocks = matching_blocks(gs, gt, f'{sub}/{source_name}->{target_name}')
        print(f'[{sub}] {source_name}->{target_name}: source={len(ys)}, target={len(yt)}, blocks={blocks.tolist()}', flush=True)

        rows = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(decode_freq_ordinal)(Xs, ys, gs, Xt, yt, gt, fi) for fi in range(len(freqs))
        )
        ordinal_r[di] = np.stack(rows)

        hls = np.isin(ys, [1., 2., 4., 5.])
        hlt = np.isin(yt, [1., 2., 4., 5.])
        Xs_hl, Xt_hl = Xs[hls], Xt[hlt]
        ys_hl, yt_hl = (ys[hls] >= 4).astype(int), (yt[hlt] >= 4).astype(int)
        gs_hl, gt_hl = gs[hls], gt[hlt]
        matching_blocks(gs_hl, gt_hl, f'{sub}/{source_name}->{target_name}/highlow')

        rows = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(decode_freq_highlow)(Xs_hl, ys_hl, gs_hl, Xt_hl, yt_hl, gt_hl, fi) for fi in range(len(freqs))
        )
        highlow_auc[di] = np.stack(rows)
        counts[di] = [len(ys), len(yt), len(ys_hl), len(yt_hl), len(blocks)]

    tmp = out.with_suffix(out.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(
            f, participant=np.array(sub), times=times, freqs=freqs,
            direction_names=np.array([f'{s}_to_{t}' for s, t in DIRECTIONS]),
            ordinal_r=ordinal_r, highlow_auc=highlow_auc, counts=counts,
        )
    os.replace(tmp, out)
    print(f'[{sub}] DONE', flush=True)


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def run_cluster(data, freqs, times, n_permutations, n_jobs):
    tmask = (times >= INFER_TMIN - 1e-12) & (times <= INFER_TMAX + 1e-12)
    ti = times[tmask]
    x = data[:, :, tmask]
    if not np.isfinite(x).all():
        raise RuntimeError('non-finite group data')
    n_sub, n_freq, n_time = x.shape
    threshold = float(stats.t.ppf(1 - CLUSTER_FORMING_ALPHA_TWO_SIDED / 2, df=n_sub - 1))
    adjacency = combine_adjacency(n_freq, n_time)
    t_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        x.reshape(n_sub, -1), threshold=threshold, n_permutations=n_permutations,
        tail=0, adjacency=adjacency, seed=RANDOM_STATE, n_jobs=n_jobs,
        out_type='indices', verbose=False,
    )
    return t_obs.reshape(n_freq, n_time), [np.asarray(c[0], int) for c in clusters], np.asarray(pvals, float), ti


def summarize_cluster(idx, tmap, freqs, times_inf):
    fi, ti = np.unravel_index(idx, tmap.shape)
    vals = tmap.reshape(-1)[idx]
    pk = int(np.argmax(np.abs(vals)))
    pfi, pti = np.unravel_index(int(idx[pk]), tmap.shape)
    return {
        'sign': 'positive' if np.mean(vals) > 0 else 'negative',
        'freq_min_hz': float(freqs[fi].min()), 'freq_max_hz': float(freqs[fi].max()),
        'time_start_s': float(times_inf[ti].min()), 'time_end_s': float(times_inf[ti].max()),
        'peak_freq_hz': float(freqs[pfi]), 'peak_time_s': float(times_inf[pti]),
        'peak_t': float(vals[pk]), 'cluster_mass_signed': float(vals.sum()),
        'cluster_mass_abs': float(np.abs(vals).sum()), 'n_points': int(len(idx)),
    }


def load_group(subjects):
    ords, hls, rows = [], [], []
    ref_times = ref_freqs = ref_dirs = None
    for sub in subjects:
        z = np.load(score_path(sub), allow_pickle=False)
        times, freqs = z['times'].astype(float), z['freqs'].astype(float)
        dirs = z['direction_names'].astype(str).tolist()
        if ref_times is None:
            ref_times, ref_freqs, ref_dirs = times, freqs, dirs
        else:
            if not np.allclose(times, ref_times) or not np.array_equal(freqs, ref_freqs) or dirs != ref_dirs:
                raise RuntimeError(f'{sub}: cached grids/labels differ')
        ords.append(z['ordinal_r']); hls.append(z['highlow_auc'])
        for di, d in enumerate(dirs):
            c = z['counts'][di]
            rows.append({'participant': sub, 'direction': d, 'source_n_ordinal': int(c[0]),
                         'target_n_ordinal': int(c[1]), 'source_n_highlow': int(c[2]),
                         'target_n_highlow': int(c[3]), 'n_blocks': int(c[4])})
    return np.stack(ords), np.stack(hls), ref_times, ref_freqs, ref_dirs, pd.DataFrame(rows)


def analyze_family(family, raw, times, freqs, directions, n_permutations, n_jobs):
    test = fisher_z(raw) if family == 'ordinal' else raw - 0.5
    chance = 0.0 if family == 'ordinal' else 0.5
    metric = 'r' if family == 'ordinal' else 'auc'
    omnibus, clusters_out, masks = [], [], {}

    for di, direction in enumerate(directions):
        tmap, clusters, pvals, times_inf = run_cluster(test[:, di], freqs, times, n_permutations, n_jobs)
        omnibus.append({'family': family, 'direction': direction,
                        'omnibus_p_within_direction_surface': float(np.min(pvals)) if len(pvals) else 1.0})
        sigmask = np.zeros_like(tmap, bool)
        for ci, (idx, p) in enumerate(zip(clusters, pvals)):
            row = {'family': family, 'direction': direction, 'cluster_id': ci,
                   'cluster_p_within_direction_surface': float(p)}
            row.update(summarize_cluster(idx, tmap, freqs, times_inf))
            clusters_out.append(row)
            if p < 0.05:
                sigmask.reshape(-1)[idx] = True
        masks[(family, direction)] = (times_inf, sigmask)

    om = pd.DataFrame(omnibus)
    om['omnibus_p_holm_across_two_directions'] = holm_adjust(om['omnibus_p_within_direction_surface'])
    om['direction_significant_holm_0p05'] = om['omnibus_p_holm_across_two_directions'] < 0.05
    cl = pd.DataFrame(clusters_out)
    if len(cl):
        cl = cl.merge(om[['family','direction','omnibus_p_holm_across_two_directions','direction_significant_holm_0p05']],
                      on=['family','direction'], how='left', validate='many_to_one')
        cl['cluster_significant_final'] = (cl['cluster_p_within_direction_surface'] < 0.05) & cl['direction_significant_holm_0p05']

    surface = []
    for di, direction in enumerate(directions):
        mean, sem = raw[:, di].mean(0), stats.sem(raw[:, di], axis=0)
        for fi, f in enumerate(freqs):
            for ti, t in enumerate(times):
                surface.append({'family': family, 'direction': direction, 'frequency_hz': float(f), 'time_s': float(t),
                                f'mean_{metric}': float(mean[fi, ti]), f'sem_{metric}': float(sem[fi, ti]),
                                'chance': chance, 'inside_primary_inference_window': bool(INFER_TMIN <= t <= INFER_TMAX)})
    return om, cl, pd.DataFrame(surface), masks


def plot_family(family, raw, times, freqs, directions, om, cl, masks):
    centre = 0.0 if family == 'ordinal' else 0.5
    label = 'Mean cross-decoded Pearson r' if family == 'ordinal' else 'Mean cross-decoded ROC-AUC'
    for di, direction in enumerate(directions):
        mean = raw[:, di].mean(0)
        lim = max(float(np.max(np.abs(mean - centre))), 1e-6)
        fig, ax = plt.subplots(figsize=(10, 5.6))
        im = ax.pcolormesh(times, freqs, mean, shading='auto', cmap='RdBu_r', vmin=centre-lim, vmax=centre+lim)
        ax.axvline(0, lw=1); ax.axvline(INFER_TMAX, lw=1, ls='--')
        ax.set(xlabel='Time from cue onset (s)', ylabel='Frequency (Hz)')
        prow = om[(om.family == family) & (om.direction == direction)].iloc[0]
        ax.set_title(f"{direction.replace('_',' ')} — {family}\nHolm p={prow.omnibus_p_holm_across_two_directions:.4g}")
        fig.colorbar(im, ax=ax, label=label)
        if len(cl):
            final = cl[(cl.family == family) & (cl.direction == direction) & (cl.cluster_significant_final == True)]
            if len(final):
                times_inf, mask = masks[(family, direction)]
                ax.contour(times_inf, freqs, mask.astype(int), levels=[0.5], linewidths=1.5)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f'{family}_{direction}_freq_time.png', dpi=240)
        plt.close(fig)


def precue_qc(family, raw, times, directions):
    pre = times < 0
    rows = []
    centred = raw if family == 'ordinal' else raw - 0.5
    for di, d in enumerate(directions):
        per_sub = centred[:, di, :, :][:, :, pre].mean((1,2))
        test = stats.ttest_1samp(per_sub, 0.0)
        rows.append({'family': family, 'direction': d, 'pre_cue_group_mean': float(per_sub.mean()),
                     'pre_cue_group_sem': float(stats.sem(per_sub)), 'pre_cue_t': float(test.statistic),
                     'pre_cue_p_two_sided_uncorrected': float(test.pvalue)})
    return pd.DataFrame(rows)


def run_group(subjects, args):
    ordinal, highlow, times, freqs, directions, counts = load_group(subjects)
    counts.to_csv(OUT_ROOT / 'participant_trial_counts.csv', index=False)
    np.savez_compressed(OUT_ROOT / 'group_score_arrays.npz', subjects=np.array(subjects), times=times, freqs=freqs,
                        direction_names=np.array(directions), ordinal_r=ordinal, highlow_auc=highlow)

    oms, cls, surfs, pre = [], [], [], []
    for family, raw in [('ordinal', ordinal), ('highlow', highlow)]:
        print(f'\n===== {family.upper()} CROSS-DECODING =====', flush=True)
        om, cl, surf, masks = analyze_family(family, raw, times, freqs, directions, args.n_permutations, args.n_jobs)
        oms.append(om); cls.append(cl); surfs.append(surf); pre.append(precue_qc(family, raw, times, directions))
        plot_family(family, raw, times, freqs, directions, om, cl, masks)
        print(om.to_string(index=False), flush=True)
        if len(cl):
            sig = cl[cl.cluster_significant_final == True]
            print('\nSignificant clusters:', flush=True)
            if len(sig):
                cols = ['direction','sign','freq_min_hz','freq_max_hz','time_start_s','time_end_s','peak_freq_hz','peak_time_s','peak_t',
                        'cluster_p_within_direction_surface','omnibus_p_holm_across_two_directions']
                print(sig[cols].to_string(index=False), flush=True)
            else:
                print('None', flush=True)

    pd.concat(oms, ignore_index=True).to_csv(OUT_ROOT / 'direction_omnibus_holm.csv', index=False)
    nonempty = [x for x in cls if len(x)]
    (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()).to_csv(OUT_ROOT / 'crossdecode_clusters.csv', index=False)
    pd.concat(surfs, ignore_index=True).to_csv(OUT_ROOT / 'crossdecode_group_surface.csv', index=False)
    pd.concat(pre, ignore_index=True).to_csv(OUT_ROOT / 'precue_qc.csv', index=False)

    params = {
        'analysis': 'PainReward passive total-power spectral cross-decoding',
        'n_subjects': len(subjects), 'subjects': list(subjects), 'directions': directions,
        'frequencies_hz': freqs.tolist(), 'saved_score_time_s': [float(times.min()), float(times.max())],
        'primary_inference_time_s': [INFER_TMIN, INFER_TMAX],
        'feature_selection': 'none; full pre-defined frequency x time TFR space',
        'block_safe_transfer': 'held-out block b: train source on blocks != b, test target block b; rotate blocks',
        'ordinal_primary': {'target': 'level 1..5', 'model': 'StandardScaler -> Ridge(alpha=1)',
                            'score': 'Pearson r on pooled target predictions', 'group_transform': 'Fisher z(r)'},
        'highlow_secondary': {'low': [1,2], 'excluded': 3, 'high': [4,5],
                              'model': 'StandardScaler -> RidgeClassifier(alpha=1)', 'score': 'ROC-AUC', 'group_null': 'AUC-0.5=0'},
        'inference': 'two-sided 2D frequency x time cluster test per direction; Holm across two directions within each family',
        'cluster_forming_alpha_two_sided': CLUSTER_FORMING_ALPHA_TWO_SIDED,
        'n_permutations': args.n_permutations, 'random_state': RANDOM_STATE,
        'reason_for_1p10_s_limit': 'cue ends ~1.5 s; low-frequency Morlet support extends around nominal centre; later scores retained but not primary inference',
    }
    with open(OUT_ROOT / 'params.json', 'w') as f:
        json.dump(params, f, indent=2)

    print('\n' + '='*78, flush=True)
    print('PASSIVE TFR CROSS-DECODING COMPLETE', flush=True)
    print(f'Subjects: {len(subjects)}', flush=True)
    print(f'Outputs: {OUT_ROOT}', flush=True)
    print('='*78, flush=True)


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True); SCORE_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    subjects = load_subject_list(args.subjects)
    print('='*78, flush=True)
    print('PAINREWARD PASSIVE TFR CROSS-DECODING', flush=True)
    print(f'Subjects: {len(subjects)}', flush=True)
    print('PRIMARY: ordinal 1..5 | SECONDARY: low(1,2) vs high(4,5)', flush=True)
    print('Directions: pain->money and money->pain', flush=True)
    print(f'Two-sided 2D cluster inference: {INFER_TMIN:.2f}-{INFER_TMAX:.2f}s', flush=True)
    print(f'Permutations: {args.n_permutations} | CPUs: {args.n_jobs}', flush=True)
    print('='*78, flush=True)
    if not args.group_only:
        for i, sub in enumerate(subjects, 1):
            print(f'\n[{i:02d}/{len(subjects)}] {sub}', flush=True)
            process_subject(sub, args.n_jobs, args.overwrite_scores)
    run_group(subjects, args)


if __name__ == '__main__':
    main()
