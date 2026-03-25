
import os, json, numpy as np, pandas as pd, mne
from os.path import join as opj
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace"))
basepath = PROJECT_DIR / "EEG" / "PainReward_sub-001-050" / "painrewardeegdata"
regressor_path = PROJECT_DIR / "Hddm_Docker_August_24" / "derviatives" / "figures_dir" / "painreward_behavioural_data_LPP_9"

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
import re
from pathlib import Path
import os

layout = BIDSLayout(basepath)

# disable Numba JIT caching & compilation
#os.environ["NUMBA_DISABLE_JIT"] = "1"
import numba
numba.config.CACHE_ENABLE = False

outpath = opj(basepath, "derivatives")
os.makedirs(outpath, exist_ok=True)

# List participants
part = [p for p in os.listdir(opj(basepath)) if "sub" in p]
part.sort()



DERIV = outpath
CSV   = r"D:/.../data_sets/your_trialwise_regressors.csv"  # <-- EDIT
EPO_REL = "eeg/erps"      # where *_cues_singletrials-epo.fif lives under each sub
EPO_NAME = "{sub}_decision_cues_singletrials-epo.fif"

TARGETS = ["painlevel","moneylevel","SV_pain","v_pain","v_money"]   # pick any subset
CONFOUNDS = ["RT"]        # add more if needed
ALPHAS = np.logspace(-3, 3, 13)
N_OUTER = 5               # outer CV
N_PERM  = 500             # permutations for p-values
SMOOTH_WIN = 5            # samples for simple moving average (set 0 to disable)
RANDOM_STATE = 23
OUTDIR = opj(DERIV, "mvpa_results")
os.makedirs(OUTDIR, exist_ok=True)

# --------- UTIL ---------
def moving_average(x, w):
    if w <= 1: return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def fit_confounds(train_conf, Y):
    # returns beta for confounds and function to apply
    # Adds intercept implicitly via column of ones
    Xc = np.c_[np.ones((len(train_conf),1)), train_conf]
    # closed-form OLS
    beta = np.linalg.pinv(Xc) @ Y
    def apply(conf, Y):
        Xc2 = np.c_[np.ones((len(conf),1)), conf]
        return Y - (Xc2 @ beta)  # residuals
    return beta, apply

def haufe_transform(X, y, model):
    # Haufe map for linear ridge: A = Sigma_X * W / var(y)
    # compute from the *training* data passed in
    W = model.coef_.ravel()         # shape (n_chans,)
    Xc = X - X.mean(0, keepdims=True)
    covX = (Xc.T @ Xc) / (X.shape[0]-1)
    vary = np.var(y, ddof=1)
    return covX @ W / (vary + 1e-12)

def time_resolved_decode(X, y, conf_train, conf_test, rs=RANDOM_STATE):
    """
    X: (trials, channels, times)
    y: (trials,)
    conf_train/conf_test: dict with arrays per confound aligned with train/test indices
    Returns: r2_time (times,), yhat_by_time (trials, times), w_haufe (channels, times), alpha_time (times,)
    """
    rng = check_random_state(rs)
    n_trials, n_ch, n_t = X.shape
    r2_t  = np.zeros(n_t)
    yhat_t = np.zeros((n_trials, n_t))
    haufe = np.zeros((n_ch, n_t))
    alpha_t = np.zeros(n_t)

    # outer CV splits on trials
    outer = KFold(n_splits=N_OUTER, shuffle=True, random_state=rs)

    for ti in range(n_t):
        Xt = X[:,:,ti]  # (trials, channels)

        # collect out-of-fold preds & haufe from each fold
        yhat_fold = np.zeros(n_trials); yhat_fold[:] = np.nan
        haufe_folds = []
        alphas_f = []

        for train_idx, test_idx in outer.split(Xt):
            # confound regression: fit on *train only*, apply to both
            Ctrain = np.column_stack([conf_train[k][train_idx] for k in conf_train])
            Ctest  = np.column_stack([conf_test[k][test_idx] for k in conf_test])
            _, apply_conf = fit_confounds(Ctrain, y[train_idx])
            y_train_res = apply_conf(Ctrain, y[train_idx])
            y_test_res  = apply_conf(Ctest,  y[test_idx])

            # pipeline: scale -> ridge with inner-CV (RidgeCV)
            pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                             ("ridge",  RidgeCV(alphas=ALPHAS, store_cv_values=False))])

            pipe.fit(Xt[train_idx,:], y_train_res)
            y_pred_res = pipe.predict(Xt[test_idx,:])
            # undo confound removal? No: evaluate on residualized y to test unique variance.
            yhat_fold[test_idx] = y_pred_res

            # derive Haufe map from training data & model weights
            ridge = Ridge(alpha=pipe.named_steps["ridge"].alpha_)
            ridge.fit(pipe.named_steps["scaler"].transform(Xt[train_idx,:]), y_train_res)
            # need weights in sensor space: compensate scaler
            w_std = ridge.coef_.ravel() / pipe.named_steps["scaler"].scale_
            # Haufe on *raw* train features
            A = haufe_transform(Xt[train_idx,:], y_train_res, Ridge().set_params(coef_=w_std, fit_intercept=False))
            haufe_folds.append(A)
            alphas_f.append(pipe.named_steps["ridge"].alpha_)

        # aggregate across outer folds
        ok = ~np.isnan(yhat_fold)
        r2_t[ti] = r2_score(y[ok], yhat_fold[ok])
        yhat_t[:,ti] = yhat_fold
        haufe[:,ti] = np.nanmean(np.vstack(haufe_folds), axis=0)
        alpha_t[ti] = np.median(alphas_f)

    return r2_t, yhat_t, haufe, alpha_t

def permutation_pvalue(X, y, conf, r2_obs, n_perm=N_PERM, rs=RANDOM_STATE):
    rng = check_random_state(rs)
    n_t = X.shape[2]
    null = np.zeros((n_perm, n_t))
    for p in range(n_perm):
        y_perm = rng.permutation(y)
        r2_p, _, _, _ = time_resolved_decode(X, y_perm, conf, conf, rs=rng.randint(1e9))
        null[p,:] = r2_p
    # p-value: proportion of null >= observed (one-sided)
    pvals = (np.sum(null >= r2_obs[None,:], axis=0) + 1) / (n_perm + 1)
    return pvals, null

# --------- MAIN ---------
if __name__ == "__main__":
    beh = pd.read_csv(CSV)  # columns: participant, trialsnum, TARGETS..., CONFOUNDS...
    subs = sorted([d for d in os.listdir(DERIV) if d.startswith("sub-")])

    for target in TARGETS:
        all_r2, all_p, all_haufe, all_alpha = [], [], [], []
        peak_times = []
        for sub in subs:
            epo_f = opj(DERIV, sub, EPO_REL, EPO_NAME.format(sub=sub))
            if not os.path.exists(epo_f): 
                continue
            epo = mne.read_epochs(epo_f, preload=True)

            # align trials: metadata has participant_id matching sub and trialsnum
            meta = epo.metadata.copy()
            df = beh[(beh["participant"]==sub)]
            # join on trialsnum
            df = df.merge(meta[["trialsnum"]], on="trialsnum", how="inner")
            # mask epochs to those trials & sort
            keep_mask = epo.metadata["trialsnum"].isin(df["trialsnum"])
            epo = epo[keep_mask]
            meta = epo.metadata.reset_index(drop=True)
            df = df.sort_values("trialsnum").reset_index(drop=True)

            # drop bad trials
            good = meta["badtrial"].to_numpy()==0
            epo = epo[good]
            meta = meta[good].reset_index(drop=True)
            df = df[good].reset_index(drop=True)

            X = epo.get_data(picks="eeg")  # (trials, channels, times)
            times = epo.times
            y = df[target].to_numpy().astype(float)
            # confounds dict -> arrays
            conf = {c: df[c].to_numpy().astype(float) for c in CONFOUNDS}

            # split conf dict into same for "train" and "test" placeholders (API expects both)
            r2_t, yhat_t, haufe, alpha_t = time_resolved_decode(X, y, conf, conf, rs=RANDOM_STATE)

            # smooth R² slightly for visualization/peak picking
            r2_s = moving_average(r2_t, SMOOTH_WIN)

            # permutations (one-sided)
            pvals_t, _ = permutation_pvalue(X, y, conf, r2_t, n_perm=N_PERM, rs=RANDOM_STATE)

            # save per-subject
            sdir = opj(OUTDIR, sub)
            os.makedirs(sdir, exist_ok=True)
            np.save(opj(sdir, f"{target}_r2_time.npy"), r2_t)
            np.save(opj(sdir, f"{target}_pvals_time.npy"), pvals_t)
            np.save(opj(sdir, f"{target}_haufe_maps.npy"), haufe)             # (channels, times)
            np.save(opj(sdir, f"{target}_yhat_trials_by_time.npy"), yhat_t)   # (trials, times)
            np.save(opj(sdir, f"{target}_alpha_by_time.npy"), alpha_t)
            with open(opj(sdir, f"{target}_times.json"), "w") as f:
                json.dump({"times": times.tolist()}, f)

            # aggregate (store peak)
            peak_ti = int(np.nanargmax(r2_s))
            peak_times.append(times[peak_ti])

            all_r2.append(r2_t); all_p.append(pvals_t); all_haufe.append(haufe); all_alpha.append(alpha_t)

        # group-level save
        if len(all_r2):
            G = opj(OUTDIR, "group"); os.makedirs(G, exist_ok=True)
            np.save(opj(G, f"{target}_r2_time_group.npy"), np.vstack(all_r2))      # (subjects, times)
            np.save(opj(G, f"{target}_p_time_group.npy"),  np.vstack(all_p))
            np.save(opj(G, f"{target}_alpha_group.npy"),   np.vstack(all_alpha))
            # for Haufe maps, keep as list (subjects of arrays) or mean later
            np.save(opj(G, f"{target}_peak_times_sec.npy"), np.array(peak_times))
            # simple report CSV
            mean_r2 = np.nanmean(np.vstack(all_r2), axis=0)
            df_rep = pd.DataFrame({"time_s": times, "r2_mean": mean_r2})
            df_rep.to_csv(opj(G, f"{target}_r2_time_group.csv"), index=False)
