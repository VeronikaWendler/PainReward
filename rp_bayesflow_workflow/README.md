# RP–drift true integrative BayesFlow workflow

This folder is a **starter workflow** for the first truly integrative model we discussed.

## Model implemented here

This is **not** a directed model.

It implements a first true integrative single-trial model where:

- pain and money are **fixed trial-wise design inputs**
- they determine the **mean** of a latent single-trial drift variable
- latent drift generates **behavior** through the DDM
- the **same latent drift** generates the single-trial RP value through a Gaussian measurement model

At the trial level:

- `mu_drift_i = v_intercept + v_pain * pain_i + v_money * money_i`
- `latent_drift_i ~ Normal(mu_drift_i, drift_sd)`
- `signed_rt_i ~ DDM(latent_drift_i, boundary, start_point, ndt)`
- `rp_i ~ Normal(rp_intercept + rp_loading * latent_drift_i, rp_noise)`

This is structurally closest to the paper's single-trial integrative drift-style models, where a latent single-trial drift variable generates both EEG and behavior, instead of EEG being plugged in as a regressor. fileciteturn15file0 fileciteturn15file4

## Files

- `config.py` — parameter names, prior ranges, defaults
- `data_utils.py` — load and clean your task CSV, build design bank and observed datasets
- `simulator.py` — prior, DDM simulator, integrative RP–drift simulator, BayesFlow batch simulator
- `model_utils.py` — BayesFlow amortizer + trainer setup
- `train_model.py` — training job with checkpoints
- `validate_recovery.py` — parameter recovery on held-out simulated data
- `fit_real_data.py` — fit the real subject datasets with the trained network
- `permutation_test.py` — shuffle RP within subject and compare posterior shifts
- `posterior_predictive.py` — simple posterior predictive RT and RP checks
- `slurm_train.sh` — example Slurm script for a 24h CPU job

## Expected input CSV

By default the code expects these columns:

- `subj_idx`
- `pain_z`
- `money_z`
- `rp_z`
- `rt`
- `response`

If your RT column is still called `choice_resp.rt`, the loader will use that automatically.

The code assumes:

- response `1` = accept / upper boundary / positive signed RT
- response `0` = reject / lower boundary / negative signed RT

Change `infer_signed_rt()` in `data_utils.py` if your coding differs.

## Core workflow

### 1) Train the model

```bash
python train_model.py \
  --data /path/to/behavioural_sv_cleaned_final_3_with_rp.csv \
  --outdir /path/to/run_001 \
  --epochs 200 \
  --batch-size 16 \
  --iterations-per-epoch 300 \
  --capacity 100 \
  --n-trials-min 80 \
  --n-trials-max 220
```

That creates:

- `checkpoints/`
- `losses.npy`
- `loss_curve.png`
- `training_config.json`

### 2) Run recovery

```bash
python validate_recovery.py \
  --data /path/to/behavioural_sv_cleaned_final_3_with_rp.csv \
  --checkpoint-dir /path/to/run_001/checkpoints \
  --outdir /path/to/run_001/recovery \
  --n-param-sets 250 \
  --n-trials 160
```

This saves true values, posterior samples, posterior means, and a `true_vs_estimated.png` plot.

### 3) Fit the real data

```bash
python fit_real_data.py \
  --data /path/to/behavioural_sv_cleaned_final_3_with_rp.csv \
  --checkpoint-dir /path/to/run_001/checkpoints \
  --outdir /path/to/run_001/real_fit \
  --n-posterior-draws 1000
```

This writes one posterior file and one summary CSV per subject.

### 4) Permutation check

```bash
python permutation_test.py \
  --data /path/to/behavioural_sv_cleaned_final_3_with_rp.csv \
  --checkpoint-dir /path/to/run_001/checkpoints \
  --outdir /path/to/run_001/permutation
```

A true single-trial integrative model should be affected when RP is shuffled relative to behavior, unlike a traditional integrative model where permuting one modality across trials leaves the fit unchanged. fileciteturn15file1 fileciteturn15file3

### 5) Posterior predictive checks

```bash
python posterior_predictive.py \
  --data /path/to/behavioural_sv_cleaned_final_3_with_rp.csv \
  --checkpoint-dir /path/to/run_001/checkpoints \
  --outdir /path/to/run_001/ppc
```

## Practical training advice for your 24h CPU cluster window

Do **not** start with the researchers' full training size.

Use a staged plan:

### smoke test
- `epochs = 10`
- `iterations_per_epoch = 100`
- `batch_size = 8`

### pilot recovery
- `epochs = 50`
- `iterations_per_epoch = 150`
- `batch_size = 16`

### first serious run
- `epochs = 150–300`
- `iterations_per_epoch = 300–500`
- `batch_size = 16 or 32`

The BayesFlow cost is mainly driven by **training updates** and **simulation cost**, not by MCMC chains. Posterior draws after training are cheap. This is exactly the training-vs-inference split emphasized in the paper. fileciteturn15file4

## Important limitations of this starter version

1. It is a **first non-hierarchical subject-level workflow**, not a full HDDM-style hierarchical model.
2. It keeps **boundary constant** across trials.
3. It estimates `ndt`, but does not make RP a non-decision-time signal.
4. It uses **single-trial scalar RP**, not the waveform.
5. It is designed to be the **first model that is likely to train and recover**, not the final best model.

## Natural next extensions

After this first model is stable, the next steps would be:

1. allow pain and money to influence boundary too
2. let the shared latent factor influence both drift and boundary
3. add subject-level hierarchy
4. compare drift-linked and boundary-linked integrative RP models

## Important honesty note

I wrote this workflow to be internally consistent with the code pattern you shared and the model family we agreed on. I did **not** run a real BayesFlow training job here, so you should expect some version-specific adjustment, especially if your cluster has a different BayesFlow/TensorFlow version than the one used in the paper.
