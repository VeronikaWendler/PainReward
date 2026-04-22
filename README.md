# PainReward

Analysis code for the **PainReward** project at Université Laval — pain and reward decision-making with EEG (N=38 participants).

---

## Running the pipeline

The entire pipeline is driven by a single script:

```bash
bash run_all.sh
```

Scripts are numbered in execution order. HDDM fitting runs inside a Docker container (`hcp4715/hddm`); everything else runs in the local Python environment.

Set the following environment variables if your data are not at the default BIDS root (`../../`, relative to `code/`):

| Variable | Default | Purpose |
|---|---|---|
| `basepath` | `../../` | BIDS data root |
| `PROJECT_DIR` | `../../` | Used by Docker-based HDDM scripts |
| `HDDM_DIR` | `<basepath>/derivatives/hddm` | HDDM output directory |
| `NJOBS` | `n_cpu - 1` | Parallel jobs for EEG permutation tests |

---

## Repository map

```
code/
├── run_all.sh                     # Single entry point — runs everything in order
│
├── behav/                         # Behavioural analyses
│   ├── 00_questionnaires_score.py
│   ├── 01a_behav_decision.py      # Decision behaviour, stats, figures
│   ├── 01b_behav_passive.py       # Passive phase behaviour
│   └── 02_behav_sv_modelling.py   # Hierarchical subjective-value modelling (PyMC)
│
├── eeg/                           # EEG analyses
│   ├── 03_eeg_preprocess.py       # Raw EEG cleaning / preprocessing
│   ├── 04_eeg_erp_prep.py         # ERP preparation and RP extraction
│   ├── 05a_eeg_erp_massunivariate_passive.py   # Mass-univariate ERP — passive phase
│   └── 05b_eeg_erp_massunivariate_decision.py  # Mass-univariate ERP — decision phase
│
├── hddm/                          # Hierarchical drift-diffusion modelling
│   ├── model_specs.py             # MODEL_SPECS dict — all model versions in one place
│   ├── 05_hddm_prep.py            # Prepare data for HDDM (outputs hddm_ready.csv)
│   ├── 06_hddm_fit.py             # Fit all model versions (runs inside Docker)
│   ├── 07_hddm_results.py         # Load chains, MAP estimates, diagnostics, DIC comparison
│   ├── 08_hddm_plot.py            # DDM schematic figure for the paper
│   └── Simulations/               # Model validation
│       ├── ppc.py                 # Posterior predictive checks
│       ├── param_recovery.py      # Parameter recovery (group and individual level)
│       └── model_recovery.py      # Model recovery — can DIC distinguish the main models?
│
└── old/                           # Archived code (not part of the active pipeline)
```

---

## Pipeline overview

### 1 — Behaviour (`behav/`)

| Script | Output |
|---|---|
| `00_questionnaires_score.py` | Questionnaire summary |
| `01a_behav_decision.py` | Decision-phase stats and figures |
| `01b_behav_passive.py` | Passive-phase stats and figures |
| `02_behav_sv_modelling.py` | Subjective-value estimates per participant |

### 2 — EEG (`eeg/`)

| Script | Output |
|---|---|
| `03_eeg_preprocess.py` | Cleaned epochs per participant |
| `04_eeg_erp_prep.py` | ERP data + RP means merged into behavioural CSV |
| `05a_eeg_erp_massunivariate_passive.py` | TFCE mass-univariate results — passive phase |
| `05b_eeg_erp_massunivariate_decision.py` | TFCE mass-univariate results — decision phase |

`05a` and `05b` must be run **after** HDDM results are available (they load trial-level parameter estimates from `hddm/07_hddm_results.py`).

### 3 — HDDM (`hddm/`)

Model versions are defined centrally in `model_specs.py`:

| Version | Model |
|---|---|
| 0 | Null HDDM (a, v, t) |
| 1 | v ~ 1 + sv_pain (intercept) |
| 2 | v ~ 0 + sv_pain (no intercept) |
| 3 | a ~ 1 + sv_pain |
| 9 | v ~ pain + money |
| 10 | a ~ pain + money |
| 11 | t ~ pain + money |
| 12 | v ~ pain × money |
| 17 | v ~ pain + money + rp + interactions |
| 18 | a ~ pain + money + rp + interactions |
| 19 | v + a ~ pain + money (**primary behavioural model**) |
| 20 | v + a ~ pain + money + rp + interactions |

`07_hddm_results.py` accepts `--version N` to process one model, or `--compare` to generate a DIC bar chart across all fitted models.

### 4 — Simulations (`hddm/Simulations/`)

Run after fitting to validate the models:

| Script | Purpose |
|---|---|
| `ppc.py` | Posterior predictive checks for a given version |
| `param_recovery.py` | Can the fitting procedure recover true parameters? |
| `model_recovery.py` | Can DIC distinguish the theoretically relevant models? |

---

## Key results (from manuscript)

- **Behaviour**: N=38; mean RT = 1.06 s (SD = 0.22); acceptance rate = 75.3% (SD = 17.8%)
- **DDM**: pain → lower drift + lower threshold; money → higher drift + higher threshold
- **EEG passive phase**: pain effect 263–1200 ms; money effect 285–351 ms + 377–1200 ms
- **EEG decision phase**: pain significant (TFCE p<.001, peak FT10 t=−79.91); money not significant
- **RP**: trial-level RP amplitude positively related to drift rate; higher thresholds predicted reduced RP amplitude

---

## Environment notes

- HDDM fitting requires the `hcp4715/hddm` Docker image (run commands are in `run_all.sh`)
- EEG scripts use MNE-Python in the local environment
- Scripts raise errors on missing files — no silent fallbacks by design
