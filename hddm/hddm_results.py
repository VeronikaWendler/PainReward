# hddm_results.py — load fitted HDDM chains, compute MAP estimates, print stats
#
# Merges the logic from DDM_EEG_load.py and MAP_estimates.py.
#
# Run: python hddm_results.py --version 9 [--model-dir PATH] [--fig-dir PATH]
# Env vars: PROJECT_DIR, MODEL_DIR

import os
import warnings
import argparse
from pathlib import Path

import pandas as pd
import pickle
import scipy.stats as stats
import kabuki

warnings.simplefilter(action="ignore", category=FutureWarning)

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import numba
numba.config.CACHE_ENABLE = False

PROJECT_DIR    = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent.parent))).resolve()
BASE_MODEL_DIR = Path(os.getenv("MODEL_DIR",
                      str(PROJECT_DIR / "Hddm_Docker_August_24" / "models_dir"))).resolve()
FIG_DIR_ROOT   = Path(os.getenv("FIG_DIR",
                      str(PROJECT_DIR / "Hddm_Docker_August_24" / "figures_dir"))).resolve()

MODEL_BASE_NAME = "painreward_behavioural_data_"
N_CHAINS = 4

# Parameters to extract per model version
VERSION_PARAMS = {
    0:  ["a", "t", "v"],
    1:  ["a", "t", "v_Intercept", "v_sv_pain_para"],
    2:  ["a", "t", "v_sv_pain_para"],
    3:  ["a", "t", "a_Intercept", "a_sv_pain_para"],
    9:  ["a", "t", "v_Intercept", "v_painlevel", "v_moneylevel"],
    10: ["a", "t", "a_Intercept", "a_painlevel", "a_moneylevel"],
    11: ["a", "v", "t_Intercept", "t_painlevel", "t_moneylevel"],
    12: ["a", "t", "v_Intercept", "v_painlevel", "v_moneylevel", "v_painlevel:moneylevel"],
    17: ["a", "t", "v_Intercept", "v_pain_z", "v_money_z", "v_rp_z",
         "v_pain_z:rp_z", "v_money_z:rp_z"],
    18: ["a", "t", "a_Intercept", "a_pain_z", "a_money_z", "a_rp_z",
         "a_pain_z:rp_z", "a_money_z:rp_z"],
    19: ["a", "t", "v_Intercept", "v_pain_z", "v_money_z",
         "a_Intercept", "a_pain_z", "a_money_z"],
    20: ["a", "t",
         "v_Intercept", "v_pain_z", "v_money_z", "v_rp_z", "v_pain_z:rp_z", "v_money_z:rp_z",
         "a_Intercept", "a_pain_z", "a_money_z", "a_rp_z", "a_pain_z:rp_z", "a_money_z:rp_z"],
}


def load_chains(version: int, model_dir: Path, n_chains: int = N_CHAINS):
    """Load all pkl chains for a model version and concatenate them."""
    model_name = MODEL_BASE_NAME + f"mod_{version}"
    models = []
    for i in range(n_chains):
        path = model_dir / f"{model_name}_{i}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            models.append(pickle.load(fh))
    return kabuki.utils.concat_models(models)


def map_estimates(combined, param_names: list) -> pd.DataFrame:
    """Compute MAP (posterior mean) and 95% HDI for group-level parameters."""
    rows = []
    stats_df = combined.gen_stats()
    available = set(stats_df.index)

    for name in param_names:
        if name not in available:
            raise KeyError(
                f"Parameter '{name}' not found in model stats. "
                f"Available: {sorted(available)}"
            )
        trace = combined.nodes_db.node[name].trace()
        rows.append({
            "parameter": name,
            "MAP":       trace.mean(),
            "HDI_2.5":   stats.mstats.mquantiles(trace, [0.025])[0],
            "HDI_97.5":  stats.mstats.mquantiles(trace, [0.975])[0],
        })
    return pd.DataFrame(rows)


def run_version(version: int, model_dir: Path, fig_dir: Path) -> None:
    print(f"\n=== Version {version} ===")
    combined = load_chains(version, model_dir)

    print("DIC:", combined.dic)
    try:
        print("BPIC:", combined.mc.BPIC)
    except AttributeError:
        pass

    params = VERSION_PARAMS.get(version, [])
    df_map = map_estimates(combined, params)
    print(df_map.to_string(index=False))

    out_dir = fig_dir / (MODEL_BASE_NAME + f"mod_{version}") / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"group_level_MAP_table_m{version}.csv"
    df_map.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load HDDM chains and compute MAP estimates")
    parser.add_argument("--version",   type=int, required=True,
                        help="Model version number (e.g. 9, 10, 19)")
    parser.add_argument("--model-dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--fig-dir",   type=Path, default=FIG_DIR_ROOT)
    args = parser.parse_args()

    run_version(args.version, args.model_dir, args.fig_dir)
