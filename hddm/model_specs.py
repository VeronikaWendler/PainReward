"""Shared model specifications for PainReward HDDM analyses.

Imported by hddm_fit.py, ppc.py, param_recovery.py, model_recovery.py.
"""
import re

def _lf(x):
    return x


MODEL_SPECS = {
    0:  {"class": "HDDM",         "regs": None},
    1:  {"class": "HDDMRegressor", "regs": [{"model": "v ~ 1 + sv_pain_para",   "link_func": _lf}]},
    2:  {"class": "HDDMRegressor", "regs": [{"model": "v ~ 0 + sv_pain_para",   "link_func": _lf}]},
    3:  {"class": "HDDMRegressor", "regs": [{"model": "a ~ 1 + sv_pain_para",   "link_func": _lf}]},
    9:  {"class": "HDDMRegressor", "regs": [{"model": "v ~ 1 + painlevel + moneylevel", "link_func": _lf}]},
    10: {"class": "HDDMRegressor", "regs": [{"model": "a ~ 1 + painlevel + moneylevel", "link_func": _lf}]},
    11: {"class": "HDDMRegressor", "regs": [{"model": "t ~ 1 + painlevel + moneylevel", "link_func": _lf}]},
    12: {"class": "HDDMRegressor", "regs": [{"model": "v ~ 1 + painlevel + moneylevel + painlevel * moneylevel", "link_func": _lf}]},
    17: {"class": "HDDMRegressor", "regs": [{"model": "v ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z", "link_func": _lf}]},
    18: {"class": "HDDMRegressor", "regs": [{"model": "a ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z", "link_func": _lf}]},
    19: {"class": "HDDMRegressor", "regs": [
        {"model": "v ~ 1 + pain_z + money_z", "link_func": _lf},
        {"model": "a ~ 1 + pain_z + money_z", "link_func": _lf},
    ]},
    20: {"class": "HDDMRegressor", "regs": [
        {"model": "v ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z", "link_func": _lf},
        {"model": "a ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z", "link_func": _lf},
    ]},
}

REQUIRED_COLS = {
    0:  ["rt", "response", "painlevel", "moneylevel"],
    1:  ["rt", "response", "painlevel", "moneylevel", "sv_pain_para"],
    2:  ["rt", "response", "painlevel", "moneylevel", "sv_pain_para"],
    3:  ["rt", "response", "painlevel", "moneylevel", "sv_pain_para"],
    9:  ["rt", "response", "painlevel", "moneylevel"],
    10: ["rt", "response", "painlevel", "moneylevel"],
    11: ["rt", "response", "painlevel", "moneylevel"],
    12: ["rt", "response", "painlevel", "moneylevel"],
    17: ["rt", "response", "pain_z", "money_z", "rp_z"],
    18: ["rt", "response", "pain_z", "money_z", "rp_z"],
    19: ["rt", "response", "pain_z", "money_z"],
    20: ["rt", "response", "pain_z", "money_z", "rp_z"],
}

MODEL_RUN_ORDER = [0, 1, 2, 3, 9, 10, 11, 12, 17, 18, 19, 20]
MODEL_BASE_NAME = "painreward_behavioural_data_"


def _parse_rhs_terms(dv: str, rhs: str) -> list:
    """Return list of (param_name, predictor_cols, is_interaction) tuples.

    param_name     — HDDM node name, e.g. "v_painlevel" or "v_painlevel:moneylevel"
    predictor_cols — list of column names multiplied together ([] for intercept)
    is_interaction — True when term is a * product
    """
    seen = set()
    out = []
    for raw in rhs.split("+"):
        term = raw.strip()
        if term == "0":
            continue
        if term == "1":
            name = f"{dv}_Intercept"
            cols = []
        elif "*" in term:
            parts = [p.strip() for p in term.split("*")]
            name = f"{dv}_{':'.join(parts)}"
            cols = parts
        else:
            name = f"{dv}_{term}"
            cols = [term]
        if name not in seen:
            out.append((name, cols))
            seen.add(name)
    return out


def get_formula_terms(version: int) -> dict:
    """Return {dv: [(param_name, predictor_cols), ...]} for a model version.

    For plain HDDM (version 0) returns an empty dict — parameters are a/v/t/z.
    """
    spec = MODEL_SPECS[version]
    if spec["class"] == "HDDM":
        return {}
    result = {}
    for reg in spec["regs"]:
        lhs, rhs = reg["model"].split("~")
        dv = lhs.strip()
        result[dv] = _parse_rhs_terms(dv, rhs)
    return result


def get_param_list(version: int) -> list[str]:
    """Return the group-level parameter names for a given model version."""
    spec = MODEL_SPECS[version]
    if spec["class"] == "HDDM":
        return ["a", "v", "t", "z"]

    formula_terms = get_formula_terms(version)
    regressed_dvs = set(formula_terms.keys())
    params = []
    for dv, terms in formula_terms.items():
        params.extend(name for name, _ in terms)
    for p in ("a", "v", "t"):
        if p not in regressed_dvs:
            params.append(p)
    return params


def build_model(data, version: int):
    """Construct (but do not sample) an HDDM model for the given version."""
    import hddm as _hddm
    spec = MODEL_SPECS[version]
    if spec["class"] == "HDDM":
        return _hddm.models.HDDM(data, p_outlier=0.05, include=["a", "t", "v", "z"])
    regs = [{"model": r["model"], "link_func": r["link_func"]} for r in spec["regs"]]
    return _hddm.models.HDDMRegressor(
        data, regs,
        p_outlier=0.05,
        include=["a", "t", "v"],
        group_only_regressors=False,
        keep_regressor_trace=True,
    )
