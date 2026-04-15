import sys, types

# Stub out all hddm-related imports so the module loads without hddm installed
for mod in ["hddm", "hddm.models", "kabuki", "arviz", "cloudpickle", "dill",
            "patsy", "statsmodels", "statsmodels.formula", "statsmodels.formula.api",
            "statsmodels.distributions", "statsmodels.distributions.empirical_distribution",
            "hddm.simulators", "hddm.simulators.hddm_dataset_generators", "numba"]:
    sys.modules.setdefault(mod, types.ModuleType(mod))

# Provide the minimal attributes the module references at import time
sys.modules["hddm"].models = types.SimpleNamespace(
    HDDM=object, HDDMRegressor=object
)
sys.modules["numba"].config = types.SimpleNamespace(CACHE_ENABLE=True)

from hddm.hddm_fit import MODEL_SPECS, REQUIRED_COLS


def test_model_specs_keys():
    """Only the defined model versions exist — no 4-8, no 13-16."""
    assert set(MODEL_SPECS.keys()) == {0, 1, 2, 3, 9, 10, 11, 12, 17, 18, 19, 20}


def test_model_specs_class_names():
    assert MODEL_SPECS[0]["class"] == "HDDM"
    for v in [1, 2, 3, 9, 10, 11, 12, 17, 18, 19, 20]:
        assert MODEL_SPECS[v]["class"] == "HDDMRegressor", f"version {v} should use HDDMRegressor"


def test_model_0_has_no_regs():
    assert MODEL_SPECS[0]["regs"] is None


def test_model_19_has_two_regs():
    """Version 19 regresses both v and a on pain_z + money_z."""
    regs = MODEL_SPECS[19]["regs"]
    assert len(regs) == 2
    formulas = [r["model"] for r in regs]
    assert any("v ~" in f for f in formulas)
    assert any("a ~" in f for f in formulas)


def test_required_cols_no_rp_z_for_basic_models():
    """Models 0, 9, 10 must NOT require rp_z."""
    for v in [0, 9, 10, 11, 12]:
        assert "rp_z" not in REQUIRED_COLS[v], f"version {v} should not need rp_z"


def test_required_cols_rp_z_for_rp_models():
    for v in [17, 18, 20]:
        assert "rp_z" in REQUIRED_COLS[v], f"version {v} should need rp_z"


def test_required_cols_sv_pain_para_only_for_sv_models():
    for v in [1, 2, 3]:
        assert "sv_pain_para" in REQUIRED_COLS[v]
    for v in [0, 9, 10, 17, 19]:
        assert "sv_pain_para" not in REQUIRED_COLS[v], \
            f"version {v} should not require sv_pain_para"


def test_required_cols_version_19_no_rp():
    """Version 19 is pain_z + money_z only — no rp_z."""
    assert "rp_z" not in REQUIRED_COLS[19]
