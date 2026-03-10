# MAP estimation code
# Veronika Wendler
# 10.03.26
# 10.03.26
# This code calculates group maximum posterior estimates of the ddm parameters in the pain-reward trade-off and their parameter comparison

#libraries as always 
import pandas as pd
import pickle
import kabuki
import scipy.stats as stats
import pickle
import kabuki
import arviz as az
import pandas as pd
from pathlib import Path
import re
import os
# disable _all_ Numba JIT caching & compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"
import numba
numba.config.CACHE_ENABLE = False


PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()


# version 9  == model: 'v ~ 1 + painlevel + moneylevel'
# version 10 == model: 'a ~ 1 + painlevel + moneylevel'
# Version 17 == model: 'v ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z'
# Version 18 == model: 'a ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z'
# Version 19 == model: 'v ~ 1 + pain_z + money_z', 
#                      'a ~ 1 + pain_z + money_z'
# Version 20 == model: 'v ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z'
#                      'a ~ 1 + pain_z + money_z + rp_z + pain_z * rp_z + money_z * rp_z'

def run_mod_9():
    #---------------------------------------------------------------------------------------------------------------
    model_paths = [
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_9_0.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_9_1.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_9_2.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_9_3.pkl",
    ]
    
    models_9 = []
    for path in model_paths:
        with open(path, "rb") as f:
            models_9.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models_9)
    
    # summary stats 
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        'a',
        't', 
        'v_Intercept',
        'v_painlevel',
        'v_moneylevel'
    ])])
    print("DIC:", combinedModels.dic)            # some diagnostics
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes for OV:
    a        = combinedModels.nodes_db.node['a']
    t        = combinedModels.nodes_db.node['t']
    v_inter  = combinedModels.nodes_db.node['v_Intercept']
    v_pain   = combinedModels.nodes_db.node['v_painlevel']
    v_money  = combinedModels.nodes_db.node['v_moneylevel']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    delta  = v_money.trace() / v_pain.trace()

    group_params = {
        "a": a.trace(),
        "t": t.trace(),
        "v_Intercept": v_inter.trace(),
        "v_Pain": v_pain.trace(),
        "v_Money": v_money.trace(),
        "delta": delta,
    }

    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params.items():
        group_results["Parameter"].append(name)
        group_results["MAP"].append(trace.mean())
        group_results["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_9/diagnostics/group_level_MAP_table_m9.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_money/ v_pain - to get the influence ratio
    delta  = v_money.trace() / v_pain.trace()
   
    rows = []
    # (group-level from t)  this obviously depends on which model you are running (4 = a varies by OV, 5 = t varies by OV - code below would need to be adjusted)
    rows.append({
        "Parameter": "a",
        "Group-level": format_estimate(a.trace()),
    })
    # a differences across OVcate
    rows.append({
        "Parameter": "t",
        "Group-level": format_estimate(t.trace()),
    })
    rows.append({
        "Parameter": "v_Intercept",
        "Group-level": format_estimate(v_inter.trace()),
    })
    rows.append({
        "Parameter": "v_Pain",
        "Group-level": format_estimate(v_pain.trace()),
    })
    # b2: differences in v_InattentionW across OVcate
    rows.append({
        "Parameter": "v_Money",
        "Group-level": format_estimate(v_money.trace()),
    })
    rows.append({
        "Parameter": "delta",
        "Group-level": format_estimate(delta),
    })

    df_combined = pd.DataFrame(rows, columns=["Parameter", "Group-level"])
    df_combined.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_9/diagnostics/combined_parameter_comparison_table_m9.csv", index=False)
    print("Combined Parameter Comparison Table:")
    print(df_combined)
    
    #---------------------------------------------------------------------------------------------------------------


def run_mod_10():
    #---------------------------------------------------------------------------------------------------------------
    model_paths = [
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_10_0.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_10_1.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_10_2.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_10_3.pkl",
    ]
    
    models_10 = []
    for path in model_paths:
        with open(path, "rb") as f:
            models_10.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models_10)
    
    # summary stats 
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        'v',
        't', 
        'a_Intercept',
        'a_painlevel',
        'a_moneylevel'
    ])])
    print("DIC:", combinedModels.dic)            # some diagnostics
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes for OV:
    v        = combinedModels.nodes_db.node['v']
    t        = combinedModels.nodes_db.node['t']
    a_inter  = combinedModels.nodes_db.node['a_Intercept']
    a_pain   = combinedModels.nodes_db.node['a_painlevel']
    a_money  = combinedModels.nodes_db.node['a_moneylevel']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    delta  = a_money.trace() / a_pain.trace()

    group_params = {
        "v": v.trace(),
        "t": t.trace(),
        "a_Intercept": a_inter.trace(),
        "a_Pain": a_pain.trace(),
        "a_Money": a_money.trace(),
        "delta": delta,
    }

    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params.items():
        group_results["Parameter"].append(name)
        group_results["MAP"].append(trace.mean())
        group_results["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_10/diagnostics/group_level_MAP_table_m10.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_money/ v_pain - to get the influence ratio
    delta  = a_money.trace() / a_pain.trace()
   
    rows = []
    rows.append({
        "Parameter": "v",
        "Group-level": format_estimate(v.trace()),
    })
    rows.append({
        "Parameter": "t",
        "Group-level": format_estimate(t.trace()),
    })
    rows.append({
        "Parameter": "a_Intercept",
        "Group-level": format_estimate(a_inter.trace()),
    })
    rows.append({
        "Parameter": "a_Pain",
        "Group-level": format_estimate(a_pain.trace()),
    })
    rows.append({
        "Parameter": "a_Money",
        "Group-level": format_estimate(a_money.trace()),
    })
    rows.append({
        "Parameter": "delta",
        "Group-level": format_estimate(delta),
    })
    
    df_combined = pd.DataFrame(rows, columns=["Parameter", "Group-level"])
    df_combined.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_10/diagnostics/combined_parameter_comparison_table_m10.csv", index=False)
    print("Combined Parameter Comparison Table:")
    print(df_combined)
    
    #---------------------------------------------------------------------------------------------------------------

def run_mod_17():
    #---------------------------------------------------------------------------------------------------------------
    model_paths = [
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_17_0.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_17_1.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_17_2.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_17_3.pkl",
    ]
    
    models_17 = []
    for path in model_paths:
        with open(path, "rb") as f:
            models_17.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models_17)
    
    # summary stats 
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        't', 
        'a', 
        'v_Intercept',
        'v_pain_z',
        'v_money_z',
        'v_rp_z',
        'v_pain_z:rp_z',
        'v_money_z:rp_z'
    ])])

    print("DIC:", combinedModels.dic)            # some diagnostics
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes:
    a        = combinedModels.nodes_db.node['a']
    t        = combinedModels.nodes_db.node['t']
    v_inter  = combinedModels.nodes_db.node['v_Intercept']
    v_pain   = combinedModels.nodes_db.node['v_pain_z']
    v_money  = combinedModels.nodes_db.node['v_money_z']
    v_rp     = combinedModels.nodes_db.node['v_rp_z']
    v_pain_rp   = combinedModels.nodes_db.node['v_pain_z:rp_z']
    v_money_rp  = combinedModels.nodes_db.node['v_money_z:rp_z']
    
    # Group-level Table (delta = money/ pain )
    delta  = v_money.trace() / v_pain.trace()
    delta_rp_p  = v_rp.trace() / v_pain.trace()
    delta_rp_m  = v_rp.trace() / v_money.trace()

    group_params = {
        "a": a.trace(),
        "t": t.trace(),
        "v_Intercept": v_inter.trace(),
        "v_Pain_z": v_pain.trace(),
        "v_Money_z": v_money.trace(),
        "v_Rp_z": v_rp.trace(),
        "v_Pain_z:rp_z": v_pain_rp.trace(),
        "v_Money_z:rp_z": v_money_rp.trace(),
        "delta": delta,
        "delta_rp_p": delta_rp_p,
        "delta_rp_m": delta_rp_m,
    }

    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params.items():
        group_results["Parameter"].append(name)
        group_results["MAP"].append(trace.mean())
        group_results["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_17/diagnostics/group_level_MAP_table_m17.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_money/ v_pain - to get the influence ratio
    delta  = v_money.trace() / v_pain.trace()
   
    rows = []
    rows.append({
        "Parameter": "a",
        "Group-level": format_estimate(a.trace()),
    })
    rows.append({
        "Parameter": "t",
        "Group-level": format_estimate(t.trace()),
    })
    rows.append({
        "Parameter": "v_Intercept",
        "Group-level": format_estimate(v_inter.trace()),
    })
    rows.append({
        "Parameter": "v_Pain_z",
        "Group-level": format_estimate(v_pain.trace()),
    })
    rows.append({
        "Parameter": "v_Money_z",
        "Group-level": format_estimate(v_money.trace()),
    })
    rows.append({
        "Parameter": "v_Rp_z",
        "Group-level": format_estimate(v_rp.trace()),
    })
    rows.append({
        "Parameter": "v_Pain_z:rp_z",
        "Group-level": format_estimate(v_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "v_Pain_z:rp_z",
        "Group-level": format_estimate(v_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "v_Money_z:rp_z",
        "Group-level": format_estimate(v_money_rp.trace()),
    })
    rows.append({
        "Parameter": "delta",
        "Group-level": format_estimate(delta),
    })
    rows.append({
        "Parameter": "delta_rp_p",
        "Group-level": format_estimate(delta_rp_p),
    })
    rows.append({
        "Parameter": "delta_rp_m",
        "Group-level": format_estimate(delta_rp_m),
    })
 

    
    df_combined = pd.DataFrame(rows, columns=["Parameter", "Group-level"])
    df_combined.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_17/diagnostics/combined_parameter_comparison_table_m17.csv", index=False)
    print("Combined Parameter Comparison Table:")
    print(df_combined)
    
    #---------------------------------------------------------------------------------------------------------------


def run_mod_18():
    #---------------------------------------------------------------------------------------------------------------
    model_paths = [
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_18_0.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_18_1.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_18_2.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_18_3.pkl",
    ]
    
    models_18 = []
    for path in model_paths:
        with open(path, "rb") as f:
            models_18.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models_18)
    
    # summary stats 
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        't', 
        'v', 
        'a_Intercept',
        'a_pain_z',
        'a_money_z',
        'a_rp_z',
        'a_pain_z:rp_z',
        'a_money_z:rp_z'
    ])])

    print("DIC:", combinedModels.dic)            # some diagnostics
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes:
    v        = combinedModels.nodes_db.node['v']
    t        = combinedModels.nodes_db.node['t']
    a_inter  = combinedModels.nodes_db.node['a_Intercept']
    a_pain   = combinedModels.nodes_db.node['a_pain_z']
    a_money  = combinedModels.nodes_db.node['a_money_z']
    a_rp     = combinedModels.nodes_db.node['a_rp_z']
    a_pain_rp   = combinedModels.nodes_db.node['a_pain_z:rp_z']
    a_money_rp  = combinedModels.nodes_db.node['a_money_z:rp_z']
    
    # Group-level Table (delta = money/ pain )
    delta  = a_money.trace() / a_pain.trace()
    delta_rp_p  = a_rp.trace() / a_pain.trace()
    delta_rp_m  = a_rp.trace() / a_money.trace()

    group_params = {
        "v": v.trace(),
        "t": t.trace(),
        "a_Intercept": a_inter.trace(),
        "a_Pain_z": a_pain.trace(),
        "a_Money_z": a_money.trace(),
        "a_Rp_z": a_rp.trace(),
        "a_Pain_z:rp_z": a_pain_rp.trace(),
        "a_Money_z:rp_z": a_money_rp.trace(),
        "delta": delta,
        "delta_rp_p": delta_rp_p,
        "delta_rp_m": delta_rp_m,
    }

    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params.items():
        group_results["Parameter"].append(name)
        group_results["MAP"].append(trace.mean())
        group_results["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_18/diagnostics/group_level_MAP_table_m18.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_money/ v_pain - to get the influence ratio
    delta  = a_money.trace() / a_pain.trace()
   
    rows = []
    rows.append({
        "Parameter": "v",
        "Group-level": format_estimate(v.trace()),
    })
    rows.append({
        "Parameter": "t",
        "Group-level": format_estimate(t.trace()),
    })
    rows.append({
        "Parameter": "a_Intercept",
        "Group-level": format_estimate(a_inter.trace()),
    })
    rows.append({
        "Parameter": "a_Pain_z",
        "Group-level": format_estimate(a_pain.trace()),
    })
    rows.append({
        "Parameter": "a_Money_z",
        "Group-level": format_estimate(a_money.trace()),
    })
    rows.append({
        "Parameter": "a_Rp_z",
        "Group-level": format_estimate(a_rp.trace()),
    })
    rows.append({
        "Parameter": "a_Pain_z:rp_z",
        "Group-level": format_estimate(a_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "a_Pain_z:rp_z",
        "Group-level": format_estimate(a_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "a_Money_z:rp_z",
        "Group-level": format_estimate(a_money_rp.trace()),
    })
    rows.append({
        "Parameter": "delta",
        "Group-level": format_estimate(delta),
    })
    rows.append({
        "Parameter": "delta_rp_p",
        "Group-level": format_estimate(delta_rp_p),
    })
    rows.append({
        "Parameter": "delta_rp_m",
        "Group-level": format_estimate(delta_rp_m),
    })
 

    df_combined = pd.DataFrame(rows, columns=["Parameter", "Group-level"])
    df_combined.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_18/diagnostics/combined_parameter_comparison_table_m18.csv", index=False)
    print("Combined Parameter Comparison Table:")
    print(df_combined)
    
    #------------------------------------------------------------------------------------
    

def run_mod_19():
    #---------------------------------------------------------------------------------------------------------------
    model_paths = [
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_19_0.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_19_1.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_19_2.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_19_3.pkl",
    ]
    
    models_19 = []
    for path in model_paths:
        with open(path, "rb") as f:
            models_19.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models_19)
    
    # summary stats 
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        't', 
        'v_Intercept',
        'v_painlevel',
        'v_moneylevel',
        'a_Intercept',
        'a_painlevel',
        'a_moneylevel',
    ])])
    print("DIC:", combinedModels.dic)          
    # some diagnostics
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes for OV:
    t        = combinedModels.nodes_db.node['t']
    v_inter  = combinedModels.nodes_db.node['v_Intercept']
    v_pain   = combinedModels.nodes_db.node['v_painlevel']
    v_money  = combinedModels.nodes_db.node['v_moneylevel']
    a_inter  = combinedModels.nodes_db.node['a_Intercept']
    a_pain   = combinedModels.nodes_db.node['a_painlevel']
    a_money  = combinedModels.nodes_db.node['a_moneylevel']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    delta_v  = v_money.trace() / v_pain.trace()
    delta_a  = a_money.trace() / a_pain.trace()

    group_params = {
        "t": t.trace(),
        "v_Intercept": v_inter.trace(),
        "v_Pain": v_pain.trace(),
        "v_Money": v_money.trace(),
        "a_Intercept": a_inter.trace(),
        "a_Pain": a_pain.trace(),
        "a_Money": a_money.trace(),
        "delta_v": delta_v,
        "delta_a": delta_a,
    }

    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params.items():
        group_results["Parameter"].append(name)
        group_results["MAP"].append(trace.mean())
        group_results["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_19/diagnostics/group_level_MAP_table_m19.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_money/ v_pain - to get the influence ratio
    delta_v  = v_money.trace() / v_pain.trace()
    delta_a  = a_money.trace() / a_pain.trace()
   
    rows = []
    rows.append({
        "Parameter": "t",
        "Group-level": format_estimate(t.trace()),
    })
    rows.append({
        "Parameter": "v_Intercept",
        "Group-level": format_estimate(v_inter.trace()),
    })
    rows.append({
        "Parameter": "v_Pain",
        "Group-level": format_estimate(v_pain.trace()),
    })
    rows.append({
        "Parameter": "v_Money",
        "Group-level": format_estimate(v_money.trace()),
    })
    rows.append({
        "Parameter": "a_Intercept",
        "Group-level": format_estimate(a_inter.trace()),
    })
    rows.append({
        "Parameter": "a_Pain",
        "Group-level": format_estimate(a_pain.trace()),
    })
    rows.append({
        "Parameter": "a_Money",
        "Group-level": format_estimate(a_money.trace()),
    })
    rows.append({
        "Parameter": "delta_v",
        "Group-level": format_estimate(delta_v),
    })
    rows.append({
        "Parameter": "delta_a",
        "Group-level": format_estimate(delta_a),
    })

    df_combined = pd.DataFrame(rows, columns=["Parameter", "Group-level"])
    df_combined.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_19/diagnostics/combined_parameter_comparison_table_m19.csv", index=False)
    print("Combined Parameter Comparison Table:")
    print(df_combined)
    
    #---------------------------------------------------------------------------------------------------------------



def run_mod_20():
    #---------------------------------------------------------------------------------------------------------------
    model_paths = [
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_20_0.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_20_1.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_20_2.pkl",
        "/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/models/painreward_behavioural_data_mod_20_3.pkl",
    ]
    
    models_20 = []
    for path in model_paths:
        with open(path, "rb") as f:
            models_20.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models_20)
    
    # summary stats 
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        't', 
        'v_Intercept',
        'v_pain_z',
        'v_money_z',
        'v_rp_z',
        'v_pain_z:rp_z',
        'v_money_z:rp_z',
        'a_Intercept',
        'a_pain_z',
        'a_money_z',
        'a_rp_z',
        'a_pain_z:rp_z',
        'a_money_z:rp_z',
    ])])

    print("DIC:", combinedModels.dic)            # some diagnostics
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes:
    t        = combinedModels.nodes_db.node['t']
    v_inter  = combinedModels.nodes_db.node['v_Intercept']
    v_pain   = combinedModels.nodes_db.node['v_pain_z']
    v_money  = combinedModels.nodes_db.node['v_money_z']
    v_rp     = combinedModels.nodes_db.node['v_rp_z']
    v_pain_rp   = combinedModels.nodes_db.node['v_pain_z:rp_z']
    v_money_rp  = combinedModels.nodes_db.node['v_money_z:rp_z']
    a_inter  = combinedModels.nodes_db.node['a_Intercept']
    a_pain   = combinedModels.nodes_db.node['a_pain_z']
    a_money  = combinedModels.nodes_db.node['a_money_z']
    a_rp     = combinedModels.nodes_db.node['a_rp_z']
    a_pain_rp   = combinedModels.nodes_db.node['a_pain_z:rp_z']
    a_money_rp  = combinedModels.nodes_db.node['a_money_z:rp_z']
    
    group_params = {
        "t": t.trace(),
        "v_Intercept": v_inter.trace(),
        "v_Pain_z": v_pain.trace(),
        "v_Money_z": v_money.trace(),
        "v_Rp_z": v_rp.trace(),
        "v_Pain_z:rp_z": v_pain_rp.trace(),
        "v_Money_z:rp_z": v_money_rp.trace(),
        "a_Intercept": a_inter.trace(),
        "a_Pain_z": a_pain.trace(),
        "a_Money_z": a_money.trace(),
        "a_Rp_z": a_rp.trace(),
        "a_Pain_z:rp_z": a_pain_rp.trace(),
        "a_Money_z:rp_z": a_money_rp.trace(),

    }

    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params.items():
        group_results["Parameter"].append(name)
        group_results["MAP"].append(trace.mean())
        group_results["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_20/diagnostics/group_level_MAP_table_m20.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_money/ v_pain - to get the influence ratio
    delta  = v_money.trace() / v_pain.trace()
   
    rows = []
    rows.append({
        "Parameter": "t",
        "Group-level": format_estimate(t.trace()),
    })
    rows.append({
        "Parameter": "v_Intercept",
        "Group-level": format_estimate(v_inter.trace()),
    })
    rows.append({
        "Parameter": "v_Pain_z",
        "Group-level": format_estimate(v_pain.trace()),
    })
    rows.append({
        "Parameter": "v_Money_z",
        "Group-level": format_estimate(v_money.trace()),
    })
    rows.append({
        "Parameter": "v_Rp_z",
        "Group-level": format_estimate(v_rp.trace()),
    })
    rows.append({
        "Parameter": "v_Pain_z:rp_z",
        "Group-level": format_estimate(v_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "v_Pain_z:rp_z",
        "Group-level": format_estimate(v_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "v_Money_z:rp_z",
        "Group-level": format_estimate(v_money_rp.trace()),
    })
    rows.append({
        "Parameter": "a_Intercept",
        "Group-level": format_estimate(a_inter.trace()),
    })
    rows.append({
        "Parameter": "a_Pain_z",
        "Group-level": format_estimate(a_pain.trace()),
    })
    rows.append({
        "Parameter": "a_Money_z",
        "Group-level": format_estimate(a_money.trace()),
    })
    rows.append({
        "Parameter": "a_Rp_z",
        "Group-level": format_estimate(a_rp.trace()),
    })
    rows.append({
        "Parameter": "a_Pain_z:rp_z",
        "Group-level": format_estimate(a_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "a_Pain_z:rp_z",
        "Group-level": format_estimate(a_pain_rp.trace()),
    })
    rows.append({
        "Parameter": "a_Money_z:rp_z",
        "Group-level": format_estimate(a_money_rp.trace()),
    })

    df_combined = pd.DataFrame(rows, columns=["Parameter", "Group-level"])
    df_combined.to_csv("/rds/projects/z/zhanglp-vwendler-core/PainReward_ULaval/derivatives/hddm/figures/painreward_behavioural_data_mod_20/diagnostics/combined_parameter_comparison_table_m20.csv", index=False)
    print("Combined Parameter Comparison Table:")
    print(df_combined)
    
    #---------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    
    version = 19
    if version == 9:
        run_mod_9()
    elif version == 10: 
        run_mod_10()
    elif version == 17: 
        run_mod_17()
    elif version == 18:
        run_mod_18()
    elif version == 19:
        run_mod_19()
    elif version == 20:
        run_mod_20()
