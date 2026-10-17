#!/usr/bin/env python3
"""
05i_eeg_compare_total_vs_evoked_removed.py

Paired group comparison of TOTAL-POWER decoding vs EVOKED-REMOVED /
INDUCED-ENRICHED decoding.

This is essential because:
    significant total + non-significant residual
does NOT by itself prove a difference.

We directly test participant-wise score differences:
- ordinal: Fisher-z(total r) - Fisher-z(evoked-removed r)
- high/low: total AUC - evoked-removed AUC

Two-sided 2D frequency x time cluster permutation, separately pain and money,
over the clean 0..1.10 s window. Holm across pain/money within each family.

Positive cluster:
    total-power decoding > evoked-removed decoding
Negative cluster:
    evoked-removed decoding > total-power decoding
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats

BASE=Path(os.getenv("basepath",Path(__file__).parent.parent.parent))
TOTAL=BASE/"derivatives"/"statistics"/"tfr_mvpa_passive_v2"/"group_score_arrays.npz"
RESID=BASE/"derivatives"/"statistics"/"tfr_mvpa_passive_evoked_removed"/"group_score_arrays.npz"
OUT=BASE/"derivatives"/"statistics"/"tfr_total_vs_evoked_removed"
TMIN,TMAX=0.,1.10
ALPHA=.05
SEED=23

def cli():
    p=argparse.ArgumentParser();p.add_argument("--n-permutations",type=int,default=5000)
    p.add_argument("--n-jobs",type=int,default=max(1,int(os.getenv("SLURM_CPUS_PER_TASK","1"))));return p.parse_args()
def holm(p):
    p=np.asarray(p,float);m=len(p);o=np.argsort(p);r=p[o];a=np.maximum.accumulate((m-np.arange(m))*r);a=np.clip(a,0,1);z=np.empty(m);z[o]=a;return z
def fisher(x):return np.arctanh(np.clip(x,-.999999,.999999))
def test(x,freqs,times,nperm,njobs):
    tm=(times>=TMIN-1e-12)&(times<=TMAX+1e-12);tt=times[tm];x=x[:,:,tm];ns,nf,nt=x.shape
    thr=float(stats.t.ppf(1-ALPHA/2,df=ns-1))
    T,C,P,_=permutation_cluster_1samp_test(x.reshape(ns,nf*nt),threshold=thr,n_permutations=nperm,tail=0,
        adjacency=combine_adjacency(nf,nt),seed=SEED,n_jobs=njobs,out_type="indices",verbose=False)
    return T.reshape(nf,nt),[np.asarray(c[0],int) for c in C],np.asarray(P,float),tt

def main():
    a=cli();OUT.mkdir(parents=True,exist_ok=True)
    zt=np.load(TOTAL,allow_pickle=False);zr=np.load(RESID,allow_pickle=False)
    for key in ["subjects","times","freqs","attribute_names"]:
        A=zt[key];B=zr[key]
        if A.dtype.kind in "f" and not np.allclose(A,B):raise RuntimeError(f"{key} mismatch")
        if A.dtype.kind not in "f" and not np.array_equal(A,B):raise RuntimeError(f"{key} mismatch")
    S=zt["subjects"].astype(str);times=zt["times"].astype(float);freqs=zt["freqs"].astype(float);attrs=zt["attribute_names"].astype(str).tolist()
    rows=[];om=[]
    for fam,kt,kr in [("ordinal","regression_freq_r","regression_freq_r"),("highlow","highlow_freq_auc","highlow_freq_auc")]:
        total=zt[kt];resid=zr[kr]
        diff=fisher(total)-fisher(resid) if fam=="ordinal" else total-resid
        per_attr=[]
        for ai,attr in enumerate(attrs):
            T,C,P,tt=test(diff[:,ai],freqs,times,a.n_permutations,a.n_jobs)
            pmin=float(P.min()) if len(P) else 1.;per_attr.append((attr,pmin,T,C,P,tt))
        hp=holm([x[1] for x in per_attr])
        for ai,(attr,pmin,T,C,P,tt) in enumerate(per_attr):
            om.append(dict(family=fam,attribute=attr,omnibus_p_within_surface=pmin,omnibus_p_holm_across_pain_money=float(hp[ai]),significant_holm_0p05=bool(hp[ai]<.05)))
            for ci,(idx,pv) in enumerate(zip(C,P)):
                fi,ti=np.unravel_index(idx,(len(freqs),len(tt)));vals=T.reshape(-1)[idx];pk=int(np.argmax(np.abs(vals)));pfi,pti=np.unravel_index(int(idx[pk]),(len(freqs),len(tt)))
                rows.append(dict(family=fam,attribute=attr,cluster_id=ci,sign="total_gt_residual" if vals.mean()>0 else "residual_gt_total",
                    cluster_p_within_surface=float(pv),omnibus_p_holm_across_pain_money=float(hp[ai]),
                    cluster_significant_final=bool(pv<.05 and hp[ai]<.05),
                    freq_min_hz=float(freqs[fi].min()),freq_max_hz=float(freqs[fi].max()),
                    time_start_s=float(tt[ti].min()),time_end_s=float(tt[ti].max()),
                    peak_freq_hz=float(freqs[pfi]),peak_time_s=float(tt[pti]),peak_t=float(vals[pk]),cluster_mass_signed=float(vals.sum())))
        print("\n",fam.upper(),flush=True)
        print(pd.DataFrame([x for x in om if x["family"]==fam]).to_string(index=False),flush=True)
    pd.DataFrame(om).to_csv(OUT/"domain_omnibus_holm.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"paired_difference_clusters.csv",index=False)
    np.savez_compressed(OUT/"paired_difference_arrays.npz",subjects=S,times=times,freqs=freqs,attribute_names=np.array(attrs),
                        ordinal_fisherz_total_minus_evokedremoved=fisher(zt["regression_freq_r"])-fisher(zr["regression_freq_r"]),
                        highlow_auc_total_minus_evokedremoved=zt["highlow_freq_auc"]-zr["highlow_freq_auc"])
    with open(OUT/"params.json","w") as f:json.dump({"comparison":"total power minus cross-fitted evoked-removed/induced-enriched power",
        "inference_s":[TMIN,TMAX],"test":"two-sided paired group 2D frequency x time cluster permutation",
        "multiplicity":"Holm across pain/money within ordinal and highlow families","n_permutations":a.n_permutations},f,indent=2)
    sig=pd.DataFrame(rows);sig=sig[sig.cluster_significant_final==True] if len(sig) else sig
    print("\nSignificant paired-difference clusters:\n",sig.to_string(index=False) if len(sig) else "None",flush=True)
    print("="*78);print("TOTAL VS EVOKED-REMOVED COMPARISON COMPLETE");print(OUT);print("="*78)
if __name__=="__main__":main()
