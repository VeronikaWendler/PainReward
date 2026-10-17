#!/usr/bin/env python3
"""
05h_eeg_erp_voltage_decode_passive.py

Straightforward passive cue-locked VOLTAGE decoding to quantify phase-locked
ERP-like magnitude information.

Data:
- cleaned passive raw
- exact good-trial identity/sample/badtrial from ERP metadata
- passive behaviour labels/blocks
- EEG only
- epoch -0.8..+1.5 s
- resample 256 Hz
- voltage baseline [-0.7,-0.5] s
- average into the same 20-ms bins as the TFR analyses

PRIMARY: level 1..5, StandardScaler -> Ridge(alpha=1), leave-one-block-out,
pooled OOF Pearson r.

SECONDARY: low(1,2) vs high(4,5), level 3 excluded, RidgeClassifier,
pooled OOF ROC-AUC.

Group temporal cluster inference is one-sided positive over 0..1.10 s,
separately pain/money, with Holm correction across the two domains within
each family.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne, numpy as np, pandas as pd
from scipy import stats
from mne.stats import permutation_cluster_1samp_test
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_PATH=Path(os.getenv("basepath",Path(__file__).parent.parent.parent))
SOURCE_BIDS=Path(os.getenv("SOURCE_BIDS",BASE_PATH))
PARTICIPANTS=BASE_PATH/"participants.tsv"
OUT_ROOT=BASE_PATH/"derivatives"/"statistics"/"erp_voltage_mvpa_passive"
SCORE_DIR=OUT_ROOT/"participant_scores"; FIG_DIR=OUT_ROOT/"figures"
ATTRS={"pain":"p","money":"m"}
RIDGE_ALPHA=1.; RANDOM_STATE=23; ALPHA=.05
EPOCH_TMIN,EPOCH_TMAX=-.8,1.5
RESAMPLE=256.; BASELINE=(-.7,-.5); BIN=.020
INFER_TMIN,INFER_TMAX=0.,1.10
DROP=["HEOGL","HEOGR","VEOGL","STI 014","Status"]

def cli():
    p=argparse.ArgumentParser(); p.add_argument("--n-permutations",type=int,default=5000)
    p.add_argument("--overwrite-scores",action="store_true"); return p.parse_args()

def subs():
    d=pd.read_csv(PARTICIPANTS,sep="\t"); return sorted(d.participant_id.dropna().astype(str).unique())
def rawp(s): return SOURCE_BIDS/"derivatives"/s/"eeg"/f"{s}_passive_cleaned-raw.fif"
def erpp(s): return SOURCE_BIDS/"derivatives"/s/"eeg"/"erps_passive"/f"{s}_passive_cues_singletrials-epo.fif"
def behp(s): return SOURCE_BIDS/s/"eeg"/f"{s}_task-passive_beh.tsv"
def scorep(s): return SCORE_DIR/f"{s}_erp_voltage_mvpa.npz"

def level15(x):
    v=pd.to_numeric(x,errors="coerce")
    if v.max(skipna=True)>5: v=v/20.
    return v.astype(float)

def metadata(s):
    e=mne.read_epochs(str(erpp(s)),preload=False,verbose="ERROR")
    if e.metadata is None: raise RuntimeError(f"{s}: no ERP metadata")
    md=e.metadata.reset_index(drop=True).copy()
    for c in ["trialsnum","sample","badtrial"]: md[c]=pd.to_numeric(md[c],errors="raise")
    md["trialsnum"]=md.trialsnum.astype(int); md["sample"]=md["sample"].astype(int); md["badtrial"]=md.badtrial.astype(int)
    b=pd.read_csv(behp(s),sep="\t"); b=b[~b["fixcross.started"].isna()].copy().reset_index(drop=True)
    b["trialsnum"]=np.arange(1,len(b)+1); b["condition"]=b.condition.astype(str).str.lower().str.strip(); b["level_1_5"]=level15(b.level)
    b=b[b.condition.isin(["p","m"])][["trialsnum","condition","level_1_5","blocks.thisRepN"]]
    m=md.merge(b,on="trialsnum",how="left",validate="1:1")
    if m[["condition","level_1_5","blocks.thisRepN"]].isna().any().any(): raise RuntimeError(f"{s}: metadata merge failed")
    return m

def binned_epochs(s):
    md=metadata(s); raw=mne.io.read_raw_fif(str(rawp(s)),preload=True,verbose="ERROR")
    dr=[c for c in DROP if c in raw.ch_names]
    if dr: raw.drop_channels(dr)
    raw.pick("eeg",exclude=[])
    good=md[md.badtrial==0].copy().reset_index(drop=True)
    ev=np.column_stack([good["sample"].to_numpy(int),np.zeros(len(good),int),np.ones(len(good),int)])
    e=mne.Epochs(raw,ev,event_id={"cue":1},tmin=EPOCH_TMIN,tmax=EPOCH_TMAX,
                 baseline=None,metadata=good,preload=True,reject=None,
                 reject_by_annotation=True,verbose="ERROR")
    if len(e)!=len(good): raise RuntimeError(f"{s}: epoch drop")
    e.resample(RESAMPLE,npad="auto",verbose="ERROR")
    X=e.get_data(copy=True).astype(np.float64); times=e.times
    bm=(times>=BASELINE[0]-1e-12)&(times<=BASELINE[1]+1e-12)
    X-=X[...,bm].mean(-1,keepdims=True)
    edges=np.arange(EPOCH_TMIN,EPOCH_TMAX+BIN*.5,BIN)
    if edges[-1]<EPOCH_TMAX-1e-9: edges=np.append(edges,EPOCH_TMAX)
    centers=[]; arr=[]
    for i in range(len(edges)-1):
        lo,hi=edges[i],edges[i+1]; ix=np.where((times>=lo-1e-12)&((times<=hi+1e-12) if i==len(edges)-2 else (times<hi-1e-12)))[0]
        if not len(ix): raise RuntimeError("empty bin")
        centers.append((lo+hi)/2); arr.append(X[...,ix].mean(-1))
    # trial x channel x timebin
    return np.stack(arr,axis=-1).astype(np.float32),np.asarray(centers),e.metadata.reset_index(drop=True),list(e.ch_names)

def reg(): return Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=RIDGE_ALPHA))])
def clf(): return Pipeline([("scaler",StandardScaler()),("ridge_classifier",RidgeClassifier(alpha=RIDGE_ALPHA))])
def cv(g):
    u=np.unique(g); k=GroupKFold(n_splits=len(u)); dummy=np.zeros(len(g))
    return list(k.split(np.zeros((len(g),1)),dummy,g))
def cr(y,p):
    if np.std(y)==0 or np.std(p)==0:return 0.
    return float(np.corrcoef(y,p)[0,1])

def decode_reg(X,y,g):
    sp=cv(g); out=np.empty(X.shape[-1],np.float32)
    for t in range(X.shape[-1]):
        pred=np.full(len(y),np.nan); xx=np.asarray(X[:,:,t],float)
        for tr,te in sp:
            m=reg();m.fit(xx[tr],y[tr]);pred[te]=m.predict(xx[te])
        out[t]=cr(y,pred)
    return out
def decode_clf(X,y,g):
    sp=cv(g); out=np.empty(X.shape[-1],np.float32)
    for t in range(X.shape[-1]):
        sc=np.full(len(y),np.nan);xx=np.asarray(X[:,:,t],float)
        for tr,te in sp:
            m=clf();m.fit(xx[tr],y[tr]);sc[te]=m.decision_function(xx[te])
        out[t]=roc_auc_score(y,sc)
    return out

def one(s,overwrite):
    p=scorep(s)
    if p.exists() and not overwrite: print(f"[{s}] cache -> skip",flush=True); return
    X,t,md,ch=binned_epochs(s); cond=md.condition.astype(str).str.lower().to_numpy()
    lev=pd.to_numeric(md.level_1_5,errors="raise").to_numpy(float); g=pd.to_numeric(md["blocks.thisRepN"],errors="raise").to_numpy(int)
    R=np.empty((2,len(t)),np.float32); A=np.empty_like(R); n1=[];n2=[]
    for ai,(a,c) in enumerate(ATTRS.items()):
        q=cond==c; xx=X[q]; y=lev[q];gg=g[q]
        R[ai]=decode_reg(xx,y,gg); keep=np.isin(y,[1.,2.,4.,5.]); yy=(y[keep]>=4).astype(int)
        A[ai]=decode_clf(xx[keep],yy,gg[keep]); n1.append(len(y));n2.append(len(yy))
    tmp=p.with_suffix(p.suffix+".tmp")
    with open(tmp,"wb") as f: np.savez_compressed(f,participant=np.array(s),times=t,channels=np.array(ch),attribute_names=np.array(list(ATTRS)),
                                                  ordinal_r=R,highlow_auc=A,n_trials_ordinal=np.array(n1),n_trials_highlow=np.array(n2))
    os.replace(tmp,p); print(f"[{s}] DONE",flush=True)

def holm(p):
    p=np.asarray(p,float);m=len(p);o=np.argsort(p);r=p[o];a=np.maximum.accumulate((m-np.arange(m))*r);a=np.clip(a,0,1);z=np.empty(m);z[o]=a;return z
def test(x,t,nperm):
    tm=(t>=INFER_TMIN-1e-12)&(t<=INFER_TMAX+1e-12);tt=t[tm];x=x[:,tm];ns=x.shape[0]
    thr=float(stats.t.ppf(1-ALPHA,df=ns-1))
    T,C,P,_=permutation_cluster_1samp_test(x,threshold=thr,n_permutations=nperm,tail=1,seed=RANDOM_STATE,out_type="indices",verbose=False)
    return T,[np.asarray(c[0],int) for c in C],np.asarray(P,float),tt

def group(S,nperm):
    zs=[]; tref=None;attrs=None;ch=None; counts=[]
    for s in S:
        z=np.load(scorep(s),allow_pickle=False); t=z["times"].astype(float)
        if tref is None: tref=t; attrs=z["attribute_names"].astype(str).tolist();ch=z["channels"].astype(str).tolist()
        elif not np.allclose(t,tref): raise RuntimeError("time mismatch")
        zs.append((z["ordinal_r"],z["highlow_auc"]))
        for ai,a in enumerate(attrs): counts.append(dict(participant=s,attribute=a,n_trials_ordinal=int(z["n_trials_ordinal"][ai]),n_trials_highlow=int(z["n_trials_highlow"][ai])))
    R=np.stack([x[0] for x in zs]);A=np.stack([x[1] for x in zs]); pd.DataFrame(counts).to_csv(OUT_ROOT/"participant_trial_counts.csv",index=False)
    np.savez_compressed(OUT_ROOT/"group_score_arrays.npz",subjects=np.array(S),times=tref,channels=np.array(ch),attribute_names=np.array(attrs),ordinal_r=R,highlow_auc=A)
    allO=[];allC=[];allTC=[]
    for fam,raw in [("ordinal",R),("highlow",A)]:
        X=np.arctanh(np.clip(raw,-.999999,.999999)) if fam=="ordinal" else raw-.5
        ors=[]; cs=[]
        for ai,a in enumerate(attrs):
            T,C,P,tt=test(X[:,ai],tref,nperm);pm=float(P.min()) if len(P) else 1.;ors.append(dict(family=fam,attribute=a,omnibus_p_within_time=pm))
            for ci,(idx,pv) in enumerate(zip(C,P)):
                vals=T[idx];pk=int(np.argmax(vals));cs.append(dict(family=fam,attribute=a,cluster_id=ci,cluster_p_within_time=float(pv),
                    time_start_s=float(tt[idx].min()),time_end_s=float(tt[idx].max()),peak_time_s=float(tt[int(idx[pk])]),peak_t=float(vals[pk])))
        O=pd.DataFrame(ors);O["omnibus_p_holm_across_pain_money"]=holm(O.omnibus_p_within_time);O["domain_significant_holm_0p05"]=O.omnibus_p_holm_across_pain_money<.05
        CDF=pd.DataFrame(cs)
        if len(CDF):
            CDF=CDF.merge(O[["family","attribute","omnibus_p_holm_across_pain_money","domain_significant_holm_0p05"]],on=["family","attribute"])
            CDF["cluster_significant_final"]=(CDF.cluster_p_within_time<.05)&CDF.domain_significant_holm_0p05
        allO.append(O);allC.append(CDF)
        mean=raw.mean(0);sem=stats.sem(raw,axis=0)
        for ai,a in enumerate(attrs):
            for ti,v in enumerate(tref): allTC.append(dict(family=fam,attribute=a,time_s=float(v),mean_score=float(mean[ai,ti]),sem_score=float(sem[ai,ti]),chance=0. if fam=="ordinal" else .5))
        print("\n",fam.upper(),"\n",O.to_string(index=False),flush=True)
        sig=CDF[CDF.cluster_significant_final==True] if len(CDF) else CDF
        print("Significant clusters:\n",sig.to_string(index=False) if len(sig) else "None",flush=True)
    pd.concat(allO).to_csv(OUT_ROOT/"domain_omnibus_holm.csv",index=False)
    pd.concat([x for x in allC if len(x)],ignore_index=True).to_csv(OUT_ROOT/"clusters.csv",index=False)
    pd.DataFrame(allTC).to_csv(OUT_ROOT/"group_timecourses.csv",index=False)

def main():
    a=cli();OUT_ROOT.mkdir(parents=True,exist_ok=True);SCORE_DIR.mkdir(parents=True,exist_ok=True);FIG_DIR.mkdir(parents=True,exist_ok=True)
    S=subs()
    for i,s in enumerate(S,1): print(f"[{i}/{len(S)}] {s}",flush=True);one(s,a.overwrite_scores)
    group(S,a.n_permutations)
    with open(OUT_ROOT/"params.json","w") as f: json.dump({"signal":"cue-locked EEG voltage","epoch_s":[EPOCH_TMIN,EPOCH_TMAX],"resample_hz":RESAMPLE,
        "baseline_s":list(BASELINE),"time_bin_s":BIN,"inference_s":[INFER_TMIN,INFER_TMAX],
        "ordinal":"Ridge level1-5 pooled OOF Pearson r","highlow":"RidgeClassifier low12/high45 pooled OOF AUC",
        "cv":"leave-one-passive-block-out","n_permutations":a.n_permutations},f,indent=2)
    print("="*78);print("ERP/VOLTAGE MVPA COMPLETE");print(OUT_ROOT);print("="*78)
if __name__=="__main__":main()
