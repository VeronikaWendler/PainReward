#!/usr/bin/env python3
"""
05g_eeg_tfr_decode_evoked_removed.py

Within-domain decoding of the cross-fitted evoked-removed / induced-enriched
passive TFRs created by 04d_eeg_tfr_prep_evoked_removed.py.

Mirrors the completed tfr_mvpa_passive_v2 analysis:
PRIMARY ordinal 1..5 Ridge regression
SECONDARY high-vs-low RidgeClassifier
frequency-resolved 4..45 Hz x time plus canonical bands.

For the mechanistic follow-up, formal inference is restricted to 0..1.10 s.
Scores are still saved across the full -0.8..+1.5 s range.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_PATH=Path(os.getenv("basepath",Path(__file__).parent.parent.parent))
PARTICIPANTS_TSV=BASE_PATH/"participants.tsv"
OUT_ROOT=BASE_PATH/"derivatives"/"statistics"/"tfr_mvpa_passive_evoked_removed"
SCORE_DIR=OUT_ROOT/"participant_scores"
FIG_DIR=OUT_ROOT/"figures"
ATTRS={"pain":"p","money":"m"}
BANDS={"theta":(4.,7.),"alpha":(8.,13.),"beta":(14.,30.),"low_gamma":(31.,45.)}
RIDGE_ALPHA=1.
RANDOM_STATE=23
ALPHA=.05
TMIN,TMAX=0.,1.10

def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--n-permutations",type=int,default=5000)
    p.add_argument("--n-jobs",type=int,default=max(1,int(os.getenv("SLURM_CPUS_PER_TASK","1"))))
    p.add_argument("--overwrite-scores",action="store_true")
    p.add_argument("--group-only",action="store_true")
    return p.parse_args()

def paths(s):
    d=BASE_PATH/"derivatives"/s/"eeg"/"tfr_evoked_removed"; stem=f"{s}_passive_cue"
    return {"power":d/f"{stem}_power_evoked_removed.npy","times":d/f"{stem}_times.npy",
            "freqs":d/f"{stem}_freqs.npy","meta":d/f"{stem}_metadata.csv"}

def scorep(s): return SCORE_DIR/f"{s}_evoked_removed_mvpa.npz"

def subs():
    d=pd.read_csv(PARTICIPANTS_TSV,sep="\t")
    out=sorted(d["participant_id"].dropna().astype(str).unique())
    for s in out:
        miss=[str(p) for p in paths(s).values() if not p.exists()]
        if miss: raise FileNotFoundError(f"{s}: missing evoked-removed TFR\n"+"\n".join(miss))
    return out

def reg(): return Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=RIDGE_ALPHA))])
def clf(): return Pipeline([("scaler",StandardScaler()),("ridge_classifier",RidgeClassifier(alpha=RIDGE_ALPHA))])

def splits(g):
    u=np.unique(g)
    if len(u)<2: raise RuntimeError("Need >=2 blocks")
    cv=GroupKFold(n_splits=len(u)); dummy=np.zeros(len(g))
    return list(cv.split(np.zeros((len(g),1)),dummy,g))

def corr(y,p):
    if np.std(y)==0 or np.std(p)==0: return 0.
    r=float(np.corrcoef(y,p)[0,1]); return r if np.isfinite(r) else 0.

def reg_tc(X,y,sp):
    out=np.empty(X.shape[2],np.float32)
    for t in range(X.shape[2]):
        xx=np.asarray(X[:,:,t],float); pred=np.full(len(y),np.nan)
        for tr,te in sp:
            m=reg(); m.fit(xx[tr],y[tr]); pred[te]=m.predict(xx[te])
        out[t]=corr(y,pred)
    return out

def clf_tc(X,y,sp):
    out=np.empty(X.shape[2],np.float32)
    for t in range(X.shape[2]):
        xx=np.asarray(X[:,:,t],float); sc=np.full(len(y),np.nan)
        for tr,te in sp:
            m=clf(); m.fit(xx[tr],y[tr]); sc[te]=m.decision_function(xx[te])
        out[t]=roc_auc_score(y,sc)
    return out

def rf(X,y,sp,fi): return reg_tc(X[:,:,fi,:],y,sp)
def cf(X,y,sp,fi): return clf_tc(X[:,:,fi,:],y,sp)

def one_sub(s,njobs,overwrite):
    op=scorep(s)
    if op.exists() and not overwrite:
        print(f"[{s}] cache exists -> skip",flush=True); return
    p=paths(s); P=np.load(p["power"],mmap_mode="r"); times=np.load(p["times"]).astype(float)
    freqs=np.load(p["freqs"]).astype(float); md=pd.read_csv(p["meta"])
    cond=md["condition"].astype(str).str.lower().to_numpy()
    lev=pd.to_numeric(md["level_1_5"],errors="coerce").to_numpy(float)
    g=pd.to_numeric(md["blocks.thisRepN"],errors="coerce").to_numpy(int)
    nr,nf,nt,nb=2,len(freqs),len(times),len(BANDS)
    R=np.empty((nr,nf,nt),np.float32); RB=np.empty((nr,nb,nt),np.float32)
    A=np.empty((nr,nf,nt),np.float32); AB=np.empty((nr,nb,nt),np.float32)
    nreg=[]; nhl=[]
    for ai,(attr,code) in enumerate(ATTRS.items()):
        sel=cond==code; X=np.asarray(P[sel],np.float32); y=lev[sel]; gg=g[sel]; sp=splits(gg)
        if len(np.unique(y))<5: raise RuntimeError(f"{s}/{attr}: missing level")
        R[ai]=np.stack(Parallel(n_jobs=njobs,prefer="threads")(delayed(rf)(X,y,sp,fi) for fi in range(nf)))
        keep=np.isin(y,[1.,2.,4.,5.]); yy=(y[keep]>=4).astype(int); g2=gg[keep]; sp2=splits(g2)
        A[ai]=np.stack(Parallel(n_jobs=njobs,prefer="threads")(delayed(cf)(X[keep],yy,sp2,fi) for fi in range(nf)))
        for bi,(band,(lo,hi)) in enumerate(BANDS.items()):
            fm=(freqs>=lo)&(freqs<=hi); XB=X[:,:,fm,:].mean(2,dtype=np.float64).astype(np.float32)
            RB[ai,bi]=reg_tc(XB,y,sp); AB[ai,bi]=clf_tc(XB[keep],yy,sp2)
        nreg.append(len(y)); nhl.append(len(yy))
    tmp=op.with_suffix(op.suffix+".tmp")
    with open(tmp,"wb") as f:
        np.savez_compressed(f,participant=np.array(s),times=times,freqs=freqs,
            attribute_names=np.array(list(ATTRS)),band_names=np.array(list(BANDS)),
            regression_freq_r=R,regression_band_r=RB,highlow_freq_auc=A,highlow_band_auc=AB,
            n_trials_reg=np.array(nreg),n_trials_highlow=np.array(nhl))
    os.replace(tmp,op); print(f"[{s}] DONE",flush=True)

def holm(p):
    p=np.asarray(p,float); m=len(p); o=np.argsort(p); r=p[o]
    a=np.maximum.accumulate((m-np.arange(m))*r); a=np.clip(a,0,1); out=np.empty(m); out[o]=a; return out
def fisher(x): return np.arctanh(np.clip(x,-.999999,.999999))
def threshold(n): return float(stats.t.ppf(1-ALPHA,df=n-1))

def tf_test(x,freqs,times,nperm,njobs):
    tm=(times>=TMIN-1e-12)&(times<=TMAX+1e-12); ti=times[tm]; x=x[:,:,tm]
    ns,nf,nt=x.shape
    T,C,P,_=permutation_cluster_1samp_test(x.reshape(ns,nf*nt),threshold=threshold(ns),
        n_permutations=nperm,tail=1,adjacency=combine_adjacency(nf,nt),seed=RANDOM_STATE,
        n_jobs=njobs,out_type="indices",verbose=False)
    return T.reshape(nf,nt),[np.asarray(c[0],int) for c in C],np.asarray(P,float),ti

def time_test(x,times,nperm,njobs):
    tm=(times>=TMIN-1e-12)&(times<=TMAX+1e-12); ti=times[tm]; x=x[:,tm]; ns=x.shape[0]
    T,C,P,_=permutation_cluster_1samp_test(x,threshold=threshold(ns),n_permutations=nperm,tail=1,
        seed=RANDOM_STATE,n_jobs=njobs,out_type="indices",verbose=False)
    return T,[np.asarray(c[0],int) for c in C],np.asarray(P,float),ti

def load_group(S):
    arr=[]; tref=fref=None; attrs=bands=None; counts=[]
    for s in S:
        z=np.load(scorep(s),allow_pickle=False); t=z["times"].astype(float); f=z["freqs"].astype(float)
        if tref is None: tref,fref=t,f; attrs=z["attribute_names"].astype(str).tolist(); bands=z["band_names"].astype(str).tolist()
        elif not np.allclose(t,tref) or not np.array_equal(f,fref): raise RuntimeError(f"{s}: grid mismatch")
        arr.append({k:z[k] for k in ["regression_freq_r","regression_band_r","highlow_freq_auc","highlow_band_auc"]})
        for ai,a in enumerate(attrs): counts.append(dict(participant=s,attribute=a,n_trials_ordinal=int(z["n_trials_reg"][ai]),n_trials_highlow=int(z["n_trials_highlow"][ai])))
    G={k:np.stack([a[k] for a in arr]) for k in arr[0]}
    return G,tref,fref,attrs,bands,pd.DataFrame(counts)

def group_family(family,F,B,times,freqs,attrs,bands,nperm,njobs):
    testF=fisher(F) if family=="ordinal" else F-.5; testB=fisher(B) if family=="ordinal" else B-.5
    drows=[]; crows=[]
    for ai,a in enumerate(attrs):
        T,C,P,ti=tf_test(testF[:,ai],freqs,times,nperm,njobs); pmin=float(P.min()) if len(P) else 1.
        drows.append(dict(family=family,attribute=a,omnibus_p_within_surface=pmin))
        for ci,(idx,pv) in enumerate(zip(C,P)):
            fi,tt=np.unravel_index(idx,(len(freqs),len(ti))); vals=T.reshape(-1)[idx]; pk=int(np.argmax(vals)); pfi,pti=np.unravel_index(int(idx[pk]),(len(freqs),len(ti)))
            crows.append(dict(family=family,attribute=a,cluster_id=ci,cluster_p_within_surface=float(pv),
                freq_min_hz=float(freqs[fi].min()),freq_max_hz=float(freqs[fi].max()),
                time_start_s=float(ti[tt].min()),time_end_s=float(ti[tt].max()),
                peak_freq_hz=float(freqs[pfi]),peak_time_s=float(ti[pti]),peak_t=float(vals[pk]),cluster_mass=float(vals.sum())))
    D=pd.DataFrame(drows); D["omnibus_p_holm_across_pain_money"]=holm(D.omnibus_p_within_surface); D["domain_significant_holm_0p05"]=D.omnibus_p_holm_across_pain_money<.05
    CDF=pd.DataFrame(crows)
    if len(CDF):
        CDF=CDF.merge(D[["family","attribute","omnibus_p_holm_across_pain_money","domain_significant_holm_0p05"]],on=["family","attribute"])
        CDF["cluster_significant_final"]=(CDF.cluster_p_within_surface<.05)&CDF.domain_significant_holm_0p05
    brows=[]; bcrows=[]
    for ai,a in enumerate(attrs):
        pmins=[]; store=[]
        for bi,b in enumerate(bands):
            T,C,P,ti=time_test(testB[:,ai,bi],times,nperm,njobs); pm=float(P.min()) if len(P) else 1.; pmins.append(pm); store.append((b,T,C,P,ti))
        hp=holm(pmins)
        for bi,(b,T,C,P,ti) in enumerate(store):
            brows.append(dict(family=family,attribute=a,band=b,band_omnibus_p_within_time=pmins[bi],band_omnibus_p_holm_within_attribute=float(hp[bi]),band_significant_holm_0p05=bool(hp[bi]<.05)))
            for ci,(idx,pv) in enumerate(zip(C,P)):
                vals=T[idx]; pk=int(np.argmax(vals))
                bcrows.append(dict(family=family,attribute=a,band=b,cluster_id=ci,cluster_p_within_band_time=float(pv),
                    time_start_s=float(ti[idx].min()),time_end_s=float(ti[idx].max()),peak_time_s=float(ti[int(idx[pk])]),peak_t=float(vals[pk]),
                    band_omnibus_p_holm_within_attribute=float(hp[bi]),cluster_significant_final=bool(pv<.05 and hp[bi]<.05)))
    return D,CDF,pd.DataFrame(brows),pd.DataFrame(bcrows)

def main():
    a=cli(); OUT_ROOT.mkdir(parents=True,exist_ok=True); SCORE_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
    S=subs()
    if not a.group_only:
        for i,s in enumerate(S,1):
            print(f"[{i}/{len(S)}] {s}",flush=True); one_sub(s,a.n_jobs,a.overwrite_scores)
    G,times,freqs,attrs,bands,counts=load_group(S); counts.to_csv(OUT_ROOT/"participant_trial_counts.csv",index=False)
    np.savez_compressed(OUT_ROOT/"group_score_arrays.npz",subjects=np.array(S),times=times,freqs=freqs,attribute_names=np.array(attrs),band_names=np.array(bands),**G)
    Ds=[]; Cs=[]; Bs=[]; BCs=[]
    for fam,fk,bk in [("ordinal","regression_freq_r","regression_band_r"),("highlow","highlow_freq_auc","highlow_band_auc")]:
        D,C,B,BC=group_family(fam,G[fk],G[bk],times,freqs,attrs,bands,a.n_permutations,a.n_jobs)
        Ds.append(D); Cs.append(C); Bs.append(B); BCs.append(BC)
        print("\n",fam.upper(),"FREQ x TIME\n",D.to_string(index=False),flush=True)
        sig=C[C.cluster_significant_final==True] if len(C) else C
        print("Significant TF clusters:\n", sig.to_string(index=False) if len(sig) else "None",flush=True)
        print("\n",fam.upper(),"BANDS\n",B.to_string(index=False),flush=True)
        sigb=BC[BC.cluster_significant_final==True] if len(BC) else BC
        print("Significant band clusters:\n",sigb.to_string(index=False) if len(sigb) else "None",flush=True)
    pd.concat(Ds).to_csv(OUT_ROOT/"freq_time_domain_omnibus_holm.csv",index=False)
    pd.concat([x for x in Cs if len(x)],ignore_index=True).to_csv(OUT_ROOT/"freq_time_clusters.csv",index=False)
    pd.concat(Bs).to_csv(OUT_ROOT/"band_omnibus_holm.csv",index=False)
    pd.concat([x for x in BCs if len(x)],ignore_index=True).to_csv(OUT_ROOT/"band_clusters.csv",index=False)
    with open(OUT_ROOT/"params.json","w") as f:
        json.dump({"signal":"cross-fitted domain-average evoked removed / induced-enriched",
                   "inference_time_s":[TMIN,TMAX],"n_permutations":a.n_permutations,
                   "ordinal":"Ridge level1-5 pooled OOF Pearson r; Fisher-z group test",
                   "highlow":"RidgeClassifier low12 vs high45; pooled OOF AUC",
                   "frequency_inference":"separate 2D cluster per pain/money; Holm across two domains",
                   "band_inference":"temporal cluster per band; Holm across 4 bands within each domain"},f,indent=2)
    print("="*78); print("EVOKED-REMOVED TFR MVPA COMPLETE"); print(OUT_ROOT); print("="*78)
if __name__=="__main__": main()
