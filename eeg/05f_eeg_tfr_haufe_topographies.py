#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne, numpy as np, pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE=23
RIDGE_ALPHA=1.0
CLUSTER_FORMING_ALPHA=0.05
BASE_PATH=Path(os.getenv("basepath", Path(__file__).parent.parent.parent))
SOURCE_BIDS=Path(os.getenv("SOURCE_BIDS", BASE_PATH))
PARTICIPANTS_TSV=BASE_PATH/"participants.tsv"
MVPA_ROOT=BASE_PATH/"derivatives"/"statistics"/"tfr_mvpa_passive_v2"
GROUP_ARRAYS=MVPA_ROOT/"group_score_arrays.npz"
OUT_ROOT=BASE_PATH/"derivatives"/"statistics"/"tfr_haufe_topographies"
FIG_DIR=OUT_ROOT/"figures"
ATTRIBUTES=["pain","money"]
BANDS={"theta":(4.,7.),"alpha":(8.,13.),"beta":(14.,30.),"low_gamma":(31.,45.)}
INFER_TMIN, INFER_TMAX = 0.0, 1.5

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--n-permutations",type=int,default=5000)
    p.add_argument("--n-jobs",type=int,default=max(1,int(os.getenv("SLURM_CPUS_PER_TASK","1"))))
    return p.parse_args()

def tfr_paths(sub):
    d=BASE_PATH/"derivatives"/sub/"eeg"/"tfr"; stem=f"{sub}_passive_cue"
    return {"power":d/f"{stem}_power.npy","times":d/f"{stem}_times.npy",
            "freqs":d/f"{stem}_freqs.npy","metadata":d/f"{stem}_metadata.csv",
            "channels":d/f"{stem}_channels.json"}

def raw_path(sub):
    return SOURCE_BIDS/"derivatives"/sub/"eeg"/f"{sub}_passive_cleaned-raw.fif"

def holm(p):
    p=np.asarray(p,float); m=len(p); order=np.argsort(p); ranked=p[order]
    a=np.maximum.accumulate((m-np.arange(m))*ranked); a=np.clip(a,0,1)
    out=np.empty(m,float); out[order]=a; return out

def thres(n):
    return float(stats.t.ppf(1-CLUSTER_FORMING_ALPHA,df=n-1))

def fisher(r): return np.arctanh(np.clip(r,-.999999,.999999))

def reconstruct_ordinal_rois(R,freqs,times,nperm,njobs):
    tmask=(times>=INFER_TMIN-1e-12)&(times<=INFER_TMAX+1e-12)
    tinf=times[tmask]; store={}; pmins=[]
    for ai,attr in enumerate(ATTRIBUTES):
        x=fisher(R[:,ai,:, :][:,:,tmask]); ns,nf,nt=x.shape
        T,C,P,_=permutation_cluster_1samp_test(
            x.reshape(ns,nf*nt),threshold=thres(ns),n_permutations=nperm,tail=1,
            adjacency=combine_adjacency(nf,nt),seed=RANDOM_STATE,n_jobs=njobs,
            out_type="indices",verbose=False)
        idx=[np.asarray(c[0],int) for c in C]; P=np.asarray(P,float)
        store[attr]=(T.reshape(nf,nt),idx,P)
        pmins.append(float(P.min()) if len(P) else 1.)
    hadj=holm(pmins); rois=[]
    for ai,attr in enumerate(ATTRIBUTES):
        if hadj[ai]>=.05: continue
        T,clusters,pvals=store[attr]
        for ci,(flat,pv) in enumerate(zip(clusters,pvals)):
            if pv>=.05: continue
            fi,ti=np.unravel_index(flat,(len(freqs),len(tinf)))
            vals=T.reshape(-1)[flat]; pk=int(np.argmax(vals))
            pfi,pti=np.unravel_index(int(flat[pk]),(len(freqs),len(tinf)))
            rid=f"ordinal_{attr}_f{int(freqs[fi].min())}-{int(freqs[fi].max())}_t{tinf[ti].min():.2f}-{tinf[ti].max():.2f}"
            rois.append(dict(roi_id=rid,family="ordinal",attribute=attr,cluster_id=ci,
                cluster_p=float(pv),domain_holm_p=float(hadj[ai]),
                freq_min_hz=float(freqs[fi].min()),freq_max_hz=float(freqs[fi].max()),
                time_start_s=float(tinf[ti].min()),time_end_s=float(tinf[ti].max()),
                peak_freq_hz=float(freqs[pfi]),peak_time_s=float(tinf[pti]),peak_t=float(vals[pk]),
                exact_flat_indices=flat,tmask=tmask))
    return rois

def reconstruct_highlow_band_rois(A,times,band_names,nperm,njobs):
    tmask=(times>=INFER_TMIN-1e-12)&(times<=INFER_TMAX+1e-12); tinf=times[tmask]; rois=[]
    for ai,attr in enumerate(ATTRIBUTES):
        st=[]; pmins=[]
        for bi,band in enumerate(band_names):
            x=A[:,ai,bi,:][:,tmask]-.5; ns=x.shape[0]
            T,C,P,_=permutation_cluster_1samp_test(
                x,threshold=thres(ns),n_permutations=nperm,tail=1,seed=RANDOM_STATE,
                n_jobs=njobs,out_type="indices",verbose=False)
            idx=[np.asarray(c[0],int) for c in C]; P=np.asarray(P,float)
            st.append((band,bi,T,idx,P)); pmins.append(float(P.min()) if len(P) else 1.)
        hadj=holm(pmins)
        for j,(band,bi,T,clusters,pvals) in enumerate(st):
            if hadj[j]>=.05: continue
            lo,hi=BANDS[band]
            for ci,(idx,pv) in enumerate(zip(clusters,pvals)):
                if pv>=.05: continue
                vals=T[idx]; pk=int(np.argmax(vals))
                rid=f"highlow_{attr}_{band}_t{tinf[idx].min():.2f}-{tinf[idx].max():.2f}"
                rois.append(dict(roi_id=rid,family="highlow",attribute=attr,band=band,
                    band_index=bi,cluster_id=ci,cluster_p=float(pv),band_holm_p=float(hadj[j]),
                    freq_min_hz=float(lo),freq_max_hz=float(hi),
                    time_start_s=float(tinf[idx].min()),time_end_s=float(tinf[idx].max()),
                    peak_time_s=float(tinf[int(idx[pk])]),peak_t=float(vals[pk]),
                    time_indices_full=np.where(tmask)[0][idx]))
    return rois

def reg():
    return Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=RIDGE_ALPHA))])
def clf():
    return Pipeline([("scaler",StandardScaler()),("ridge_classifier",RidgeClassifier(alpha=RIDGE_ALPHA))])

def haufe(X,s):
    X=np.asarray(X,float); s=np.asarray(s,float).reshape(-1)
    Xc=X-X.mean(0,keepdims=True); sc=s-s.mean()
    den=float(np.dot(sc,sc)/max(1,len(sc)-1))
    if not np.isfinite(den) or den<=1e-15: raise RuntimeError("Degenerate Haufe model-output variance")
    return (Xc.T@sc/max(1,len(sc)-1))/den

def pattern_reg(X,y,g):
    out=[]
    for b in np.unique(g):
        tr=g!=b; m=reg(); m.fit(X[tr],y[tr]); out.append(haufe(X[tr],m.predict(X[tr])))
    return np.mean(out,0)

def pattern_clf(X,y,g):
    out=[]
    for b in np.unique(g):
        tr=g!=b
        if len(np.unique(y[tr]))<2: raise RuntimeError("One-class fold")
        m=clf(); m.fit(X[tr],y[tr]); out.append(haufe(X[tr],m.decision_function(X[tr])))
    return np.mean(out,0)

def rms(v):
    d=float(np.sqrt(np.mean(np.square(v))))
    return np.zeros_like(v) if d<=1e-15 else v/d

def participant_roi(sub,roi,freqs_ref,times_ref):
    p=tfr_paths(sub); P=np.load(p["power"],mmap_mode="r")
    freqs=np.load(p["freqs"]).astype(float); times=np.load(p["times"]).astype(float)
    md=pd.read_csv(p["metadata"])
    with open(p["channels"],encoding="utf-8") as f: ch=json.load(f)
    if not np.array_equal(freqs,freqs_ref) or not np.allclose(times,times_ref): raise RuntimeError(f"{sub}: grid mismatch")
    code="p" if roi["attribute"]=="pain" else "m"; sel=md["condition"].astype(str).str.lower().to_numpy()==code
    y=pd.to_numeric(md["level_1_5"],errors="coerce").to_numpy(float)[sel]
    g=pd.to_numeric(md["blocks.thisRepN"],errors="coerce").to_numpy(int)[sel]
    Xall=np.asarray(P[sel],dtype=np.float32); pats=[]
    if roi["family"]=="ordinal":
        tmask=roi["tmask"]; tinf=times[tmask]; flat=np.asarray(roi["exact_flat_indices"],int)
        fi,ti=np.unravel_index(flat,(len(freqs),len(tinf))); fullti=np.where(tmask)[0][ti]
        for fidx,tidx in zip(fi,fullti):
            pats.append(pattern_reg(np.asarray(Xall[:,:,int(fidx),int(tidx)],float),y,g))
    else:
        keep=np.isin(y,[1.,2.,4.,5.]); yy=(y[keep]>=4).astype(int); gg=g[keep]
        fmask=(freqs>=roi["freq_min_hz"]-1e-12)&(freqs<=roi["freq_max_hz"]+1e-12)
        for tidx in roi["time_indices_full"]:
            X=np.asarray(Xall[keep,:,:,int(tidx)][:,:,fmask].mean(2,dtype=np.float64),float)
            pats.append(pattern_clf(X,yy,gg))
    raw=np.mean(np.stack(pats),0)
    return ch,raw,rms(raw)

def plot_info(sub,ch):
    p=raw_path(sub)
    raw=mne.io.read_raw_fif(str(p),preload=False,verbose="ERROR")
    dr=[x for x in ["HEOGL","HEOGR","VEOGL","STI 014","Status"] if x in raw.ch_names]
    if dr: raw.drop_channels(dr)
    raw.pick("eeg",exclude=[])
    if list(raw.ch_names)!=list(ch): raise RuntimeError("Raw/TFR channel order mismatch")
    return raw.info.copy()

def public_roi(r):
    return {k:v for k,v in r.items() if k not in {"exact_flat_indices","tmask","time_indices_full"}}

def main():
    a=args(); OUT_ROOT.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)
    z=np.load(GROUP_ARRAYS,allow_pickle=False)
    subjects=z["subjects"].astype(str).tolist(); freqs=z["freqs"].astype(float); times=z["times"].astype(float)
    attrs=z["attribute_names"].astype(str).tolist(); bands=z["band_names"].astype(str).tolist()
    if attrs!=ATTRIBUTES or bands!=list(BANDS): raise RuntimeError("Unexpected v2 label order")
    rois=reconstruct_ordinal_rois(z["regression_freq_r"],freqs,times,a.n_permutations,a.n_jobs)
    rois+=reconstruct_highlow_band_rois(z["highlow_band_auc"],times,bands,a.n_permutations,a.n_jobs)
    if not rois: raise RuntimeError("No significant ROIs reconstructed")
    pd.DataFrame([public_roi(r) for r in rois]).to_csv(OUT_ROOT/"roi_definitions.csv",index=False)
    print(f"ROIs: {len(rois)}",flush=True)
    rows=[]; maps={}; chref=None
    for ri,r in enumerate(rois,1):
        print(f"[{ri}/{len(rois)}] {r['roi_id']}",flush=True); nm=[]
        for si,sub in enumerate(subjects,1):
            ch,raw,norm=participant_roi(sub,r,freqs,times)
            if chref is None: chref=ch
            elif ch!=chref: raise RuntimeError(f"{sub}: channel mismatch")
            nm.append(norm)
            for c,rv,nv in zip(ch,raw,norm):
                rows.append(dict(participant=sub,roi_id=r["roi_id"],family=r["family"],attribute=r["attribute"],
                                 channel=c,haufe_raw=float(rv),haufe_rms_normalized=float(nv)))
            if si%10==0 or si==len(subjects): print(f"  {si}/{len(subjects)}",flush=True)
        maps[r["roi_id"]]=np.stack(nm)
    long=pd.DataFrame(rows); long.to_csv(OUT_ROOT/"participant_roi_haufe_patterns_long.csv",index=False)
    grows=[]
    for r in rois:
        d=long[long.roi_id==r["roi_id"]]
        for c in chref:
            rr=d.loc[d.channel==c,"haufe_raw"].to_numpy(float); nn=d.loc[d.channel==c,"haufe_rms_normalized"].to_numpy(float)
            grows.append(dict(roi_id=r["roi_id"],family=r["family"],attribute=r["attribute"],channel=c,
                              mean_haufe_raw=float(rr.mean()),sem_haufe_raw=float(stats.sem(rr)),
                              mean_haufe_rms_normalized=float(nn.mean()),sem_haufe_rms_normalized=float(stats.sem(nn))))
    gd=pd.DataFrame(grows); gd.to_csv(OUT_ROOT/"group_roi_haufe_patterns.csv",index=False)
    info=plot_info(subjects[0],chref)
    for r in rois:
        vals=gd[gd.roi_id==r["roi_id"]].set_index("channel").loc[chref,"mean_haufe_rms_normalized"].to_numpy(float)
        vmax=float(np.max(np.abs(vals))) or 1.; fig,ax=plt.subplots(figsize=(5.5,5.0))
        mne.viz.plot_topomap(vals,info,axes=ax,show=False,contours=6,vlim=(-vmax,vmax),cmap="RdBu_r",sensors=True)
        ax.set_title(f"{r['attribute'].capitalize()} — {r['family']}\n{r['freq_min_hz']:.0f}-{r['freq_max_hz']:.0f} Hz, {r['time_start_s']:.2f}-{r['time_end_s']:.2f} s")
        fig.tight_layout(); fig.savefig(FIG_DIR/f"{r['roi_id']}_topomap.png",dpi=250); plt.close(fig)
    lookup={r["roi_id"]:r for r in rois}; ids=list(lookup); sims=[]
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            A,B=ids[i],ids[j]
            if lookup[A]["attribute"]==lookup[B]["attribute"]: continue
            rs=[]
            for s in range(len(subjects)):
                va,vb=maps[A][s],maps[B][s]
                rs.append(np.nan if np.std(va)<=0 or np.std(vb)<=0 else float(np.corrcoef(va,vb)[0,1]))
            rs=np.asarray(rs); ok=np.isfinite(rs); zz=np.arctanh(np.clip(rs[ok],-.999999,.999999))
            sims.append(dict(roi_a=A,roi_b=B,n_participants=int(ok.sum()),
                             mean_r_fisher=float(np.tanh(zz.mean())) if len(zz) else np.nan,
                             median_r=float(np.nanmedian(rs)),mean_r_arithmetic=float(np.nanmean(rs)),
                             note="descriptive only; ROIs were selected from decoding effects"))
    pd.DataFrame(sims).to_csv(OUT_ROOT/"pain_money_roi_pattern_similarity_descriptive.csv",index=False)
    with open(OUT_ROOT/"params.json","w") as f:
        json.dump({"analysis":"Haufe activation patterns for completed passive TFR MVPA",
                   "formula":"cov(X_original, model_output)/var(model_output)",
                   "training_only":True,"folds":"leave-one-block-out",
                   "group_scaling":"RMS normalize each participant ROI map across channels",
                   "topography_inference":"none; descriptive follow-up",
                   "roi_reconstruction_permutations":a.n_permutations,"seed":RANDOM_STATE},f,indent=2)
    print("="*78); print("HAUFE TOPOGRAPHIES COMPLETE"); print(OUT_ROOT); print("="*78)
if __name__=="__main__": main()
