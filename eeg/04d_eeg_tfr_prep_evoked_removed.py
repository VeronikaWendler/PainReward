#!/usr/bin/env python3
"""
04d_eeg_tfr_prep_evoked_removed.py

Create a leakage-safe EVOKED-REMOVED / INDUCED-ENRICHED single-trial TFR.

For each participant, domain (pain/money), and held-out passive block b:
    template = mean EEG waveform from SAME DOMAIN, blocks != b
    residual trials in block b = original EEG - template

The test block never contributes to its own template and cue LEVEL is never
used to construct the template. This removes the domain-general phase-locked
evoked response without using magnitude labels. It does NOT guarantee removal
of every level-specific phase-locked component, so outputs are deliberately
called "evoked_removed" / "induced_enriched", not pure induced activity.

TFR parameters are matched to 04b_eeg_tfr_prep.py:
4-45 Hz, 1-Hz steps, n_cycles=f/2, -2..+2 s working epoch, 256 Hz,
Morlet decim=2, mean-log-ratio baseline [-0.7,-0.5], 20-ms bins,
save -0.8..+1.5 s.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import mne, numpy as np, pandas as pd
from mne.time_frequency import tfr_array_morlet

mne.set_log_level("ERROR")
BASE_PATH=Path(os.getenv("basepath",Path(__file__).parent.parent.parent))
SOURCE_BIDS=Path(os.getenv("SOURCE_BIDS",BASE_PATH))
FREQS=np.arange(4.,46.,1.)
N_CYCLES_FACTOR=.5
EPOCH_TMIN,EPOCH_TMAX=-2.,2.
EPOCH_SFREQ=256.
TFR_DECIM=2
BASELINE=(-.7,-.5)
SAVE_TMIN,SAVE_TMAX=-.8,1.5
TIME_BIN_S=.020
FREQ_CHUNK_SIZE=6
LOG_FLOOR=np.finfo(np.float64).tiny
DROP=["HEOGL","HEOGR","VEOGL","STI 014","Status"]

def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--subject",required=True)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--n-jobs",type=int,default=max(1,int(os.getenv("SLURM_CPUS_PER_TASK","1"))))
    return p.parse_args()

def beh_path(s): return SOURCE_BIDS/s/"eeg"/f"{s}_task-passive_beh.tsv"
def raw_path(s): return SOURCE_BIDS/"derivatives"/s/"eeg"/f"{s}_passive_cleaned-raw.fif"
def erp_path(s): return SOURCE_BIDS/"derivatives"/s/"eeg"/"erps_passive"/f"{s}_passive_cues_singletrials-epo.fif"
def outdir(s): return BASE_PATH/"derivatives"/s/"eeg"/"tfr_evoked_removed"

def level15(v):
    x=pd.to_numeric(v,errors="coerce")
    if x.isna().all(): raise RuntimeError("level entirely nonnumeric")
    if x.max(skipna=True)>5: x=x/20.
    return x.astype(float)

def load_beh(s):
    p=beh_path(s)
    if not p.exists(): raise FileNotFoundError(p)
    b=pd.read_csv(p,sep="\t")
    req=["fixcross.started","cue.started","condition","level","blocks.thisRepN"]
    miss=[c for c in req if c not in b.columns]
    if miss: raise KeyError(f"{s}: missing behaviour columns {miss}")
    b=b[~b["fixcross.started"].isna()].copy().reset_index(drop=True)
    b["trialsnum"]=np.arange(1,len(b)+1,dtype=int)
    b["condition"]=b["condition"].astype(str).str.lower().str.strip()
    b["level_1_5"]=level15(b["level"])
    b["fixation_duration_s"]=pd.to_numeric(b["cue.started"],errors="coerce")-pd.to_numeric(b["fixcross.started"],errors="coerce")
    b=b[b["condition"].isin(["p","m"])].copy()
    return b[["trialsnum","condition","level_1_5","blocks.thisRepN","fixation_duration_s"]]

def metadata(s):
    p=erp_path(s)
    if not p.exists(): raise FileNotFoundError(p)
    e=mne.read_epochs(str(p),preload=False,verbose="ERROR")
    if e.metadata is None: raise RuntimeError(f"{s}: ERP metadata missing")
    md=e.metadata.reset_index(drop=True).copy()
    req=["trialsnum","sample","badtrial"]
    miss=[c for c in req if c not in md.columns]
    if miss: raise KeyError(f"{s}: ERP metadata missing {miss}")
    for c in req: md[c]=pd.to_numeric(md[c],errors="coerce")
    if md[req].isna().any().any(): raise RuntimeError(f"{s}: bad trialsnum/sample/badtrial")
    md["trialsnum"]=md["trialsnum"].astype(int); md["sample"]=md["sample"].astype(int); md["badtrial"]=md["badtrial"].astype(int)
    if md["trialsnum"].duplicated().any(): raise RuntimeError(f"{s}: duplicate trialsnum")
    m=md.merge(load_beh(s),on="trialsnum",how="left",validate="1:1")
    if m[["condition","level_1_5","blocks.thisRepN","fixation_duration_s"]].isna().any().any():
        raise RuntimeError(f"{s}: unlabeled ERP trials after behaviour merge")
    if (m["fixation_duration_s"].astype(float)<abs(BASELINE[0])).any():
        raise RuntimeError(f"{s}: baseline not safely inside fixation")
    return m

def long_epochs(s,md):
    p=raw_path(s)
    if not p.exists(): raise FileNotFoundError(p)
    raw=mne.io.read_raw_fif(str(p),preload=True,verbose="ERROR")
    dr=[c for c in DROP if c in raw.ch_names]
    if dr: raw.drop_channels(dr)
    raw.pick("eeg",exclude=[])
    good=md[md["badtrial"]==0].copy().reset_index(drop=True)
    ev=np.column_stack([good["sample"].to_numpy(int),np.zeros(len(good),int),np.ones(len(good),int)])
    epo=mne.Epochs(raw,events=ev,event_id={"cue":1},tmin=EPOCH_TMIN,tmax=EPOCH_TMAX,
                   baseline=None,metadata=good,preload=True,reject=None,
                   reject_by_annotation=True,verbose="ERROR")
    if len(epo)!=len(good): raise RuntimeError(f"{s}: long epoch drop {len(epo)}/{len(good)}")
    epo.resample(EPOCH_SFREQ,npad="auto",verbose="ERROR")
    return epo

def time_bins(times):
    edges=np.arange(SAVE_TMIN,SAVE_TMAX+TIME_BIN_S*.5,TIME_BIN_S)
    if edges[-1]<SAVE_TMAX-1e-9: edges=np.append(edges,SAVE_TMAX)
    if abs(edges[-1]-SAVE_TMAX)>1e-9: edges=np.append(edges[edges<SAVE_TMAX],SAVE_TMAX)
    centers=[]; masks=[]
    for i in range(len(edges)-1):
        lo,hi=float(edges[i]),float(edges[i+1])
        idx=np.where((times>=lo-1e-12)&((times<=hi+1e-12) if i==len(edges)-2 else (times<hi-1e-12)))[0]
        if not len(idx): raise RuntimeError(f"No TFR samples in {lo},{hi}")
        centers.append((lo+hi)/2); masks.append(idx)
    return np.asarray(centers),masks

def atomic_npy(path,a):
    t=Path(str(path)+".tmp")
    with open(t,"wb") as f: np.save(f,a,allow_pickle=False)
    os.replace(t,path)

def crossfit_remove_evoked(data,md):
    residual=np.empty_like(data,dtype=np.float64)
    assigned=np.zeros(len(md),bool)
    cond=md["condition"].astype(str).str.lower().to_numpy()
    blocks=pd.to_numeric(md["blocks.thisRepN"],errors="raise").to_numpy(int)
    diagnostics=[]
    for domain in ["p","m"]:
        dm=cond==domain
        bs=np.unique(blocks[dm])
        for b in bs:
            train=dm&(blocks!=b); test=dm&(blocks==b)
            if not train.any() or not test.any(): raise RuntimeError(f"{domain} block {b}: empty train/test")
            template=data[train].mean(axis=0)
            residual[test]=data[test]-template
            assigned[test]=True
            before=float(np.sqrt(np.mean(data[test]**2)))
            after=float(np.sqrt(np.mean(residual[test]**2)))
            diagnostics.append(dict(condition=domain,heldout_block=int(b),n_template_trials=int(train.sum()),
                                    n_test_trials=int(test.sum()),rms_before=before,rms_after=after))
    if not assigned.all(): raise RuntimeError("Some trials were not assigned evoked-removed data")
    return residual,pd.DataFrame(diagnostics)

def main():
    a=cli(); s=a.subject; out=outdir(s); out.mkdir(parents=True,exist_ok=True)
    powerp=out/f"{s}_passive_cue_power_evoked_removed.npy"
    timesp=out/f"{s}_passive_cue_times.npy"; freqsp=out/f"{s}_passive_cue_freqs.npy"
    metap=out/f"{s}_passive_cue_metadata.csv"; chanp=out/f"{s}_passive_cue_channels.json"
    paramsp=out/f"{s}_passive_cue_params.json"; sump=out/f"{s}_passive_cue_prep_summary.json"
    diagp=out/f"{s}_evoked_removal_diagnostics.csv"
    outs=[powerp,timesp,freqsp,metap,chanp,paramsp,sump,diagp]
    if all(p.exists() for p in outs) and not a.overwrite:
        print(f"[{s}] complete evoked-removed outputs exist -> skip",flush=True); return
    mdall=metadata(s); epo=long_epochs(s,mdall); md=epo.metadata.reset_index(drop=True).copy()
    ch=list(epo.ch_names); data=epo.get_data(copy=False).astype(np.float64,copy=False)
    residual,diag=crossfit_remove_evoked(data,md); diag.to_csv(diagp,index=False)
    tf_times=epo.times[::TFR_DECIM]
    bm=(tf_times>=BASELINE[0]-1e-12)&(tf_times<=BASELINE[1]+1e-12)
    centers,masks=time_bins(tf_times)
    final=np.empty((len(md),len(ch),len(FREQS),len(centers)),np.float32)
    print(f"[{s}] evoked-removed TFR trials={len(md)} channels={len(ch)} freqs={len(FREQS)} bins={len(centers)}",flush=True)
    for st in range(0,len(FREQS),FREQ_CHUNK_SIZE):
        en=min(st+FREQ_CHUNK_SIZE,len(FREQS)); fq=FREQS[st:en]
        print(f"[{s}] {fq[0]:.0f}-{fq[-1]:.0f} Hz",flush=True)
        pw=tfr_array_morlet(residual,sfreq=EPOCH_SFREQ,freqs=fq,n_cycles=fq*N_CYCLES_FACTOR,
                            output="power",use_fft=True,zero_mean=True,decim=TFR_DECIM,
                            n_jobs=max(1,a.n_jobs),verbose=False)
        lp=np.log(np.maximum(pw,LOG_FLOOR)); norm=lp-lp[...,bm].mean(axis=-1,keepdims=True)
        for bi,ix in enumerate(masks): final[:,:,st:en,bi]=norm[...,ix].mean(-1).astype(np.float32)
        del pw,lp,norm
    if not np.isfinite(final).all(): raise RuntimeError(f"{s}: nonfinite final power")
    atomic_npy(powerp,final); atomic_npy(timesp,centers.astype(float)); atomic_npy(freqsp,FREQS.astype(float))
    md.to_csv(metap,index=False)
    with open(chanp,"w") as f: json.dump(ch,f,indent=2)
    with open(paramsp,"w") as f:
        json.dump({"signal":"cross-fitted domain-average evoked removed; induced-enriched",
                   "evoked_removal":"for held-out block b, subtract same-domain mean waveform estimated from blocks != b; level never used",
                   "frequencies_hz":FREQS.tolist(),"n_cycles":"f/2","working_epoch_s":[EPOCH_TMIN,EPOCH_TMAX],
                   "resample_hz":EPOCH_SFREQ,"tfr_decim":TFR_DECIM,"baseline_s":list(BASELINE),
                   "baseline_normalization":"log(power)-mean(log(power_baseline))",
                   "saved_time_s":[SAVE_TMIN,SAVE_TMAX],"time_bin_s":TIME_BIN_S},f,indent=2)
    with open(sump,"w") as f:
        json.dump({"participant":s,"n_trials":len(md),"n_channels":len(ch),"shape":list(final.shape),
                   "mean_rms_before":float(diag.rms_before.mean()),"mean_rms_after":float(diag.rms_after.mean())},f,indent=2)
    print(f"[{s}] DONE",flush=True)
if __name__=="__main__": main()
