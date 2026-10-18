#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import mne, numpy as np, pandas as pd
from mne.time_frequency import tfr_array_morlet

mne.set_log_level('ERROR')
BASE=Path(os.getenv('basepath',Path(__file__).parent.parent.parent))
SOURCE=Path(os.getenv('SOURCE_BIDS',BASE))
DROP=['HEOGL','HEOGR','VEOGL','STI 014','Status']
LOG_FLOOR=np.finfo(np.float64).tiny

def cli():
    p=argparse.ArgumentParser(); p.add_argument('--subject',required=True); p.add_argument('--overwrite',action='store_true')
    p.add_argument('--n-jobs',type=int,default=max(1,int(os.getenv('SLURM_CPUS_PER_TASK','1')))); return p.parse_args()

def rawp(s): return SOURCE/'derivatives'/s/'eeg'/f'{s}_decision_cleaned-raw.fif'
def eventsp(s):
    canon=SOURCE/s/'eeg'/f'{s}_task-decision_events.tsv'
    if canon.exists(): return canon
    hits=sorted((SOURCE/s/'eeg').glob(f'*task-decision*events.tsv'))
    if len(hits)!=1:
        raise FileNotFoundError(f'{s}: expected exactly one decision events TSV, got {hits}')
    return hits[0]
def behp(s): return SOURCE/s/'eeg'/f'{s}_task-decision_beh.tsv'
def refp(s): return BASE/'derivatives'/s/'eeg'/'tfr'/f'{s}_passive_cue_params.json'
def outd(s): return BASE/'derivatives'/s/'eeg'/'tfr_decision'

def num(x,name):
    y=pd.to_numeric(x,errors='coerce')
    if y.isna().any(): raise RuntimeError(f'{name}: missing/nonnumeric values')
    return y.astype(float)

def money_level(df):
    if 'moneylevel' in df: return num(df['moneylevel'],'moneylevel')
    if 'moneystim' not in df: raise KeyError('Need moneylevel or moneystim')
    vals=[]
    for v in df['moneystim'].astype(str):
        m=re.search(r'(-?\d+(?:\.\d+)?)',v)
        if not m: raise RuntimeError(f'Cannot parse moneystim={v!r}')
        vals.append(float(m.group(1)))
    x=pd.Series(vals,index=df.index); u=np.sort(x.unique())
    if len(u)!=5: raise RuntimeError(f'Expected 5 money levels, got {u}')
    if np.array_equal(u,np.arange(1.,6.)): return x.astype(float)
    mp={v:i+1. for i,v in enumerate(u)}; return x.map(mp).astype(float)

def load_beh(s):
    df=pd.read_csv(behp(s),sep='\t')
    df=df[~df['fixcross.started'].isna()].copy().reset_index(drop=True)
    if len(df)!=125: raise RuntimeError(f'{s}: expected 125 decision trials, got {len(df)}')
    df['trialsnum']=np.arange(1,126)
    df['painlevel']=num(df['painlevel'],f'{s} painlevel'); df['moneylevel']=money_level(df)
    rtcol=next((c for c in ['choice_resp.rt','rt','response.rt'] if c in df.columns),None)
    if rtcol is None: raise KeyError(f'{s}: no RT column')
    df['rt']=pd.to_numeric(df[rtcol],errors='coerce')
    df['accepted']=pd.to_numeric(df['accepted'],errors='coerce') if 'accepted' in df else np.nan
    if 'blocks.thisRepN' in df: df['block']=num(df['blocks.thisRepN'],f'{s} block').astype(int)
    else: df['block']=np.arange(125)//25
    out=df[['trialsnum','painlevel','moneylevel','block','rt','accepted']].copy(); out.attrs['rtcol']=rtcol; return out

def load_meta(s):
    """
    Reconstruct the decision-cue metadata directly from the cleaned raw,
    decision events TSV and behaviour TSV.

    This deliberately mirrors eeg/04_eeg_erp_prep.py for the pieces the TFR
    pipeline actually needs:
      - cue event = off+
      - chronological trialsnum = 1..125
      - exact event sample from events.tsv
      - badtrial from a 150-uV peak-to-peak rejection rule on the ERP-style
        -0.2..1.5-s cue epoch after a 30-Hz low-pass and -0.2..0 baseline

    No decision ERP FIF is required.
    """
    evp=eventsp(s)
    evdf=pd.read_csv(evp,sep='\t')
    if 'trial_type' not in evdf or 'sample' not in evdf:
        raise KeyError(f'{s}: decision events TSV needs trial_type and sample')
    cue=evdf[evdf['trial_type'].astype(str)=='off+'].copy()
    cue=cue.sort_values('sample').reset_index(drop=True)
    if len(cue)!=125:
        raise RuntimeError(f'{s}: expected 125 off+ decision cues, got {len(cue)}')
    cue['sample']=pd.to_numeric(cue['sample'],errors='coerce')
    if cue['sample'].isna().any():
        raise RuntimeError(f'{s}: nonnumeric off+ sample values')
    cue['sample']=cue['sample'].astype(int)

    md=pd.DataFrame({
        'trialsnum':np.arange(1,126,dtype=int),
        'sample':cue['sample'].to_numpy(int),
    })

    # Recreate the ERP-prep badtrial flag from the cleaned decision raw.
    rp=rawp(s)
    if not rp.exists():
        raise FileNotFoundError(rp)
    raw_qc=mne.io.read_raw_fif(str(rp),preload=True,verbose='ERROR')
    raw_qc.drop_channels([c for c in DROP if c in raw_qc.ch_names])
    raw_qc.pick('eeg',exclude=[])
    raw_qc.filter(l_freq=None,h_freq=30.,method='fir',verbose='ERROR')

    # Use unique event IDs exactly as the ERP-prep single-trial construction did.
    ev=np.column_stack([
        md['sample'].to_numpy(int),
        np.zeros(125,dtype=int),
        np.arange(1,126,dtype=int),
    ])
    event_id={f'trial_{i:03d}':i for i in range(1,126)}
    qc_ep=mne.Epochs(
        raw_qc,ev,event_id=event_id,
        tmin=-.2,tmax=1.5,baseline=(-.2,0),
        metadata=md.copy(),preload=True,verbose='ERROR'
    )
    if len(qc_ep)!=125:
        raise RuntimeError(
            f'{s}: ERP-style QC epoching retained {len(qc_ep)}/125 trials; '
            'inspect raw annotations/events before continuing'
        )
    tmp=qc_ep.copy().drop_bad(reject=dict(eeg=150e-6))
    bad_flags=np.asarray([1 if len(x)>0 else 0 for x in tmp.drop_log],dtype=int)
    if len(bad_flags)!=125:
        raise RuntimeError(f'{s}: badtrial flag length {len(bad_flags)} != 125')
    md['badtrial']=bad_flags

    beh=load_beh(s)
    rtcol=beh.attrs['rtcol']
    z=md.merge(beh,on='trialsnum',how='left',validate='1:1')
    if z[['painlevel','moneylevel','block']].isna().any().any():
        raise RuntimeError(f'{s}: behavior merge failed')
    z['invalid_rt']=(~np.isfinite(pd.to_numeric(z['rt'],errors='coerce').to_numpy(float))).astype(int)
    z.attrs['rtcol']=rtcol
    return z

def read_ref(s):
    with open(refp(s),'r',encoding='utf-8') as f: q=json.load(f)
    need=['working_epoch_s','epoch_resample_hz','frequencies_hz','tfr_decim','baseline_s','saved_time_range_s','time_bin_s','frequency_chunk_size']
    miss=[k for k in need if k not in q]
    if miss: raise KeyError(f'{s}: passive params missing {miss}')
    return q

def bins(times,a,b,w):
    edges=np.arange(a,b+w*.5,w)
    if edges[-1]<b-1e-9: edges=np.append(edges,b)
    if abs(edges[-1]-b)>1e-9: edges=np.append(edges[edges<b],b)
    ctr=[]; masks=[]
    for i in range(len(edges)-1):
        lo,hi=float(edges[i]),float(edges[i+1]); idx=np.where((times>=lo-1e-12)&(times<(hi-1e-12 if i<len(edges)-2 else hi+1e-12)))[0]
        if len(idx)==0: raise RuntimeError(f'Empty time bin {lo},{hi}')
        ctr.append((lo+hi)/2); masks.append(idx)
    return np.asarray(ctr),masks

def atomic(path,a):
    t=Path(str(path)+'.tmp');
    with open(t,'wb') as f: np.save(f,a,allow_pickle=False)
    os.replace(t,path)

def main():
    a=cli(); s=a.subject; q=read_ref(s); out=outd(s); out.mkdir(parents=True,exist_ok=True); stem=f'{s}_decision_cue'
    P={k:out/f'{stem}_{v}' for k,v in {'power':'power.npy','times':'times.npy','freqs':'freqs.npy','meta':'metadata.csv','channels':'channels.json','params':'params.json','summary':'prep_summary.json'}.items()}
    if any(p.exists() for p in P.values()) and not a.overwrite: raise FileExistsError(f'{s}: outputs exist; use --overwrite')
    allmd=load_meta(s); rtcol=allmd.attrs['rtcol']; good=allmd[(allmd.badtrial==0)&(allmd.invalid_rt==0)].copy().reset_index(drop=True)
    raw=mne.io.read_raw_fif(str(rawp(s)),preload=True,verbose='ERROR'); raw.drop_channels([c for c in DROP if c in raw.ch_names]); raw.pick('eeg',exclude=[])
    ev=np.column_stack([good['sample'].to_numpy(int),np.zeros(len(good),int),np.ones(len(good),int)])
    tmin,tmax=map(float,q['working_epoch_s']); ep=mne.Epochs(raw,ev,event_id={'cue':1},tmin=tmin,tmax=tmax,baseline=None,metadata=good,preload=True,reject=None,reject_by_annotation=True,verbose='ERROR')
    if len(ep)!=len(good): raise RuntimeError(f'{s}: long epoching dropped trials')
    ep.resample(float(q['epoch_resample_hz']),npad='auto',verbose='ERROR'); data=ep.get_data(copy=False).astype(np.float64,copy=False)
    freqs=np.asarray(q['frequencies_hz'],float); decim=int(q['tfr_decim']); tt=ep.times[::decim]; bl=tuple(map(float,q['baseline_s'])); bm=(tt>=bl[0]-1e-12)&(tt<=bl[1]+1e-12)
    ctr,masks=bins(tt,*map(float,q['saved_time_range_s']),float(q['time_bin_s'])); final=np.empty((len(ep),len(ep.ch_names),len(freqs),len(ctr)),np.float32); chunk=int(q['frequency_chunk_size'])
    print(f'[{s}] trials={len(ep)} channels={len(ep.ch_names)} freqs={len(freqs)} bins={len(ctr)}',flush=True)
    for st in range(0,len(freqs),chunk):
        en=min(st+chunk,len(freqs)); ff=freqs[st:en]; print(f'[{s}] {ff[0]:.0f}-{ff[-1]:.0f} Hz',flush=True)
        pw=tfr_array_morlet(data,sfreq=float(ep.info['sfreq']),freqs=ff,n_cycles=ff*.5,output='power',use_fft=True,zero_mean=True,decim=decim,n_jobs=max(1,a.n_jobs),verbose=False)
        lp=np.log(np.maximum(pw,LOG_FLOOR)); norm=lp-lp[...,bm].mean(axis=-1,keepdims=True)
        for bi,idx in enumerate(masks): final[:,:,st:en,bi]=norm[...,idx].mean(axis=-1).astype(np.float32)
        del pw,lp,norm
    atomic(P['power'],final); atomic(P['times'],ctr.astype(float)); atomic(P['freqs'],freqs.astype(float)); good.to_csv(P['meta'],index=False)
    P['channels'].write_text(json.dumps(ep.ch_names,indent=2))
    pars={'passive_parameter_reference':str(refp(s)),'rt_source_column':rtcol,'working_epoch_s':q['working_epoch_s'],'epoch_resample_hz':q['epoch_resample_hz'],'frequencies_hz':q['frequencies_hz'],'n_cycles_rule':'n_cycles = frequency * 0.5','tfr_decim':q['tfr_decim'],'baseline_s':q['baseline_s'],'saved_time_range_s':q['saved_time_range_s'],'time_bin_s':q['time_bin_s'],'metadata_source':'decision events.tsv + decision behaviour.tsv; badtrial reconstructed with ERP-prep rule','badtrial_rule':'30-Hz LP; cue epoch -0.2..1.5 s; baseline -0.2..0; EEG reject 150e-6','data_shape_order':['trial','channel','frequency','time']}; P['params'].write_text(json.dumps(pars,indent=2))
    sm={'participant':s,'n_tfr_trials':len(good),'n_badtrial':int((allmd.badtrial==1).sum()),'n_invalid_rt':int(allmd.invalid_rt.sum()),'rt_min_s':float(good.rt.min()),'rt_p05_s':float(good.rt.quantile(.05)),'rt_median_s':float(good.rt.median()),'rt_p95_s':float(good.rt.quantile(.95))}; P['summary'].write_text(json.dumps(sm,indent=2)); print(f'[{s}] DONE',flush=True)
if __name__=='__main__': main()
