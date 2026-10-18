#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE=Path(os.getenv('basepath',Path(__file__).parent.parent.parent)); PART=BASE/'participants.tsv'
OUT=BASE/'derivatives'/'statistics'/'tfr_mvpa_decision_native'; SCORES=OUT/'participant_scores'
ATTRS=['pain','money']; BANDS={'theta':(4.,7.),'alpha':(8.,13.),'beta':(14.,30.),'low_gamma':(31.,45.)}
ALPHA=.05; SEED=23; TMIN,TMAX=0.,1.10; HALF=5.*(.5/(2*np.pi))

def cli():
 p=argparse.ArgumentParser(); p.add_argument('--n-permutations',type=int,default=5000); p.add_argument('--n-jobs',type=int,default=max(1,int(os.getenv('SLURM_CPUS_PER_TASK','1')))); p.add_argument('--overwrite-scores',action='store_true'); p.add_argument('--group-only',action='store_true'); return p.parse_args()
def paths(s):
 d=BASE/'derivatives'/s/'eeg'/'tfr_decision'; st=f'{s}_decision_cue'; return {'power':d/f'{st}_power.npy','times':d/f'{st}_times.npy','freqs':d/f'{st}_freqs.npy','meta':d/f'{st}_metadata.csv'}
def scorep(s): return SCORES/f'{s}_decision_native_mvpa.npz'
def subs():
 ss=sorted(pd.read_csv(PART,sep='\t').participant_id.dropna().astype(str).unique())
 for s in ss:
  miss=[str(p) for p in paths(s).values() if not p.exists()]
  if miss: raise FileNotFoundError(f'{s}: missing\n'+'\n'.join(miss))
 return ss
def model(): return make_pipeline(StandardScaler(),Ridge(alpha=1.))
def corr(y,p): return float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else np.nan

def nuisance(rt,responded):
 lr=np.log(np.asarray(rt,float)); return np.column_stack([lr,lr**2,np.asarray(responded,float)])
def resid(Xtr,Xte,Ztr0,Zte0):
 mu=Ztr0.mean(0); sd=Ztr0.std(0); use=sd>1e-12
 if not np.any(use): return Xtr,Xte
 Ztr=(Ztr0[:,use]-mu[use])/sd[use]; Zte=(Zte0[:,use]-mu[use])/sd[use]; Ztr=np.c_[np.ones(len(Ztr)),Ztr]; Zte=np.c_[np.ones(len(Zte)),Zte]
 B=np.linalg.lstsq(Ztr,Xtr,rcond=None)[0]; return Xtr-Ztr@B,Xte-Zte@B

def score(X,y,g,rt,responded,rtres=False):
 pred=np.full(len(y),np.nan); cv=GroupKFold(n_splits=len(np.unique(g)))
 for tr,te in cv.split(X,y,g):
  A=np.asarray(X[tr],float); B=np.asarray(X[te],float)
  if rtres: A,B=resid(A,B,nuisance(rt[tr],responded[tr]),nuisance(rt[te],responded[te]))
  m=model().fit(A,y[tr]); pred[te]=m.predict(B)
 if not np.isfinite(pred).all(): raise RuntimeError('nonfinite OOF predictions')
 return corr(y,pred)

def one_subject(s):
 p=paths(s); power=np.load(p['power'],mmap_mode='r'); times=np.load(p['times']).astype(float); freqs=np.load(p['freqs']).astype(float); md=pd.read_csv(p['meta'])
 pain=pd.to_numeric(md.painlevel,errors='coerce').to_numpy(float); money=pd.to_numeric(md.moneylevel,errors='coerce').to_numpy(float); g=pd.to_numeric(md.block,errors='coerce').to_numpy(int); rt=pd.to_numeric(md.rt,errors='coerce').to_numpy(float)
 if not (np.isfinite(pain).all() and np.isfinite(money).all() and np.isfinite(rt).all()): raise RuntimeError(f'{s}: bad metadata')
 resp=np.stack([rt <= (t+HALF) for t in times],axis=1)
 raw=np.empty((2,len(freqs),len(times)),np.float32); rr=np.empty_like(raw)
 for ai,y in enumerate([pain,money]):
  for fi in range(len(freqs)):
   for ti in range(len(times)):
    X=np.asarray(power[:,:,fi,ti],float); raw[ai,fi,ti]=score(X,y,g,rt,resp[:,ti],False); rr[ai,fi,ti]=score(X,y,g,rt,resp[:,ti],True)
 bn=list(BANDS); rawb=np.empty((2,len(bn),len(times)),np.float32); rrb=np.empty_like(rawb)
 for ai,y in enumerate([pain,money]):
  for bi,b in enumerate(bn):
   lo,hi=BANDS[b]; fm=(freqs>=lo)&(freqs<=hi)
   for ti in range(len(times)):
    X=np.asarray(power[:,:,fm,ti],float).mean(2); rawb[ai,bi,ti]=score(X,y,g,rt,resp[:,ti],False); rrb[ai,bi,ti]=score(X,y,g,rt,resp[:,ti],True)
 np.savez_compressed(scorep(s),raw_freq_r=raw,rtres_freq_r=rr,raw_band_r=rawb,rtres_band_r=rrb,times=times,freqs=freqs,band_names=np.asarray(bn),attribute_names=np.asarray(ATTRS),n_trials=np.asarray(len(md)),rt_min=np.asarray(rt.min()),rt_p05=np.asarray(np.quantile(rt,.05)),rt_median=np.asarray(np.median(rt)),response_overlap_fraction=resp.mean(0).astype(np.float32))

def holm(p):
 p=np.asarray(p,float); m=len(p); o=np.argsort(p); r=p[o]; a=np.maximum.accumulate((m-np.arange(m))*r); a=np.clip(a,0,1); z=np.empty(m); z[o]=a; return z
def fisher(x): return np.arctanh(np.clip(x,-.999999,.999999))
def thr(n): return float(stats.t.ppf(1-ALPHA,df=n-1))

def surf_test(x,freqs,times,nperm,njobs):
 tm=(times>=TMIN)&(times<=TMAX); tt=times[tm]; z=fisher(x[:,:,tm]); ns,nf,nt=z.shape
 T,C,P,_=permutation_cluster_1samp_test(z.reshape(ns,nf*nt),threshold=thr(ns),n_permutations=nperm,tail=1,adjacency=combine_adjacency(nf,nt),seed=SEED,n_jobs=njobs,out_type='indices',verbose=False)
 return T.reshape(nf,nt),[np.asarray(c[0],int) for c in C],np.asarray(P,float),tt
def time_test(x,times,nperm,njobs):
 tm=(times>=TMIN)&(times<=TMAX); tt=times[tm]; z=fisher(x[:,tm]); ns=z.shape[0]
 T,C,P,_=permutation_cluster_1samp_test(z,threshold=thr(ns),n_permutations=nperm,tail=1,seed=SEED,n_jobs=njobs,out_type='indices',verbose=False)
 return T,[np.asarray(c[0],int) for c in C],np.asarray(P,float),tt

def surfaces(arr,fam,freqs,times,nperm,njobs):
 store={}; pmin=[]; om=[]; cl=[]
 for ai,a in enumerate(ATTRS):
  T,C,P,tt=surf_test(arr[:,ai],freqs,times,nperm,njobs); pm=float(P.min()) if len(P) else 1.; pmin.append(pm); store[a]=(T,C,P,tt)
 adj=holm(pmin)
 for ai,a in enumerate(ATTRS):
  om.append({'family':fam,'attribute':a,'omnibus_p_within_surface':pmin[ai],'omnibus_p_holm_across_pain_money':float(adj[ai]),'domain_significant_holm_0p05':bool(adj[ai]<.05)})
  T,C,P,tt=store[a]; nf,nt=T.shape
  if adj[ai]<.05:
   for ci,(idx,pv) in enumerate(zip(C,P)):
    if pv>=.05: continue
    fi,ti=np.unravel_index(idx,(nf,nt)); vals=T.reshape(-1)[idx]; pk=int(np.argmax(vals)); pfi,pti=np.unravel_index(int(idx[pk]),(nf,nt))
    cl.append({'family':fam,'attribute':a,'cluster_id':ci,'cluster_p_within_surface':float(pv),'freq_min_hz':float(freqs[fi].min()),'freq_max_hz':float(freqs[fi].max()),'time_start_s':float(tt[ti].min()),'time_end_s':float(tt[ti].max()),'peak_freq_hz':float(freqs[pfi]),'peak_time_s':float(tt[pti]),'peak_t':float(vals[pk]),'omnibus_p_holm_across_pain_money':float(adj[ai])})
 return pd.DataFrame(om),pd.DataFrame(cl)

def bands(arr,fam,times,bnames,nperm,njobs):
 om=[]; cl=[]
 for ai,a in enumerate(ATTRS):
  st=[]; pmin=[]
  for bi,b in enumerate(bnames):
   T,C,P,tt=time_test(arr[:,ai,bi],times,nperm,njobs); pm=float(P.min()) if len(P) else 1.; pmin.append(pm); st.append((b,T,C,P,tt))
  adj=holm(pmin)
  for bi,(b,T,C,P,tt) in enumerate(st):
   om.append({'family':fam,'attribute':a,'band':b,'band_omnibus_p_within_time':pmin[bi],'band_omnibus_p_holm_within_attribute':float(adj[bi]),'band_significant_holm_0p05':bool(adj[bi]<.05)})
   if adj[bi]<.05:
    for ci,(idx,pv) in enumerate(zip(C,P)):
     if pv>=.05: continue
     vals=T[idx]; pk=int(np.argmax(vals)); cl.append({'family':fam,'attribute':a,'band':b,'cluster_id':ci,'cluster_p_within_band_time':float(pv),'time_start_s':float(tt[idx].min()),'time_end_s':float(tt[idx].max()),'peak_time_s':float(tt[int(idx[pk])]),'peak_t':float(vals[pk]),'band_omnibus_p_holm_within_attribute':float(adj[bi])})
 return pd.DataFrame(om),pd.DataFrame(cl)

def main():
 a=cli(); OUT.mkdir(parents=True,exist_ok=True); SCORES.mkdir(parents=True,exist_ok=True); ss=subs()
 if not a.group_only:
  for i,s in enumerate(ss,1):
   if scorep(s).exists() and not a.overwrite_scores: print(f'[{i}/{len(ss)}] {s} SKIP',flush=True); continue
   print(f'[{i}/{len(ss)}] {s}',flush=True); one_subject(s); print(f'[{s}] DONE',flush=True)
 Z=[np.load(scorep(s),allow_pickle=False) for s in ss]; times=Z[0]['times']; freqs=Z[0]['freqs']; bn=Z[0]['band_names'].astype(str).tolist()
 raw=np.stack([z['raw_freq_r'] for z in Z]); rr=np.stack([z['rtres_freq_r'] for z in Z]); rawb=np.stack([z['raw_band_r'] for z in Z]); rrb=np.stack([z['rtres_band_r'] for z in Z]); ov=np.stack([z['response_overlap_fraction'] for z in Z])
 np.savez_compressed(OUT/'group_score_arrays.npz',subjects=np.asarray(ss),times=times,freqs=freqs,band_names=np.asarray(bn),attribute_names=np.asarray(ATTRS),raw_freq_r=raw,rtres_freq_r=rr,raw_band_r=rawb,rtres_band_r=rrb,response_overlap_fraction=ov)
 SO=[];SC=[];BO=[];BC=[]
 for fam,fa,ba in [('raw',raw,rawb),('rt_residualized',rr,rrb)]:
  x,y=surfaces(fa,fam,freqs,times,a.n_permutations,a.n_jobs); SO.append(x); SC.append(y); x,y=bands(ba,fam,times,bn,a.n_permutations,a.n_jobs); BO.append(x); BC.append(y)
 so=pd.concat(SO,ignore_index=True); sc=pd.concat(SC,ignore_index=True) if any(len(x) for x in SC) else pd.DataFrame(); bo=pd.concat(BO,ignore_index=True); bc=pd.concat(BC,ignore_index=True) if any(len(x) for x in BC) else pd.DataFrame()
 so.to_csv(OUT/'surface_omnibus_holm.csv',index=False); sc.to_csv(OUT/'surface_significant_clusters.csv',index=False); bo.to_csv(OUT/'band_omnibus_holm.csv',index=False); bc.to_csv(OUT/'band_significant_clusters.csv',index=False)
 pd.DataFrame([{'participant':s,'n_trials':int(z['n_trials']),'rt_min_s':float(z['rt_min']),'rt_p05_s':float(z['rt_p05']),'rt_median_s':float(z['rt_median'])} for s,z in zip(ss,Z)]).to_csv(OUT/'participant_rt_summary.csv',index=False)
 pd.DataFrame({'time_s':times,'mean_fraction_trials_response_overlaps_wavelet':ov.mean(0),'sem_fraction_trials_response_overlaps_wavelet':stats.sem(ov,axis=0)}).to_csv(OUT/'response_overlap_by_time.csv',index=False)
 pars={'primary':'raw leave-one-block-out ordinal decoding','rt_sensitivity':'training-fold-only EEG residualization for logRT, logRT^2, and response-before-wavelet-end','morlet_half_support_s':HALF,'formal_inference_s':[TMIN,TMAX],'n_permutations':a.n_permutations,'note':'RT control is sensitivity only because RT is downstream and may remove genuine decision variance'}; (OUT/'params.json').write_text(json.dumps(pars,indent=2))
 print('\nPRIMARY RAW FREQ x TIME'); print(so[so.family=='raw'].to_string(index=False)); print('\nRT-RESIDUALIZED FREQ x TIME'); print(so[so.family=='rt_residualized'].to_string(index=False));
 if len(sc): print('\nSIGNIFICANT SURFACE CLUSTERS'); print(sc.to_string(index=False))
 print('\nPRIMARY RAW BANDS'); print(bo[bo.family=='raw'].to_string(index=False)); print('\nRT-RESIDUALIZED BANDS'); print(bo[bo.family=='rt_residualized'].to_string(index=False));
 if len(bc): print('\nSIGNIFICANT BAND CLUSTERS'); print(bc.to_string(index=False))
 print(f'\nOutputs: {OUT}')
if __name__=='__main__': main()
