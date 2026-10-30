"""Daily synthetic cohort calibrated to local public files, not clinical validation."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sqlite3
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'public'
N_VISITS, DAYS = 9, 730


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def calibrate(out, data_dir=None):
    out = Path(out)
    data_root = DATA if data_dir is None else Path(data_dir)
    cycle = data_root / 'NHANES_2017-2018'
    files = [cycle / f'{name}_J.xpt' for name in ('DEMO','GHB','GLU','BMX','FASTQX')]
    frames = [pd.read_sas(path, format='xport') for path in files]
    keep = [['SEQN','RIDAGEYR','RIAGENDR'], ['SEQN','LBXGH'],
            ['SEQN','LBXGLU','WTSAF2YR'], ['SEQN','BMXWT'], ['SEQN','PHAFSTHR']]
    frame = frames[0][keep[0]]
    for table, columns in zip(frames[1:], keep[1:]):
        frame = frame.merge(table[columns], on='SEQN', validate='one_to_one')
    frame = frame.dropna()
    frame = frame[(frame.RIDAGEYR >= 18) & (frame.PHAFSTHR >= 8)
                  & (frame.PHAFSTHR < 24) & (frame.WTSAF2YR > 0)].copy()
    # Donor groups are separated before creating any virtual clone.
    donor_order = np.random.default_rng(12026).permutation(len(frame))
    split = int(.8 * len(frame))
    dev, heldout = frame.iloc[donor_order[:split]], frame.iloc[donor_order[split:]]
    fitx = np.c_[np.ones(len(dev)), dev.LBXGLU, dev.BMXWT]
    weighted_x = fitx * np.sqrt(dev.WTSAF2YR.to_numpy())[:, None]
    weighted_y = dev.LBXGH.to_numpy() * np.sqrt(dev.WTSAF2YR.to_numpy())
    coef = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
    hall = data_root / 'Hall2018_glucotypes' / 'S5_Data_clinical.sqlite'
    with sqlite3.connect(hall) as con:
        hs = pd.read_sql_query('select userID,FBG,mean_glucose,sd_glucose from clinical', con).dropna()
    rng = np.random.default_rng(32026)
    order = rng.permutation(len(hs))
    hs = hs.iloc[order[:int(.8 * len(hs))]]
    bridge = np.linalg.lstsq(np.c_[np.ones(len(hs)), hs.FBG], hs.mean_glucose, rcond=None)[0]
    cv = (hs.sd_glucose / hs.mean_glucose).to_numpy(dtype=np.float32)
    # Fixed source-derived scaling: no statistics from future synthetic dates.
    ymean = float(np.average(dev.LBXGH, weights=dev.WTSAF2YR))
    xmean = np.array([150,30,180,45,30,75,0,0,0,150,35,.5,ymean,180,50,0,0,0,0,0],np.float32)
    xscale = np.array([50,20,60,30,30,20,3,1,1,50,20,.5,1.2,180,20,1,1,1,1,1],np.float32)
    xmean[0], xmean[5], xmean[14] = dev.LBXGLU.mean(), dev.BMXWT.mean(), dev.RIDAGEYR.mean()
    result = {
        'profile':'E1_daily_moment_simulation', 'clinical_validation':False,
        'nhanes_fasting_adults':len(frame),'development_donors':len(dev),
        'heldout_donors':len(heldout),'hall_bridge_donors':len(hs),
        'glucose_weight_linear_coefficients':coef.tolist(),
        'fasting_to_mean_bridge':bridge.tolist(), 'bridge_time_alignment_verified':False,
        'calibration_limitations':['Cross-sectional NHANES, no real repeated A1c.',
            'Hall bridge is a short-record cross-sectional assumption, not a validated timing model.',
            'Daily CGM moments only; no full 5-minute waveform or activity data generated.',
            'Long-term dynamics, participation and sensor bias are scenario assumptions.'],
        'source_hashes':{str(Path('data/public') / p.relative_to(data_root)):sha256(p) for p in files+[hall]},
        'y_mean':ymean,'x_mean':xmean.tolist(),'x_scale':xscale.tolist(),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/'calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dev.reset_index(drop=True),heldout.reset_index(drop=True),cv,result


@dataclass
class World:
    x: np.ndarray
    y: np.ndarray
    draw: np.ndarray
    arrival: np.ndarray
    available: np.ndarray
    dev_n: int
    groups: dict
    metadata: dict

    def save(self,path):
        np.savez_compressed(path,x=self.x,y=self.y,draw=self.draw,arrival=self.arrival,
            available=np.packbits(self.available,axis=1),dev_n=self.dev_n,
            metadata=json.dumps(self.metadata),**{'group_'+k:v for k,v in self.groups.items()})

    @classmethod
    def load(cls,path):
        with np.load(path) as z:
            return cls(z['x'],z['y'],z['draw'],z['arrival'],
                np.unpackbits(z['available'],axis=1)[:,:DAYS].astype(bool),int(z['dev_n']),
                {k[6:]:z[k] for k in z.files if k.startswith('group_')},json.loads(str(z['metadata'])))


def make_world(n,seed,beta,attendance,phi,calibration,progress=print):
    start=time.perf_counter()
    dev,heldout,cv,cal=calibration
    dev_n=int(.8*n)
    rng=np.random.default_rng(seed)
    sources=[]
    for table,count in ((dev,dev_n),(heldout,n-dev_n)):
        prob=table.WTSAF2YR.to_numpy();prob=prob/prob.sum()
        sources.append(table.iloc[rng.choice(len(table),count,p=prob)])
    donors=pd.concat(sources,ignore_index=True)
    fpg=donors.LBXGLU.to_numpy(np.float32)
    weight=donors.BMXWT.to_numpy(np.float32)
    a1c=donors.LBXGH.to_numpy(np.float32)
    age=donors.RIDAGEYR.to_numpy(np.float32)
    q=rng.beta(2,2,n).astype(np.float32)
    sensor=rng.random(n)<.2
    cv_person=rng.choice(cv,n)
    # Persistent sensor/input context, not a claim about a named device's accuracy.
    domain=rng.integers(0,3,n)
    sensor_bias=np.array([-.04,0,.04],np.float32)[domain]
    phase=rng.integers(0,90,n)
    planned=(phase[:,None]+90*np.arange(N_VISITS)[None,:]+rng.integers(-14,15,(n,N_VISITS))).clip(0)
    x=np.full((n,N_VISITS,20),np.nan,np.float32)
    y=np.full((n,N_VISITS),np.nan,np.float32)
    draw=np.full((n,N_VISITS),-1,np.int16)
    arrival=np.full((n,N_VISITS),32767,np.int16)
    available=np.zeros((n,DAYS),bool)
    group={'donor_id':donors.SEQN.to_numpy(np.int32),'high_a1c':a1c>=8,
           'low_engagement':q<.3,'sensor':sensor,'domain':domain.astype(np.int8),
           'baseline_a1c':a1c,'age':age}
    transition=np.array([[.98,.999,1],[.04,.99,1],[.005,.02,1]],np.float32)
    # Separate reproducible block streams; processing order inside each block is fixed.
    for first in range(0,n,4096):
        end=min(first+4096,n);size=end-first;sl=slice(first,end)
        r=np.random.default_rng(np.random.SeedSequence([seed,first,1729]))
        base=fpg[sl];bw=weight[sl];ba=a1c[sl];qq=q[sl]
        z=r.choice(3,size,p=[.6,.3,.1]);alive=np.ones(size,bool)
        u=r.normal(0,.08,size).astype(np.float32)
        v=r.normal(0,1,size).astype(np.float32)
        fast=np.zeros((90,size),np.float32);post=fast.copy();latent=fast.copy()
        weights=fast.copy()
        cg=np.zeros((14,size),np.float32);cgvar=cg.copy()
        fsum=np.zeros(size);fsq=np.zeros(size);fcnt=np.zeros(size,np.int16)
        psum=np.zeros(size);pcnt=np.zeros(size,np.int16)
        csum=np.zeros(size);csq=np.zeros(size);ccnt=np.zeros(size,np.int16)
        wlast=np.full(size,np.nan,np.float32);wday=np.full(size,-1000)
        gday=np.full(size,-1000)
        bridge=np.array(cal['fasting_to_mean_bridge'])
        baseline_mean=bridge[0]+bridge[1]*base
        baseline_mean=np.maximum(baseline_mean,40)
        decay=np.exp(-1/45);den=(1-decay**90)/(1-decay)
        ew=baseline_mean*den
        latent[:]=baseline_mean
        visit_schedule=planned[sl]
        for day in range(DAYS):
            # Predict/draw before today's measurements; only prior days enter features.
            hit_i,hit_j=np.where(visit_schedule==day)
            if len(hit_i):
                attended=r.random(len(hit_i))<expit(logit(attendance)+.7*(qq[hit_i]-.5))
                attended &= alive[hit_i]
                ii,jj=hit_i[attended],hit_j[attended]
                if len(ii):
                    denom=np.maximum(fcnt[ii],1);pdenom=np.maximum(pcnt[ii],1)
                    ff=fsum[ii]/denom
                    fsd=np.sqrt(np.maximum(fsq[ii]/denom-ff**2,0))
                    pp=psum[ii]/pdenom
                    cc=csum[ii]/np.maximum(ccnt[ii],1)
                    csd=np.sqrt(np.maximum(csq[ii]/np.maximum(ccnt[ii],1)-cc**2,0))
                    previous=np.where((arrival[first+ii]<day)&(draw[first+ii]>=0),draw[first+ii],-1).argmax(axis=1)
                    prev_exists=(arrival[first+ii,previous]<day)
                    previous_y=y[first+ii,previous]
                    previous_age=day-draw[first+ii,previous]
                    ordered=weights[(np.arange(90)+day%90)%90][:,ii]
                    first_valid=np.argmax(ordered!=0,axis=0)
                    wfirst=ordered[first_valid,np.arange(len(ii))]
                    wcount=(ordered!=0).sum(axis=0)
                    values=np.column_stack([ff,fsd,pp,fcnt[ii],day-gday[ii],
                        wlast[ii],wlast[ii]-wfirst,np.zeros(len(ii)),np.zeros(len(ii)),
                        cc,csd,ccnt[ii]/14,previous_y,previous_age,age[sl][ii],
                        fcnt[ii]==0,pcnt[ii]==0,(wcount<2)|(day-wday[ii]>30),
                        np.ones(len(ii)),~prev_exists]).astype(np.float32)
                    values[fcnt[ii]==0,0:2]=np.nan
                    values[fcnt[ii]<2,1]=np.nan
                    values[pcnt[ii]==0,2]=np.nan
                    values[day-wday[ii]>30,5]=np.nan
                    values[wcount<2,6]=np.nan
                    values[:,7]=np.nan
                    values[ccnt[ii]==0,9:11]=np.nan
                    values[~prev_exists,12:14]=np.nan
                    change=(ew[ii]/max(den,1e-9))-baseline_mean[ii]
                    yy=ba[ii]+cal['glucose_weight_linear_coefficients'][1]*change+r.normal(0,.12,len(ii))
                    # No current Y enters values. Eligibility still requires some glucose observation.
                    x[first+ii,jj]=values;y[first+ii,jj]=yy
                    draw[first+ii,jj]=day;arrival[first+ii,jj]=day+2
            h=np.clip((ba-5.8)/1.2+u/.08,-3,3)
            alive &= r.random(size)>=.0002
            z=np.sum(r.random(size)[:,None]>transition[z,:2],axis=1)
            p_rec=expit(logit(np.array([.85,.35,.02])[z])+(qq-.5)+beta*h)
            # Some locally recorded data may upload late: simplify to same-day visible data,
            # explicitly logged as the E1 execution profile rather than full design equivalence.
            rec=(r.random(size)<p_rec)&alive
            av=expit(logit(np.array([.7,.25,.02])[z])+(qq-.5)+beta*h)
            available[sl,day]=(r.random(size)<av)&alive
            u=phi*u+.08*np.sqrt(1-phi*phi)*r.normal(size=size)
            v=.995*v+np.sqrt(1-.995**2)*r.normal(size=size)
            true_fast=base*np.exp(u)
            observed=true_fast*np.exp(r.normal(0,.05,size))*(1+sensor_bias[sl])
            bad=r.random(size)<.01
            observed[bad]*=r.choice([18.,1/18],bad.sum())
            rec &= (observed>=20)&(observed<=600)
            now=np.where(rec,observed,0)
            pos=day%90;old=fast[pos].copy()
            fsum+=now-old;fsq+=now**2-old**2;fcnt+=(now!=0).astype(np.int16)-(old!=0).astype(np.int16)
            fast[pos]=now;gday[rec]=day
            pobs=np.where(rec&(r.random(size)<.3),true_fast*1.2*np.exp(r.normal(0,.07,size)),0)
            oldpost=post[pos].copy();psum+=pobs-oldpost;pcnt+=(pobs!=0).astype(np.int16)-(oldpost!=0).astype(np.int16);post[pos]=pobs
            wr=rec&(r.random(size)<.6)
            wn=np.where(wr,bw+v+r.normal(0,.2,size),0)
            weights[pos]=wn;wlast[wr]=wn[wr];wday[wr]=day
            mean_g=np.maximum(40,baseline_mean+bridge[1]*(true_fast-base))
            oldlatent=latent[pos].copy();latent[pos]=mean_g
            ew=decay*ew+mean_g-(decay**90)*oldlatent
            cr=sensor[sl]&(((day-phase[sl])%90)<14)&(r.random(size)>=.1)&alive
            cg_now=np.where(cr,mean_g*(1+sensor_bias[sl]),0)
            var_now=np.where(cr,(mean_g*cv_person[sl])**2+cg_now**2,0)
            cp=day%14;oc=cg[cp].copy();ov=cgvar[cp].copy()
            csum+=cg_now-oc;csq+=var_now-ov;ccnt+=(cg_now!=0).astype(np.int16)-(oc!=0).astype(np.int16)
            cg[cp]=cg_now;cgvar[cp]=var_now
        if first==0 or end==n:
            progress(f'WORLD generated {end}/{n} seed={seed} beta={beta:.3f} attendance={attendance} phi={phi}',flush=True)
    valid=(draw>=90)&(x[:,:,3]>0)
    y[~valid]=np.nan
    raw_mean=np.array(cal['x_mean'],np.float32);scale=np.array(cal['x_scale'],np.float32)
    x=(np.where(np.isfinite(x),x,raw_mean)-raw_mean)/scale
    x=np.clip(x,-10,10).astype(np.float32)
    meta={'n':n,'seed':seed,'beta':beta,'attendance':attendance,'phi':phi,
          'generation_seconds':time.perf_counter()-start,'profile':cal['profile'],
          'valid_labs':int(np.isfinite(y).sum()),'valid_people':int(np.isfinite(y).any(axis=1).sum()),
          'source_donors':cal['development_donors']+cal['heldout_donors'],
          'same_day_measurement_visibility_simplification':True,
          'no_real_longitudinal_validation':True,'label_noise_sd_pp':.12,
          'y_mean':cal['y_mean']}
    return World(x,y,draw,arrival,available,dev_n,group,meta)


def training_schedule(world,k=2000):
    rng=np.random.default_rng(world.metadata['seed']+48001)
    last_consumed=np.full(world.dev_n,-1,np.int16)
    arr=world.arrival[:world.dev_n];valid=np.isfinite(world.y[:world.dev_n])
    result=[];selection_count=np.zeros(len(world.x),np.int32)
    for day in range(90,500):
        latest=np.where(valid&(arr<=day),arr,-1).max(axis=1)
        candidates=np.flatnonzero((latest>last_consumed)&(latest>=day-30)&world.available[:world.dev_n,day])
        invited=rng.choice(candidates,min(len(candidates),int(np.ceil(k/.9))),replace=False)
        done=(rng.random(len(invited))<.9)&(rng.lognormal(np.log(10),.6,len(invited))<=60)
        accepted=invited[done][:k]
        if len(accepted)<20:accepted=np.array([],np.int64)
        last_consumed[accepted]=latest[accepted]
        selection_count[accepted]+=1
        result.append((day,accepted))
    world.groups['training_visits']=selection_count
    world.metadata['accepted_updates']=int(selection_count.sum())
    world.metadata['ever_trained_people']=int((selection_count>0).sum())
    return result
