"""Independent batched local SGD, FedProx, SCAFFOLD option II and local BN."""
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

DEVICE=torch.device(os.environ.get('DEFAI_DEVICE', 'cpu'))
DTYPE=torch.float32


def initial_parameters(seed,hidden=64,bn=False,device=DEVICE,dtype=DTYPE):
    generator=torch.Generator(device=device).manual_seed(seed)
    p=[torch.randn(20,hidden,generator=generator,device=device,dtype=dtype)*(2/20)**.5,
       torch.zeros(hidden,device=device,dtype=dtype),
       torch.randn(hidden,1,generator=generator,device=device,dtype=dtype)/hidden**.5,
       torch.zeros(1,device=device,dtype=dtype)]
    if bn:p.extend([torch.ones(hidden,device=device,dtype=dtype),torch.zeros(hidden,device=device,dtype=dtype)])
    return p


def local_fit(x,y,mask,start,epochs,lr,mu=0.,control=None,bn_state=None,momentum=.1):
    """One parameter copy per client. Sum of individual mean losses, not global mean."""
    if epochs < 1 or lr <= 0 or not np.isfinite(lr) or mu < 0 or not np.isfinite(mu):
        raise ValueError('epochs >= 1, finite positive lr and finite nonnegative mu are required')
    if x.ndim != 3 or y.shape != x.shape[:2] or mask.shape != y.shape or len(x) == 0:
        raise ValueError('Expected nonempty x[clients,samples,features] and aligned y/mask')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(mask).all():
        raise ValueError('Training tensors must be finite; zero-fill unavailable targets before masking')
    if not ((mask == 0) | (mask == 1)).all() or (mask.sum(1) == 0).any():
        raise ValueError('Each client needs at least one observed sample and a binary mask')
    count=len(x)
    p=[v.detach().clone().requires_grad_(True) for v in start]
    n=mask.sum(1).clamp_min(1)
    initial_loss=None
    running=None if bn_state is None else [v.detach().clone() for v in bn_state]
    for _ in range(epochs):
        h=torch.bmm(x,p[0])+p[1][:,None,:]
        if running is not None:
            mean=(h*mask[:,:,None]).sum(1)/n[:,None]
            var=((h-mean[:,None,:])**2*mask[:,:,None]).sum(1)/n[:,None]
            enough=(n>=2)[:,None]
            used_mean=torch.where(enough,mean,running[0])
            used_var=torch.where(enough,var,running[1])
            h=(h-used_mean[:,None,:])/torch.sqrt(used_var[:,None,:]+1e-5)
            h=h*p[4][:,None,:]+p[5][:,None,:]
            with torch.no_grad():
                running[0]=torch.where(enough,(1-momentum)*running[0]+momentum*mean.detach(),running[0])
                unbiased=var.detach()*n[:,None]/(n-1).clamp_min(1)[:,None]
                running[1]=torch.where(enough,(1-momentum)*running[1]+momentum*unbiased,running[1])
        prediction=(torch.bmm(torch.relu(h),p[2])+p[3][:,None,:]).squeeze(-1)
        individual=(((prediction-y)**2)*mask).sum(1)/n
        if initial_loss is None:initial_loss=individual.detach()
        loss=individual.sum()
        if mu:
            loss=loss+mu*.5*sum(((a-b)**2).sum() for a,b in zip(p,start))
        grad=torch.autograd.grad(loss,p)
        if control is not None:
            grad=[g+c for g,c in zip(grad,control)]
        p=[(a-lr*g).detach().requires_grad_(True) for a,g in zip(p,grad)]
    return [v.detach() for v in p],running,initial_loss


class Learner:
    def __init__(self,world,method,lr,mu=0.,seed=0,hidden=64,device=DEVICE):
        self.method,self.lr,self.mu=method,lr,mu
        self.bn=method in ('fedavg_bn','fedbn')
        self.global_p=initial_parameters(seed,hidden,self.bn,device)
        self.n=world.dev_n;self.total_n=len(world.x);self.device=device
        self.global_running=[torch.zeros(hidden,device=device),torch.ones(hidden,device=device)] if self.bn else None
        self.local_bn=None
        if method=='fedbn':
            self.local_bn=[torch.ones(self.total_n,hidden),torch.zeros(self.total_n,hidden),
                           torch.zeros(self.total_n,hidden),torch.ones(self.total_n,hidden)]
        self.client_c=None;self.server_c=None
        if method=='scaffold':
            self.client_c=[torch.zeros((self.n,*p.shape)) for p in self.global_p]
            self.server_c=[torch.zeros_like(p) for p in self.global_p]
        self.epochs=5;self.steps=0;self.bytes=0;self.single_sample_updates=0

    def step(self,ids,x,y,mask):
        if len(ids)==0:return None
        host_ids=torch.as_tensor(ids,dtype=torch.long,device='cpu')
        ids=host_ids.to(self.device)
        size=len(ids)
        start=[p.unsqueeze(0).expand(size,*p.shape).clone() for p in self.global_p]
        state=None
        if self.bn:
            state=[p.unsqueeze(0).expand(size,*p.shape).clone() for p in self.global_running]
        if self.local_bn is not None:
            start[4],start[5]=self.local_bn[0][host_ids].to(self.device),self.local_bn[1][host_ids].to(self.device)
            state=[self.local_bn[2][host_ids].to(self.device),self.local_bn[3][host_ids].to(self.device)]
        correction=None
        if self.client_c is not None:
            correction=[c.unsqueeze(0)-ci[host_ids].to(self.device) for c,ci in zip(self.server_c,self.client_c)]
        updated,running,loss=local_fit(x,y,mask,start,self.epochs,self.lr,
                                     self.mu if self.method=='fedprox' else 0.,correction,state)
        with torch.no_grad():
            if self.client_c is not None:
                for index,(old,new) in enumerate(zip(start,updated)):
                    delta_c=-self.server_c[index].unsqueeze(0)+(old-new)/(self.epochs*self.lr)
                    self.client_c[index][host_ids]+=delta_c.cpu()
                    # Fixed-population correction: server c remains the mean of all client c_i.
                    self.server_c[index]+=delta_c.sum(0)/self.n
            shared=4 if self.method=='fedbn' else len(updated)
            for index in range(shared):self.global_p[index].copy_(updated[index].mean(0))
            if self.local_bn is not None:
                for target,value in zip(self.local_bn,[updated[4],updated[5],running[0],running[1]]):target[host_ids]=value.cpu()
            elif self.bn:
                for target,value in zip(self.global_running,running):target.copy_(value.mean(0))
        transmitted=sum(p.numel() for p in self.global_p[:shared])
        if self.method=='scaffold':transmitted*=2
        elif self.method=='fedavg_bn':transmitted+=sum(p.numel() for p in self.global_running)
        self.bytes+=2*4*size*transmitted
        self.steps+=1
        self.single_sample_updates+=int((mask.sum(1)<2).sum().item())
        if not torch.isfinite(updated[0]).all():raise FloatingPointError('Non-finite local update')
        return float(loss.mean().item())

    def predict(self,x,people):
        with torch.no_grad():
            p=self.global_p
            h=x@p[0]+p[1]
            if self.bn:
                if self.local_bn is not None:
                    gamma,beta,mean,var=[a[people.cpu()].to(self.device) for a in self.local_bn]
                else:gamma,beta,mean,var=p[4],p[5],self.global_running[0],self.global_running[1]
                h=(h-mean)/torch.sqrt(var+1e-5)*gamma+beta
            return (torch.relu(h)@p[2]+p[3]).squeeze(-1)

    def save(self,path):
        values={f'p_{j}':v.cpu().numpy() for j,v in enumerate(self.global_p)}
        if self.global_running is not None:values.update({f'running_{j}':v.cpu().numpy() for j,v in enumerate(self.global_running)})
        if self.local_bn is not None:values.update({f'bn_{j}':v.cpu().numpy() for j,v in enumerate(self.local_bn)})
        np.savez_compressed(path,**values)

    def load(self,path):
        with np.load(path) as z:
            for j,p in enumerate(self.global_p):p.copy_(torch.as_tensor(z[f'p_{j}'],device=self.device))
            if self.global_running is not None:
                for j,p in enumerate(self.global_running):p.copy_(torch.as_tensor(z[f'running_{j}'],device=self.device))
            if self.local_bn is not None:
                for j,p in enumerate(self.local_bn):p.copy_(torch.as_tensor(z[f'bn_{j}']))


def evaluate(learner,world,split,save_errors=None):
    if split=='validation':keep=(world.draw>=500)&(world.draw<=589)&(world.arrival<=589)
    elif split=='test':keep=(world.draw>=590)&(world.draw<=729)
    else:raise ValueError(split)
    keep &= np.isfinite(world.y)
    people,visits=np.where(keep)
    abs_sum=np.zeros(len(world.x),np.float64);sq_sum=abs_sum.copy();signed_sum=abs_sum.copy()
    counts=np.zeros(len(world.x),np.int32)
    all_errors=[]
    for offset in range(0,len(people),16384):
        ii=people[offset:offset+16384];jj=visits[offset:offset+16384]
        xx=torch.as_tensor(world.x[ii,jj],device=learner.device)
        pred=learner.predict(xx,torch.as_tensor(ii,device=learner.device)).cpu().numpy()+world.metadata['y_mean']
        errors=pred-world.y[ii,jj]
        if not np.isfinite(errors).all():raise FloatingPointError('Non-finite predictions')
        np.add.at(abs_sum,ii,np.abs(errors));np.add.at(sq_sum,ii,errors**2)
        np.add.at(signed_sum,ii,errors);np.add.at(counts,ii,1)
        all_errors.append(errors)
    per_person=abs_sum/np.maximum(counts,1)
    groups={'all':np.ones(len(counts),bool),'seen':np.arange(len(counts))<world.dev_n,
            'new':np.arange(len(counts))>=world.dev_n,'high_a1c':world.groups['high_a1c'],
            'low_engagement':world.groups['low_engagement'],
            'never_trained':world.groups.get('training_visits',np.zeros(len(counts)))==0,
            'ever_trained':world.groups.get('training_visits',np.zeros(len(counts)))>0,
            'seen_never_trained':(np.arange(len(counts))<world.dev_n)&(world.groups.get('training_visits',np.zeros(len(counts)))==0)}
    result={}
    for name,group in groups.items():
        valid=group&(counts>0)
        result[name]={'people':int(valid.sum()),'population':int(group.sum()),
            'mae_pp':float(per_person[valid].mean()) if valid.any() else None,
            'rmse_pp':float(np.sqrt((sq_sum[valid]/counts[valid]).mean())) if valid.any() else None,
            'bias_pp':float((signed_sum[valid]/counts[valid]).mean()) if valid.any() else None,
            'p90_person_mae_pp':float(np.quantile(per_person[valid],.9)) if valid.any() else None}
    if save_errors is not None:
        np.savez_compressed(save_errors,mae=per_person.astype(np.float32),counts=counts)
    return result


def run_training(world,schedule,method,lr,mu,out,seed,round_limit=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    learner=Learner(world,method,lr,mu,seed)
    curve=[];validation_curve=[]
    if DEVICE.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for index,(day,ids) in enumerate(schedule):
        if round_limit is not None and index>=round_limit:break
        if len(ids):
            xx=torch.as_tensor(world.x[ids],device=DEVICE)
            yy=torch.as_tensor(np.nan_to_num(world.y[ids]-world.metadata['y_mean']),device=DEVICE)
            valid=np.isfinite(world.y[ids])
            mask=torch.as_tensor(valid&(world.arrival[ids]<=day)&(world.arrival[ids]>=day-365),device=DEVICE,dtype=torch.float32)
            loss=learner.step(ids,xx,yy,mask)
            if index%20==0:curve.append({'day':day,'round':index,'people':len(ids),'initial_local_mse':loss})
        if index in (49,149,249,349,409):
            validation_curve.append({'day':day,'round':index+1,
                                     'offline_validation_mae_pp':evaluate(learner,world,'validation')['all']['mae_pp']})
    if DEVICE.type=='cuda':torch.cuda.synchronize()
    training_seconds=time.perf_counter()-start
    validation=evaluate(learner,world,'validation')
    learner.save(out/'model.npz')
    result={'method':method,'lr':lr,'mu':mu,'seed':seed,'world':world.metadata,
        'actual_aggregations':learner.steps,'single_sample_updates':learner.single_sample_updates,
        'training_seconds':training_seconds,'validation':validation,'curve':curve,
        'validation_curve':validation_curve,
        'payload_bytes_theoretical':learner.bytes,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if DEVICE.type=='cuda' else 0,
        'process_rss_bytes':psutil.Process().memory_info().rss,
        'parameters':sum(p.numel() for p in learner.global_p),
        'aggregation':'equal_completed_client',
        'scaffold_control_mean_max_error':max(float((c-ci.mean(0).to(c.device)).abs().max().item()) for c,ci in zip(learner.server_c,learner.client_c)) if method=='scaffold' else None,
        'device':str(DEVICE),'persistent_client_state_device':'cpu','gpu_memory_fraction_cap':None,
        'bn_single_sample_rule':'freeze_running_stats_and_use_prior_stats_when_n_lt_2' if learner.bn else None}
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return learner,result
