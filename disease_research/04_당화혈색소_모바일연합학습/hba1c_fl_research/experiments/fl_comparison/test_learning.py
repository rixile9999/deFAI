import json
from types import SimpleNamespace
import numpy as np
import torch

from learning import initial_parameters,local_fit,Learner

torch.manual_seed(91)
dtype=torch.float64
x=torch.randn(3,4,20,dtype=dtype)
y=torch.randn(3,4,dtype=dtype)
mask=torch.tensor([[1,1,1,1],[1,1,0,0],[1,0,0,0]],dtype=dtype)
base=initial_parameters(9,hidden=5,device='cpu',dtype=dtype)
start=[v[None].expand(3,*v.shape).clone() for v in base]
checks=[]
for mu in (0.,.1):
    updated,_,_=local_fit(x,y,mask,start,5,.01,mu)
    for i in range(3):
        one,_,_=local_fit(x[i:i+1],y[i:i+1],mask[i:i+1],[v[i:i+1] for v in start],5,.01,mu)
        for a,b in zip(updated,one):torch.testing.assert_close(a[i],b[0],rtol=1e-10,atol=1e-10)
checks.append('Batched local SGD/FedProx equals separate masked clients')

# Compare BN gradients and unbiased running variance against PyTorch's reference layer.
bnbase=initial_parameters(9,hidden=5,bn=True,device='cpu',dtype=dtype)
bnstart=[v[None].clone() for v in bnbase]
updated,running,_=local_fit(x[:1],y[:1],mask[:1],bnstart,1,.01,
                          bn_state=[torch.zeros(1,5,dtype=dtype),torch.ones(1,5,dtype=dtype)])
layer=torch.nn.BatchNorm1d(5,eps=1e-5,momentum=.1,dtype=dtype)
reference=[v.detach().clone().requires_grad_() for v in bnbase[:4]]
hidden=x[0]@reference[0]+reference[1]
prediction=(torch.relu(layer(hidden))@reference[2]+reference[3]).squeeze(-1)
loss=((prediction-y[0])**2).mean()
grads=torch.autograd.grad(loss,reference+[layer.weight,layer.bias])
for a,p,g in zip(updated,bnbase,grads):torch.testing.assert_close(a[0],p-.01*g,rtol=1e-9,atol=1e-10)
torch.testing.assert_close(running[0][0],layer.running_mean)
torch.testing.assert_close(running[1][0],layer.running_var)
checks.append('BN gradient and running-stat update match torch.nn.BatchNorm1d')

# Single-observation clients retain running statistics; their affine parameters may learn.
single=[v[None].clone() for v in bnbase]
oldmean=torch.randn(1,5,dtype=dtype);oldvar=torch.ones(1,5,dtype=dtype)*1.7
_,after,_=local_fit(x[2:3],y[2:3],mask[2:3],single,3,.01,bn_state=[oldmean,oldvar])
torch.testing.assert_close(after[0],oldmean);torch.testing.assert_close(after[1],oldvar)
checks.append('Single-sample BN freezes statistics instead of computing zero variance')

world=SimpleNamespace(dev_n=6,x=np.zeros((8,1,20)))
batchx=x[:2].float();batchy=y[:2].float();batchmask=mask[:2].float()
for method in ('fedavg','fedprox','scaffold','fedavg_bn','fedbn'):
    a=Learner(world,method,.01,mu=0.,seed=9,hidden=5,device=torch.device('cpu'))
    a.step([0,2],batchx,batchy,batchmask)
    if method=='scaffold':
        a.step([2,3],batchx,batchy,batchmask)
        for c,ci in zip(a.server_c,a.client_c):
            torch.testing.assert_close(c,ci.mean(0),rtol=1e-5,atol=1e-7)
            assert torch.count_nonzero(ci[1])==0
        checks.append('SCAFFOLD population control mean invariant under partial participation')
    if method=='fedbn':
        for state,initial in zip(a.local_bn,[1,0,0,1]):
            torch.testing.assert_close(state[1],torch.full_like(state[1],initial))
        torch.testing.assert_close(a.global_p[4],torch.ones_like(a.global_p[4]))
        torch.testing.assert_close(a.global_p[5],torch.zeros_like(a.global_p[5]))
        checks.append('FedBN excludes all BN affine and running state from server averaging')

a=Learner(world,'fedavg',.01,seed=9,hidden=5,device=torch.device('cpu'))
b=Learner(world,'fedprox',.01,mu=0,seed=9,hidden=5,device=torch.device('cpu'))
a.step([0,2],batchx,batchy,batchmask);b.step([0,2],batchx,batchy,batchmask)
for p,q in zip(a.global_p,b.global_p):torch.testing.assert_close(p,q)
checks.append('FedProx mu=0 exactly matches FedAvg')

# Future and unavailable slots must have no effect, including BN statistics.
altered_x=x.clone();altered_y=y.clone()
altered_x[mask==0]=torch.randn_like(altered_x[mask==0])*1000
altered_y[mask==0]=torch.randn_like(altered_y[mask==0])*1000
for bn in (False,True):
    initial=initial_parameters(9,hidden=5,bn=bn,device='cpu',dtype=dtype)
    copies=[v[None].expand(3,*v.shape).clone() for v in initial]
    state=[torch.zeros(3,5,dtype=dtype),torch.ones(3,5,dtype=dtype)] if bn else None
    for mu,controlled in ((0.,False),(.1,False),(0.,True)):
        control=[torch.randn_like(v)*.01 for v in copies] if controlled else None
        first,stats1,_=local_fit(x,y,mask,copies,5,.001,mu,control,state)
        second,stats2,_=local_fit(altered_x,altered_y,mask,copies,5,.001,mu,control,state)
        for p,q in zip(first,second):torch.testing.assert_close(p,q,rtol=1e-10,atol=1e-10)
        if bn:
            for p,q in zip(stats1,stats2):torch.testing.assert_close(p,q,rtol=1e-10,atol=1e-10)
checks.append('Unavailable future slots cannot change gradients or BN statistics')

print(json.dumps({'passed':len(checks),'checks':checks},indent=2))
