"""Core VADiffAD noise field, deterministic diffusion paths, and normal reference model."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    window: int = 64
    stride: int = 16
    eval_stride: int = 100
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = .02
    rho: float = .5
    cmin: float = .5
    cmax: float = 2.
    coarse_steps: int = 40
    tau_r: float = .2
    tau_delta: float = .02
    nodes: int = 10  # K responses, K+1 states, 2K-1 signature dimensions
    shrinkage: float = .1
    eps: float = 1e-6
    q: int = 3
    hidden: int = 64
    epochs: int = 50
    batch: int = 64
    lr: float = .001
    seed: int = 42
    uniform: bool = False
    fixed_depth: bool = False
    fixed_depth_fraction: float = 1.0
    weights: tuple = (1/3,1/3,1/3)
    revision: str = "revised"
    activity_floor: float = 1e-4
    score_floor_ratio: float = .05
    reconstruction_steps: tuple = (100, 250, 500)
    reconstruction_repeats: int = 2
    context_weight: float = .5
    mask_probability: float = .25
    prototype_points: int = 50000
    calibration_quantile: float = .99
    geometry_version: str = "v1"
    prototype_components: int = 1
    path_response: str = "epsilon"
    # The original absolute rule is retained for backward compatibility.  The
    # relative rule supports a controlled adaptive-range experiment without
    # assuming that retention has the same numerical scale on every dataset.
    adaptive_depth_rule: str = "absolute"
    relative_destruction_fraction: float = .5
    minimum_depth_fraction: float = .05
    retention_reference_median: tuple = ()
    retention_reference_scale: tuple = ()
    retention_deviation_threshold: float = 0.0
    depth_conditioned_prototype: bool = False


def variable_coefficients(x, cfg):
    """Equations (2)-(6): difference MAD, power ratio, bounded mean normalization."""
    delta=np.diff(x,axis=0)
    centered=abs(delta-np.median(delta,axis=0))
    r=np.median(centered,axis=0)
    # Quantized telemetry often has a zero MAD although it is not constant.
    # The revised method uses a deterministic hierarchy that preserves MAD
    # whenever it is informative and only falls back for degenerate columns.
    if cfg.revision == "revised":
        q75=np.quantile(centered,.75,axis=0)
        sd=delta.std(axis=0)
        r=np.where(r>cfg.eps,r,np.where(q75>cfg.eps,q75,sd))
        active=r>cfg.eps
        reference=float(np.median(r[active])) if active.any() else 1.
        r=np.where(active,r,reference)
    else:
        reference=float(np.median(r))
    raw=((reference+cfg.eps)/(r+cfg.eps))**cfg.rho
    # Project by a common positive scale so bounds AND mean=1 are satisfied.
    if not 0<cfg.cmin<=1<=cfg.cmax:raise ValueError('bounds must contain one')
    lo,hi=0.,max(1.,cfg.cmax/raw.min())*2
    for _ in range(80):
        mid=(lo+hi)/2
        if np.clip(mid*raw,cfg.cmin,cfg.cmax).mean()<1:lo=mid
        else:hi=mid
    c=np.clip((lo+hi)/2*raw,cfg.cmin,cfg.cmax)
    return (np.ones_like(c) if cfg.uniform else c).astype('float32')


class Schedule:
    def __init__(self,c,cfg,device):
        self.cfg=cfg
        self.c=torch.as_tensor(c,device=device)
        beta=torch.linspace(cfg.beta_start,cfg.beta_end,cfg.timesteps,device=device,dtype=torch.float64)[:,None]*self.c.double()[None]
        if (beta<=0).any() or (beta>=1).any():raise ValueError('invalid beta')
        self.log_alpha=torch.cat([torch.zeros_like(beta[:1]),torch.log1p(-beta).cumsum(0)],0).float()

    def alpha(self,t):
        """Discrete cumulative product; fractional nodes use log-alpha interpolation."""
        t=t.clamp(0,self.cfg.timesteps)
        lo=t.long();hi=(lo+1).clamp(max=self.cfg.timesteps);w=(t-lo)[:,None]
        return torch.exp(self.log_alpha[lo]*(1-w)+self.log_alpha[hi]*w)

    def coordinates(self,t):
        a=self.alpha(t)
        return a,torch.sqrt((1-a)/a)


class ResidualBlock(nn.Module):
    def __init__(self,h,dilation):
        super().__init__()
        self.norm=nn.GroupNorm(8,h)
        self.conv=nn.Conv1d(h,h,3,padding=dilation,dilation=dilation)
        self.out=nn.Conv1d(h,h,1)
    def forward(self,x,embedding):
        return (x+self.out(torch.nn.functional.silu(self.conv(torch.nn.functional.silu(self.norm(x)+embedding[:,:,None])))))/2**.5


class NormalScoreField(nn.Module):
    """Noise predictor epsilon_theta(X_t,t,c), equation (12)."""
    def __init__(self,d,cfg):
        super().__init__();self.cfg=cfg
        h=cfg.hidden
        self.input=nn.Conv1d(2*d,h,1)
        self.embed=nn.Sequential(nn.Linear(32,h),nn.SiLU(),nn.Linear(h,h))
        self.blocks=nn.ModuleList([ResidualBlock(h,dil) for dil in [1,2,4,8]])
        self.output=nn.Conv1d(h,d,1)
    def forward(self,x,t,c):
        f=torch.exp(torch.linspace(0,np.log(1000),16,device=x.device))
        angle=t[:,None]/self.cfg.timesteps*f[None]
        emb=self.embed(torch.cat([angle.sin(),angle.cos()],-1))
        cc=c.log()[None,None].expand(x.shape[0],x.shape[1],-1)
        h=self.input(torch.cat([x,cc],-1).transpose(1,2))
        for block in self.blocks:h=block(h,emb)
        return self.output(h).transpose(1,2)


class ContextReconstructor(nn.Module):
    """Bidirectional masked-context head trained only on normal windows."""
    def __init__(self,d,cfg):
        super().__init__()
        h=cfg.hidden
        self.input=nn.Conv1d(2*d,h,3,padding=1)
        self.blocks=nn.ModuleList([ResidualBlock(h,dilation) for dilation in [1,2,4,8,16]])
        self.output=nn.Conv1d(h,d,1)

    def forward(self,x,mask):
        observed=x*(1-mask)
        h=self.input(torch.cat([observed,mask],-1).transpose(1,2))
        embedding=torch.zeros(h.shape[0],h.shape[1],device=h.device,dtype=h.dtype)
        for block in self.blocks:
            h=block(h,embedding)
        return self.output(h).transpose(1,2)


def structure_retention(original,current,eps=1e-6):
    """Eq (13), absolute cosine of separately centered variable sequences.

    Scaling to unit std cancels algebraically in cosine. Constant variables
    contribute zero as prescribed by the epsilon-stabilized denominator.
    """
    a=original-original.mean(1,keepdim=True)
    b=current-current.mean(1,keepdim=True)
    return ((a*b).sum(1).abs()/(a.square().sum(1).sqrt()*b.square().sum(1).sqrt()+eps)).mean(-1)


def activity_aware_retention(original,current,eps=1e-6,activity_floor=1e-4):
    """Revised retention: exclude constant channels from the window average.

    A channel with no centered energy contains no structure to retain. Counting
    it as zero made sparse telemetry appear destroyed at t=0. Active channels
    still use the exact absolute centered cosine from equation (13).
    """
    a=original-original.mean(1,keepdim=True)
    b=current-current.mean(1,keepdim=True)
    ea=a.square().sum(1);eb=b.square().sum(1)
    cosine=(a*b).sum(1).abs()/(ea.sqrt()*eb.sqrt()+eps)
    active=ea>activity_floor
    return (cosine*active).sum(-1)/(active.sum(-1).clamp(min=1))


def first_saturation(times,retention,tau_r,tau_delta):
    """Eq (14)-(15), first joint crossing; no crossing => final time."""
    delta=abs(retention[:,1:]-retention[:,:-1])
    valid=(retention[:,1:]<=tau_r)&(delta<=tau_delta)
    first=valid.to(torch.int64).argmax(1)+1
    return torch.where(valid.any(1),times[first],times[-1]),valid.any(1)


def relative_destruction_depth(
    times, retention, fraction=.5, minimum_fraction=.05, eps=1e-6,
):
    """Choose an input-relative endpoint from structural destruction.

    Retention may oscillate along the pilot trajectory.  Its largest decline
    observed so far forms a monotone envelope.  The endpoint is the first
    probe at which that envelope reaches ``fraction`` of the same window's
    maximum decline.  Windows with no measurable decline use the full range.
    """
    if not 0 < fraction <= 1:
        raise ValueError("relative destruction fraction must be in (0, 1]")
    if not 0 <= minimum_fraction <= 1:
        raise ValueError("minimum depth fraction must be in [0, 1]")
    destruction=(retention[:,:1]-retention).clamp_min(0)
    envelope=torch.cummax(destruction,dim=1).values
    total=envelope[:,-1]
    target=total[:,None]*fraction
    valid=(envelope+eps>=target) & (
        times[None,:]+eps>=times[-1]*minimum_fraction
    )
    first=valid.to(torch.int64).argmax(1)
    hit=(total>eps)&valid.any(1)
    return torch.where(hit,times[first],times[-1]),hit


@torch.no_grad()
def pilot_retention(model,schedule,x,cfg):
    """Return the deterministic coarse grid and its retention curves."""
    batch=len(x);device=x.device
    grid=torch.linspace(0,cfg.timesteps,cfg.coarse_steps+1,device=device)
    retention=(lambda a,b: activity_aware_retention(a,b,cfg.eps,cfg.activity_floor)) \
        if cfg.revision == "revised" else (lambda a,b: structure_retention(a,b,cfg.eps))
    z=x.clone();rs=[retention(x,x)]
    for k in range(cfg.coarse_steps):
        t=grid[k].expand(batch);tn=grid[k+1].expand(batch)
        a,g=schedule.coordinates(t);an,gn=schedule.coordinates(tn)
        xt=z*a[:,None].sqrt()
        e=model(xt,t,schedule.c)
        z=z+(gn-g)[:,None]*e
        rs.append(retention(x,z*an[:,None].sqrt()))
    return grid,torch.stack(rs,-1)


def geometry(responses,dgamma,eps):
    """Eq (22)-(30). responses [B,L,D,K]; dgamma [B,D,K]."""
    v=responses.abs()
    w=dgamma.abs()/(dgamma.abs().sum(-1,keepdim=True)+eps)
    velocity=(v.square()*w[:,None]).sum(-1)
    acc=torch.diff(responses,dim=-1)/(dgamma[:,None,:,:-1]+eps)
    wc=dgamma[:,:,:-1].abs()
    wc=wc/(wc.sum(-1,keepdim=True)+eps)
    curvature=(acc.square()*wc[:,None]).sum(-1)
    return velocity,curvature,torch.cat([v,acc],-1)


def transport_geometry(responses,dgamma,eps):
    """Stable geometry based on actual path transport increments.

    The original curvature divides by very small early delta-gamma values.
    V2 records signed state increments and changes of normalized direction,
    avoiding a singularity at the clean endpoint.
    """
    increments=responses*dgamma[:,None]
    signed_log=torch.sign(increments)*torch.log1p(increments.abs())
    velocity=signed_log.square().mean(-1)
    direction=increments/(increments.abs().sum(-1,keepdim=True)+eps)
    turning=torch.diff(direction,dim=-1)
    curvature=turning.square().mean(-1)
    return velocity,curvature,torch.cat([signed_log,turning],-1)


def residual_geometry(responses,eps):
    """Geometry of input-consistent clean-estimate residuals."""
    signed_log=torch.sign(responses)*torch.log1p(responses.abs())
    velocity=signed_log.square().mean(-1)
    direction=responses/(responses.abs().sum(-1,keepdim=True)+eps)
    turning=torch.diff(direction,dim=-1)
    curvature=turning.square().mean(-1)
    return velocity,curvature,torch.cat([signed_log,turning],-1)


class PathExtractor:
    def __init__(self,model,schedule,cfg):self.model,self.schedule,self.cfg=model,schedule,cfg

    @torch.no_grad()
    def __call__(self,x):
        cfg=self.cfg;s=self.schedule;batch=len(x);device=x.device
        if cfg.fixed_depth:
            if not 0 < cfg.fixed_depth_fraction <= 1:
                raise ValueError('fixed depth fraction must be in (0, 1]')
            # A fixed endpoint does not depend on the pilot.  Skipping those
            # coarse field calls is algebraically exact and makes the control
            # variants measure the intended path cost.
            depth=torch.full(
                (batch,),cfg.timesteps*cfg.fixed_depth_fraction,
                device=device,dtype=x.dtype,
            )
            hit=torch.zeros(batch,device=device,dtype=torch.bool)
            guard=torch.zeros(batch,device=device,dtype=torch.bool)
            r=torch.empty((batch,0),device=device,dtype=x.dtype)
        else:
            # A coarse deterministic pilot realizes the actual R_X(t), not a proxy.
            grid,r=pilot_retention(self.model,s,x,cfg)
            guard=torch.zeros(batch,device=device,dtype=torch.bool)
            if cfg.adaptive_depth_rule == "absolute":
                depth,hit=first_saturation(grid,r,cfg.tau_r,cfg.tau_delta)
            elif cfg.adaptive_depth_rule in {
                "relative_destruction","relative_guarded",
            }:
                depth,hit=relative_destruction_depth(
                    grid,r,cfg.relative_destruction_fraction,
                    cfg.minimum_depth_fraction,cfg.eps,
                )
                if cfg.adaptive_depth_rule == "relative_guarded":
                    reference=torch.as_tensor(
                        cfg.retention_reference_median,device=device,dtype=r.dtype,
                    )
                    scale=torch.as_tensor(
                        cfg.retention_reference_scale,device=device,dtype=r.dtype,
                    )
                    if reference.numel()!=r.shape[1] or scale.numel()!=r.shape[1]:
                        raise ValueError("guarded rule requires one reference value per probe")
                    standardized=(r-reference[None]).abs()/(scale[None]+cfg.eps)
                    count=max(1,int(np.ceil(r.shape[1]*.25)))
                    deviation=standardized.topk(count,dim=1).values.mean(1)
                    guard=deviation>cfg.retention_deviation_threshold
                    depth=torch.where(guard,torch.full_like(depth,cfg.timesteps),depth)
            else:
                raise ValueError("unknown adaptive depth rule: " + cfg.adaptive_depth_rule)
        # K integration intervals and K response observations.
        z=x.clone();responses=[];dg=[]
        if cfg.path_response == "clean_residual":
            t=torch.zeros_like(depth)
            a,g=s.coordinates(t)
            e=self.model(x,t,s.c)
            for k in range(cfg.nodes):
                tn=depth*(k+1)/cfg.nodes
                an,gn=s.coordinates(tn)
                z=z+(gn-g)[:,None]*e
                e_next=self.model(z*an[:,None].sqrt(),tn,s.c)
                # In tilde coordinates x0_hat = z_t - gamma_t epsilon_theta.
                responses.append(z-gn[:,None]*e_next-x)
                dg.append(gn-g)
                t,a,g,e=tn,an,gn,e_next
        else:
            for k in range(cfg.nodes):
                t=depth*k/cfg.nodes;tn=depth*(k+1)/cfg.nodes
                a,g=s.coordinates(t);an,gn=s.coordinates(tn)
                e=self.model(z*a[:,None].sqrt(),t,s.c)
                responses.append(e);dg.append(gn-g)
                z=z+(gn-g)[:,None]*e
        response_tensor=torch.stack(responses,-1)
        dgamma_tensor=torch.stack(dg,-1)
        if cfg.path_response == "clean_residual":
            v,c,h=residual_geometry(response_tensor,cfg.eps)
        elif cfg.geometry_version == "v2":
            v,c,h=transport_geometry(response_tensor,dgamma_tensor,cfg.eps)
        else:
            v,c,h=geometry(response_tensor,dgamma_tensor,cfg.eps)
        return v,c,h,dict(
            depth=depth.cpu().numpy(),hit=hit.cpu().numpy(),
            guard=guard.cpu().numpy(),retention=r.cpu().numpy(),
        )


class NormalPrototype:
    """Eq (31)-(37), exact full-sample covariance and medians, no truncation."""
    def __init__(self,cfg):self.cfg=cfg
    def fit(self,v,c,h,compute_device=None):
        # h [N,D,2K-1], includes ALL time-variable signatures in training windows.
        n,d,p=h.shape
        if self.cfg.prototype_components<=1:
            self.mu=h.mean(0,dtype=np.float64);self.inv=[]
            for j in range(d):
                cov=np.cov(h[:,j].astype('float64'),rowvar=False)
                cov=(1-self.cfg.shrinkage)*cov+self.cfg.shrinkage*np.diag(np.diag(cov))+self.cfg.eps*np.eye(p)
                self.inv.append(np.linalg.inv(cov))
            self.inv=np.array(self.inv)
        else:
            from sklearn.cluster import MiniBatchKMeans
            k=min(self.cfg.prototype_components,max(1,n//(p+2)))
            means=[];inverses=[]
            for j in range(d):
                values=h[:,j].astype('float64')
                km=MiniBatchKMeans(
                    n_clusters=k,random_state=self.cfg.seed+j,n_init=3,
                    batch_size=min(2048,n),reassignment_ratio=.01,
                ).fit(values)
                labels=km.labels_;global_cov=np.cov(values,rowvar=False)
                variable_means=[];variable_inverses=[]
                for cluster in range(k):
                    selected=values[labels==cluster]
                    cov=np.cov(selected,rowvar=False) if len(selected)>p+1 else global_cov
                    cov=(1-self.cfg.shrinkage)*cov+self.cfg.shrinkage*np.diag(np.diag(cov))+self.cfg.eps*np.eye(p)
                    variable_means.append(selected.mean(0) if len(selected) else km.cluster_centers_[cluster])
                    variable_inverses.append(np.linalg.inv(cov))
                means.append(variable_means);inverses.append(variable_inverses)
            self.mu=np.asarray(means);self.inv=np.asarray(inverses)
        if compute_device is None:
            components=self.components(v,c,h)
        else:
            parts=[]
            # Bound temporary [N,D,K,P] tensors for multi-prototype distance.
            for start in range(0,len(v),2048):
                vv=torch.as_tensor(v[start:start+2048],device=compute_device)
                cc=torch.as_tensor(c[start:start+2048],device=compute_device)
                hh=torch.as_tensor(h[start:start+2048],device=compute_device)
                parts.append(self.components_torch(vv,cc,hh).cpu().numpy())
            components=np.concatenate(parts)
        self.median=np.median(components,axis=0)
        self.mad=np.median(abs(components-self.median),axis=0)
        if self.cfg.revision == "revised":
            for component in range(self.mad.shape[-1]):
                positive=self.mad[:,component][self.mad[:,component]>self.cfg.eps]
                reference=float(np.median(positive)) if len(positive) else 1.
                self.mad[:,component]=np.maximum(
                    self.mad[:,component],self.cfg.score_floor_ratio*reference
                )
        return self
    def components(self,v,c,h):
        output=np.empty(v.shape+(3,),dtype=np.float64)
        output[...,0]=v;output[...,1]=c
        for j in range(v.shape[-1]):
            if self.cfg.prototype_components<=1:
                delta=h[:,j].astype('float64')-self.mu[j]
                output[:,j,2]=np.einsum('ni,ij,nj->n',delta,self.inv[j],delta,optimize=True)
            else:
                distances=[]
                for cluster in range(self.mu.shape[1]):
                    delta=h[:,j].astype('float64')-self.mu[j,cluster]
                    distances.append(np.einsum(
                        'ni,ij,nj->n',delta,self.inv[j,cluster],delta,optimize=True
                    ))
                output[:,j,2]=np.min(np.stack(distances,axis=1),axis=1)
        return output
    def components_torch(self,v,c,h):
        """Vectorized equivalent of :meth:`components` on a torch device."""
        dtype=h.dtype;device=h.device
        mu=torch.as_tensor(self.mu,device=device,dtype=dtype)
        inv=torch.as_tensor(self.inv,device=device,dtype=dtype)
        if self.cfg.prototype_components<=1:
            delta=h-mu[None]
            distance=torch.einsum('ndp,dpq,ndq->nd',delta,inv,delta)
        else:
            delta=h[:,:,None,:]-mu[None]
            distance=torch.einsum(
                'ndkp,dkpq,ndkq->ndk',delta,inv,delta
            ).amin(-1)
        return torch.stack((v,c,distance),dim=-1)
    def score(self,v,c,h):
        return self.standardized_components(v,c,h)@np.asarray(self.cfg.weights)

    def standardized_components(self,v,c,h):
        """Return the three path terms before their fixed linear fusion.

        The last axis is ordered as velocity, curvature, and prototype
        deviation.  Exposing this intermediate representation makes it
        possible to compute the main method score without using test labels.
        """
        return (self.components(v,c,h)-self.median)/(1.4826*self.mad+self.cfg.eps)

    def standardized_components_torch(self,v,c,h):
        components=self.components_torch(v,c,h)
        median=torch.as_tensor(self.median,device=h.device,dtype=h.dtype)
        mad=torch.as_tensor(self.mad,device=h.device,dtype=h.dtype)
        return (components-median)/(1.4826*mad+self.cfg.eps)


def aggregate_overlaps(local,starts,n):
    """Eq (40)-(41); local may be [W,L] or [W,L,D]. Exact medians."""
    local=np.asarray(local);starts=np.asarray(starts,dtype=int)
    if np.any(np.diff(starts)<0):raise ValueError('window starts must be sorted')
    coverage=np.zeros(n,dtype=int)
    length=local.shape[1]
    for start in starts:coverage[start:start+length]+=1
    if (coverage==0).any():raise ValueError('uncovered timestamps')
    # Constructing the full [N, max_coverage, ...] cube and then calling
    # np.nanmedian duplicates the cube internally.  Long SMAP streams
    # can therefore fail despite the final output being small.  Process exact
    # medians in bounded chunks while preserving the original overlap order.
    tail=local.shape[2:]
    max_coverage=int(coverage.max())
    values_per_time=max_coverage*int(np.prod(tail,dtype=int) if tail else 1)
    chunk=max(512,min(50000,(16*1024*1024)//max(4*values_per_time,1)))
    output=np.empty((n,)+tail,dtype='float32')
    for lo in range(0,n,chunk):
        hi=min(n,lo+chunk)
        cube=np.full((hi-lo,max_coverage)+tail,np.nan,dtype='float32')
        slots=np.zeros(hi-lo,dtype=int)
        first=np.searchsorted(starts,lo-length+1,side='left')
        last=np.searchsorted(starts,hi,side='left')
        for row,start in zip(local[first:last],starts[first:last]):
            a=max(int(start),lo);b=min(int(start)+length,hi)
            if a>=b:continue
            ids=np.arange(a-lo,b-lo)
            cube[ids,slots[ids]]=row[a-int(start):b-int(start)]
            slots[ids]+=1
        output[lo:hi]=np.nanmedian(cube,axis=1,overwrite_input=True)
    return output


def make_windows(x,length,stride,boundaries=None):
    segments=[(0,len(x))] if boundaries is None else boundaries
    starts=[]
    for a,b in segments:
        if b-a<length:raise ValueError('entity segment shorter than window')
        starts.extend(np.unique(np.r_[np.arange(a,b-length+1,stride),b-length]).tolist())
    return np.stack([x[i:i+length] for i in starts]).astype('float32'),np.array(starts)
