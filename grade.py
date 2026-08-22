"""Editorial grade + a ramp baked into the foot of the frame.

The ramp is solved per image so white type clears 4.5:1 where the label sits, and
because the grain is laid down after it, it reads as light falling off rather than
as a panel dropped over the picture."""
from PIL import Image
import numpy as np, os

RAW='/private/tmp/claude-501/-Users-namratatripathi-Claude/3490c0ff-647b-4ede-9a48-d3c4a82f290a/scratchpad/img_raw'

def _lum(x):
    c=np.where(x<=.03928,x/12.92,((x+.055)/1.055)**2.4)
    return (c*np.array([.2126,.7152,.0722],np.float32)).sum(-1)

def base(x, exposure=.88, contrast=1.16, knee=.58, ceiling=.86, lift=.030, sat=.88):
    x=x*exposure
    x=np.clip(.5+(x-.5)*contrast,0,4)
    span=max(ceiling-knee,1e-4)
    x=np.where(x>knee, knee+span*(1-np.exp(-(x-knee)/span)), x)   # nothing ever clips
    x=lift+(1-lift)*x                                            # matte black
    g=(x*np.array([.2126,.7152,.0722],np.float32)).sum(2,keepdims=True)
    x=g+(x-g)*sat
    w=1-np.clip(g,0,1)
    x[...,0]+=.020*w[...,0]; x[...,2]-=.011*w[...,0]
    x[...,2]+=.010*np.clip(g,0,1)[...,0]
    return np.clip(x,0,1)

def ramp_mask(h,w,start=.42,strength=1.0):
    """1.0 down to (1-strength) between `start` of the height and the bottom, eased."""
    y=np.linspace(0,1,h,dtype=np.float32)
    t=np.clip((y-start)/max(1-start,1e-4),0,1)
    t=t*t*(3-2*t)                                   # smoothstep, so there is no visible edge
    return (1.0-strength*t)[:,None,None]

def grade(name,out,label_band=.30,target=.19,grain=4.4,seed=7,quality=84,
          with_ramp=True,start=.42,**kw):
    im=Image.open(os.path.join(RAW,name)).convert('RGB')
    x=base(np.asarray(im).astype(np.float32)/255.0,**kw)
    h,w,_=x.shape
    s=0.0
    if with_ramp:
        band=slice(int(h*(1-label_band)),h)
        lo,hi=0.0,0.92
        for _ in range(26):                          # solve the ramp for this picture
            s=(lo+hi)/2
            trial=x[band]*ramp_mask(h,w,start,s)[band]
            if np.percentile(_lum(trial),99.5)>target: lo=s
            else: hi=s
        s=hi
        x=x*ramp_mask(h,w,start,s)
    rng=np.random.default_rng(seed)                  # grain last, so it lives in the shadow too
    g=_lum(x)
    n=rng.normal(0,grain/255.0,(h,w)).astype(np.float32)
    x=np.clip(x+(n*((1-np.abs(np.clip(g,0,1)*2-1))*.65+.35))[...,None],0,1)
    Image.fromarray((x*255+.5).astype(np.uint8)).save(out,quality=quality,optimize=True,progressive=True)
    band=x[int(h*(1-label_band)):]
    return s, 1.05/(np.percentile(_lum(band),99.5)+.05)
