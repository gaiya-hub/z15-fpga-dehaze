"""Input-only guard for a neural residual. No GT-dependent decisions."""
import numpy as np
from PIL import Image

def protect(base,candidate,config):
    if base.size!=candidate.size:raise ValueError('Base and candidate must have identical geometry')
    a=np.asarray(base,dtype=np.float32)/255;b=np.asarray(candidate,dtype=np.float32)/255;delta=b-a
    floor=float(config['dark_floor']);ceiling=float(config['bright_headroom']);limit=float(config['max_delta'])/255
    blend=float(config['blend'])
    if not (0<=floor<=1 and 0<=ceiling<=1 and 0<=blend<=1 and 0<limit<=1):raise ValueError('Invalid guard configuration')
    # One common mixing factor for RGB avoids per-channel clipping of the correction.
    allowance=np.where(delta<0,(1-floor)*a,(1-ceiling)*(1-a))
    ratios=np.minimum(allowance,limit)/np.maximum(np.abs(delta),1e-8)
    weight=np.minimum(1,ratios.min(2))*blend
    out=a+delta*weight[:,:,None]
    return Image.fromarray(np.clip(np.rint(out*255),0,255).astype(np.uint8))
