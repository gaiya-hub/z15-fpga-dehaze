"""DCP-conditioned low-resolution coefficient CNN, with full-size RGB recovery."""
import numpy as np,cv2,torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from fused_dcp import process as dcp_process,features,strong_reference,estimate_strong_air

class GridNet(nn.Module):
    def __init__(self):
        super().__init__();self.frame=nn.Linear(22,24);self.input=nn.Conv2d(19,24,1)
        self.body=nn.Sequential(nn.ReLU(),nn.Conv2d(24,24,5,padding=2,groups=24),nn.ReLU(),nn.Conv2d(24,24,1),nn.ReLU(),nn.Conv2d(24,24,5,padding=2,groups=24),nn.ReLU(),nn.Conv2d(24,4,1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
    def forward(self,p,g):return self.body(self.input(p)+self.frame(g)[:,:,None,None])

def down4(a):
    h,w=a.shape[:2];ys=np.arange(0,h,4);xs=np.arange(0,w,4);cnt=np.minimum(4,h-ys)[:,None]*np.minimum(4,w-xs)[None,:]
    value=np.add.reduceat(np.add.reduceat(a,ys,axis=0),xs,axis=1)
    return value/cnt[:,:,None] if a.ndim==3 else value/cnt

def make_features(image,base_model):
    base,_=dcp_process(image,base_model);strong,info=strong_reference(image,estimate_strong_air(image))
    orig=np.asarray(image,dtype=np.float32)/255;old=np.asarray(base,dtype=np.float32)/255
    a=down4(orig).astype(np.float32);b=down4(old).astype(np.float32);s=down4(np.asarray(strong,dtype=np.float32)/255).astype(np.float32)
    avg=lambda field:cv2.boxFilter(field,cv2.CV_32F,(5,5),borderType=cv2.BORDER_REFLECT_101)
    am=avg(a);bm=avg(b);std=np.sqrt(np.maximum(0,avg(a*a)-am*am));t=down4(info['transmission'].astype(np.float32)/255)[:,:,None]
    local=np.concatenate([a,b,s,am,bm,std,t],2).astype(np.float32)
    # Block-center coordinates for partial blocks follow the fixed 4-pixel grid.
    h,w=orig.shape[:2];xx=(np.arange(w,dtype=np.float32)-1.5)/4;yy=(np.arange(h,dtype=np.float32)-1.5)/4
    mx,my=np.meshgrid(xx,yy);blur=cv2.remap(am,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return local,old,orig-blur

def recover(old,detail,coeff):
    # Coefficients are interpolated, while original RGB texture remains full size.
    h,w=old.shape[:2];xx=(np.arange(w,dtype=np.float32)-1.5)/4;yy=(np.arange(h,dtype=np.float32)-1.5)/4;mx,my=np.meshgrid(xx,yy)
    q=cv2.remap(coeff,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    delta=np.clip(q[:,:,:3],-.375,.375)+(np.clip(q[:,:,3:4]+1,.5,5)-1)*detail
    return old+delta

class GridDehazer:
    def __init__(self,checkpoint,base_model):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True);self.net=GridNet();self.net.load_state_dict(state['state']);self.net.eval()
        self.mean=np.asarray(state['mean']);self.scale=np.asarray(state['scale']);self.base_model=base_model
    def process(self,image,blend=1):
        p,old,detail=make_features(image,self.base_model);g=np.clip((features(image)-self.mean)/self.scale,-3,3).astype(np.float32)
        with torch.inference_mode():q=self.net(torch.from_numpy(p.transpose(2,0,1).copy())[None],torch.from_numpy(g)[None])[0].permute(1,2,0).numpy()
        raw=recover(old,detail,q);out=old+blend*(raw-old)
        return Image.fromarray(np.clip(np.rint(out*255),0,255).astype(np.uint8)),Image.fromarray(np.rint(old*255).astype(np.uint8))
