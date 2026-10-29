"""DCP + bounded neural detail correction. CPU candidate; no FPGA deployment claim."""
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image
from fused_dcp import process as dcp_process,features,strong_reference,estimate_strong_air

class DetailNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(41,32),nn.ReLU(),nn.Linear(32,24),nn.ReLU(),nn.Linear(24,4))
    def coefficients(self,p,g):
        return self.layers(torch.cat([p,g],-1))
    def forward(self,p,g):
        q=self.coefficients(p,g);gain=(q[...,3:4]+1).clamp(.5,5)
        return (p[...,3:6]+q[...,:3].clamp(-.375,.375)+(gain-1)*(p[...,:3]-p[...,9:12])).clamp(0,1)

def local_features(image,base_model):
    base,_=dcp_process(image,base_model);strong,info=strong_reference(image,estimate_strong_air(image))
    orig=np.asarray(image,dtype=np.float32)/255;old=np.asarray(base,dtype=np.float32)/255;ss=np.asarray(strong,dtype=np.float32)/255
    avg=lambda a:cv2.boxFilter(a,cv2.CV_32F,(15,15),borderType=cv2.BORDER_REFLECT_101)
    om=avg(orig);bm=avg(old);std=np.sqrt(np.maximum(0,avg(orig*orig)-om*om))
    local=np.concatenate([orig,old,ss,om,bm,std,info['transmission'][:,:,None]/255],2).astype(np.float32)
    return local,base

class DetailDehazer:
    def __init__(self,checkpoint,base_model):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.net=DetailNet();self.net.load_state_dict(state['state']);self.net.eval()
        self.mean=np.asarray(state['mean']);self.scale=np.asarray(state['scale']);self.base_model=base_model
    def process(self,image,blend=1.0):
        if not 0<=blend<=1:raise ValueError('blend must be in [0,1]')
        local,base=local_features(image,self.base_model);p=torch.from_numpy(local.reshape(-1,19))
        g=torch.tensor(np.clip((features(image)-self.mean)/self.scale,-3,3),dtype=torch.float32);out=[]
        with torch.inference_mode():
            for k in range(0,len(p),32768):
                part=p[k:k+32768];pred=self.net(part,g.expand(len(part),-1));out.append(pred.numpy())
        a=np.concatenate(out).reshape(image.height,image.width,3)
        a=a*blend+np.asarray(base,dtype=np.float32)/255*(1-blend)
        return Image.fromarray(np.clip(np.rint(a*255),0,255).astype(np.uint8)),base
