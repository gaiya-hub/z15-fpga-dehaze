"""Experimental DCP + bounded pointwise residual neural network (not board deployed)."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from PIL import Image
from fused_dcp import features,process as dcp_process

class ResidualNet(nn.Module):
    def __init__(self,global_count=22,hidden=24):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(6+global_count,hidden),nn.ReLU(),nn.Linear(hidden,16),nn.ReLU(),nn.Linear(16,3))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,pixels,global_features):
        # Training uses flattened pixels; the frame features are broadcast explicitly.
        values=torch.cat([pixels,global_features],dim=-1)
        delta=self.layers(values).clamp(-.375,.375)
        return (pixels[...,3:6]+delta).clamp(0,1)

def infer(image,checkpoint_path,base_model,blend=1.0):
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    net=ResidualNet();net.load_state_dict(state['state']);net.eval()
    old,_=dcp_process(image,base_model)
    a=np.asarray(image,dtype=np.float32)/255;b=np.asarray(old,dtype=np.float32)/255
    p=np.concatenate([a,b],axis=2).reshape(-1,6)
    g=(features(image)-np.asarray(state['mean']))/np.asarray(state['scale'])
    output=[]
    with torch.inference_mode():
        for start in range(0,len(p),65536):
            local=torch.from_numpy(p[start:start+65536]);glob=torch.tensor(g,dtype=torch.float32).expand(len(local),-1)
            result=net(local,glob).numpy()
            output.append(result*blend+p[start:start+len(local),3:6]*(1-blend))
    return Image.fromarray(np.clip(np.rint(np.concatenate(output).reshape(a.shape)*255),0,255).astype(np.uint8))
