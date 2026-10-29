"""Small DCP-conditioned residual CNN. Experimental until explicitly board-verified."""
import torch
from torch import nn

class SpatialNet(nn.Module):
    def __init__(self,hidden=16):
        super().__init__()
        self.g=nn.Linear(22,hidden)
        self.input=nn.Conv2d(9,hidden,1)
        self.body=nn.Sequential(nn.ReLU(),nn.Conv2d(hidden,hidden,5,padding=2,groups=hidden),nn.ReLU(),nn.Conv2d(hidden,hidden,1),nn.ReLU(),nn.Conv2d(hidden,hidden,5,padding=2,groups=hidden),nn.ReLU(),nn.Conv2d(hidden,3,1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
    def forward(self,pixels,g):
        residual=self.body(self.input(pixels)+self.g(g)[:,:,None,None]).clamp(-.5,.5)
        return (pixels[:,3:6]+residual).clamp(0,1)
