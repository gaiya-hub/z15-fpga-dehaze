"""Software candidates for quality-first FPGA research, NOT deployed hardware."""
import numpy as np
import cv2
from PIL import Image,ImageOps
from model import estimate_air

def air_light(image,method):
    if method=='baseline':return np.asarray(estimate_air(image),dtype=np.float32)/255
    small=ImageOps.contain(image,(320,320))
    rgb=np.asarray(small,dtype=np.float32)/255
    dark=cv2.erode(rgb.min(axis=2),np.ones((15,15),np.uint8))
    count=max(16,dark.size//500)
    indices=np.argpartition(dark.reshape(-1),-count)[-count:]
    candidates=rgb.reshape(-1,3)[indices]
    if method=='robust':air=np.median(candidates,axis=0)
    elif method=='brightest':air=candidates[np.argmax(candidates.sum(axis=1))]
    else:raise ValueError('Unknown atmospheric estimator')
    return np.clip(air,64/255,250/255)

def guided_transmission(gray,transmission,subsample=4,window=41,eps=.001):
    height,width=gray.shape
    size=(max(1,(width+subsample-1)//subsample),max(1,(height+subsample-1)//subsample))
    guide=cv2.resize(gray,size,interpolation=cv2.INTER_AREA) if subsample>1 else gray
    target=cv2.resize(transmission,size,interpolation=cv2.INTER_AREA) if subsample>1 else transmission
    kernel=max(3,int(round(window/subsample))|1)
    box=lambda value:cv2.boxFilter(value,cv2.CV_32F,(kernel,kernel),borderType=cv2.BORDER_REFLECT_101)
    mean_i=box(guide);mean_t=box(target)
    variance=np.maximum(0,box(guide*guide)-mean_i*mean_i)
    covariance=box(guide*target)-mean_i*mean_t
    coefficient=covariance/(variance+eps);offset=mean_t-coefficient*mean_i
    coefficient=box(coefficient);offset=box(offset)
    if subsample>1:
        coefficient=cv2.resize(coefficient,(width,height),interpolation=cv2.INTER_LINEAR)
        offset=cv2.resize(offset,(width,height),interpolation=cv2.INTER_LINEAR)
    return np.clip(coefficient*gray+offset,0,1)

def process(image,config):
    rgb=np.asarray(image,dtype=np.float32)/255
    air=air_light(image,config['air'])
    dark=cv2.erode((rgb/air).min(axis=2),np.ones((15,15),np.uint8))
    transmission=np.clip(1-config['omega']*dark,0,1)
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    if config.get('guided',True):
        transmission=guided_transmission(gray,transmission,config['subsample'],41,config.get('eps',.001))
    transmission=np.maximum(config['floor'],transmission)
    if config.get('range_constraint',True):
        # Smallest t consistent with J in [0,1] for the selected A and observed I.
        # It limits clipping; it does not claim semantic sky detection.
        lower_dark=(1-rgb/air).max(axis=2)
        lower_bright=((rgb-air)/(1-air)).max(axis=2)
        transmission=np.maximum(transmission,np.clip(np.maximum(lower_dark,lower_bright),0,1))
    result=np.clip((rgb-air)/transmission[:,:,None]+air,0,1)
    return Image.fromarray(np.rint(result*255).astype(np.uint8)),{'air':air.tolist(),'transmission_mean':float(transmission.mean())}

def configurations():
    configs=[]
    for air in ['baseline','robust','brightest']:
        for omega,floor in [(.90,.15),(.95,.10)]:
            configs.append({'name':f'fast_guided_{air}_{omega:.2f}','air':air,'omega':omega,'floor':floor,'subsample':4,'guided':True,'range_constraint':True})
    configs += [
        {'name':'full_guided_robust_0.90','air':'robust','omega':.90,'floor':.15,'subsample':1,'guided':True,'range_constraint':True},
        {'name':'full_guided_brightest_0.95','air':'brightest','omega':.95,'floor':.10,'subsample':1,'guided':True,'range_constraint':True},
        {'name':'ablation_no_range','air':'robust','omega':.90,'floor':.15,'subsample':4,'guided':True,'range_constraint':False},
        {'name':'ablation_no_guided','air':'robust','omega':.90,'floor':.15,'subsample':4,'guided':False,'range_constraint':True}]
    return configs
