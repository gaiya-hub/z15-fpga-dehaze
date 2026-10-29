"""Two physical-prior DCP branches; no pretrained network and no GT at runtime."""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image,ImageOps
from model import estimate_air,reference

def features(image):
    rgb=np.asarray(ImageOps.contain(image,(160,160)),dtype=np.float64)/255
    gray=.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2]
    maximum=rgb.max(axis=2);minimum=rgb.min(axis=2);sat=(maximum-minimum)/(maximum+1e-6)
    values=[]
    for field in [gray,minimum,sat]:
        values.extend([field.mean(),field.std(),*np.quantile(field,[.1,.5,.9])])
    for region in [gray[:max(1,gray.shape[0]//3)],sat[:max(1,sat.shape[0]//3)]]:
        values.extend([region.mean(),region.std()])
    values.extend(rgb.mean(axis=(0,1)).tolist())
    return np.asarray(values,dtype=np.float64)

def estimate_strong_air(image,method='brightest'):
    rgb=np.asarray(ImageOps.contain(image,(320,320)),dtype=np.uint8)
    dark=cv2.erode(rgb.min(axis=2),np.ones((15,15),np.uint8))
    count=max(16,dark.size//500);ids=np.argpartition(dark.reshape(-1),-count)[-count:]
    pixels=rgb.reshape(-1,3)[ids]
    air=pixels[np.argmax(pixels.astype(np.int32).sum(axis=1))] if method=='brightest' else np.median(pixels,axis=0)
    return tuple(np.clip(np.rint(air),64,250).astype(int).tolist())

def box_sums(fields,radius):
    k=radius*2+1
    return cv2.boxFilter(fields,cv2.CV_64F,(k,k),normalize=False,borderType=cv2.BORDER_CONSTANT)

def strong_reference(image,air,strength=230,t_min=38):
    """Integer-oriented guided branch, used as the intended hardware specification."""
    rgb=np.asarray(image,dtype=np.int32);height,width=rgb.shape[:2]
    av=np.asarray(air,dtype=np.int32);gain=(65280+av//2)//av
    norm=np.clip((rgb*gain+128)>>8,0,255).min(axis=2).astype(np.uint8)
    dark=cv2.erode(norm,np.ones((15,15),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=255).astype(np.int32)
    t=255-((strength*dark+127)>>8)
    gray=((77*rgb[:,:,0]+150*rgb[:,:,1]+29*rgb[:,:,2]+128)>>8).astype(np.int32)
    # Block-mean subsampling with partial edge blocks, matching a straightforward HLS implementation.
    lh=(height+3)//4;lw=(width+3)//4
    small_i=np.empty((lh,lw),np.int64);small_t=np.empty((lh,lw),np.int64)
    # Vectorized integral sums avoid a slow per-block Python loop.
    def downsample(field):
        integral=cv2.integral(field.astype(np.float64),sdepth=cv2.CV_64F)
        y0=np.arange(lh)*4;y1=np.minimum(y0+4,height);x0=np.arange(lw)*4;x1=np.minimum(x0+4,width)
        total=integral[np.ix_(y1,x1)]-integral[np.ix_(y0,x1)]-integral[np.ix_(y1,x0)]+integral[np.ix_(y0,x0)]
        n=(y1-y0)[:,None]*(x1-x0)[None,:]
        return ((total+n//2)//n).astype(np.int64)
    small_i=downsample(gray);small_t=downsample(t)
    n=box_sums(np.ones((lh,lw),np.float64),5).astype(np.int64)
    fields=np.stack([small_i,small_t,small_i*small_i,small_i*small_t],axis=2)
    sums=np.rint(box_sums(fields.astype(np.float64),5)).astype(np.int64)
    si,st,sii,sit=[sums[:,:,i] for i in range(4)]
    variance=np.maximum(0,sii*n-si*si);covariance=sit*n-si*st
    denominator=variance+65*n*n
    a=np.trunc(covariance.astype(np.float64)*4096/denominator).astype(np.int64)
    inverse=((1<<20)+n//2)//n
    product=((st<<12)-a*si)*inverse
    b=((np.abs(product)+(1<<19))>>20)*np.sign(product)
    summed=np.rint(box_sums(np.stack([a,b],axis=2).astype(np.float64),5)).astype(np.int64)
    product=summed*inverse[:,:,None]
    averaged=(((np.abs(product)+(1<<19))>>20)*np.sign(product)).astype(np.int32)
    # Grid coordinates are block centers; hardware bilinear interpolation uses fractions /256.
    def axis(length,small_length):
        position=(np.arange(length,dtype=np.int32)*2-3)*32
        left=np.floor_divide(position,256);fraction=position-left*256
        right=np.clip(left+1,0,small_length-1);left=np.clip(left,0,small_length-1)
        return left,right,fraction
    x0,x1,fx=axis(width,lw);y0,y1,fy=axis(height,lh)
    coeff=[]
    for channel in range(2):
        p=averaged[:,:,channel].astype(np.int64)
        top=p[np.ix_(y0,x0)]*(256-fx)[None,:]+p[np.ix_(y0,x1)]*fx[None,:]
        bottom=p[np.ix_(y1,x0)]*(256-fx)[None,:]+p[np.ix_(y1,x1)]*fx[None,:]
        value=top*(256-fy)[:,None]+bottom*fy[:,None]
        coeff.append(((np.abs(value)+32768)>>16)*np.sign(value))
    value=coeff[0]*gray+coeff[1]
    refined=np.clip(((np.abs(value)+2048)>>12)*np.sign(value),t_min,255).astype(np.int32)
    normal=(rgb*gain+128)>>8
    upper_gain=(65280+(255-av)//2)//(255-av)
    lower=np.maximum((255-normal).max(axis=2),(((rgb-av)*upper_gain+128)>>8).max(axis=2))
    refined=np.clip(np.maximum(refined,lower),t_min,255)
    inv=(1044480+refined//2)//refined
    product=(rgb-av)*inv[:,:,None]
    delta=((np.abs(product)+2048)>>12)*np.sign(product)
    output=np.clip(av+delta,0,255).astype(np.uint8)
    return Image.fromarray(output),{'transmission':refined,'coefficients':averaged,'dark':norm,'air':air}

def predict_weight(image,model):
    if model['kind']=='constant':return int(model['weight_q8'])
    x=(features(image)-np.asarray(model['mean']))/np.asarray(model['scale'])
    value=float(np.dot(x,np.asarray(model['weights']))+model['bias'])
    return int(np.clip(round(value*256),0,256))

def process(image,model):
    weak_air=estimate_air(image);strong_air=estimate_strong_air(image,model['air_method'])
    baseline=reference(image,weak_air,160,96,64)
    strong,_=strong_reference(image,strong_air)
    weight=predict_weight(image,model)
    a=np.asarray(baseline,dtype=np.int32);b=np.asarray(strong,dtype=np.int32)
    output=((a*(256-weight)+b*weight+128)>>8).astype(np.uint8)
    return Image.fromarray(output),{'weak_air':weak_air,'strong_air':strong_air,'weight_q8':weight,'controller':'development-calibrated statistics, not a neural network'}
