"""Bit-exact host reference for the RGB hardware; also used for verification.
No FFA-Net inference or generated image enhancement is used here.
"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

def min_axis(array,radius,axis):
    pad=[(0,0)]*array.ndim;pad[axis]=(radius,radius)
    padded=np.pad(array,pad,mode='constant',constant_values=255)
    result=np.full_like(array,255)
    for offset in range(2*radius+1):
        selection=[slice(None)]*array.ndim
        selection[axis]=slice(offset,offset+array.shape[axis])
        np.minimum(result,padded[tuple(selection)],out=result)
    return result

def load_image(path,max_edge=1280):
    image=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    original=image.size
    scale=min(1.0,max_edge/max(image.size),1920/image.width,1280/image.height)
    if scale<1:image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
    return image,original

def estimate_air(image):
    small=ImageOps.contain(image,(320,320))
    rgb=np.asarray(small,dtype=np.uint8)
    dark=min_axis(min_axis(rgb.min(axis=2),7,1),7,0)
    count=max(16,dark.size//500)
    candidates=np.argpartition(dark.reshape(-1),-count)[-count:]
    pixels=rgb.reshape(-1,3)[candidates]
    # Median of the brightest dark-channel candidates reduces isolated highlight sensitivity.
    air=np.median(pixels,axis=0).astype(int)
    return tuple(np.clip(air,160,250).tolist())

def reference(image,air,strength=160,t_min=96,lift=64):
    rgb=np.asarray(image,dtype=np.int32)
    a=np.asarray(air,dtype=np.int32)
    gain=(65280+a//2)//a
    normal=np.clip((rgb*gain+128)>>8,0,255).min(axis=2).astype(np.uint8)
    dark=min_axis(min_axis(normal,7,1),7,0).astype(np.int32)
    t=np.maximum(t_min,255-((strength*dark+127)>>8))
    inverse=(1044480+t//2)//t
    product=(rgb-a)*inverse[:,:,None]
    delta=((np.abs(product)+2048)>>12)*np.sign(product)
    result=np.clip(a+delta,0,255)
    result=np.clip(result+((result*(255-result)*lift+32768)>>16),0,255).astype(np.uint8)
    return Image.fromarray(result,'RGB')

def pack(image):
    rgb=np.asarray(image,dtype=np.uint8)
    out=np.empty((image.height,image.width,4),dtype=np.uint8)
    out[:,:,:3]=rgb;out[:,:,3]=255
    return out.tobytes()

def unpack(data,width,height):
    return Image.fromarray(np.frombuffer(data,dtype=np.uint8).reshape(height,width,4)[:,:,:3].copy(),'RGB')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='电脑同算法参考，仅验证像素算法，不运行FPGA')
    parser.add_argument('image');parser.add_argument('--edge',type=int,default=1280)
    parser.add_argument('--output',default=str(Path(__file__).parent/'outputs/cpu_reference.png'))
    args=parser.parse_args();image,_=load_image(args.image,args.edge)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    reference(image,estimate_air(image)).save(destination);print(destination)
