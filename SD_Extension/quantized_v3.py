"""Integer golden specification for PL V3; GT is not an inference input."""
from pathlib import Path
import json
import numpy as np,cv2
from PIL import Image
from fused_dcp import process,strong_reference,estimate_strong_air,features
ROOT=Path(__file__).resolve().parent

def rounding(a,bits):
    a=np.asarray(a,dtype=np.int64);return np.sign(a)*((np.abs(a)+(1<<(bits-1)))>>bits)

class QuantizedV3:
    def __init__(self,root=ROOT):
        root=Path(root);self.model=json.loads((root/'fixed_v3.json').read_text());self.base_model=json.loads((root/'frozen_model.json').read_text())
        import hashlib
        expected_tag=int(hashlib.sha256(np.array(self.model['weights'],dtype='<i2').tobytes()+np.array(self.model['bias'],dtype='<i4').tobytes()).hexdigest()[:8],16)
        if expected_tag!=self.model['tag']:raise ValueError('Quantized weights/tag mismatch')
        self.w=np.array(self.model['weights'],np.int64);self.b=np.array(self.model['bias'],np.int64);self.norm=np.array(self.model['pixel_normalization'],np.int64);self.inv=np.array(self.model['guard_inverse_q24'],np.int64)
    def globals(self,image):
        f=np.clip((features(image)-np.array(self.model['global_mean']))/np.array(self.model['global_scale']),-3,3)
        q=np.rint(f*16384).astype(np.int32)
        if self.model.get('output_mode')=='physical-air-gain':
            air=estimate_strong_air(image,self.base_model['air_method']);q=np.concatenate([q,self.norm[np.asarray(air,dtype=np.int32)].astype(np.int32)])
        return q
    def inputs(self,image,base=None,strong=None,info=None):
        if base is None:base,_=process(image,self.base_model)
        if strong is None or info is None:strong,info=strong_reference(image,estimate_strong_air(image))
        a=np.asarray(image,np.uint8);b=np.asarray(base,np.uint8);s=np.asarray(strong,np.uint8)
        fields=np.zeros((*a.shape[:2],19),np.int64);fields[:,:,:3]=self.norm[a];fields[:,:,3:6]=self.norm[b];fields[:,:,6:9]=self.norm[s];fields[:,:,18]=self.norm[info['transmission']]
        for c in range(3):
            x=a[:,:,c].astype(np.float64);sums=cv2.boxFilter(x,cv2.CV_64F,(15,15),normalize=False,borderType=cv2.BORDER_REFLECT_101).astype(np.int64)
            squares=cv2.boxFilter(x*x,cv2.CV_64F,(15,15),normalize=False,borderType=cv2.BORDER_REFLECT_101).astype(np.int64)
            bs=cv2.boxFilter(b[:,:,c].astype(np.float64),cv2.CV_64F,(15,15),normalize=False,borderType=cv2.BORDER_REFLECT_101).astype(np.int64)
            variance=np.maximum(0,squares*225-sums*sums);root=np.floor(np.sqrt(variance.astype(np.float64))).astype(np.int64)
            root=np.where((root+1)*(root+1)<=variance,root+1,root);root=np.where(root*root>variance,root-1,root)
            fields[:,:,9+c]=(sums*16384+28687)//57375;fields[:,:,12+c]=(bs*16384+28687)//57375;fields[:,:,15+c]=(root*16384+28687)//57375
        return fields,base,(s.astype(np.uint32)[:,:,0]|s.astype(np.uint32)[:,:,1]<<8|s.astype(np.uint32)[:,:,2]<<16|info['transmission'].astype(np.uint32)<<24)
    def infer(self,fields,global_q):
        shape=fields.shape[:2];local=fields.reshape(-1,19);outputs=[]
        # Float64 BLAS is exact here: every integer sum/product is bounded below 2**53.
        frame=rounding(np.asarray(global_q,np.int64)[:22]@self.w[:32,19:41].T+(self.b[:32]<<14),14)
        for start in range(0,len(local),16384):
            p=local[start:start+16384];a=rounding(p.astype(np.float64)@self.w[:32,:19].T.astype(np.float64)+(frame<<14),14);a=np.maximum(a,0)
            b=rounding(a.astype(np.float64)@self.w[32:56,:32].T.astype(np.float64)+(self.b[32:56]<<14),14);b=np.maximum(b,0)
            q=rounding(b.astype(np.float64)@self.w[56:60,:24].T.astype(np.float64)+(self.b[56:60]<<14),14)
            gain=np.clip(q[:,3:4],-8192,65536);res=np.clip(q[:,:3],-6144,6144)
            detail=p[:,6:9]-np.asarray(global_q,np.int64)[22:25] if self.model.get('output_mode')=='physical-air-gain' else p[:,:3]-p[:,9:12]
            raw=np.clip(p[:,3:6]+res+rounding(gain*detail,14),0,16384)
            candidate=((raw*255+8192)>>14);base=((p[:,3:6]*255+8192)>>14);delta=candidate-base
            allowance=np.minimum(np.where(delta<0,3*base,3*(255-base)),256)
            ratios=np.where(delta==0,65536,(allowance*self.inv[np.abs(delta)]+512)>>10)
            alpha=np.minimum(65536,ratios.min(1));weight=(alpha*3+2)>>2
            value=(base<<16)+delta*weight[:,None];floor=value>>16;remainder=value&65535
            out=floor+((remainder>32768)|((remainder==32768)&((floor&1)!=0)))
            outputs.append(np.clip(out,0,255).astype(np.uint8))
        return Image.fromarray(np.concatenate(outputs).reshape(*shape,3))
    def process(self,image):
        x,base,aux=self.inputs(image);return self.infer(x,self.globals(image)),base
    def packed_fields(self,fields):
        flat=fields.reshape(-1,19).astype(np.uint32);p=np.zeros((len(flat),20),np.uint32);p[:,:19]=flat
        return (p[:,::2]|(p[:,1::2]<<16)).astype('<u4')

    def packed_network(self):
        if np.max(np.abs(self.w[:56]))>8192 or np.max(np.abs(self.b))>16384:raise ValueError('Model exceeds supported weight/bias bounds')
        frame_abs=(np.abs(self.w[:32,19:41]).sum(1)*49152+np.abs(self.b[:32])*16384+8192)//16384
        first_upper=(np.maximum(self.w[:32,:19],0).sum(1)*16384+frame_abs*16384+8192)//16384
        second_upper=np.maximum(0,(np.maximum(self.w[32:56,:32],0)@first_upper+self.b[32:56]*16384+8192)//16384)
        if max(frame_abs.max(),first_upper.max(),second_upper.max())>262143:raise ValueError('Model exceeds proven signed 19-bit activation bounds')
        weights=np.zeros((32,3,64),dtype='<i2');biases=np.zeros((32,3),dtype='<i4')
        weights[:,0,:]=self.w[:32,:];biases[:,0]=self.b[:32]
        weights[:24,1,:]=self.w[32:56,:];biases[:24,1]=self.b[32:56]
        for row in range(4):
            biases[row*8,2]=self.b[56+row]
            for lane in range(8):weights[row*8+lane,2,:3]=self.w[56+row,lane:24:8]
        return weights.tobytes()+biases.tobytes()
