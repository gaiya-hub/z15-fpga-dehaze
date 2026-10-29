"""Deterministic native/HLS test inputs and independent NumPy expected results."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quantized_v3 import QuantizedV3
from model import pack,estimate_air
from fused_dcp import estimate_strong_air,predict_weight
model=QuantizedV3(ROOT);fixtures=ROOT/'hls/fixtures';fixtures.mkdir(exist_ok=True)
cases=[]
for w,h in [(1,1),(5,3),(13,11),(37,29),(96,64),(640,360)]:
 rng=np.random.default_rng(w*100+h)
 if w==640:
  y,x=np.indices((h,w));rgb=np.stack([70+x*90//w,80+y*110//h,110+(x+y)*60//(w+h)],2).astype(np.uint8)
  rgb=np.clip(rgb.astype(np.int16)+rng.integers(-15,16,(h,w,3)),0,255).astype(np.uint8)
 else:rgb=rng.integers(25,245,(h,w,3),dtype=np.uint8)
 image=Image.fromarray(rgb);fields,base,aux=model.inputs(image);glob=model.globals(image);output=model.infer(fields,glob)
 stem=fixtures/f'{w}x{h}'
 for suffix,data in [('.input',pack(image)),('.base',pack(base)),('.expected',pack(output)),('.aux',aux.astype('<u4').tobytes()),('.features',model.packed_fields(fields).tobytes()),('.global',glob.astype('<i4').tobytes()),('.network',model.packed_network())]:Path(str(stem)+suffix).write_bytes(data)
 cases.append(' '.join(map(str,[w,h,*estimate_air(image),*estimate_strong_air(image),predict_weight(image,model.base_model)])))
(fixtures/'cases.txt').write_text('\n'.join(cases),encoding='utf-8')
print('NEURAL_FIXTURES_READY',len(cases))
