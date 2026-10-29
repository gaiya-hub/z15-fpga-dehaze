"""Generate deterministic synthetic reference fixtures; no external image/model required."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from model import reference,estimate_air,pack
from fused_dcp import process
model=json.loads((ROOT/'frozen_model.json').read_text(encoding='utf-8'))
out=ROOT/'hls/fixtures';out.mkdir(exist_ok=True);cases=[]
sizes=[(1,1),(5,3),(17,11),(64,43),(127,65),(320,213),(640,360)]
if '--extended' in sys.argv:sizes.extend([(1920,1280),(65,129),(7,19)])
for index,(w,h) in enumerate(sizes):
    rng=np.random.default_rng(w);rgb=rng.integers(30,240,(h,w,3),dtype=np.uint8)
    if index==8:rgb[:]=[249,250,251];rgb[h//2:]=[20,24,17]
    if index==9:rgb[:]=0
    im=Image.fromarray(rgb);weight=[0,64,128,192,256,91,207,193,193,193][index]
    golden,params=process(im,{'kind':'constant','weight_q8':weight,'air_method':model['air_method']})
    air=estimate_air(im);name=f'{w}x{h}'
    (out/(name+'.input')).write_bytes(pack(im));(out/(name+'.expected')).write_bytes(pack(golden))
    (out/(name+'.baseline')).write_bytes(pack(reference(im,air,160,96,64)))
    cases.append(' '.join(map(str,[w,h,*air,*params['strong_air'],weight])))
(out/'cases.txt').write_text('\n'.join(cases),encoding='utf-8')
print('REFERENCE_FIXTURES_READY',len(cases))
