from pathlib import Path
import sys,json,hashlib,time
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'quality_v3'/'physical';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from fused_dcp import process,features,strong_reference,estimate_strong_air
from model import load_image,reference,estimate_air
from evaluation import reference_image,image_scores
rows=[r for r in __import__('research_config').read_manifest(ROOT) if r['split']=='development']
old_model=json.loads((ROOT/'frozen_model.json').read_text())
air_methods=['brightest','median','white220','white240','white250']
data=[]
for i,r in enumerate(rows):
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);v2,_=process(im,old_model);base=reference(im,estimate_air(im),160,96,64)
 old=image_scores(gt,v2);variants={};g=np.asarray(gt,dtype=np.int32)
 for air_method in air_methods:
  air=(int(air_method[5:]),)*3 if air_method.startswith('white') else estimate_strong_air(im,air_method)
  s,_=strong_reference(im,air);s=np.asarray(s,dtype=np.int32)
  for basetype,b in [('weak',np.asarray(base,dtype=np.int32)),('input',np.asarray(im,dtype=np.int32))]:
   for weight in [0,64,128,192,256]:
    a=((b*(256-weight)+s*weight+128)>>8).astype(np.uint8);pic=Image.fromarray(a);score=image_scores(gt,pic)
    variants[f'{air_method}:{basetype}:{weight}']=score
 data.append({'row':r,'features':features(im).tolist(),'v2':old,'variants':variants})
 print('PHYSICAL_SCREEN',i+1,flush=True)
 (OUT/'development.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
summaries=[]
for name in data[0]['variants']:
 means={d:{k:float(np.mean([r['variants'][name][k] for r in data if r['row']['dataset']==d])) for k in ['ssim','psnr_db','mae']} for d in ['natural','synthetic']}
 summaries.append({'name':name,'means':means,'macro_ssim':float(np.mean([v['ssim'] for v in means.values()]))})
summaries.sort(key=lambda r:r['macro_ssim'],reverse=True)
(OUT/'constants.json').write_text(json.dumps(summaries,indent=2))
print('BEST_CONSTANTS',json.dumps(summaries[:8]),flush=True)
print('ORACLE_DIAGNOSTIC',json.dumps({d:float(np.mean([max(v['ssim'] for v in r['variants'].values()) for r in data if r['row']['dataset']==d])) for d in ['natural','synthetic']}),flush=True)
