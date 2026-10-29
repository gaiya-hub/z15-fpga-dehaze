from pathlib import Path
import sys,json,time
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'quality_v3'/'scale';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from PIL import Image
from fused_dcp import process,estimate_strong_air
from model import load_image
from evaluation import reference_image,image_scores
source=(ROOT/'fused_dcp.py').read_text(encoding='utf-8');funcs={}
for k in [3,7,15]:
 for radius in [2,5]:
  ns={};exec(compile(source.replace('np.ones((15,15),np.uint8)','np.ones(('+str(k)+','+str(k)+'),np.uint8)').replace('np.float64),5)','np.float64),'+str(radius)+')'),'<research-'+str(k)+'-'+str(radius)+'>','exec'),ns)
  funcs[k,radius]=ns['strong_reference']
def score(gt,a):
 x=np.asarray(gt,dtype=np.float32);y=a.astype(np.float32);mse=float(np.mean((x-y)**2));mu=lambda f:cv2.GaussianBlur(f,(11,11),1.5,borderType=cv2.BORDER_REFLECT)
 u=mu(x);v=mu(y);vx=mu(x*x)-u*u;vy=mu(y*y)-v*v;cov=mu(x*y)-u*v
 s=((2*u*v+6.5025)*(2*cov+58.5225))/((u*u+v*v+6.5025)*(vx+vy+58.5225))
 return {'ssim':float(s[5:-5,5:-5].mean()),'psnr_db':float(10*np.log10(65025/mse)),'mae':float(np.abs(x-y).mean())}
rows=[r for r in __import__('research_config').read_manifest(ROOT) if r['split']=='development'];old_model=json.loads((ROOT/'frozen_model.json').read_text());data=[]
for i,r in enumerate(rows):
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);v2,_=process(im,old_model);old=np.asarray(v2,dtype=np.int32);variants={}
 for (k,radius),fn in funcs.items():
  for air_method in ['brightest','white240']:
   air=estimate_strong_air(im) if air_method=='brightest' else (240,240,240)
   for strength in [230,243]:
    strong,_=fn(im,air,strength=strength);s=np.asarray(strong,dtype=np.int32)
    for weight in [128,256]:
     pic=((old*(256-weight)+s*weight+128)>>8).astype(np.uint8)
     variants[f'{k}:{radius}:{air_method}:{strength}:{weight}']=score(gt,pic)
 data.append({'row':r,'v2':score(gt,np.asarray(v2)),'variants':variants});print('SCALE_SCREEN',i+1,flush=True)
 (OUT/'development.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
summaries=[]
for name in data[0]['variants']:
 means={d:{k:float(np.mean([r['variants'][name][k] for r in data if r['row']['dataset']==d])) for k in ['ssim','psnr_db','mae']} for d in ['natural','synthetic']}
 eligible=all(means[d]['psnr_db']>=np.mean([r['v2']['psnr_db'] for r in data if r['row']['dataset']==d])-.25 and means[d]['mae']<=np.mean([r['v2']['mae'] for r in data if r['row']['dataset']==d])*1.02 for d in means)
 summaries.append({'name':name,'means':means,'macro_ssim':float(np.mean([v['ssim'] for v in means.values()])),'eligible':bool(eligible)})
summaries.sort(key=lambda r:r['macro_ssim'],reverse=True)
(OUT/'summary.json').write_text(json.dumps(summaries,indent=2));print('SCALE_BEST',json.dumps(summaries[:6]),flush=True)
