from pathlib import Path
import sys,json,time,itertools
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'quality_v3'/'safe_detail';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT))
import torch,numpy as np,cv2
from PIL import Image
from model import load_image
from evaluation import reference_image,image_scores
from detail_dehaze import DetailDehazer
from safe_detail_v3 import protect
torch.set_num_threads(6)
protocol=json.loads((ROOT/'quality_v3/detail/protocol.json').read_text(encoding='utf-8'));groups=set(protocol['validation_groups'])
rows=[r for r in __import__('research_config').read_manifest(ROOT) if r['split']=='development' and r['group'] in groups]
model=DetailDehazer(ROOT/'quality_v3/detail/model.pt',json.loads((ROOT/'frozen_model.json').read_text()));data=[]
def scores(gt,pic):
 s=image_scores(gt,pic);a=np.asarray(gt,dtype=np.float32)/255;b=np.asarray(pic,dtype=np.float32)/255
 s['delta_e76_mean']=float(np.linalg.norm(cv2.cvtColor(a,cv2.COLOR_RGB2Lab)-cv2.cvtColor(b,cv2.COLOR_RGB2Lab),axis=2).mean());s['luma_bias_abs']=float(abs(((b-a)*np.array([.299,.587,.114])).sum(2).mean())*255)
 return s
for r in rows:
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);pic,base=model.process(im);data.append({'row':r,'gt':gt,'pic':pic,'base':base,'old':scores(gt,base)})
results=[]
for floor,blend,limit in itertools.product([.25,.5,.75],[.5,.75,1],[32,64]):
 config={'dark_floor':floor,'bright_headroom':.25,'blend':blend,'max_delta':limit};values=[]
 for r in data:values.append({'dataset':r['row']['dataset'],'old':r['old'],'new':scores(r['gt'],protect(r['base'],r['pic'],config))})
 means={d:{v:{k:float(np.mean([r[v][k] for r in values if r['dataset']==d])) for k in ['ssim','psnr_db','mae','delta_e76_mean','luma_bias_abs']} for v in ['old','new']} for d in ['natural','synthetic']}
 eligible=all(m['new']['psnr_db']>=m['old']['psnr_db'] and m['new']['mae']<=m['old']['mae'] and m['new']['ssim']>=m['old']['ssim'] and m['new']['delta_e76_mean']<=m['old']['delta_e76_mean'] and m['new']['luma_bias_abs']<=m['old']['luma_bias_abs'] for m in means.values())
 severe=sum(r['new']['psnr_db']<r['old']['psnr_db']-3 or r['new']['ssim']<r['old']['ssim']-.03 for r in values)
 results.append({'config':config,'means':means,'eligible':bool(eligible and severe==0),'severe_new':severe,'macro_ssim':float(np.mean([m['new']['ssim'] for m in means.values()]))})
 print('GUARD',floor,blend,limit,eligible,severe,results[-1]['macro_ssim'],flush=True)
eligible=[r for r in results if r['eligible']];selected=max(eligible,key=lambda r:r['macro_ssim']) if eligible else None
(OUT/'selection.json').write_text(json.dumps({'selected':selected,'candidates':results,'selection_split':'internal development validation only','regression_used_for_guard_selection':False},ensure_ascii=False,indent=2),encoding='utf-8');print('GUARD_SELECTED',json.dumps(selected),flush=True)
