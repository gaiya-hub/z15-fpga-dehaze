from pathlib import Path
import sys,json,time,hashlib,shutil
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'quality_v3'/'detail_final';sys.path.insert(0,str(ROOT))
import torch,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps
from model import load_image
from detail_dehaze import DetailDehazer
from evaluation import reference_image,image_scores
torch.set_num_threads(6);checkpoint=OUT/'model.pt'
dehazer=DetailDehazer(checkpoint,json.loads((ROOT/'frozen_model.json').read_text()))
manifest=__import__('research_config').read_manifest(ROOT);rows=[r for r in manifest if r['split']=='holdout']
def extra(clear,pic):
 a=np.asarray(clear,np.float32);b=np.asarray(pic,np.float32);diff=b-a
 la=cv2.cvtColor(a/255,cv2.COLOR_RGB2Lab);lb=cv2.cvtColor(b/255,cv2.COLOR_RGB2Lab)
 return {'rgb_mean_bias':diff.mean((0,1)).tolist(),'luma_bias_abs':float(abs((diff*np.array([.299,.587,.114])).sum(2).mean())),'delta_e76_mean':float(np.linalg.norm(la-lb,axis=2).mean()),'extreme_channel_fraction':float(((b==0)|(b==255)).mean())}
results=[]
for i,r in enumerate(rows):
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size)
 start=time.perf_counter();output,old=dehazer.process(im);ms=(time.perf_counter()-start)*1000
 results.append({**r,'v2':{**image_scores(gt,old),**extra(gt,old)},'candidate':{**image_scores(gt,output),**extra(gt,output)},'hazy':image_scores(gt,im),'cpu_total_ms':ms})
 if i%20==0:print('DETAIL_REGRESSION',i+1,flush=True)
summary={}
for d in ['natural','synthetic','all']:
 part=[r for r in results if d=='all' or r['dataset']==d]
 summary[d]={'n':len(part),**{v:{k:float(np.mean([r[v][k] for r in part])) for k in ['ssim','psnr_db','mae','luma_bias_abs','delta_e76_mean','extreme_channel_fraction']} for v in ['v2','candidate']},'ssim_wins':sum(r['candidate']['ssim']>r['v2']['ssim'] for r in part),'psnr_wins':sum(r['candidate']['psnr_db']>r['v2']['psnr_db'] for r in part),'severe_below_input':sum(r['candidate']['psnr_db']<r['hazy']['psnr_db']-3 for r in part),'ssim_drop_gt_03':sum(r['candidate']['ssim']<r['v2']['ssim']-.03 for r in part)}
(OUT/'regression.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('DETAIL_SUMMARY',json.dumps(summary),flush=True)
examples=[]
for d in ['natural','synthetic']:
 part=sorted([r for r in results if r['dataset']==d],key=lambda r:r['candidate']['ssim']-r['v2']['ssim'])
 for label,r in [('worst',part[0]),('median',part[len(part)//2]),('best',part[-1])]:
  im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);output,old=dehazer.process(im)
  dest=OUT/'examples'/(d+'_'+label);dest.mkdir(parents=True,exist_ok=True)
  for name,pic in [('hazy',im),('gt',gt),('v2',old),('candidate',output)]:pic.save(dest/(name+'.png'))
  w,h=im.size;canvas=Image.new('RGB',(w*4,h+90),'#eef3f8');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
  for c,(title,pic) in enumerate([('有雾输入',im),('V2算法参考（电脑计算）',old),('神经校正候选（电脑）',output),('无雾参考',gt)]):
   draw.text((c*w+8,8),title,font=font,fill='#142b45');canvas.paste(pic,(c*w,60))
   if c in [1,2]:scores=r['v2' if c==1 else 'candidate'];draw.text((c*w+8,32),f'SSIM {scores["ssim"]:.4f}  PSNR {scores["psnr_db"]:.2f}',font=font,fill='#142b45')
  canvas.save(dest/'comparison.png');examples.append({'dataset':d,'selection':label,'row':r,'directory':str(dest)})
(OUT/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
for file in ['detail_dehaze.py','train_detail_v3.py','evaluate_detail_v3.py']:shutil.copy2(Path(__file__).parent/file,OUT/file)
print('DETAIL_EXAMPLES_READY',flush=True)
