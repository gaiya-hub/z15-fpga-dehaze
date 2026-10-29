from pathlib import Path
import sys,json,time,hashlib,random,os
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'quality_v3'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from model import load_image
from fused_dcp import process,features
from evaluation import reference_image,image_scores
from quality_v3_net import ResidualNet
torch.set_num_threads(6);torch.manual_seed(31025);np.random.seed(31025);random.seed(31025)
OUT.mkdir(parents=True,exist_ok=True)
rows=__import__('research_config').read_manifest(ROOT)
base_model=json.loads((ROOT/'frozen_model.json').read_text())
development=[r for r in rows if r['split']=='development']
groups=sorted({r['group'] for r in development},key=lambda g:hashlib.sha256(g.encode()).hexdigest())
valid_groups=set(groups[::5]);train=[];valid=[]
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
save('protocol.json',{'target_mean_ssim':.90,'evaluation_edge':640,'regression_count':134,
 'train_groups':[g for g in groups if g not in valid_groups],'internal_validation_groups':sorted(valid_groups),
 'regression_is_previously_viewed':True,'base_v2_controller_used_all_49_development_images':True,
 'no_regression_GT_for_training':True,'loss':'MSE + 0.08*(1-SSIM) + 0.05*channel-mean squared error',
 'network':'(6 local RGB + 22 frame statistics) -> 24 ReLU -> 16 ReLU -> 3; bounded residual +/-95.625 RGB levels',
 'board_deployed':False,'seed':31025,'same_metrics_as_v2':True})
for i,r in enumerate(development):
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);base,_=process(im,base_model)
 item={'row':r,'pixels':np.concatenate([np.asarray(im),np.asarray(base)],axis=2),'gt':np.asarray(gt),'global':features(im),'baseline':image_scores(gt,base)}
 (valid if r['group'] in valid_groups else train).append(item)
 if i%10==0:print('CACHE',i+1,len(development),flush=True)
mean=np.stack([r['global'] for r in train]).mean(0);scale=np.maximum(.03,np.stack([r['global'] for r in train]).std(0))
for row in train+valid:row['g']=((row['global']-mean)/scale).astype(np.float32)
net=ResidualNet();optimizer=torch.optim.Adam(net.parameters(),lr=.001)
grid=torch.arange(11,dtype=torch.float32)-5;kernel=torch.exp(-grid.square()/(2*1.5**2));kernel/=kernel.sum();window=(kernel[:,None]*kernel[None,:])[None,None].expand(3,1,11,11).contiguous()
def ssim(pred,target):
 mu=F.conv2d(pred,window,groups=3);mt=F.conv2d(target,window,groups=3)
 var=F.conv2d(pred*pred,window,groups=3)-mu*mu;vt=F.conv2d(target*target,window,groups=3)-mt*mt
 cov=F.conv2d(pred*target,window,groups=3)-mu*mt
 return (((2*mu*mt+.01**2)*(2*cov+.03**2))/((mu*mu+mt*mt+.01**2)*(var+vt+.03**2))).mean()
def validation():
 net.eval();scores=[]
 with torch.inference_mode():
  for row in valid:
   p=torch.tensor(row['pixels'].reshape(-1,6),dtype=torch.float32)/255;g=torch.from_numpy(row['g'])
   predictions=[]
   for start in range(0,len(p),65536):predictions.append(net(p[start:start+65536],g.expand(min(65536,len(p)-start),-1)).numpy())
   out=np.clip(np.rint(np.concatenate(predictions).reshape(row['gt'].shape)*255),0,255).astype(np.uint8)
   scores.append({'dataset':row['row']['dataset'],'group':row['row']['group'],**image_scores(Image.fromarray(row['gt']),Image.fromarray(out))})
 means={d:{k:float(np.mean([s[k] for s in scores if s['dataset']==d])) for k in ['ssim','psnr_db','mae']} for d in ['natural','synthetic']}
 baseline={d:{k:float(np.mean([r['baseline'][k] for r in valid if r['row']['dataset']==d])) for k in ['ssim','psnr_db','mae']} for d in ['natural','synthetic']}
 eligible=all(means[d]['psnr_db']>=baseline[d]['psnr_db']-.25 and means[d]['mae']<=baseline[d]['mae']*1.025 for d in means)
 net.train();return {'means':means,'baseline':baseline,'eligible':eligible,'ssim':float(np.mean([m['ssim'] for m in means.values()])),'scores':scores}
history=[];best=-1;started=time.time()
for step in range(1,1601):
 pp=[];tt=[];gg=[]
 for j in range(8):
  row=random.choice(train);h,w=row['gt'].shape[:2];y=random.randrange(h-48+1);x=random.randrange(w-48+1)
  p=row['pixels'][y:y+48,x:x+48].copy();t=row['gt'][y:y+48,x:x+48].copy()
  if random.random()<.5:p=p[:,::-1].copy();t=t[:,::-1].copy()
  pp.append(p);tt.append(t);gg.append(row['g'])
 pixels=torch.from_numpy(np.stack(pp)).float()/255;target=torch.from_numpy(np.stack(tt)).float()/255
 glob=torch.from_numpy(np.stack(gg))[:,None,None,:].expand(-1,48,48,-1)
 pred=net(pixels,glob);mse=F.mse_loss(pred,target);ss=ssim(pred.permute(0,3,1,2),target.permute(0,3,1,2))
 loss=mse+.08*(1-ss)+.05*((pred.mean((1,2))-target.mean((1,2)))**2).mean()
 optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
 if step in [600,1000,1400]:
  for group in optimizer.param_groups:group['lr']*=.5
 if step%100==0:
  result=validation();result.update(step=step,loss=float(loss),seconds=round(time.time()-started,1));history.append(result);save('training.json',history)
  print('VALID',json.dumps({k:v for k,v in result.items() if k not in ['scores','baseline']}),flush=True)
  if result['eligible'] and result['ssim']>best:
   best=result['ssim'];torch.save({'state':net.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'step':step,'validation':result},OUT/'residual_v3.pt')
save('training_complete.json',{'best_validation_ssim':best,'seconds':time.time()-started,'checkpoint':str(OUT/'residual_v3.pt')})
print('TRAINING_COMPLETE',best,flush=True)
