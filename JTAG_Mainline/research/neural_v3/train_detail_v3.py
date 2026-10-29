from pathlib import Path
import sys,json,time,hashlib,random,argparse
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from model import load_image
from fused_dcp import process,features,strong_reference,estimate_strong_air
from evaluation import reference_image,image_scores
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=4000);parser.add_argument('--all',action='store_true');args=parser.parse_args()
OUT=ROOT/'quality_v3'/('detail_final' if args.all else 'detail');OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6);torch.manual_seed(73125);random.seed(73125);np.random.seed(73125)
rows=__import__('research_config').read_manifest(ROOT);base_model=json.loads((ROOT/'frozen_model.json').read_text())
dev=[r for r in rows if r['split']=='development'];val_groups=set()
for d in ['natural','synthetic']:
 groups=sorted({r['group'] for r in dev if r['dataset']==d},key=lambda g:hashlib.sha256(g.encode()).hexdigest());val_groups.update(groups[::4])
train=[];valid=[]
for i,r in enumerate(dev):
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);v2,_=process(im,base_model);strong,info=strong_reference(im,estimate_strong_air(im))
 orig=np.asarray(im).astype(np.float32)/255;old=np.asarray(v2).astype(np.float32)/255;ss=np.asarray(strong).astype(np.float32)/255
 avg=lambda a:cv2.boxFilter(a,cv2.CV_32F,(15,15),borderType=cv2.BORDER_REFLECT_101)
 om=avg(orig);bm=avg(old);std=np.sqrt(np.maximum(0,avg(orig*orig)-om*om))
 local=np.concatenate([orig,old,ss,om,bm,std,info['transmission'][:,:,None]/255],2).astype(np.float32)
 # Pass high-pass input and its low-pass reference separately to keep texture explicit.
 data={'row':r,'p':local,'t':np.asarray(gt),'f':features(im),'v2':image_scores(gt,v2)}
 (valid if r['group'] in val_groups and not args.all else train).append(data)
 if i%10==0:print('DETAIL_CACHE',i+1,flush=True)
mean=np.stack([r['f'] for r in train]).mean(0);scale=np.maximum(.04,np.stack([r['f'] for r in train]).std(0))
for r in train+valid:r['g']=np.clip((r['f']-mean)/scale,-3,3).astype(np.float32)
class DetailNet(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Linear(19+22,32),nn.ReLU(),nn.Linear(32,24),nn.ReLU(),nn.Linear(24,4));nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
 def forward(self,p,g):
  q=self.layers(torch.cat([p,g],-1));gain=(q[...,3:4]+1).clamp(.5,5)
  # At zero residual the model equals V2. Extra detail is bounded, without GAN/text synthesis.
  detail=p[...,:3]-p[...,9:12]
  return (p[...,3:6]+q[...,:3].clamp(-.375,.375)+(gain-1)*detail).clamp(0,1)
net=DetailNet();opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.001)
grid=torch.arange(11,dtype=torch.float32)-5;k=torch.exp(-grid.square()/4.5);k/=k.sum();window=(k[:,None]*k[None,:])[None,None].expand(3,1,11,11).contiguous()
def ssim(pred,t):
 u=F.conv2d(pred,window,groups=3);v=F.conv2d(t,window,groups=3);a=F.conv2d(pred*pred,window,groups=3)-u*u;b=F.conv2d(t*t,window,groups=3)-v*v;c=F.conv2d(pred*t,window,groups=3)-u*v
 return (((2*u*v+.0001)*(2*c+.0009))/((u*u+v*v+.0001)*(a+b+.0009))).mean()
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
save('protocol.json',{'train_groups':sorted({r['row']['group'] for r in train}),'validation_groups':sorted({r['row']['group'] for r in valid}),'regression_used_for_training':False,'steps':args.steps,'network':'local DCP/box-statistics 19 + frame22 ->32ReLU->24ReLU->4; RGB residual and bounded shared detail gain','loss':'MSE+.12*(1-SSIM)+.1*color mean error','board_deployed':False,'base_v2_uses_all_development':True})
@torch.inference_mode()
def validation():
 net.eval();scores=[]
 for r in valid:
  p=torch.from_numpy(r['p'].reshape(-1,19));g=torch.from_numpy(r['g']);out=[]
  for k in range(0,len(p),32768):out.append(net(p[k:k+32768],g.expand(min(32768,len(p)-k),-1)).numpy())
  pic=Image.fromarray(np.clip(np.rint(np.concatenate(out).reshape(r['t'].shape)*255),0,255).astype(np.uint8))
  scores.append({'group':r['row']['group'],'dataset':r['row']['dataset'],'v2':r['v2'],'new':image_scores(Image.fromarray(r['t']),pic)})
 means={d:{v:{k:float(np.mean([r[v][k] for r in scores if r['dataset']==d])) for k in ['ssim','psnr_db','mae']} for v in ['v2','new']} for d in ['natural','synthetic']}
 eligible=all(m['new']['psnr_db']>=m['v2']['psnr_db']-.2 and m['new']['mae']<=m['v2']['mae']*1.02 and m['new']['ssim']>=m['v2']['ssim'] for m in means.values())
 result={'means':means,'eligible':eligible,'macro_ssim':float(np.mean([m['new']['ssim'] for m in means.values()])),'rows':scores};net.train();return result
history=[];best=-1;start=time.time();by_dataset={d:[r for r in train if r['row']['dataset']==d] for d in ['natural','synthetic']}
for step in range(1,args.steps+1):
 pp=[];tt=[];gg=[]
 for j in range(8):
  r=random.choice(by_dataset['natural' if j%2 else 'synthetic']);h,w=r['t'].shape[:2];y=random.randrange(h-48+1);x=random.randrange(w-48+1)
  p=r['p'][y:y+48,x:x+48];t=r['t'][y:y+48,x:x+48]
  if random.random()<.5:p=p[:,::-1];t=t[:,::-1]
  pp.append(p.copy());tt.append(t.copy());gg.append(r['g'])
 p=torch.from_numpy(np.stack(pp));t=torch.from_numpy(np.stack(tt)).float()/255;g=torch.from_numpy(np.stack(gg))[:,None,None,:].expand(-1,48,48,-1)
 pred=net(p,g);loss=F.mse_loss(pred,t)+.12*(1-ssim(pred.permute(0,3,1,2),t.permute(0,3,1,2)))+.1*((pred.mean((1,2))-t.mean((1,2)))**2).mean()
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
 if step in [args.steps//2,args.steps*3//4]:
  for v in opt.param_groups:v['lr']*=.4
 if step%250==0 or step==args.steps:
  if valid:
   result=validation();result.update(step=step,loss=float(loss.detach()),seconds=round(time.time()-start,1));history.append(result);save('training.json',history)
   print('DETAIL_VALID',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
   if result['eligible'] and result['macro_ssim']>best:
    best=result['macro_ssim'];torch.save({'state':net.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'step':step,'validation':result},OUT/'model.pt')
  else:print('DETAIL_TRAIN',step,float(loss.detach()),round(time.time()-start,1),flush=True)
if args.all:torch.save({'state':net.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'step':args.steps},OUT/'model.pt')
save('complete.json',{'seconds':time.time()-start,'best_macro_ssim':best,'steps':args.steps});print('DETAIL_COMPLETE',flush=True)
