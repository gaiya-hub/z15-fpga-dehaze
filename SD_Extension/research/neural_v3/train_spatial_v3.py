from pathlib import Path
import sys,json,time,hashlib,random,argparse
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from model import load_image
from fused_dcp import process,features,strong_reference,estimate_strong_air
from evaluation import reference_image,image_scores
from quality_v3_spatial import SpatialNet
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=3000);parser.add_argument('--all-development',action='store_true');args=parser.parse_args()
OUT=ROOT/'quality_v3'/('spatial_final' if args.all_development else 'spatial');OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6);torch.manual_seed(45125);random.seed(45125);np.random.seed(45125)
rows=__import__('research_config').read_manifest(ROOT);base_model=json.loads((ROOT/'frozen_model.json').read_text())
dev=[r for r in rows if r['split']=='development'];val_groups=set()
for d in ['natural','synthetic']:
 groups=sorted({r['group'] for r in dev if r['dataset']==d},key=lambda g:hashlib.sha256(g.encode()).hexdigest());val_groups.update(groups[::4])
train=[];valid=[]
for i,r in enumerate(dev):
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);base,_=process(im,base_model);strong,_=strong_reference(im,estimate_strong_air(im))
 row={'row':r,'p':np.concatenate([np.asarray(im),np.asarray(base),np.asarray(strong)],axis=2),'t':np.asarray(gt),'f':features(im),'v2':image_scores(gt,base)}
 (valid if r['group'] in val_groups and not args.all_development else train).append(row)
 if i%10==0:print('SPATIAL_CACHE',i+1,flush=True)
mean=np.stack([r['f'] for r in train]).mean(0);scale=np.maximum(.03,np.stack([r['f'] for r in train]).std(0))
for r in train+valid:r['g']=np.clip((r['f']-mean)/scale,-4,4).astype(np.float32)
net=SpatialNet();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
grid=torch.arange(11,dtype=torch.float32)-5;k=torch.exp(-grid.square()/4.5);k/=k.sum();window=(k[:,None]*k[None,:])[None,None].expand(3,1,11,11).contiguous()
def ssim(pred,t):
 u=F.conv2d(pred,window,groups=3);v=F.conv2d(t,window,groups=3);a=F.conv2d(pred*pred,window,groups=3)-u*u;b=F.conv2d(t*t,window,groups=3)-v*v;c=F.conv2d(pred*t,window,groups=3)-u*v
 return (((2*u*v+.0001)*(2*c+.0009))/((u*u+v*v+.0001)*(a+b+.0009))).mean()
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
save('protocol.json',{'network':'DCP -> bounded residual CNN; 16 channels, two depthwise5x5, pointwise1x1; global22 stats','train_groups':sorted({r['row']['group'] for r in train}),'validation_groups':sorted({r['row']['group'] for r in valid}),'regression_used_for_training':False,'steps':args.steps,'base_v2_controller_used_all_49_development':True,'board_deployed':False,'ssim_evaluation_unchanged':True})
@torch.inference_mode()
def validation():
 net.eval();scores=[]
 for r in valid:
  p=torch.from_numpy(r['p'].transpose(2,0,1).copy()).float()[None]/255;g=torch.from_numpy(r['g'])[None]
  # Full-width stripes with a 4-pixel halo exactly match the network receptive field.
  out=[]
  for y in range(0,p.shape[2],64):
   lo=max(0,y-4);hi=min(p.shape[2],y+64+4)
   part=net(p[:,:,lo:hi],g)[0,:,y-lo:min(y+64,p.shape[2])-lo];out.append(part)
  a=torch.cat(out,1).permute(1,2,0).numpy();pic=Image.fromarray(np.clip(np.rint(a*255),0,255).astype(np.uint8))
  scores.append({'group':r['row']['group'],'dataset':r['row']['dataset'],'v2':r['v2'],'new':image_scores(Image.fromarray(r['t']),pic)})
 means={d:{v:{k:float(np.mean([r[v][k] for r in scores if r['dataset']==d])) for k in ['ssim','psnr_db','mae']} for v in ['v2','new']} for d in ['natural','synthetic']}
 eligible=all(m['new']['psnr_db']>=m['v2']['psnr_db']-.2 and m['new']['mae']<=m['v2']['mae']*1.02 and m['new']['ssim']>=m['v2']['ssim'] for m in means.values())
 result={'means':means,'eligible':eligible,'macro_ssim':float(np.mean([m['new']['ssim'] for m in means.values()])),'rows':scores};net.train();return result
history=[];best=-1;start=time.time()
for step in range(1,args.steps+1):
 pp=[];tt=[];gg=[]
 for j in range(6):
  # Equal chance for each scene group, reducing domination by repeated synthetic frames.
  group=random.choice(list(dict.fromkeys(r['row']['group'] for r in train)));r=random.choice([v for v in train if v['row']['group']==group]);h,w=r['t'].shape[:2];y=random.randrange(h-64+1);x=random.randrange(w-64+1)
  p=r['p'][y:y+64,x:x+64];t=r['t'][y:y+64,x:x+64]
  if random.random()<.5:p=p[:,::-1];t=t[:,::-1]
  pp.append(p.transpose(2,0,1).copy());tt.append(t.transpose(2,0,1).copy());gg.append(r['g'])
 p=torch.from_numpy(np.stack(pp)).float()/255;t=torch.from_numpy(np.stack(tt)).float()/255;g=torch.from_numpy(np.stack(gg))
 pred=net(p,g);mse=F.mse_loss(pred,t);loss=mse+.08*(1-ssim(pred,t))+.05*((pred.mean((2,3))-t.mean((2,3)))**2).mean()
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
 if step in [args.steps//2,args.steps*3//4]:
  for v in opt.param_groups:v['lr']*=.4
 if step%200==0 or step==args.steps:
  if valid:
   result=validation();result.update(step=step,loss=float(loss.detach()),seconds=round(time.time()-start,1));history.append(result);save('training.json',history)
   print('SPATIAL_VALID',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
   if result['eligible'] and result['macro_ssim']>best:
    best=result['macro_ssim'];torch.save({'state':net.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'step':step,'validation':result},OUT/'model.pt')
  else:print('SPATIAL_TRAIN',step,float(loss.detach()),round(time.time()-start,1),flush=True)
if args.all_development:torch.save({'state':net.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'step':args.steps},OUT/'model.pt')
save('complete.json',{'seconds':time.time()-start,'best_macro_ssim':best,'steps':args.steps});print('SPATIAL_COMPLETE',flush=True)
