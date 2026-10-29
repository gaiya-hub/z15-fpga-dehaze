from pathlib import Path
import sys,json,time,hashlib,random,argparse
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from model import load_image
from fused_dcp import features
from evaluation import reference_image,image_scores
from grid_dehaze import GridNet,make_features,recover
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=4000);parser.add_argument('--all',action='store_true');args=parser.parse_args()
OUT=ROOT/'quality_v3'/('grid_final' if args.all else 'grid');OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6);torch.manual_seed(93125);random.seed(93125);np.random.seed(93125)
manifest=__import__('research_config').read_manifest(ROOT);base_model=json.loads((ROOT/'frozen_model.json').read_text());dev=[r for r in manifest if r['split']=='development'];val_groups=set()
for d in ['natural','synthetic']:
 groups=sorted({r['group'] for r in dev if r['dataset']==d},key=lambda g:hashlib.sha256(g.encode()).hexdigest());val_groups.update(groups[::4])
train=[];valid=[]
for i,r in enumerate(dev):
 im,_=load_image(r['source'],640);gt,_=reference_image(r['source'],r['gt'],im.size);p,old,detail=make_features(im,base_model)
 row={'row':r,'p':p,'old':old,'detail':detail,'t':np.asarray(gt,dtype=np.float32)/255,'f':features(im),'v2':image_scores(gt,Image.fromarray(np.rint(old*255).astype(np.uint8)))}
 (valid if r['group'] in val_groups and not args.all else train).append(row)
 if i%10==0:print('GRID_CACHE',i+1,flush=True)
mean=np.stack([r['f'] for r in train]).mean(0);scale=np.maximum(.04,np.stack([r['f'] for r in train]).std(0))
for r in train+valid:r['g']=np.clip((r['f']-mean)/scale,-3,3).astype(np.float32)
net=GridNet();opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.001)
grid=torch.arange(11,dtype=torch.float32)-5;k=torch.exp(-grid.square()/4.5);k/=k.sum();window=(k[:,None]*k[None,:])[None,None].expand(3,1,11,11).contiguous()
def ssim(p,t):
 u=F.conv2d(p,window,groups=3);v=F.conv2d(t,window,groups=3);a=F.conv2d(p*p,window,groups=3)-u*u;b=F.conv2d(t*t,window,groups=3)-v*v;c=F.conv2d(p*t,window,groups=3)-u*v
 return (((2*u*v+.0001)*(2*c+.0009))/((u*u+v*v+.0001)*(a+b+.0009))).mean()
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
save('protocol.json',{'train_groups':sorted({r['row']['group'] for r in train}),'validation_groups':sorted({r['row']['group'] for r in valid}),'regression_used_for_training':False,'steps':args.steps,'network':'quarter-resolution19 features + frame22 ->24; depthwise5x5 twice; pointwise24; 4 correction coefficients; full RGB/detail recovery','loss':'unclipped MSE+.1*(1-SSIM)+.15*color mean error+.5*range overflow','board_deployed':False,'base_v2_uses_all_development':True})
@torch.inference_mode()
def validation():
 net.eval();scores=[]
 for r in valid:
  p=torch.from_numpy(r['p'].transpose(2,0,1).copy())[None];g=torch.from_numpy(r['g'])[None];q=net(p,g)[0].permute(1,2,0).numpy();out=recover(r['old'],r['detail'],q);pic=Image.fromarray(np.clip(np.rint(out*255),0,255).astype(np.uint8));gt=Image.fromarray(np.rint(r['t']*255).astype(np.uint8))
  scores.append({'group':r['row']['group'],'dataset':r['row']['dataset'],'v2':r['v2'],'new':image_scores(gt,pic)})
 means={d:{v:{k:float(np.mean([r[v][k] for r in scores if r['dataset']==d])) for k in ['ssim','psnr_db','mae']} for v in ['v2','new']} for d in ['natural','synthetic']};eligible=all(m['new']['psnr_db']>=m['v2']['psnr_db']-.2 and m['new']['mae']<=m['v2']['mae']*1.02 and m['new']['ssim']>=m['v2']['ssim'] for m in means.values());net.train()
 return {'means':means,'eligible':eligible,'macro_ssim':float(np.mean([m['new']['ssim'] for m in means.values()])),'rows':scores}
history=[];best=-1;start=time.time();by_dataset={d:[r for r in train if r['row']['dataset']==d] for d in ['natural','synthetic']}
for step in range(1,args.steps+1):
 pp=[];tt=[];oo=[];dd=[];gg=[]
 for j in range(8):
  r=random.choice(by_dataset['natural' if j%2 else 'synthetic']);h,w=r['t'].shape[:2];y=random.randrange((h-64)//4+1)*4;x=random.randrange((w-64)//4+1)*4
  pp.append(r['p'][y//4:y//4+16,x//4:x//4+16].transpose(2,0,1).copy());tt.append(r['t'][y:y+64,x:x+64].transpose(2,0,1).copy());oo.append(r['old'][y:y+64,x:x+64].transpose(2,0,1).copy());dd.append(r['detail'][y:y+64,x:x+64].transpose(2,0,1).copy());gg.append(r['g'])
 p=torch.from_numpy(np.stack(pp));t=torch.from_numpy(np.stack(tt));old=torch.from_numpy(np.stack(oo));detail=torch.from_numpy(np.stack(dd));g=torch.from_numpy(np.stack(gg))
 q=F.interpolate(net(p,g),size=(64,64),mode='bilinear',align_corners=False);raw=old+q[:,:3].clamp(-.375,.375)+((q[:,3:4]+1).clamp(.5,5)-1)*detail
 loss=F.mse_loss(raw,t)+.1*(1-ssim(raw.clamp(0,1),t))+.15*((raw.mean((2,3))-t.mean((2,3)))**2).mean()+.5*(F.relu(-raw).square()+F.relu(raw-1).square()).mean()
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
 if step in [args.steps//2,args.steps*3//4]:
  for v in opt.param_groups:v['lr']*=.4
 if step%250==0 or step==args.steps:
  if valid:
   result=validation();result.update(step=step,loss=float(loss.detach()),seconds=round(time.time()-start,1));history.append(result);save('training.json',history);print('GRID_VALID',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
   if result['eligible'] and result['macro_ssim']>best:best=result['macro_ssim'];torch.save({'state':net.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'step':step,'validation':result},OUT/'model.pt')
  else:print('GRID_TRAIN',step,float(loss.detach()),round(time.time()-start,1),flush=True)
if args.all:torch.save({'state':net.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'step':args.steps},OUT/'model.pt')
save('complete.json',{'seconds':time.time()-start,'best_macro_ssim':best,'steps':args.steps});print('GRID_COMPLETE',flush=True)
