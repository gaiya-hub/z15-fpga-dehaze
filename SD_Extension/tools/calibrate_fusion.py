from pathlib import Path
import sys,json,time,hashlib
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent.parent;STAGE=ROOT
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(STAGE))
import numpy as np
from PIL import Image
from model import load_image,reference,estimate_air
from evaluation import reference_image,image_scores
from fused_dcp import strong_reference,estimate_strong_air,features,predict_weight,process
import argparse
parser=argparse.ArgumentParser(description='Fit the frame-level fusion controller on development GT only.')
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,default=ROOT/'quality_upgrade')
args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads(args.manifest.read_text(encoding='utf-8'))['manifest']
for item in manifest:
    for key in ['source','gt']:
        p=Path(item[key]);item[key]=str(p if p.is_absolute() else args.manifest.resolve().parent/p)
development=[m for m in manifest if m['split']=='development'];holdout=[m for m in manifest if m['split']=='holdout']
if len({m['group'] for m in development})<5:raise ValueError('At least 5 development scene groups are required')
if {m['group'] for m in development}&{m['group'] for m in holdout}:raise ValueError('Scene groups must not cross development and regression splits')
datasets=sorted({m['dataset'] for m in development})
if not holdout or {m['dataset'] for m in holdout}!=set(datasets):raise ValueError('Regression split must cover the same named datasets')
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
save('protocol.json',{'development_only_fitting':True,'old_holdout_is_regression':True,'edge':640,
    'models':['constant blend 0.25/0.50/0.75','ridge controller alpha 1/10/100, grouped development cross-validation'],
    'air_methods':['brightest','median'],'hardware_status':'integer-oriented specification, not yet deployed',
    'model_selection':'development out-of-fold macro PSNR with per-dataset SSIM >= baseline -0.005',
    'fused_source_sha256':hashlib.sha256((STAGE/'fused_dcp.py').read_bytes()).hexdigest()})
dev=[]
for i,item in enumerate(m for m in manifest if m['split']=='development'):
    im,_=load_image(item['source'],640);gt,_=reference_image(item['source'],item['gt'],im.size)
    base=reference(im,estimate_air(im),160,96,64);branches={}
    for method in ['brightest','median']:
        strong,_=strong_reference(im,estimate_strong_air(im,method));branches[method]=np.asarray(strong,dtype=np.float64)
    dev.append({'item':item,'image':im,'gt':gt,'base':np.asarray(base,dtype=np.float64),'branches':branches,'features':features(im),'baseline_scores':image_scores(gt,base)})
    if i%10==0:print('DEV_BRANCHES',i+1,'/',len(development),flush=True)
X=np.stack([r['features'] for r in dev]);groups=sorted({r['item']['group'] for r in dev},key=lambda name:hashlib.sha256(name.encode()).hexdigest())
fold={g:i%5 for i,g in enumerate(groups)};fold_ids=np.array([fold[r['item']['group']] for r in dev])
def fit(indices,targets,alpha):
    mean=X[indices].mean(axis=0);scale=np.maximum(X[indices].std(axis=0),.02)
    x=(X[indices]-mean)/scale;y=targets[indices];bias=float(y.mean())
    weights=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@(y-bias))
    return {'kind':'ridge','mean':mean.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),'bias':bias}
models=[]
for method in ['brightest','median']:
    targets=[]
    for row in dev:
        difference=row['branches'][method]-row['base'];target=np.asarray(row['gt'],dtype=np.float64)-row['base']
        targets.append(float(np.clip(np.sum(target*difference)/max(1,np.sum(difference*difference)),0,1)))
    targets=np.array(targets)
    for choice in [.25,.5,.75,1,10,100]:
        if choice<1:
            model={'kind':'constant','weight_q8':int(choice*256)};predicted=np.full(len(dev),choice)
        else:
            predicted=np.empty(len(dev))
            for f in range(5):
                train=np.where(fold_ids!=f)[0];test=np.where(fold_ids==f)[0];m=fit(train,targets,choice)
                predicted[test]=np.clip(((X[test]-m['mean'])/m['scale'])@np.asarray(m['weights'])+m['bias'],0,1)
            model=fit(np.arange(len(dev)),targets,choice);model['alpha']=choice
        scores=[]
        for row,w in zip(dev,predicted):
            weight=int(np.clip(round(float(w)*256),0,256))
            array=np.clip((row['base']*(256-weight)+row['branches'][method]*weight+128)//256,0,255).astype(np.uint8)
            scores.append(image_scores(row['gt'],Image.fromarray(array)))
        means={d:{key:float(np.mean([v[key] for r,v in zip(dev,scores) if r['item']['dataset']==d])) for key in ['psnr_db','ssim']} for d in datasets}
        eligible=all(means[d]['ssim']>=np.mean([r['baseline_scores']['ssim'] for r in dev if r['item']['dataset']==d])-.005 for d in means)
        models.append({'model':{**model,'air_method':method},'development_cv':means,'macro_psnr':float(np.mean([v['psnr_db'] for v in means.values()])),'eligible':eligible})
eligible=[m for m in models if m['eligible']]
save('development_models.json',models)
if not eligible:raise RuntimeError('No model passed development screening')
selected=max(eligible,key=lambda m:m['macro_psnr']);save('frozen_model.json',selected['model']);save('selection.json',selected)
print('FROZEN',json.dumps(selected),flush=True)
rows=[]
for i,item in enumerate(m for m in manifest if m['split']=='holdout'):
    im,_=load_image(item['source'],640);gt,_=reference_image(item['source'],item['gt'],im.size)
    baseline=reference(im,estimate_air(im),160,96,64)
    started=time.perf_counter();output,parameters=process(im,selected['model']);milliseconds=(time.perf_counter()-started)*1000
    rows.append({**item,'baseline':image_scores(gt,baseline),'fused':image_scores(gt,output),'hazy':image_scores(gt,im),'ms':milliseconds,'parameters':parameters})
    if i%20==0:print('REGRESSION',i+1,'/',len(holdout),flush=True)
save('regression.json',rows)
summary={}
for d in datasets+['all']:
    part=[r for r in rows if d=='all' or r['dataset']==d]
    summary[d]={'n':len(part),'baseline':{k:float(np.mean([r['baseline'][k] for r in part])) for k in ['psnr_db','ssim','mae']},
        'fused':{k:float(np.mean([r['fused'][k] for r in part])) for k in ['psnr_db','ssim','mae']},
        'improved_over_baseline':sum(r['fused']['psnr_db']>r['baseline']['psnr_db'] for r in part),
        'severe_below_input':sum(r['fused']['psnr_db']<r['hazy']['psnr_db']-3 for r in part),
        'mean_ms':float(np.mean([r['ms'] for r in part]))}
save('summary.json',{'model':selected,'regression':summary,'status':'software result; no FPGA claim yet'})
print('FUSION_RESULT',json.dumps(summary),flush=True)
