"""Frame-level parameter selection; no GT, network or file matching at inference."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import ImageOps
from model import min_axis

PATH=Path(__file__).resolve().parent/'selection'/'parameter_policy.json'

def haze_proxy(image,air):
    small=ImageOps.contain(image,(320,320))
    rgb=np.asarray(small,dtype=np.uint8)
    dark=min_axis(min_axis(rgb.min(axis=2),7,1),7,0)
    return float(np.mean(dark)/max(1,float(np.mean(air))))

def select_parameters(image,air,policy=None):
    started=time.perf_counter()
    if policy is None:
        raw=PATH.read_bytes();policy=json.loads(raw)
        digest=hashlib.sha256(raw).hexdigest()
    else:digest=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    if policy.get('schema')!=1:raise ValueError('参数策略版本不受支持')
    thresholds=policy['thresholds'];configs=policy['parameters']
    if len(thresholds)!=2 or thresholds[0]>thresholds[1] or len(configs)!=3:raise ValueError('参数策略格式错误')
    proxy=haze_proxy(image,air)
    index=int(proxy>thresholds[0])+int(proxy>thresholds[1])
    values=configs[index]
    strength,tmin,lift=map(int,values)
    if not (0<=strength<=243 and 32<=tmin<=255 and 0<=lift<=128):raise ValueError('参数策略越界')
    metadata={'mode':'calibrated-frame-parameters','policy_version':policy['version'],'policy_sha256':digest,
              'proxy':proxy,'bin':index,'strength':strength,'t_min':tmin,'lift':lift,
              'controller_ms':(time.perf_counter()-started)*1000,
              'execution':'PC selects frame parameters; FPGA executes unchanged fixed-point recovery',
              'ground_truth_used_at_runtime':False,'neural_network':False}
    return (strength,tmin,lift),metadata
