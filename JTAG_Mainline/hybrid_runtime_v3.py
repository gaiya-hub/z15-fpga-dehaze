"""V3: verified FPGA DCP output + PC auxiliary features, neural correction and guard."""
from pathlib import Path
import json,hashlib,time,threading
ROOT=Path(__file__).resolve().parent
ALGORITHM='fused-neural-v3-hybrid'
LABEL='优化V3（FPGA＋电脑神经校正）'
_lock=threading.RLock();_processor=None;_fingerprint=None

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def artifacts():
    paths=[ROOT/'neural_v3/model.pt',ROOT/'neural_v3/candidate.json',ROOT/'frozen_model.json',ROOT/'detail_dehaze.py',ROOT/'safe_detail_v3.py',Path(__file__)]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}

class Processor:
    def __init__(self):
        import torch
        from detail_dehaze import DetailDehazer
        self.metadata=json.loads((ROOT/'neural_v3/candidate.json').read_text(encoding='utf-8'))
        if digest(ROOT/'neural_v3/model.pt')!=self.metadata['checkpoint_sha256']:raise RuntimeError('V3神经权重校验失败，请恢复neural_v3目录。')
        if digest(ROOT/'frozen_model.json')!=self.metadata['base_controller_sha256']:raise RuntimeError('V3使用的V2控制器已改变，请恢复对应版本后再处理。')
        torch.set_num_threads(min(6,torch.get_num_threads()))
        self.model=DetailDehazer(ROOT/'neural_v3/model.pt',json.loads((ROOT/'frozen_model.json').read_text(encoding='utf-8')))

    def correct(self,image,base):
        import numpy as np,torch
        from PIL import Image
        from fused_dcp import features
        from detail_dehaze import local_features
        from safe_detail_v3 import protect
        if image.size!=base.size:raise ValueError('V3输入与FPGA基础输出尺寸不一致。')
        begin=time.perf_counter()
        local,_=local_features(image,self.model.base_model,base_image=base)
        g=torch.tensor(np.clip((features(image)-self.model.mean)/self.model.scale,-3,3),dtype=torch.float32)
        p=torch.from_numpy(local.reshape(-1,19));middle=time.perf_counter();output=[]
        with torch.inference_mode():
            for start in range(0,len(p),32768):
                part=p[start:start+32768];output.append(self.model.net(part,g.expand(len(part),-1)).numpy())
        pic=Image.fromarray(np.clip(np.rint(np.concatenate(output).reshape(image.height,image.width,3)*255),0,255).astype(np.uint8))
        neural_done=time.perf_counter();pic=protect(base,pic,self.metadata['guard']);end=time.perf_counter()
        return pic,{'host_feature_ms':(middle-begin)*1000,'host_neural_ms':(neural_done-middle)*1000,
                    'host_guard_ms':(end-neural_done)*1000,'host_post_ms':(end-begin)*1000}

def processor():
    global _processor,_fingerprint
    with _lock:
        fingerprint=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in [ROOT/'neural_v3/model.pt',ROOT/'neural_v3/candidate.json',ROOT/'frozen_model.json'])
        if _processor is None or fingerprint!=_fingerprint:
            try:_processor=Processor()
            except ImportError as error:raise RuntimeError('优化V3需要PyTorch。请使用项目启动器；也可在下拉框切换自适应增强V2。') from error
            _fingerprint=fingerprint
        return _processor

def process_pc(image,base_model):
    from fused_dcp import process
    base,parameters=process(image,base_model)
    out,timing=processor().correct(image,base)
    return out,{'base_parameters':parameters,**timing,'execution':'PC software for the complete same algorithm'}

def explain():
    return ('当前默认优化V3：Z15 FPGA完成增强V2暗通道基础去雾，电脑生成辅助强分支/透射率与局部统计，'
            '运行轻量神经网络和防失真保护。神经网络尚未部署到板上。\n'
            '134张既有回归平均SSIM 0.8499→0.8667，PSNR 18.41→20.18dB；目标0.90尚未达到。'
            '仍有3张相对V2的PSNR下降超过3dB，可以在下拉框切回V2。\n'
            'FPGA板上时间只表示基础去雾；完整算法计算合计另外包含电脑参数与校正。JTAG与文件读写单列。')
