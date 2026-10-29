"""Same fixed V3 algorithm: whole PC versus on-board PS/PL pipeline."""
from pathlib import Path
import json,time,hashlib,platform
import numpy as np
from PIL import Image
from model import estimate_air,reference
from fused_dcp import estimate_strong_air,predict_weight,strong_reference
from evaluation import open_rgb,reference_image,image_scores
ROOT=Path(__file__).resolve().parent
def measure(record_dir,ground_truth=None,progress=lambda _:None):
    from pl_backend_v3 import get_model,artifacts
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(4)
    except ImportError:pass
    directory=Path(record_dir);record=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if record.get('batch_compacted'):raise ValueError('批处理精简记录不能用于完整图片对比。')
    if artifacts()!=record.get('quantized_artifacts'):raise ValueError('定点模型或源文件与FPGA记录不一致，请重新处理后比较。')
    image=open_rgb(directory/'input.png');hardware=open_rgb(directory/'fpga_output.png');model=get_model()
    def compute():
        begin=time.perf_counter();air=estimate_air(image);strong_air=estimate_strong_air(image,model.base_model['air_method']);weight=predict_weight(image,model.base_model);g=model.globals(image)
        if tuple(air)!=tuple(record['air']) or tuple(strong_air)!=tuple(record['strong_air']) or weight!=record['blend_q8']:raise ValueError('帧参数与FPGA记录不一致。')
        middle=time.perf_counter();weak=reference(image,air,160,96,64);strong,info=strong_reference(image,strong_air)
        base=Image.fromarray(((np.asarray(weak,np.int32)*(256-weight)+np.asarray(strong,np.int32)*weight+128)>>8).astype(np.uint8));dcp_done=time.perf_counter()
        fields,_,_=model.inputs(image,base=base,strong=strong,info=info);output=model.infer(fields,g);end=time.perf_counter()
        return output,{'parameters_ms':(middle-begin)*1000,'dcp_ms':(dcp_done-middle)*1000,'post_ms':(end-dcp_done)*1000,'pixel_ms':(end-middle)*1000,'algorithm_ms':(end-begin)*1000}
    progress('电脑完整定点V3预热一次');compute();samples=[]
    for i in range(3):progress(f'电脑定点V3计时 {i+1}/3');pc,sample=compute();samples.append(sample)
    diff=np.abs(np.asarray(pc,np.int16)-np.asarray(hardware,np.int16));mismatch=int(np.count_nonzero(diff));maximum=int(diff.max())
    run=ROOT/'outputs'/('same_algorithm_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000000));run.mkdir(parents=True)
    pc.save(run/'pc_output.png');hardware.save(run/'fpga_output.png');image.save(run/'input.png')
    clear,reason=reference_image(record['source'],ground_truth or record.get('ground_truth'),image.size)
    if clear:clear.save(run/'ground_truth.png')
    mean={k:float(np.mean([s[k] for s in samples])) for k in samples[0]};f=record
    data={'version':'same-algorithm-pl-v3','algorithm':record['algorithm'],'source':record['source'],'processing_size':list(image.size),'linked_fpga_result':str(directory/'result.json'),
          'platform':platform.platform(),'pc':{'mean':mean,'samples':samples,'scores':image_scores(clear,pc) if clear else None},
          'fpga':{'board_ms':f['fpga_ms'],'kernel_ms':f['fpga_start_done_ms'],'dcp_ms':f['dcp_ms'],'arm_feature_ms':f['arm_feature_ms'],'neural_ms':f['neural_ms'],
                  'host_post_ms':0,'post_stage_ms':f['arm_feature_ms']+f['neural_ms'],'host_parameter_ms':f['host_parameter_ms'],'algorithm_ms':f['pipeline_compute_ms'],'algorithm_ms_estimated':False,
                  'upload_ms':f['upload_ms'],'readback_ms':f['readback_ms'],'record_wall_seconds':f['wall_seconds'],'transport':f.get('transport'),'sd_read_and_input_check_ms':f.get('sd_read_and_input_check_ms'),'sd_output_write_ms':f.get('sd_output_write_ms'),'dual_arm':f.get('dual_arm',False),'dual_active':f.get('dual_active',False),'dual_startup_ms':f.get('dual_startup_ms'),'dual_bootstrap_ms':f.get('dual_bootstrap_ms'),'scores':image_scores(clear,hardware) if clear else None,'neural_location':'FPGA PL','local_feature_location':f.get('local_feature_location','Zynq ARM')},
          'channel_mismatches':mismatch,'max_channel_error':maximum,'quality_reason':reason,'algorithm_compute_ratio':mean['algorithm_ms']/f['pipeline_compute_ms'],
          'pixel_compute_ratio':mean['pixel_ms']/f['fpga_ms'],'timing_scope':'Both run the same Q14 network, DCP and guard. PC mean of 3 after warmup; board time from linked real record includes board statistics and PL DCP/neural. JTAG, file I/O and verification excluded from compute ratio; no continuous video or end-to-end speedup asserted.'}
    if f.get('transport')=='sd-card':data['timing_scope']+=' SD IO, input/output CRC, first-batch warmup/startup and task preparation are excluded from compute ratio, separately recorded; not a task end-to-end ratio.'
    (run/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if mismatch:raise RuntimeError('完整定点V3同算法校验失败，差异'+str(mismatch)+'通道。')
    return run,data,pc,hardware
