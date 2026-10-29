"""Real Z15 V3: DCP + neural inference / guard in PL, local statistics on board ARM."""
from pathlib import Path
import json,time,re,hashlib
import numpy as np
from model import load_image,estimate_air,pack,unpack
from fused_dcp import estimate_strong_air,predict_weight
from evaluation import auto_ground_truth,attach_quality,export_comparison
ROOT=Path(__file__).resolve().parent
ALGORITHM='fused-neural-v3-pl'
_model=None;_model_hash=None
def get_model():
    global _model,_model_hash
    from quantized_v3 import QuantizedV3
    digest=hashlib.sha256((ROOT/'fixed_v3.json').read_bytes()).hexdigest()
    if _model is None or digest!=_model_hash:_model=QuantizedV3(ROOT);_model_hash=digest
    return _model
def artifacts():
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['fixed_v3.json','quantized_v3.py','fused_dcp.py','model.py','frozen_model.json']}
def run_image(source,max_edge=1280,strength=160,t_min=96,lift=64,repeats=1,run_cpu=False,ground_truth=None,progress=lambda _:None,export_sheet=True,read_base=False,pl_statistics=False):
    if pl_statistics:raise ValueError("此紧凑硬件使用优化ARM统计，不支持PL统计模式。")
    from backend import invoke,path_tcl
    from dual_arm import load_protocol,reset_worker_commands,start_commands,diagnostics
    if not 1<=int(repeats)<=10:raise ValueError('重复次数必须在1..10。')
    wall=time.monotonic();protocol=load_protocol(ROOT);model=get_model();prepare=time.perf_counter();image,original=load_image(source,max_edge);decode_ms=(time.perf_counter()-prepare)*1000
    w,h=image.size;count=w*h;begin=time.perf_counter();air=estimate_air(image);strong_air=estimate_strong_air(image,model.base_model['air_method']);weight=predict_weight(image,model.base_model);global_q=model.globals(image);parameters_ms=(time.perf_counter()-begin)*1000
    run=ROOT/'outputs'/(time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000000:09d}');run.mkdir(parents=True)
    image.save(run/'input.png');(run/'input.bin').write_bytes(pack(image));model.packed_network()  # Validate the frozen 19-bit bounds; weights are already in PL ROM.
    params=np.zeros(40,dtype='<u4');params[:22]=global_q[:22].astype(np.int32).view(np.uint32);params[22]=model.model['tag'];params[32:35]=global_q[22:25] if len(global_q)>22 else 0;params[35]=int(model.model.get('output_mode')=='physical-air-gain');(run/'frame_parameters.bin').write_bytes(params.tobytes())
    values=[0,w,h,*air,160,96,64,int(repeats),int(run_cpu),*strong_air,weight,2]
    script=run/'process.tcl';cmd=['connect -url tcp:127.0.0.1:3121','targets -set -filter {name =~ "ARM*#0"}','catch {stop}','rst -processor','dow '+path_tcl(ROOT/'board/demo.elf'),
    'puts "PHASE: Uploading RGB image to DDR"','set upload_begin [clock milliseconds]','dow -data '+path_tcl(run/'input.bin')+' 0x10000000','dow -data '+path_tcl(run/'frame_parameters.bin')+' 0x15000000',f'dow -data {path_tcl(run/"network_q14.bin")} {0x15000000+4*(256+11*count):#x}', 'puts "UPLOAD_MS: [expr {[clock milliseconds]-$upload_begin}]"']
    cmd=[command for command in cmd if 'network_q14.bin' not in command]
    cmd[1:1]=reset_worker_commands()
    cmd += [f'mwr {0x14000000+4*i:#x} {v}' for i,v in enumerate(values)];cmd += [f'mwr 0x1400008c {int(pl_statistics)}']
    cmd += ['puts "PHASE: Board DCP, dual ARM statistics and FPGA neural inference"',*start_commands(protocol),'targets -set -filter {name =~ "APU*"}',
    'set deadline [expr {[clock milliseconds]+240000}]','set status 0','while {[clock milliseconds]<$deadline} {','after 100','set status [lindex [mrd -address-space AP0 -value 0x14000000] 0]','if {$status!=0} {break}','}',
    'puts "STATUS: [format 0x%08x $status]"','if {$status!=0x600d600d} {targets -set -filter {name =~ "ARM*#1"}; catch {stop}; targets -set -filter {name =~ "ARM*#0"}; catch {stop}; disconnect; error "Board validation failed; initialize board and check dual firmware/model"}',
    'set stats [mrd -address-space AP0 -value 0x14000000 48]','puts "STATS: [join $stats ,]"','targets -set -filter {name =~ "ARM*#1"}','stop','targets -set -filter {name =~ "ARM*#0"}','stop','targets -set -filter {name =~ "APU*"}',
    'puts "PHASE: Reading actual FPGA output from DDR"','set read_begin [clock milliseconds]',f'mrd -address-space AP0 -bin -file {path_tcl(run/"fpga.bin")} 0x11000000 {count}']
    if read_base:cmd += [f'mrd -address-space AP0 -bin -file {path_tcl(run/"fpga_v2.bin")} 0x15000400 {count}']
    cmd += ['puts "READBACK_MS: [expr {[clock milliseconds]-$read_begin}]"','puts "PL_V3_OUTPUT_READY"','disconnect','exit']
    script.write_text('\n'.join(cmd)+'\n',encoding='utf-8');text=invoke(script,progress,timeout=600)
    match=re.search(r'STATS: ([^\r\n]+)',text)
    if not match or 'PL_V3_OUTPUT_READY' not in text:raise RuntimeError('未获得完整板上神经推理结果。日志：'+str(script.with_suffix('.log')))
    words=[int(v,0) for v in match.group(1).split(',')];frequency=words[23]
    if not frequency or words[31]!=model.model['tag'] or words[34]!=0x504c5633:raise RuntimeError('FPGA神经模型标记未通过校验，请重新初始化对应硬件。')
    ticks=lambda i:words[i]+(words[i+1]<<32);ms=lambda i:ticks(i)/frequency*1000
    dual=diagnostics(words,frequency,int(repeats))
    dual['dual_bootstrap_ms']=int(re.search(r'DUAL_BOOTSTRAP_MS: (\d+)',text).group(1))
    output=unpack((run/'fpga.bin').read_bytes(),w,h);output.save(run/'fpga_output.png')
    progress('核对独立定点神经参考；最终显示图保持FPGA回读数据')
    verify=time.perf_counter();golden,base=model.process(image);golden.save(run/'reference.png')
    diff=np.abs(np.asarray(output,np.int16)-np.asarray(golden,np.int16));mismatches=int(np.count_nonzero(diff));verify_ms=(time.perf_counter()-verify)*1000
    base_mismatch=None
    if read_base:
        actual_base=unpack((run/'fpga_v2.bin').read_bytes(),w,h);actual_base.save(run/'fpga_v2_output.png');base_mismatch=int(np.count_nonzero(np.asarray(actual_base)!=np.asarray(base)))
    result={'source':str(Path(source).resolve()),'original_size':list(original),'processing_size':[w,h],'algorithm':ALGORITHM,
            'execution':'FPGA PL DCP + FPGA PL local statistics / quantized neural inference / guard; PC frame parameters and verification only' if pl_statistics else 'FPGA PL DCP and neural; optimized ARM local statistics',
            'neural_network_location':'FPGA PL','local_feature_location':'FPGA PL' if pl_statistics else 'Zynq ARM','controller_location':'PC frame statistics','implementation':'streaming PL statistics and 32-lane PL NN' if pl_statistics else 'ARM vector RGB statistics, exact NEON normalization, 16-row ring and corrected sqrt; 32-lane PL NN with fixed ROM weights, feature prefetch / independent result write and DSP guard','ground_truth_used_at_runtime':False,
            'air':air,'strong_air':strong_air,'blend_q8':weight,'strength':160,'t_min':96,'lift':64,'repeats':repeats,'fclk_mhz':100,
            'arm_ms':ms(16) if run_cpu else None,'fpga_ms':ms(18),'fpga_start_done_ms':ms(20),'dcp_ms':ms(25),'arm_feature_ms':ms(27),'neural_ms':ms(29),'neural_kernel_ms':ms(32),
            'host_post_ms':0,'host_parameter_ms':parameters_ms,'decode_resize_ms':decode_ms,'verification_ms':verify_ms,
            'board_pixel_mismatches':words[22] if run_cpu else None,'host_channel_mismatches':mismatches,'max_channel_error':int(diff.max()),'raw_v2_channel_mismatches':base_mismatch,
            'upload_ms':int(re.search(r'UPLOAD_MS: (\d+)',text).group(1)),'readback_ms':int(re.search(r'READBACK_MS: (\d+)',text).group(1)),
            'fpga_model_tag':hex(words[31]),'quantized_artifacts':artifacts(),'output':str(run/'fpga_output.png'),
            'bit_sha256':hashlib.sha256((ROOT/'board/design_1_wrapper.bit').read_bytes()).hexdigest(),'firmware_sha256':hashlib.sha256((ROOT/'board/demo.elf').read_bytes()).hexdigest(),
            'weight_storage':'FPGA fixed ROM','model_transfer_per_image_bytes':0,'neural_engine_lanes':32,'firmware_optimization':'-O3 -mfpu=neon; NEON RGB/statistics/packing, exact reciprocal normalization and integer-corrected sqrt; no fast-math','pl_statistics':bool(pl_statistics),'neural_stage_includes_statistics':bool(pl_statistics),'timing_scope':'Board total includes DCP PL, board local statistics (PL or ARM selected), neural PL and cache/control. PL kernel sum excludes ARM feature preparation. Pipeline compute adds PC frame parameter time; excludes JTAG, file I/O and independent verification. ARM scalar full-pipeline reference optional. No continuous video FPS asserted.'}
    result.update(dual)
    result['implementation']='ARM0/ARM1 private 16-row rings with aligned row split and exact NEON statistics; unchanged 32-lane PL neural/guard'
    result['firmware_optimization']='-O3 -mfpu=neon; two ARM A9 private contexts, exact integer math; aligned fallback to ARM0'
    result['timing_scope']+=' CPU1 bootstrap/startup is separately reported; included in wall_seconds, excluded from board steady compute.'
    result['speedup']=result['arm_ms']/result['fpga_ms'] if run_cpu else None;result['pipeline_compute_ms']=parameters_ms+result['fpga_ms']
    if mismatches or (run_cpu and words[22]) or (base_mismatch not in [None,0]):
        (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');raise RuntimeError('板上神经输出与独立定点参考不一致，结果已留存：'+str(run))
    clear=attach_quality(result,run,image,output,ground_truth or auto_ground_truth(source));result['wall_seconds']=time.monotonic()-wall
    (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if export_sheet:export_comparison(run,result,image,output,clear)
    progress('完成：最终图片直接来自板上FPGA神经推理回读，定点参考校验通过')
    return image,output,result,run

def process_pc(image,base_model=None):
    output,_=get_model().process(image)
    return output,{'execution':'PC complete Q14 same algorithm','model_tag':hex(get_model().model['tag'])}
