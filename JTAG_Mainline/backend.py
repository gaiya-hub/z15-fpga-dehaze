from pathlib import Path
import subprocess,socket,time,json,re,os,queue,threading,msvcrt
import numpy as np
from model import load_image,estimate_air,pack,unpack,reference
from evaluation import auto_ground_truth,attach_quality,export_comparison

ROOT=Path(__file__).parent
from portable_settings import tool_executable
XSDB=tool_executable('xsdb')
SERVER=tool_executable('hw_server')
HIDDEN=getattr(subprocess,'CREATE_NO_WINDOW',0)

def server_ready():
    try:
        with socket.create_connection(('127.0.0.1',3121),timeout=1):return True
    except OSError:return False

def ensure_server():
    if server_ready():return
    log=ROOT/'logs';log.mkdir(exist_ok=True)
    with (log/'hw_server.log').open('wb') as output:
        subprocess.Popen(['cmd.exe','/d','/c',str(SERVER),'-s','tcp::3121'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,creationflags=HIDDEN)
    for _ in range(40):
        if server_ready():return
        time.sleep(.25)
    raise RuntimeError('硬件服务未启动，请检查 Vivado 2024.2 安装。')

def hardware_failure_message(text,log):
    lower=text.lower()
    if 'dr shift through all zeroes' in lower or 'dr shift through all zeros' in lower:
        return ('未检测到Z15上的Zynq芯片：JTAG扫描链返回全零。\n'
                '请先断开板卡电源，检查JTAG排线是否插紧、是否接在JTAG口、1脚方向是否对应；'
                '确认BOOT_CFG两个开关为ON，再接配套电源并打开板卡开关。'
                '下载器USB亮灯或PS_UART串口出现，都不代表板卡JTAG链已连通。\n日志：'+str(log))
    if 'dr shift through all ones' in lower:
        return '未检测到有效JTAG芯片，扫描链返回全一。请检查板卡供电、排线接口与方向。\n日志：'+str(log)
    if 'no targets found' in lower:
        return '调试服务已连接，但没有找到需要的Zynq/ARM目标。请检查板卡供电与JTAG连接。\n日志：'+str(log)+'\n'+text[-1000:]
    if 'while executing' in lower and 'invoked from within' in lower:
        return '硬件工具执行出错，初始化尚未完成。\n日志：'+str(log)+'\n'+text[-1400:]
    return None

def _invoke_unlocked(script,progress=lambda _:None,timeout=600):
    ensure_server()
    log=script.with_suffix('.log')
    started=time.monotonic()
    p=subprocess.Popen(['cmd.exe','/d','/c',str(XSDB),str(script)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,creationflags=HIDDEN)
    lines=[]
    incoming=queue.Queue()
    def read_lines():
        for raw in p.stdout:incoming.put(raw)
        incoming.put(None)
    threading.Thread(target=read_lines,daemon=True).start()
    with log.open('w',encoding='utf-8') as output:
        while True:
            try:raw=incoming.get(timeout=.2)
            except queue.Empty:raw=b''
            if raw is None:break
            if raw:
                line=raw.decode('utf-8',errors='replace');lines.append(line);output.write(line);output.flush()
                if line.startswith('PHASE:'):
                    phase=line.partition(':')[2].strip()
                    progress({'JTAG connected; loading hardware':'正在检测板卡并加载硬件',
                              'Detecting board JTAG target':'调试服务已连接，正在检测板卡JTAG目标',
                              'JTAG target verified; loading hardware':'已识别Zynq芯片，正在加载FPGA硬件',
                              'Uploading RGB image to DDR':'正在向开发板传输高清图片',
                              'Running ARM benchmark and FPGA':'正在板上运行 ARM 基准和 FPGA 处理',
                              'Reading actual FPGA output from DDR':'正在回读 FPGA 处理结果'}.get(phase,phase))
            if time.monotonic()-started>timeout:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,creationflags=HIDDEN)
                raise RuntimeError('操作超时，请检查板卡供电和 JTAG 连接。')
    text=''.join(lines)
    exit_status=p.wait()
    failure=hardware_failure_message(text,log)
    if failure:raise RuntimeError(failure)
    if exit_status!=0:raise RuntimeError('板卡操作失败：\n'+text[-1800:]+'\n日志：'+str(log))
    return text

def invoke(script,progress=lambda _:None,timeout=600):
    lock_path=(ROOT.parent if ROOT.name=='guided_v2' else ROOT)/'device.lock'
    with lock_path.open('a+b') as handle:
        handle.seek(0)
        if lock_path.stat().st_size==0:
            handle.write(b'0');handle.flush();handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise RuntimeError('开发板正在被另一个测试或窗口使用，请等待当前操作完成后重试。')
        try:
            return _invoke_unlocked(script,progress,timeout)
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

def path_tcl(path):
    value=str(path.resolve()).replace('\\','/')
    if any(c in value for c in '{}\n\r'):raise ValueError('演示程序目录不能包含花括号或换行。')
    return '{'+value+'}'

def initialize(progress=lambda _:None):
    for name in ['design_1_wrapper.bit','ps7_init.tcl','demo.elf']:
        if not (ROOT/'board'/name).is_file():raise RuntimeError('缺少板卡构建文件：'+name)
    scripts=ROOT/'outputs';scripts.mkdir(exist_ok=True)
    script=scripts/'initialize.tcl'
    script.write_text('\n'.join([
        'connect -url tcp:127.0.0.1:3121',
        'puts "PHASE: Detecting board JTAG target"',
        'targets -set -filter {name =~ "APU*"}',
        'puts "PHASE: JTAG target verified; loading hardware"','rst -system','after 1000',
        'fpga -file '+path_tcl(ROOT/'board/design_1_wrapper.bit'),
        'targets -set -filter {name =~ "ARM*#0"}','catch {stop}',
        'source '+path_tcl(ROOT/'board/ps7_init.tcl'),'ps7_init','ps7_post_config',
        'puts "HD_INITIALIZED"','disconnect','exit'])+'\n',encoding='utf-8')
    text=invoke(script,progress)
    if 'HD_INITIALIZED' not in text:
        raise RuntimeError('硬件初始化未完成。详细日志：'+str(script.with_suffix('.log'))+'\n'+text[-1400:])
    progress('硬件已就绪，100 MHz 彩色去雾模块')

def _run_board_image(source,max_edge=1280,strength=160,t_min=96,lift=64,repeats=3,run_cpu=True,ground_truth=None,progress=lambda _:None,algorithm="fused",export_sheet=True):
    start=time.monotonic()
    prepare_started=time.perf_counter()
    image,original_size=load_image(source,max_edge)
    decode_resize_ms=(time.perf_counter()-prepare_started)*1000
    width,height=image.size
    enhanced=algorithm=='fused'
    model=json.loads((ROOT/'frozen_model.json').read_text(encoding='utf-8')) if enhanced else None
    parameter_started=time.perf_counter()
    air=estimate_air(image)
    if enhanced:
        from fused_dcp import estimate_strong_air,predict_weight
        strong_air=estimate_strong_air(image,model['air_method']);weight=predict_weight(image,model)
        strength,t_min,lift=160,96,64
    else:strong_air=(220,220,220);weight=0
    host_parameter_ms=(time.perf_counter()-parameter_started)*1000
    run=ROOT/'outputs'/(time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000000:09d}')
    run.mkdir(parents=True,exist_ok=False)
    (run/'input.bin').write_bytes(pack(image));image.save(run/'input.png')
    count=width*height
    values=[0,width,height,*air,int(strength),int(t_min),int(lift),int(repeats),int(run_cpu),*strong_air,weight,int(enhanced)]
    script=run/'process.tcl'
    commands=['connect -url tcp:127.0.0.1:3121','targets -set -filter {name =~ "ARM*#0"}',
              'catch {stop}','rst -processor','dow '+path_tcl(ROOT/'board/demo.elf'),
              'puts "PHASE: Uploading RGB image to DDR"','set upload_begin [clock milliseconds]',
              'dow -data '+path_tcl(run/'input.bin')+' 0x10000000',
              'puts "UPLOAD_MS: [expr {[clock milliseconds]-$upload_begin}]"']
    commands += [f'mwr {0x14000000+4*i:#x} {v}' for i,v in enumerate(values)]
    commands += ['puts "PHASE: Running ARM benchmark and FPGA"','con',
                 'targets -set -filter {name =~ "APU*"}',
                 'set deadline [expr {[clock milliseconds]+120000}]','set status 0',
                 'while {[clock milliseconds]<$deadline} {',
                 'after 100','set status [lindex [mrd -address-space AP0 -value 0x14000000] 0]',
                 'if {$status!=0} {break}','}',
                 'puts "STATUS: [format 0x%08x $status]"',
                 'if {$status!=0x600d600d} {disconnect; error "Board validation failed"}',
                 'set stats [mrd -address-space AP0 -value 0x14000000 32]',
                 'puts "STATS: [join $stats ,]"',
                 'targets -set -filter {name =~ "ARM*#0"}','stop',
                 'targets -set -filter {name =~ "APU*"}',
                 'puts "PHASE: Reading actual FPGA output from DDR"','set read_begin [clock milliseconds]',
                 f'mrd -address-space AP0 -bin -file {path_tcl(run/"fpga.bin")} 0x11000000 {count}',
                 'puts "READBACK_MS: [expr {[clock milliseconds]-$read_begin}]"',
                 'puts "HD_OUTPUT_READY"','disconnect','exit']
    script.write_text('\n'.join(commands)+'\n',encoding='utf-8')
    from dual_arm import reset_worker_commands
    commands[1:1]=reset_worker_commands()
    script.write_text('\n'.join(commands)+'\n',encoding='utf-8')
    text=invoke(script,progress)
    match=re.search(r'STATS: ([^\r\n]+)',text)
    if 'HD_OUTPUT_READY' not in text or match is None:
        raise RuntimeError('没有收到完整的板上处理结果，板卡可能已被复位或断电。请重新初始化后重试。\n'+text[-1200:])
    words=[int(v,0) for v in match.group(1).split(',')]
    def ticks(i):return words[i]+(words[i+1]<<32)
    frequency=words[23]
    if not frequency:raise RuntimeError('板上计时器频率无效。')
    output=unpack((run/'fpga.bin').read_bytes(),width,height)
    output.save(run/'fpga_output.png')
    progress('核对独立软件参考')
    if enhanced:
        from fused_dcp import process
        golden,_=process(image,model)
    else:golden=reference(image,air,strength,t_min,lift)
    delta=np.abs(np.asarray(output,dtype=np.int16)-np.asarray(golden,dtype=np.int16))
    mismatches=int(np.count_nonzero(delta))
    golden.save(run/'reference.png')
    result={'source':str(Path(source).resolve()),'original_size':original_size,'processing_size':[width,height],
            'algorithm':'fused-guided-v2' if enhanced else 'baseline','strong_air':strong_air,'blend_q8':weight,'ground_truth_used_at_runtime':False,'controller_location':'PC','refinement_location':'FPGA PL' if enhanced else None,'air':air,'strength':strength,'t_min':t_min,'lift':lift,'repeats':repeats,
            'arm_ms':ticks(16)/frequency*1000 if run_cpu else None,
            'fpga_ms':ticks(18)/frequency*1000,'fpga_start_done_ms':ticks(20)/frequency*1000,
            'board_pixel_mismatches':words[22] if run_cpu else None,
            'host_channel_mismatches':mismatches,'max_channel_error':int(delta.max()),
            'upload_ms':int(re.search(r'UPLOAD_MS: (\d+)',text).group(1)),
            'readback_ms':int(re.search(r'READBACK_MS: (\d+)',text).group(1)),
            'wall_seconds':time.monotonic()-start,'output':str(run/'fpga_output.png'),
            'timing_scope':'board FPGA includes cache maintenance and control; excludes PC/JTAG transfers; ARM scalar reference -O2, caches enabled',
            'fclk_mhz':100,'host_parameter_ms':host_parameter_ms,'decode_resize_ms':decode_resize_ms}
    result['speedup']=result['arm_ms']/result['fpga_ms'] if run_cpu else None
    if mismatches or (run_cpu and words[22]):
        (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        raise RuntimeError('FPGA 与独立参考不一致，已保存日志和结果：'+str(run))
    progress('计算配对无雾参考的 PSNR / SSIM / MAE')
    gt=Path(ground_truth) if ground_truth else auto_ground_truth(source)
    clear=attach_quality(result,run,image,output,gt)
    (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if export_sheet:export_comparison(run,result,image,output,clear)
    progress('完成：实际 FPGA 回读，逐像素验证通过')
    return image,output,result,run


def run_image(source,max_edge=1280,strength=160,t_min=96,lift=64,repeats=1,run_cpu=False,ground_truth=None,progress=lambda _:None,algorithm="pl_v3",export_sheet=True):
    if algorithm=='pl_v3':
        from pl_backend_v3 import run_image as run_pl
        return run_pl(source,max_edge,strength,t_min,lift,repeats,run_cpu,ground_truth,progress,export_sheet)
    if algorithm not in ['baseline','fused','hybrid']:raise ValueError('未知去雾算法。')
    if algorithm!='hybrid':
        return _run_board_image(source,max_edge,strength,t_min,lift,repeats,run_cpu,ground_truth,progress,algorithm,export_sheet)
    import copy
    from hybrid_runtime_v3 import processor,ALGORITHM,artifacts
    started=time.monotonic();load_begin=time.perf_counter();neural=processor();load_ms=(time.perf_counter()-load_begin)*1000
    image,base,result,run=_run_board_image(source,max_edge,strength,t_min,lift,repeats,run_cpu,ground_truth,progress,'fused',False)
    progress('真实FPGA基础输出已验证，正在电脑生成辅助特征并进行神经校正')
    try:output,timing=neural.correct(image,base)
    except Exception as error:
        result['postprocessing_error']=str(error)
        (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        raise RuntimeError('FPGA基础去雾已完成，但V3电脑校正失败；未生成V3成功结果。'+str(error)) from error
    output.save(run/'hybrid_output.png')
    result.update(algorithm=ALGORITHM,fpga_algorithm='fused-guided-v2',execution='FPGA DCP + PC auxiliary features / neural correction / range guard',
                  neural_network_location='PC CPU',host_model_load_ms=load_ms,neural_artifacts=artifacts(),**timing,
                  fpga_output=str(run/'fpga_output.png'),output=str(run/'hybrid_output.png'),fpga_quality=copy.deepcopy(result.get('quality')),
                  raw_fpga_rgb_sha256=__import__('hashlib').sha256(base.tobytes()).hexdigest(),
                  final_rgb_sha256=__import__('hashlib').sha256(output.tobytes()).hexdigest(),
                  quality_release_status='user-selected preview; 0.90 target not achieved; per-image regressions remain')
    result['pipeline_compute_ms']=result['host_parameter_ms']+result['fpga_ms']+timing['host_post_ms']
    result['timing_scope']='FPGA/ARM times and ARM speedup describe DCP V2 only. Pipeline compute = PC frame parameters + board DCP + PC auxiliary features/neural/guard; excludes model load, JTAG, file I/O and verification. Wall time includes current-call processing and validation; collage export excluded.'
    clear=attach_quality(result,run,image,output,ground_truth or auto_ground_truth(source))
    result['wall_seconds']=time.monotonic()-started
    (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if export_sheet:export_comparison(run,result,image,output,clear)
    progress('完成：FPGA基础去雾＋电脑神经校正，原始回读与最终输出均已保存')
    return image,output,result,run

if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--init',action='store_true');p.add_argument('--image');p.add_argument('--edge',type=int,default=1280);p.add_argument('--ground-truth');p.add_argument('--algorithm',choices=['baseline','fused','hybrid','pl_v3'],default='pl_v3')
    args=p.parse_args()
    if args.init:initialize(print)
    if args.image:
        _,_,result,_=run_image(args.image,max_edge=args.edge,ground_truth=args.ground_truth,progress=print,algorithm=args.algorithm)
        print(json.dumps(result,ensure_ascii=False,indent=2))
