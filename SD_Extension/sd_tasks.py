"""SD task interchange. Imported output is board bytes, never a PC correction."""
from pathlib import Path
import hashlib,json,struct,time,uuid,zlib
import numpy as np
from PIL import Image
from model import load_image,estimate_air,pack,unpack
from fused_dcp import estimate_strong_air,predict_weight
from pl_backend_v3 import get_model,artifacts
ROOT=Path(__file__).resolve().parent
INPUT_MAGIC=0x31495a44;INDEX_MAGIC=0x31585a44;RESULT_MAGIC=0x31525a44;BATCH_MAGIC=0x31425a44;VERSION=1
def crc(data):return zlib.crc32(data)&0xffffffff
def words(data):return list(struct.unpack('<'+'I'*(len(data)//4),data))
def binary(values):return struct.pack('<'+'I'*len(values),*values)
def sealed(values):
    values=list(values);values[-1]=crc(binary(values[:-1]));return binary(values)
def checked(data,size,magic):
    if len(data)!=size:raise ValueError('任务或结果长度不正确。')
    values=words(data)
    if values[0]!=magic or values[1]!=VERSION or crc(data[:-4])!=values[-1]:raise ValueError('任务或结果校验失败。')
    return values
def read_input(path):
    data=Path(path).read_bytes()
    if len(data)<288:raise ValueError('输入文件不完整。')
    head=checked(data[:64],64,INPUT_MAGIC);w,h,n=head[2:5]
    if not 1<=w<=1920 or not 1<=h<=1280 or n!=w*h or head[12]!=n*4 or head[13]!=len(data) or len(data)!=288+n*4:raise ValueError('输入尺寸或长度不正确。')
    regs=data[64:128];params=data[128:288];pixels=data[288:]
    if crc(pixels)!=head[5] or crc(params)!=head[6] or crc(regs)!=head[7]:raise ValueError('输入内容CRC不匹配。')
    return head,words(regs),words(params),pixels
def _is_removable(path):
    import os
    if os.name!='nt':return False
    import ctypes
    return ctypes.windll.kernel32.GetDriveTypeW(str(Path(path).anchor))==2

def _allocation_unit(path):
    import os
    if os.name=='nt':
        import ctypes
        values=[ctypes.c_ulong() for _ in range(4)]
        ok=ctypes.windll.kernel32.GetDiskFreeSpaceW(str(Path(path).anchor),*(ctypes.byref(v) for v in values))
        if ok:return max(512,values[0].value*values[1].value)
    return 4096

def _export_plan(files,base,max_edge):
    import shutil
    existing=base
    while not existing.exists():existing=existing.parent
    free=shutil.disk_usage(existing).free;unit=_allocation_unit(existing)
    allocated=lambda n:((n+unit-1)//unit)*unit
    costs=allocated(64)+allocated(256)+unit*4;input_bytes=output_bytes=0;sizes=[];fits=0
    for i,path in enumerate(files,1):
        with Image.open(path) as image:
            w,h=image.size
            if image.getexif().get(274,1) in [5,6,7,8]:w,h=h,w
        scale=min(1.0,max_edge/max(w,h),1920/w,1280/h)
        if scale<1:w,h=max(1,round(w*scale)),max(1,round(h*scale))
        sizes.append((w,h));pixels=w*h*4
        input_bytes+=288+pixels;output_bytes+=pixels+512
        costs+=allocated(288+pixels)+allocated(pixels)+allocated(512)+allocated(2048+2*len(str(path).encode('utf-8')))
        required=costs+max(2*1024*1024,(costs+19)//20)
        if required<=free:fits=i
    return {'source_files':len(files),'input_bytes':input_bytes,'output_bytes':output_bytes+256,
            'required_bytes':required,'free_bytes':free,'conservative_fit_count':fits,'sizes':sizes}

def export_task(input_dir,destination,max_edge=640,repeats=1,recursive=True,progress=lambda _:None):
    import errno,os,shutil
    source=Path(input_dir).resolve();base=Path(destination).resolve();target=base/'DEHAZE'
    if not source.is_dir() or not 1<=int(repeats)<=10 or not 1<=int(max_edge)<=1920:raise ValueError('输入目录、最长边或重复次数无效。')
    if target.is_relative_to(source):raise ValueError('SD任务目录必须位于输入图片目录之外。')
    if _is_removable(base) and base!=Path(base.anchor).resolve():
        raise ValueError(f'板卡只读取卡根目录的DEHAZE。请将保存位置选为 {base.anchor} 根目录，不要选择归档文件夹或DEHAZE文件夹。')
    candidates=source.rglob('*') if recursive else source.iterdir()
    files=sorted(p for p in candidates if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff'])
    if not files or len(files)>100000:raise ValueError('任务须包含1..100000张图片。')
    if target.exists() and any(target.iterdir()):raise ValueError('目标已有DEHAZE任务，请先将旧任务改名保存，再选择卡根目录导出。')
    progress(f'预检查{len(files)}张图片的尺寸及输入＋输出空间，尚未写入任务。')
    plan=_export_plan(files,base,int(max_edge))
    if plan['required_bytes']>plan['free_bytes']:
        mb=lambda n:n/1024/1024
        raise ValueError(f"空间不足，未写入任何任务文件。本批{len(files)}张需预留约{mb(plan['required_bytes']):.1f}MB（含输入、输出、记录和余量），当前仅{mb(plan['free_bytes']):.1f}MB。按本批尺寸，当前空间建议最多约{plan['conservative_fit_count']}张；请选择较小图片文件夹或整理存储空间。")
    model=get_model();tag=model.model['tag']
    if model.model.get('output_mode')=='physical-air-gain':raise ValueError('此SD协议仅支持当前已部署的V3模型。')
    rows=[];started=time.perf_counter();created=[];new_target=not target.exists();made_dirs=[]
    assert target.resolve().is_relative_to(base) and target.name=='DEHAZE'
    def write_new(path,data):
        with path.open('xb') as output:
            created.append(path);output.write(data);output.flush();os.fsync(output.fileno())
    try:
        target.mkdir(parents=True,exist_ok=True)
        write_new(target/'EXPORT.LCK',str(os.getpid()).encode('ascii'))
        for name in ['IN','OUT']:
            folder=target/name;folder.mkdir();made_dirs.append(folder)
        for i,path in enumerate(files,1):
            progress(f'准备 {i}/{len(files)}：{path.name}')
            image,original=load_image(path,max_edge);w,h=image.size
            if (w,h)!=plan['sizes'][i-1]:raise ValueError('预检查后输入图片尺寸发生变化，请重新导出。')
            t=time.perf_counter();air=estimate_air(image);strong=estimate_strong_air(image,model.base_model['air_method']);blend=predict_weight(image,model.base_model);g=model.globals(image)
            parameter_ms=(time.perf_counter()-t)*1000
            registers=[0,w,h,*air,160,96,64,int(repeats),0,*strong,blend,2]
            frame=np.zeros(40,dtype='<u4');frame[:22]=g[:22].astype(np.int32).view(np.uint32);frame[22]=tag;frame[32:35]=g[22:25] if len(g)>22 else 0
            pixels=pack(image);register_data=binary(registers);params=frame.tobytes()
            head=[INPUT_MAGIC,VERSION,w,h,w*h,crc(pixels),crc(params),crc(register_data),tag,int(repeats),0,i,len(pixels),288+len(pixels),0,0]
            packet=sealed(head)+register_data+params+pixels;name=f'I{i:07d}.BIN';write_new(target/'IN'/name,packet)
            rows.append({'index':i,'input_file':name,'source':str(path),'relative_source':path.relative_to(source).as_posix(),
              'original_size':list(original),'processing_size':[w,h],'air':air,'strong_air':strong,'blend_q8':blend,
              'host_parameter_ms':parameter_ms,'input_crc32':head[5],'parameters_crc32':head[6],'registers_crc32':head[7],'input_sha256':hashlib.sha256(pixels).hexdigest()})
        nonce=uuid.uuid4();index=[INDEX_MAGIC,VERSION,len(rows),tag,int(repeats),*struct.unpack('<4I',nonce.bytes),0,0,0,0,0,0,0];index_data=sealed(index)
        if shutil.disk_usage(base).free<plan['output_bytes']+1024*1024:raise OSError(errno.ENOSPC,'空间在导出期间减少，无法预留输出空间。')
        manifest={'version':VERSION,'task_id':nonce.hex,'model_tag':tag,'quantized_artifacts':artifacts(),'index_crc32':words(index_data)[15],
          'repeats':int(repeats),'max_edge':int(max_edge),'jobs':rows,'export_seconds':time.perf_counter()-started,
          'space_preflight':{k:v for k,v in plan.items() if k!='sizes'},
          'scope':'PC decoded and resized images and precomputed frame parameters. Board reads SD, computes PS/PL and writes output. Not full JPG-to-output standalone.'}
        write_new(target/'TASK.JSON',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
        write_new(target/'INDEX.BIN',index_data)
        (target/'EXPORT.LCK').unlink();created.remove(target/'EXPORT.LCK')
    except Exception as error:
        # Only files created by this invocation are eligible for cleanup.
        expected=(base/'DEHAZE').resolve();assert target.resolve()==expected and expected.is_relative_to(base)
        for path in reversed(created):
            assert path.resolve().is_relative_to(expected)
            if path.is_file():path.unlink()
        for folder in reversed(made_dirs):
            assert folder.resolve().is_relative_to(expected)
            if folder.exists() and not any(folder.iterdir()):folder.rmdir()
        if new_target and target.exists() and not any(target.iterdir()):target.rmdir()
        if isinstance(error,OSError) and error.errno==errno.ENOSPC:raise ValueError('导出期间空间不足，本次半成品已撤销；输入原图与已有归档任务保留。请减少本批图片数量。') from error
        raise
    progress('SD任务已准备：'+str(target));return target,manifest


def import_task(task_dir,destination=None,verify=True,progress=lambda _:None):
    task=Path(task_dir);task=task/'DEHAZE' if (task/'DEHAZE').is_dir() else task
    manifest=json.loads((task/'TASK.JSON').read_text(encoding='utf-8'));index=checked((task/'INDEX.BIN').read_bytes(),64,INDEX_MAGIC)
    if index[15]!=manifest['index_crc32'] or index[2]!=len(manifest['jobs']) or index[3]!=manifest['model_tag']:raise ValueError('任务索引与清单不匹配。')
    if artifacts()!=manifest['quantized_artifacts']:raise ValueError('当前算法模型与SD任务不一致。')
    root=Path(destination) if destination else ROOT/'outputs'/('sd_import_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
    root.mkdir(parents=True,exist_ok=False);records=[];missing=[]
    for job in manifest['jobs']:
        number=job['index'];rpath=task/'OUT'/f'R{number:07d}.BIN';opath=task/'OUT'/f'O{number:07d}.BIN'
        if not rpath.is_file() or not opath.is_file():missing.append(number);continue
        progress(f'导入 {number}/{len(manifest["jobs"])}：{Path(job["source"]).name}')
        r=checked(rpath.read_bytes(),512,RESULT_MAGIC);w,h=job['processing_size'];n=w*h
        if r[2]!=0 or r[3]!=number or r[4:7]!=[w,h,n] or r[7]!=manifest['model_tag'] or r[8:11]!=[job['input_crc32'],job['parameters_crc32'],job['registers_crc32']] or r[77]!=index[15]:raise ValueError('结果不属于当前任务：'+str(number))
        head,registers,params,pixels=read_input(task/'IN'/job['input_file'])
        if hashlib.sha256(pixels).hexdigest()!=job['input_sha256'] or head[11]!=number:raise ValueError('输入图像与清单不一致。')
        output_bytes=opath.read_bytes()
        if len(output_bytes)!=n*4 or crc(output_bytes)!=r[11]:raise ValueError('板卡输出不完整或损坏：'+str(number))
        mailbox=r[16:64];freq=r[12]
        if not freq or mailbox[31]!=manifest['model_tag'] or mailbox[34]!=0x504c5633 or mailbox[46]!=0x4455414c or mailbox[47]!=1 or mailbox[36]!=1:raise ValueError('板上执行标记未通过。')
        ticks=lambda values,i:values[i]+(values[i+1]<<32)
        ms=lambda i:ticks(mailbox,i)/freq*1000
        sdms=lambda i:ticks(r,i)/freq*1000
        active=bool(mailbox[37]);expected=(r[74]+r[13]) if active else 0
        if r[75]!=expected:raise ValueError('第二核作业数与当前任务不一致。')
        run=root/f'{number:07d}';run.mkdir()
        image=unpack(pixels,w,h);output=unpack(output_bytes,w,h);image.save(run/'input.png');output.save(run/'fpga_output.png');(run/'fpga.bin').write_bytes(output_bytes)
        mismatch=maximum=None
        if verify:
            golden,_=get_model().process(image);golden.save(run/'reference.png');delta=np.abs(np.asarray(output,np.int16)-np.asarray(golden,np.int16));mismatch=int(np.count_nonzero(delta));maximum=int(delta.max())
        result={'source':job['source'],'original_size':job['original_size'],'processing_size':[w,h],'algorithm':'fused-neural-v3-pl',
          'transport':'sd-card','execution':'FPGA DCP/neural/guard + ARM0/ARM1 statistics; SD image IO; precomputed PC frame parameters',
          'neural_network_location':'FPGA PL','local_feature_location':'Zynq ARM0 + ARM1' if active else 'Zynq ARM0 (alignment fallback)',
          'controller_location':'PC precomputed frame parameters','ground_truth_used_at_runtime':False,'air':job['air'],'strong_air':job['strong_air'],'blend_q8':job['blend_q8'],
          'strength':160,'t_min':96,'lift':64,'repeats':r[13],'warmup_runs':r[74],'fclk_mhz':100,'host_parameter_ms':job['host_parameter_ms'],
          'arm_ms':None,'fpga_ms':ms(18),'fpga_start_done_ms':ms(20),'dcp_ms':ms(25),'arm_feature_ms':ms(27),'neural_ms':ms(29),'neural_kernel_ms':ms(32),
          'host_post_ms':0,'upload_ms':None,'readback_ms':None,'wall_seconds':None,'board_pixel_mismatches':None,'host_channel_mismatches':mismatch,'max_channel_error':maximum,'raw_v2_channel_mismatches':None,
          'dual_arm':True,'dual_active':active,'arm_cores_used':2 if active else 1,'dual_split_y':mailbox[38],'cpu1_completed_jobs':r[75],'dual_startup_ms':(mailbox[40]+(mailbox[43]<<32))/freq*1000 if number==1 else 0,'dual_bootstrap_ms':None,
          'sd_read_and_input_check_ms':sdms(64),'sd_output_write_ms':sdms(66),'sd_output_crc_ms':sdms(68),'sd_job_before_telemetry_ms':sdms(70),'sd_compute_including_warmup_ms':sdms(72),
          'sd_build_id':hex(r[14]),'task_id':manifest['task_id'],'model_tag':hex(r[7]),'fpga_model_tag':hex(r[7]),'neural_engine_lanes':32,'pl_statistics':False,
          'quantized_artifacts':manifest['quantized_artifacts'],'output':str(run/'fpga_output.png'),'speedup':None,
          'timing_scope':'Board steady compute excludes SD IO, CRC, first batch warmup/startup and PC precomputed parameters. SD file-read/input-check, output-write, output-CRC and job-before-telemetry timers separately recorded. No JTAG image data transfer; no full task end-to-end speedup asserted.'}
        result['pipeline_compute_ms']=result['fpga_ms']+job['host_parameter_ms']
        from evaluation import attach_quality,auto_ground_truth
        attach_quality(result,run,image,output,auto_ground_truth(job['source']))
        (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        if mismatch:raise ValueError('板卡结果与独立参考不一致，已保留原始输出：'+str(run))
        records.append({'run':str(run),'result':result})
    batch_summary=None
    if (task/'OUT/BATCH.BIN').is_file():
        summary=checked((task/'OUT/BATCH.BIN').read_bytes(),256,BATCH_MAGIC)
        if summary[2]!=len(manifest['jobs']) or summary[3]!=0 or summary[4]!=manifest['model_tag'] or not summary[5] or summary[7]!=index[15]:raise ValueError('整批结束记录与任务不匹配。')
        measured=(summary[8]+(summary[9]<<32))/summary[5]*1000
        lower_bound=sum(row['result']['sd_job_before_telemetry_ms'] for row in records)
        valid=measured+0.1>=lower_bound
        batch_summary={'board_before_final_summary_ms':measured if valid else None,'raw_timer_ms':measured,
          'minimum_from_imported_jobs_ms':lower_bound,'timing_valid':valid,
          'timing_warning':None if valid else '整批计时小于逐图时间合计，计时基准不连续；该整批时间不可用于性能结论。',
          'jobs':summary[2],'build_id':hex(summary[6]),
          'scope':'Board mount, index/input reads/checks, CPU1 startup/warmup, all image compute, outputs, per-job records and serial progress; excludes final BATCH.BIN write, JTAG config, PC export/import and moving the card.'}
    report={'task_id':manifest['task_id'],'records':records,'missing_jobs':missing,'verified':bool(verify),'board_results_imported':len(records),'board_batch_completed':batch_summary is not None and not missing,'batch_summary':batch_summary}
    (root/'sd_import.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return root,report
def explain_sd(result):
    return ('SD卡实测导入：图片从SD读入，经板上ARM局部统计与FPGA暗通道、神经推理和保护，结果写回SD。电脑仅提前解码、缩放和估计帧参数，导入后可核验与评价；显示图保持板卡文件原始像素。\n'
      '本批启动时JTAG仅负责一次配置与双核启动；图片数据不经过JTAG。本阶段并非SD上电自启动。\n'
      '算法时间不含SD读写、CRC、首图预热和启动；这些开销另列。首图以外沿用已启动CPU1。没有测量完整PC导出/搬卡/板卡运行/导入全任务加速比。')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('export');e.add_argument('input');e.add_argument('destination');e.add_argument('--edge',type=int,default=640);e.add_argument('--repeats',type=int,default=1)
    i=sub.add_parser('import');i.add_argument('task');i.add_argument('--destination');i.add_argument('--skip-verification',action='store_true')
    a=p.parse_args()
    if a.action=='export':print(export_task(a.input,a.destination,a.edge,a.repeats,progress=print)[0])
    else:print(import_task(a.task,a.destination,not a.skip_verification,progress=print)[0])
