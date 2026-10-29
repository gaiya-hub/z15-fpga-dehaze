"""Sequential folder processing. PC and actual FPGA are explicitly different engines."""
from pathlib import Path
import csv,hashlib,json,time,threading,argparse
from model import load_image
from fused_dcp import process
from evaluation import IMAGE_EXTENSIONS,auto_ground_truth,evaluate_quality

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def find_gt(source,relative,gt_root):
    if not gt_root:return auto_ground_truth(source)
    folder=Path(gt_root)/relative.parent
    if not folder.is_dir():return None
    stem=source.stem[:-5] if source.stem.lower().endswith('_hazy') else source.stem
    for candidate in dict.fromkeys([source.stem,stem]):
        matches=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.lower()==candidate.lower() and p.resolve()!=source.resolve()]
        if len(matches)==1:return matches[0]
        if len(matches)>1:return None
    return None

def _run_batch_unlocked(input_dir,output_dir,engine='pc',max_edge=1280,ground_truth_dir=None,recursive=True,
              arm_benchmark=False,keep_debug=False,initialize_board=True,stop=None,progress=lambda _:None,algorithm='fused'):
    if algorithm not in ['fused','hybrid','pl_v3']:raise ValueError('未知批处理算法。')
    start=time.perf_counter();stop=stop or threading.Event()
    source_root=Path(input_dir).resolve();destination=Path(output_dir).resolve()
    if not source_root.is_dir():raise ValueError('有雾图片文件夹不存在。')
    if destination.is_relative_to(source_root):raise ValueError('输出文件夹必须放在输入文件夹之外，避免把结果再次当作输入。')
    if engine not in ['pc','fpga']:raise ValueError('未知处理模式。')
    if max_edge not in [640,1280,1920]:raise ValueError('最长边请选择640、1280或1920。')
    gt_root=Path(ground_truth_dir).resolve() if ground_truth_dir else None
    if gt_root and not gt_root.is_dir():raise ValueError('无雾参考文件夹不存在。')
    files=sorted((p for p in (source_root.rglob('*') if recursive else source_root.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),key=lambda p:p.relative_to(source_root).as_posix().lower())
    if not files:raise ValueError('文件夹中没有支持的图片。')
    model_path=ROOT/'frozen_model.json';model=json.loads(model_path.read_text(encoding='utf-8'))
    config={'version':'folder-v1','input':str(source_root),'engine':engine,'max_edge':max_edge,'gt_root':str(gt_root) if gt_root else None,
            'recursive':bool(recursive),'arm_benchmark':bool(arm_benchmark) if engine=='fpga' else False,'keep_debug':bool(keep_debug) if engine=='fpga' else False,'model_sha256':digest(model_path),
            'algorithm_source_sha256':digest(ROOT/'fused_dcp.py'),'reference_source_sha256':digest(ROOT/'model.py'),
            'algorithm':('fused-neural-v3-pl' if algorithm=='pl_v3' else ('fused-neural-v3-hybrid' if algorithm=='hybrid' else 'fused-guided-v2')),'resize_policy':'keep aspect, no upsampling, at most 1920x1280'}
    if algorithm=='pl_v3':
        from pl_backend_v3 import artifacts,get_model
        config['quantized_artifacts']=artifacts();get_model()
    if algorithm=='hybrid':
        from hybrid_runtime_v3 import artifacts,processor
        config['neural_artifacts']=artifacts();processor()
    if engine=='fpga':
        config['bit_sha256']=digest(ROOT/'board/design_1_wrapper.bit');config['firmware_sha256']=digest(ROOT/'board/demo.elf')
    config_path=destination/'batch_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8'))!=config:raise ValueError('此输出目录属于不同配置或版本的任务。请用新目录，或恢复原来的输入、模式、尺寸、参考目录及基准/调试选项。')
    elif destination.exists() and any(p.name!='batch.lock' for p in destination.iterdir()):raise ValueError('请选择空的输出文件夹，或此前同一批任务的输出文件夹。')
    destination.mkdir(parents=True,exist_ok=True);save_json(config_path,config)
    state={'engine':engine,'total':len(files),'succeeded':0,'skipped':0,'failed':0,'processed':0,'status':'running','output_dir':str(destination),'board_initialized':False}
    columns=['source','status','output','original_size','processing_size','pc_ms','fpga_ms','arm_ms','upload_ms','readback_ms','item_seconds','psnr_input','psnr_output','ssim_input','ssim_output','mae_output','host_channel_mismatches','host_post_ms','pipeline_compute_ms','dcp_ms','arm_feature_ms','neural_ms','algorithm','source_sha256','output_sha256','error']
    def event(message):
        with (destination/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),**message},ensure_ascii=False)+'\n')
    def emit(message):progress({**state,'elapsed_seconds':time.perf_counter()-start,'message':message})
    def summary():
        state['elapsed_seconds']=time.perf_counter()-start
        save_json(destination/'summary.json',state)
    summary();emit('找到 '+str(len(files))+' 张图片。')
    backend=None;board_initialized=False
    try:
        with (destination/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();f.flush()
            for source in files:
                if stop.is_set():state['status']='stopped';break
                relative=source.relative_to(source_root);record_path=destination/'records'/Path(str(relative)+'.json')
                output_path=destination/'images'/Path(str(relative)+'.dehazed.png')
                if not output_path.resolve().is_relative_to(destination) or not record_path.resolve().is_relative_to(destination):raise ValueError('输出路径超出任务目录。')
                began=time.perf_counter();record={'source':str(source),'relative_source':relative.as_posix(),'output':str(output_path),'engine':engine}
                try:
                    source_hash=digest(source);gt=find_gt(source,relative,gt_root);gt_hash=digest(gt) if gt else None
                    previous=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else None
                    reused=bool(previous and previous.get('status')=='success' and previous.get('source_sha256')==source_hash and previous.get('gt_sha256')==gt_hash and output_path.is_file() and digest(output_path)==previous.get('output_sha256'))
                    if reused:
                        record=previous;state['skipped']+=1
                    else:
                        emit('正在处理 '+relative.as_posix());pc_ms=None;hardware_run=None
                        if engine=='fpga':
                            if backend is None:import backend as backend_module;backend=backend_module
                            if not board_initialized:
                                if initialize_board:backend.initialize(lambda text:emit(text))
                                board_initialized=True
                                state['board_initialized']=True
                            image,output,result,hardware_run=backend.run_image(source,max_edge=max_edge,repeats=1,run_cpu=arm_benchmark,ground_truth=gt,
                                    progress=lambda text:emit(text),algorithm=algorithm,export_sheet=False)
                        else:
                            image,original_size=load_image(source,max_edge);t=time.perf_counter();output,parameters=(__import__('pl_backend_v3').process_pc(image,model) if algorithm=='pl_v3' else (__import__('hybrid_runtime_v3').process_pc(image,model) if algorithm=='hybrid' else process(image,model)));pc_ms=(time.perf_counter()-t)*1000
                            quality,_=evaluate_quality(source,image,output,gt)
                            if 'fpga' in quality:quality['output']=quality.pop('fpga')
                            result={'source':str(source),'original_size':original_size,'processing_size':list(image.size),'quality':quality,'parameters':parameters,'algorithm':config['algorithm'],'execution':'PC software; not FPGA output'}
                        output_path.parent.mkdir(parents=True,exist_ok=True)
                        temporary=output_path.with_name(output_path.stem+'.partial.png');output.save(temporary,format='PNG');temporary.replace(output_path)
                        record.update(status='success',source_sha256=source_hash,gt_sha256=gt_hash,output_sha256=digest(output_path),result=result,
                                      pc_ms=pc_ms,item_seconds=time.perf_counter()-began,hardware_run=str(hardware_run) if hardware_run else None)
                        save_json(record_path,record);state['succeeded']+=1
                        if hardware_run and not keep_debug:
                            # Only remove known files in this newly created, already verified run.
                            run=Path(hardware_run).resolve();assert run.is_relative_to((backend.ROOT/'outputs').resolve())
                            compacted=dict(result,batch_compacted=True,batch_output=str(output_path))
                            save_json(run/'result.json',compacted)
                            for name in ['input.bin','fpga.bin','input.png','fpga_output.png','hybrid_output.png','fpga_v2.bin','fpga_v2_output.png','frame_parameters.bin','reference.png','ground_truth.png']:
                                p=run/name
                                try:
                                    if p.is_file() and not p.is_symlink():p.unlink()
                                except OSError as cleanup_error:event({'warning':'调试文件未清理','file':str(p),'error':str(cleanup_error)})
                    r=record['result'];q=r.get('quality') or {};before=q.get('input') or {};after=q.get('output') or q.get('fpga') or {}
                    row={'source':relative.as_posix(),'status':'reused' if reused else 'success','output':str(output_path),
                         'original_size':str(r['original_size']),'processing_size':str(r['processing_size']),'pc_ms':record.get('pc_ms'),
                         **{k:r.get(k) for k in ['fpga_ms','arm_ms','upload_ms','readback_ms','host_channel_mismatches']},
                         'host_post_ms':r.get('host_post_ms'),'pipeline_compute_ms':r.get('pipeline_compute_ms'),'dcp_ms':r.get('dcp_ms'),'arm_feature_ms':r.get('arm_feature_ms'),'neural_ms':r.get('neural_ms'),'algorithm':r.get('algorithm'),'item_seconds':record.get('item_seconds'),'psnr_input':before.get('psnr_db'),'psnr_output':after.get('psnr_db'),
                         'ssim_input':before.get('ssim'),'ssim_output':after.get('ssim'),'mae_output':after.get('mae'),
                         'source_sha256':record['source_sha256'],'output_sha256':record['output_sha256']}
                    writer.writerow(row)
                except Exception as error:
                    state['failed']+=1;record.update(status='failed',error=str(error),item_seconds=time.perf_counter()-began)
                    save_json(record_path,record);writer.writerow({'source':relative.as_posix(),'status':'failed','error':str(error)})
                    event({'source':relative.as_posix(),'status':'failed','error':str(error)})
                    if engine=='fpga':state['status']='hardware_error'
                state['processed']+=1;f.flush();summary();emit(relative.as_posix())
                if state['status']=='hardware_error':break
            if state['status']=='running':state['status']='completed'
    finally:
        summary();event(state);emit('批处理结束：'+state['status'])
    return state

def run_batch(input_dir,output_dir,*args,**kwargs):
    import msvcrt
    source=Path(input_dir).resolve();destination=Path(output_dir).resolve()
    if not source.is_dir():raise ValueError('有雾图片文件夹不存在。')
    if destination.is_relative_to(source):raise ValueError('输出文件夹必须放在输入文件夹之外。')
    if destination.exists() and not (destination/'batch_config.json').exists() and any(p.name!='batch.lock' for p in destination.iterdir()):
        raise ValueError('请选择空的输出文件夹，或此前同一批任务的输出文件夹。')
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'batch.lock').open('a+b') as handle:
        if handle.seek(0,2)==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('这个批处理输出目录正被另一个任务使用，请等待后再继续。')
        try:return _run_batch_unlocked(input_dir,output_dir,*args,**kwargs)
        finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Folder dehazing: PC software or real sequential FPGA processing.')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--engine',choices=['pc','fpga'],default='pc')
    p.add_argument('--algorithm',choices=['fused','hybrid','pl_v3'],default='pl_v3')
    p.add_argument('--edge',type=int,choices=[640,1280,1920],default=1280);p.add_argument('--gt');p.add_argument('--no-recursive',action='store_true')
    p.add_argument('--arm-benchmark',action='store_true');p.add_argument('--keep-debug',action='store_true')
    a=p.parse_args();print(json.dumps(run_batch(a.input,a.output,a.engine,a.edge,a.gt,not a.no_recursive,a.arm_benchmark,a.keep_debug,algorithm=a.algorithm,progress=lambda v:print(v['processed'],v['total'],v['message'],flush=True)),ensure_ascii=False))
