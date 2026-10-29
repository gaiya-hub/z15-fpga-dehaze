"""PC-versus-FPGA comparison for one identical algorithm and recorded parameters."""
from pathlib import Path
import json,time,hashlib,threading,queue,platform
import numpy as np
from PIL import Image,ImageTk,ImageOps
from model import estimate_air,reference
from fused_dcp import estimate_strong_air,predict_weight,strong_reference
from evaluation import open_rgb,reference_image,image_scores,display_number
ROOT=Path(__file__).resolve().parent

def measure(record_dir,ground_truth=None,progress=lambda _:None):
    directory=Path(record_dir);record=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if record.get('algorithm')=='fused-neural-v3-pl':
        from pl_same_algorithm_v3 import measure as measure_pl
        return measure_pl(record_dir,ground_truth,progress)
    if record.get('batch_compacted'):raise ValueError('请使用单图处理的完整记录进行同算法对比。')
    hybrid=record.get('algorithm')=='fused-neural-v3-hybrid'
    image=open_rgb(directory/'input.png');raw=open_rgb(directory/'fpga_output.png')
    hardware=open_rgb(directory/('hybrid_output.png' if hybrid else 'fpga_output.png'))
    if image.size!=hardware.size or list(image.size)!=record['processing_size']:raise ValueError('记录的输入输出尺寸不一致。')
    fused=hybrid or record.get('algorithm')=='fused-guided-v2'
    model=json.loads((ROOT/'frozen_model.json').read_text(encoding='utf-8')) if fused else None
    if hybrid:
        from hybrid_runtime_v3 import processor,artifacts
        if artifacts()!=record.get('neural_artifacts'):raise ValueError('V3权重或源文件与记录不一致，请重新处理后对比。')
        neural=processor()
    def compute():
        begin=time.perf_counter();air=estimate_air(image)
        if fused:
            strong_air=estimate_strong_air(image,model['air_method']);weight=predict_weight(image,model)
            if tuple(air)!=tuple(record['air']) or tuple(strong_air)!=tuple(record['strong_air']) or weight!=record['blend_q8']:
                raise ValueError('当前参数控制器与FPGA记录不一致，请重新处理当前图片。')
        elif tuple(air)!=tuple(record['air']):raise ValueError('空气光估计与记录不一致，请重新处理。')
        middle=time.perf_counter();base=reference(image,air,record['strength'],record['t_min'],record['lift'])
        if fused:
            strong,_=strong_reference(image,strong_air);a=np.asarray(base,dtype=np.int32);b=np.asarray(strong,dtype=np.int32)
            base=Image.fromarray(((a*(256-weight)+b*weight+128)>>8).astype(np.uint8))
        dcp_done=time.perf_counter();output=base
        if hybrid:output,_=neural.correct(image,base)
        end=time.perf_counter()
        return output,{'parameters_ms':(middle-begin)*1000,'dcp_ms':(dcp_done-middle)*1000,'post_ms':(end-dcp_done)*1000 if hybrid else 0,
                       'pixel_ms':(end-middle)*1000,'algorithm_ms':(end-begin)*1000},base
    progress('电脑完整同算法预热一次');compute();samples=[]
    for i in range(3):progress(f'电脑完整同算法计时 {i+1}/3');pc,sample,pc_base=compute();samples.append(sample)
    difference=np.abs(np.asarray(pc,dtype=np.int16)-np.asarray(hardware,dtype=np.int16))
    base_difference=np.abs(np.asarray(pc_base,dtype=np.int16)-np.asarray(raw,dtype=np.int16))
    mismatch=int(np.count_nonzero(difference));maximum=int(difference.max());base_mismatch=int(np.count_nonzero(base_difference))
    run=ROOT/'outputs'/('same_algorithm_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000000));run.mkdir(parents=True)
    pc.save(run/'pc_output.png');hardware.save(run/'scheme_output.png');raw.save(run/'fpga_output.png');image.save(run/'input.png')
    gt=ground_truth or record.get('ground_truth');clear,reason=reference_image(record['source'],gt,image.size)
    if clear:clear.save(run/'ground_truth.png')
    means={key:float(np.mean([sample[key] for sample in samples])) for key in samples[0]}
    actual_parameter_ms=record.get('host_parameter_ms');shared_parameters=actual_parameter_ms if actual_parameter_ms is not None else means['parameters_ms']
    post_ms=record.get('host_post_ms',0) if hybrid else 0
    hardware_algorithm=shared_parameters+record['fpga_ms']+post_ms
    data={'version':'same-algorithm-v3','algorithm':record.get('algorithm','baseline'),'source':record['source'],
          'processing_size':list(image.size),'linked_fpga_result':str((directory/'result.json').resolve()),
          'input_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'platform':platform.platform(),
          'pc':{'mean':means,'samples':samples,'scores':image_scores(clear,pc) if clear else None},
          'fpga':{'board_ms':record['fpga_ms'],'kernel_ms':record['fpga_start_done_ms'],'host_parameter_ms':actual_parameter_ms,
                  'host_post_ms':post_ms,'pixel_pipeline_ms':record['fpga_start_done_ms']+post_ms,
                  'algorithm_ms':hardware_algorithm,'algorithm_ms_estimated':actual_parameter_ms is None,
                  'upload_ms':record['upload_ms'],'readback_ms':record['readback_ms'],'record_wall_seconds':record['wall_seconds'],
                  'scores':image_scores(clear,hardware) if clear else None,'neural_location':'PC CPU' if hybrid else None},
          'channel_mismatches':mismatch,'max_channel_error':maximum,'raw_fpga_channel_mismatches':base_mismatch,'quality_reason':reason,
          'pixel_compute_ratio':means['pixel_ms']/(record['fpga_start_done_ms']+post_ms),
          'algorithm_compute_ratio':means['algorithm_ms']/hardware_algorithm,
          'timing_scope':'PC: full identical algorithm, 1 warmup, mean of 3. Hardware-assisted V3 includes PC auxiliary feature generation/neural/guard after real FPGA DCP. FPGA timing is from the linked record. Model loading, disk I/O, JTAG and verification excluded from compute ratios. No end-to-end acceleration asserted.'}
    (run/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if mismatch or base_mismatch:raise RuntimeError(f'同算法校验未通过：最终差异{mismatch}通道、FPGA基础差异{base_mismatch}通道。记录：{run}')
    return run,data,pc,hardware

def open_window(parent,record_dir,ground_truth=None):
    import tkinter as tk
    from tkinter import ttk,messagebox
    from detail_view import show_detail
    win=tk.Toplevel(parent);win.title('同算法对比 · 电脑纯软件 / FPGA');win.geometry('1180x850');win.minsize(1050,740)
    top=ttk.Frame(win,padding=12);top.pack(fill='x')
    ttk.Label(top,text='同一算法 · 电脑纯软件与FPGA',font=('Microsoft YaHei UI',17,'bold')).pack(side='left')
    state={'images':[None,None],'photos':[],'run':None,'data':None,'busy':False};events=queue.Queue()
    ttk.Button(top,text='同步放大',command=lambda:show_detail(win,state['images'],['电脑纯软件（同算法）','板卡输出（执行位置见记录）'])).pack(side='right',padx=8)
    info=tk.StringVar(value='使用当前FPGA记录中的同一输入、同一尺寸和同一算法参数；不运行其他去雾算法。');win.info_var=info
    ttk.Label(win,textvariable=info,padding=(12,0),wraplength=1140).pack(fill='x')
    holder=ttk.Frame(win,padding=12);holder.pack(fill='both',expand=True);holder.rowconfigure(1,weight=1);canvases=[]
    for i,title in enumerate(['电脑完整同算法输出','FPGA参与方案输出']):
        holder.columnconfigure(i,weight=1,uniform='output');ttk.Label(holder,text=title,anchor='center').grid(row=0,column=i,sticky='ew')
        canvas=tk.Canvas(holder,bg='#172638',height=220,highlightthickness=0);canvas.grid(row=1,column=i,sticky='nsew',padx=5,pady=5);canvases.append(canvas)
    table=ttk.Treeview(win,columns=('metric','pc','fpga','scope'),show='headings',height=10)
    for key,title,width in [('metric','项目',155),('pc','电脑纯软件',165),('fpga','FPGA方案',165),('scope','计时与评价口径',585)]:
        table.heading(key,text=title);table.column(key,width=width,anchor='center' if key!='scope' else 'w')
    table.pack(fill='x',padx=12)
    note=tk.StringVar(value='FPGA一侧来自当前处理记录；电脑一侧在本窗口重新计时。计算比不代表包含JTAG的整批速度。')
    ttk.Label(win,textvariable=note,wraplength=1140,padding=12).pack(fill='x')
    bottom=ttk.Frame(win,padding=12);bottom.pack(fill='x');status=tk.StringVar(value='等待电脑同算法计时')
    ttk.Label(bottom,textvariable=status,wraplength=850).pack(side='left')
    ttk.Button(bottom,text='结果目录',command=lambda:__import__('os').startfile(state['run']) if state['run'] else None).pack(side='right')
    def render(event=None):
        state['photos']=[]
        for canvas,image in zip(canvases,state['images']):
            canvas.delete('all');w=max(1,canvas.winfo_width());h=max(1,canvas.winfo_height())
            if image is None:canvas.create_text(w/2,h/2,text='正在准备同算法对比',fill='white');continue
            fitted=ImageOps.contain(image,(max(1,w-10),max(1,h-10)),Image.Resampling.LANCZOS)
            photo=ImageTk.PhotoImage(fitted);state['photos'].append(photo);canvas.create_image(w/2,h/2,image=photo)
    for canvas in canvases:canvas.bind('<Configure>',render)
    def display(result):
        run,data,pc,hardware=result;state.update(run=run,data=data,images=[pc,hardware])
        for item in table.get_children():table.delete(item)
        p=data['pc'];f=data['fpga'];n=lambda value,digits=2:display_number(value,digits)
        rows=[('基础DCP计算 (ms)',n(p['mean']['dcp_ms']),n(f['kernel_ms']),'FPGA仅基础DCP；下面单列电脑辅助与神经校正'),
              ('电脑辅助＋校正 (ms)',n(p['mean']['post_ms']),n(f['host_post_ms']),'V3两侧均在电脑CPU执行；V2为0'),
              ('参数估计 (ms)',n(p['mean']['parameters_ms']),n(f['host_parameter_ms']),'均在电脑执行；旧记录未计时则留空'),
              ('算法合计 (ms)',n(p['mean']['algorithm_ms']),n(f['algorithm_ms'])+('（估算）' if f['algorithm_ms_estimated'] else ''),'参数＋基础DCP＋电脑辅助/神经校正；不含传输与文件读写'),
              ('JTAG上传 / 回读 (ms)','—',n(f['upload_ms'])+' / '+n(f['readback_ms']),'单列传输耗时，不计入算法计算比')]
        if data.get('algorithm')=='fused-neural-v3-pl':
            rows[0]=('基础DCP计算 (ms)',n(p['mean']['dcp_ms']),n(f['dcp_ms']),'电脑CPU参考；板卡端FPGA暗通道阶段含控制与缓存')
            rows[1]=('统计＋神经阶段(ms)',n(p['mean']['post_ms']),n(f['post_stage_ms']),'电脑CPU参考；板卡端统计＋FPGA神经推理；执行位置见下方')
            rows[3]=('算法合计 (ms)',n(p['mean']['algorithm_ms']),n(f['algorithm_ms']),'电脑：完整定点算法；板卡：帧参数＋板上暗通道、统计、神经及保护')
        if f.get('dual_arm'):
            rows.insert(4,('双核启动握手(ms)','—',n(f.get('dual_bootstrap_ms')),'逐图启动开销；计入实际等待，未计入算法计算耗时比'))
        for key,label,digits in [('psnr_db' ,'PSNR ↑ (dB)',2),('ssim','SSIM ↑',4),('mae','MAE ↓',3)]:
            rows.append((label,n((p['scores'] or {}).get(key),digits),n((f['scores'] or {}).get(key),digits),'完整最终输出对同一无雾参考评价；缺GT留空'))
        rows += [('计算一致性','差异通道 '+str(data['channel_mismatches']),'最大通道误差 '+str(data['max_channel_error']),'仅核对算法实现，不等于无雾图画质'),
                 ('算法计算耗时比',n(data['algorithm_compute_ratio'])+'×','电脑 / FPGA方案','旧记录缺参数计时则为估算；不是整批端到端加速比')]
        table.configure(height=len(rows))
        for row in rows:table.insert('','end',values=row)
        note.set('V3神经校正在电脑，FPGA仅基础DCP；V2无神经校正。相同输入与尺寸 '+str(data['processing_size'])+'；电脑预热1次、平均3次。FPGA来自记录：'+str(record_dir)+'。'+(data['quality_reason'] or ''))
        if data.get('algorithm')=='fused-neural-v3-pl':note.set('局部统计位置：'+data['fpga'].get('local_feature_location','Zynq ARM')+'；神经推理和防失真保护均在FPGA；最终图片直接回读。电脑侧预热后测3次，板卡侧引用当前记录；JTAG另计。'+('双核启动握手另列，不计入算法计算比。' if data['fpga'].get('dual_arm') else '')+'')
        status.set('同算法对比完成；逐像素一致。');render()
    def start():
        if state['busy']:return
        state['busy']=True;button.configure(state='disabled')
        def work():
            try:events.put(('done',measure(record_dir,ground_truth,lambda text:events.put(('status',text)))))
            except Exception as error:events.put(('error',str(error)))
        threading.Thread(target=work,daemon=True).start()
    button=ttk.Button(top,text='重新测量电脑同算法',command=start);button.pack(side='right')
    def drain():
        try:
            while True:
                kind,value=events.get_nowait()
                if kind=='status':status.set(value)
                else:
                    state['busy']=False;button.configure(state='normal')
                    if kind=='done':display(value)
                    else:status.set('对比未完成');messagebox.showerror('同算法对比',value,parent=win)
        except queue.Empty:pass
        if win.winfo_exists():win.after(100,drain)
    win.after(100,drain);win.after(200,start);win.comparison_state=state;win.metric_table=table;return win
