"""PC-only baselines. Never programs the FPGA or uses GT during inference."""
from pathlib import Path
import json,time,hashlib,importlib.util,sys,threading,queue,platform,copy
import numpy as np
from PIL import Image,ImageOps,ImageTk
from model import load_image,estimate_air,reference
from evaluation import open_rgb,reference_image,image_scores,display_number

ROOT=Path(__file__).resolve().parent
from portable_settings import ffa_root
FFA_ROOT=ffa_root()
NETWORKS={'none':'仅同算法软件','its':'FFA-Net 室内权重','ots':'FFA-Net 室外权重'}
_models={}

def load_ffa(task):
    import torch
    from numpy.core.multiarray import scalar
    checkpoint=FFA_ROOT/f'FFA_trained_models/{task}_train_ffa_3_19.pk'
    # Only definitions are imported: local test.py has plotting/extra enhancement side effects.
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        spec=importlib.util.spec_from_file_location('_demo_ffa',FFA_ROOT/'net/models/FFA.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    # Old checkpoints contain scalar training statistics. Keep restricted loading enabled.
    allowed=[scalar,np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(allowed):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
    net=module.FFA(gps=3,blocks=19)
    net.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    net.eval().to('cpu')
    metadata={'architecture':'FFA-Net, gps=3, blocks=19, dim=64','parameters':sum(p.numel() for p in net.parameters()),
              'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'device':'CPU','torch_version':torch.__version__,'threads':torch.get_num_threads(),
              'preprocessing':'RGB /255; mean=[0.64,0.60,0.58], std=[0.14,0.15,0.152]',
              'postprocessing':'clamp [0,1], round to uint8; no gamma/CLAHE/saturation changes; no tiling/resizing'}
    return net,metadata

def run_comparison(source,ground_truth=None,max_edge=1280,strength=160,t_min=96,lift=64,
                   task='ots',fpga_run=None,progress=lambda message:None,algorithm='fused'):
    if task not in NETWORKS:raise ValueError('未知网络选项')
    linked=None
    if fpga_run:
        fpga_run=Path(fpga_run)
        linked=json.loads((fpga_run/'result.json').read_text(encoding='utf-8'))
        algorithm='fused' if linked.get('algorithm')=='fused-guided-v2' else 'baseline'
        image=open_rgb(fpga_run/'input.png');source=linked['source']
        air=tuple(linked['air']);strength=linked['strength'];t_min=linked['t_min'];lift=linked['lift']
        fpga=open_rgb(fpga_run/'fpga_output.png')
        if image.size!=fpga.size:raise ValueError('历史记录输入输出尺寸不一致')
    else:
        image,_=load_image(source,max_edge);air=estimate_air(image);fpga=None
    if algorithm=='fused':
        from fused_dcp import process as enhanced_process
        fusion_model=json.loads((ROOT/'frozen_model.json').read_text(encoding='utf-8'))
        def pc_compute():return enhanced_process(image,fusion_model)[0]
    else:
        def pc_compute():return reference(image,air,strength,t_min,lift)
    run=ROOT/'outputs'/('software_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000000))
    run.mkdir(parents=True)
    data={'source':str(source),'ground_truth':str(ground_truth) if ground_truth else None,
          'processing_size':list(image.size),'air':list(air),'strength':strength,'t_min':t_min,'lift':lift,
          'linked_fpga_result':str(fpga_run/'result.json') if linked else None,
          'input_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
          'platform':platform.platform(),'processor':platform.processor(),'numpy_version':np.__version__,
          'timing_note':'CPU: 1 warmup, mean of 3 compute-only runs; excludes input/air estimation and PNG/metrics. FFA includes normalization, inference and uint8 output; model loading separately. ARM/FPGA use existing board record. Different machines/implementations must not be described as equal-platform acceleration.',
          'methods':{}}
    image.save(run/'input.png')
    clear,reason=reference_image(source,ground_truth,image.size)
    data['quality_reason']=reason
    if clear is not None:
        clear.save(run/'ground_truth.png')
        data['ground_truth_sha256']=hashlib.sha256(Path(ground_truth).read_bytes()).hexdigest()
    scores=lambda im:image_scores(clear,im) if clear is not None else None
    data['methods']['hazy']={'label':'有雾输入','scores':scores(image),'file':'input.png'}
    progress(f'电脑同算法软件测量：{image.width}×{image.height}，预热 1 次、平均 3 次')
    pc_compute()
    samples=[]
    for _ in range(3):
        start=time.perf_counter();pc=pc_compute();samples.append((time.perf_counter()-start)*1000)
    pc.save(run/'pc_dcp.png')
    data['methods']['pc_dcp']={'label':'电脑 CPU · 增强V2参考（含参数估计）' if algorithm=='fused' else '电脑 CPU · 旧版参考','ms':float(np.mean(samples)),
        'samples_ms':samples,'scores':scores(pc),'file':'pc_dcp.png'}
    if algorithm=='fused':
        start=time.perf_counter();old=reference(image,air,160,96,64);old_ms=(time.perf_counter()-start)*1000
        old.save(run/'baseline.png')
        data['methods']['baseline']={'label':'旧版暗通道 · 电脑参考（单次）','ms':old_ms,'scores':scores(old),'file':'baseline.png'}
        data['timing_note']+=' Enhanced PC reference includes both air estimates and controller; board ARM/FPGA excludes PC frame parameter estimation.'
    from local_routes import SOURCE as ADAPTIVE_SOURCE
    if ADAPTIVE_SOURCE.is_file():
        from local_routes import adaptive_image,adaptive_metadata
        progress('测量你的自适应暗通道（传统增强，无神经网络），保持相同尺寸')
        adaptive_image(image)
        adaptive_samples=[]
        for _ in range(3):
            start=time.perf_counter();adapted=adaptive_image(image);adaptive_samples.append((time.perf_counter()-start)*1000)
        adapted.save(run/'adaptive_output.png')
        data['methods']['adaptive']={'label':'电脑 CPU · 你的自适应暗通道（非网络）','ms':float(np.mean(adaptive_samples)),
            'samples_ms':adaptive_samples,'scores':scores(adapted),'file':'adaptive_output.png',**adaptive_metadata()}
    else:
        data.setdefault('unavailable_methods',{})['adaptive']='未配置外部自适应算法；当前只运行自研同算法参考。'
    if linked:
        difference=np.abs(np.asarray(pc,dtype=np.int16)-np.asarray(fpga,dtype=np.int16))
        data['pc_fpga_channel_mismatches']=int(np.count_nonzero(difference))
        data['pc_fpga_max_error']=int(difference.max())
        fpga.save(run/'fpga_output.png')
        data['methods']['arm']={'label':'板上 ARM · 同算法 C/O2（历史实测）','ms':linked.get('arm_ms'),'scores':scores(fpga) if linked.get('board_pixel_mismatches')==0 else None}
        data['methods']['fpga']={'label':'FPGA · 同算法（历史实测）','ms':linked['fpga_ms'],'scores':scores(fpga),'file':'fpga_output.png'}
        data['board_speedup']=linked.get('speedup')
    # Persist the useful CPU result even if neural inference fails.
    (run/'software_result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if task!='none':
        progress(f'加载 {NETWORKS[task]}；电脑 CPU 推理可能较慢，保持实际处理尺寸，不缩成小图')
        import torch
        start=time.perf_counter()
        if task not in _models:_models[task]=load_ffa(task)
        net,metadata=_models[task]
        load_ms=(time.perf_counter()-start)*1000
        def predict():
            rgb=np.asarray(image,dtype=np.float32)/255.0
            x=torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0)
            mean=torch.tensor([.64,.60,.58]).view(1,3,1,1)
            std=torch.tensor([.14,.15,.152]).view(1,3,1,1)
            with torch.inference_mode():y=net((x-mean)/std).clamp(0,1)[0].permute(1,2,0).numpy()
            return Image.fromarray(np.rint(y*255).astype(np.uint8))
        progress('FFA-Net 预热 1/1（首次可能较慢）');predict()
        samples=[]
        for index in range(3):
            progress(f'FFA-Net CPU 实际推理 {index+1}/3 · {image.width}×{image.height}')
            start=time.perf_counter();neural=predict();samples.append((time.perf_counter()-start)*1000)
        neural.save(run/'ffa_output.png')
        data['methods']['ffa']={'label':NETWORKS[task]+' · 电脑 CPU','ms':float(np.mean(samples)),
            'samples_ms':samples,'model_load_or_cache_ms':load_ms,'scores':scores(neural),'file':'ffa_output.png',**metadata}
    (run/'software_result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    export_sheet(run,data)
    return run,data

def export_sheet(run,data):
    from PIL import ImageDraw,ImageFont
    w,h=data['processing_size'];margin=20;head=90;foot=105
    panels=[]
    if (run/'ground_truth.png').exists():panels.append(('无雾参考（仅评价）','ground_truth.png',None))
    for key in ['hazy','baseline','pc_dcp','adaptive','fpga','ffa']:
        method=data['methods'].get(key)
        if method and method.get('file'):panels.append((method['label'],method['file'],method))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    sheet=Image.new('RGB',((w+margin)*len(panels)+margin,h+head+foot),'#eef2f6');draw=ImageDraw.Draw(sheet)
    for index,(title,file,method) in enumerate(panels):
        x=margin+index*(w+margin);draw.text((x,12),title,font=font,fill='#142b45')
        if method:
            s=method.get('scores')
            if s:draw.text((x,40),f'PSNR {display_number(s["psnr_db"],2)} dB / SSIM {display_number(s["ssim"],4)}',font=small,fill='#42556b')
            if method.get('ms') is not None:draw.text((x,62),f'{method["ms"]:.2f} ms',font=small,fill='#42556b')
        sheet.paste(open_rgb(run/file),(x,head))
    y=head+h+15
    draw.text((margin,y),f'实际处理 {w}×{h} RGB，全部同尺寸；FFA-Net 在电脑 CPU 运行，尚未部署到 FPGA。',font=font,fill='#142b45')
    draw.text((margin,y+30),'同算法 ARM/FPGA 用于板上加速比较；不同算法和不同电脑的耗时分别展示。',font=small,fill='#42556b')
    draw.text((margin,y+55),'FFA 原始输出不附加增强；你的自适应暗通道含手工增强，未运行训练网络。各项预热一次、平均三次。',font=small,fill='#42556b')
    sheet.save(run/'software_comparison.png')

def open_window(parent,source,ground_truth=None,fpga_run=None,**parameters):
    import tkinter as tk
    from tkinter import ttk,messagebox,filedialog
    window=tk.Toplevel(parent);window.title('纯软件与神经网络对比 · CPU / ARM / FPGA')
    window.geometry('1240x860');window.minsize(1050,740)
    q=queue.Queue();state={'busy':False,'run':None,'data':None,'images':[None,None,None],'photos':[]}
    top=ttk.Frame(window,padding=12);top.pack(fill='x')
    ttk.Label(top,text='三种算法对比 · 同尺寸实测',font=('Microsoft YaHei UI',17,'bold')).pack(side='left')
    choice=tk.StringVar(value=NETWORKS['ots'])
    combo=ttk.Combobox(top,textvariable=choice,values=list(NETWORKS.values()),state='readonly',width=22);combo.pack(side='left',padx=12)
    from detail_view import show_detail
    ttk.Button(top,text='同步细节',command=lambda:show_detail(window,state['images'],['当前算法参考','自适应暗通道','FFA-Net'])).pack(side='right')
    info=tk.StringVar(value='采用当前 FPGA 实测记录的尺寸与参数；本窗口只在电脑运行。' if fpga_run else '采用当前输入尺寸与参数；不需要初始化板卡。')
    ttk.Label(window,textvariable=info,padding=(12,0)).pack(fill='x')
    status=tk.StringVar(value='神经网络在电脑 CPU 运行；没有额外缩小图片，也不宣称 FPGA 已运行网络。')
    holder=ttk.Frame(window,padding=12);holder.pack(fill='both',expand=True);holder.rowconfigure(1,weight=1)
    canvases=[]
    def full(index):
        image=state['images'][index]
        if image is None:return
        view=tk.Toplevel(window);view.title(f'软件对比原生像素 · {image.width}×{image.height}');view.geometry('1050x740')
        canvas=tk.Canvas(view);xs=ttk.Scrollbar(view,orient='horizontal',command=canvas.xview);ys=ttk.Scrollbar(view,orient='vertical',command=canvas.yview)
        xs.pack(side='bottom',fill='x');ys.pack(side='right',fill='y');canvas.pack(fill='both',expand=True)
        canvas.configure(xscrollcommand=xs.set,yscrollcommand=ys.set,scrollregion=(0,0,*image.size))
        view.photo=ImageTk.PhotoImage(image);canvas.create_image(0,0,anchor='nw',image=view.photo)
    def render(event=None):
        state['photos']=[]
        for canvas,image in zip(canvases,state['images']):
            canvas.delete('all');w=max(1,canvas.winfo_width()-12);h=max(1,canvas.winfo_height()-12)
            if image is None:canvas.create_text(w//2,h//2,text='运行后显示实际结果',fill='white');continue
            photo=ImageTk.PhotoImage(ImageOps.contain(image,(w,h),Image.Resampling.LANCZOS));state['photos'].append(photo)
            canvas.create_image((w+12)//2,(h+12)//2,image=photo)
    for index,title in enumerate(['当前 FPGA 算法 · 电脑软件参考','你的自适应暗通道 · 非神经网络','FFA-Net · 电脑 CPU']):
        holder.columnconfigure(index,weight=1,uniform='software')
        header=ttk.Frame(holder);header.grid(row=0,column=index,sticky='ew',padx=5)
        ttk.Label(header,text=title,font=('Microsoft YaHei UI',11,'bold')).pack(side='left')
        ttk.Button(header,text='100%',width=5,command=lambda i=index:full(i)).pack(side='right')
        canvas=tk.Canvas(holder,bg='#172638',height=300,highlightthickness=0);canvas.grid(row=1,column=index,sticky='nsew',padx=5,pady=5)
        canvas.bind('<Configure>',render);canvases.append(canvas)
    table=ttk.Treeview(window,columns=('method','psnr','ssim','mae','ms'),show='headings',height=8)
    for key,label,width in [('method','执行位置与算法',360),('psnr','PSNR ↑ (dB)',120),('ssim','SSIM ↑',120),('mae','MAE ↓',120),('ms','计算耗时 (ms)',150)]:
        table.heading(key,text=label);table.column(key,width=width,anchor='center')
    table.pack(fill='x',padx=12)
    notes=tk.StringVar(value='无雾图、有雾图与 FPGA 在主页展示；本窗口对比三种软件算法。同算法软件应与 FPGA 一致；你的自适应暗通道没有运行训练网络。')
    ttk.Label(window,textvariable=notes,wraplength=1180,padding=12).pack(fill='x')
    bottom=ttk.Frame(window,padding=12);bottom.pack(fill='x')
    ttk.Label(bottom,textvariable=status,wraplength=950).pack(side='left')
    def show(run,data,history=False):
        state.update(run=Path(run),data=data)
        state['images']=[open_rgb(Path(run)/file) if (Path(run)/file).exists() else None for file in ['pc_dcp.png','adaptive_output.png','ffa_output.png']]
        for row in table.get_children():table.delete(row)
        for method in data['methods'].values():
            s=method.get('scores') or {}
            table.insert('','end',values=(method['label'],display_number(s.get('psnr_db'),2),display_number(s.get('ssim'),4),display_number(s.get('mae'),3),display_number(method.get('ms'),2)))
        message=f'实际尺寸 {data["processing_size"][0]}×{data["processing_size"][1]}。'
        if 'pc_fpga_channel_mismatches' in data:
            message+=f'电脑同算法与 FPGA 相差 {data["pc_fpga_channel_mismatches"]} 个通道值。'
            if data.get('board_speedup') is not None:message+=f'板上 ARM/FPGA 加速比 {data["board_speedup"]:.2f}×。'
        if data.get('quality_reason'):message+=data['quality_reason']
        message+=' CPU 预热一次、平均三次；模型加载和文件传输另计。不同算法的速度差不作为 FPGA 加速比。'
        notes.set(message);status.set(('历史软件实测；' if history else '软件实测完成；')+str(run));render()
    def open_saved(path):
        if state['busy']:raise RuntimeError('正在运行，请等待完成再打开历史记录。')
        run=Path(path)
        if run.is_file():run=run.parent
        data=json.loads((run/'software_result.json').read_text(encoding='utf-8'))
        if 'methods' not in data or not (run/'pc_dcp.png').exists():raise ValueError('不是有效的软件实测目录。')
        show(run,data,history=True)
        state['history_only']=True;button.configure(state='disabled')
        info.set('历史软件实测，只读查看。需要现场复测时，请从主页重新打开软件对比。')

    def browse_saved():
        name=filedialog.askopenfilename(parent=window,title='打开 software_result.json',initialdir=str(ROOT/'outputs'),filetypes=[('软件实测','software_result.json'),('JSON','*.json')])
        if not name:return
        try:open_saved(name)
        except Exception as error:messagebox.showerror('历史记录读取失败',str(error),parent=window)

    def start():
        if state['busy'] or state.get('history_only'):return
        state['busy']=True;button.configure(state='disabled');combo.configure(state='disabled');history_button.configure(state='disabled')
        task=next(k for k,v in NETWORKS.items() if v==choice.get())
        def work():
            try:q.put(('done',run_comparison(source,ground_truth,task=task,fpga_run=fpga_run,progress=lambda text:q.put(('status',text)),**parameters)))
            except Exception as error:q.put(('error',str(error)))
        threading.Thread(target=work,daemon=True).start()
    button=ttk.Button(top,text='运行软件对比',command=start);button.pack(side='left')
    history_button=ttk.Button(top,text='打开历史软件实测',command=browse_saved);history_button.pack(side='left',padx=8)
    def open_outputs():
        import os
        if state['run']:os.startfile(state['run'])
    ttk.Button(bottom,text='对比图与结果目录',command=open_outputs).pack(side='right')
    def drain():
        try:
            while True:
                kind,payload=q.get_nowait()
                if kind=='status':status.set(payload)
                else:
                    state['busy']=False;button.configure(state='normal');combo.configure(state='readonly');history_button.configure(state='normal')
                    if kind=='done':show(*payload)
                    else:status.set('软件测量失败');messagebox.showerror('软件对比失败',payload,parent=window)
        except queue.Empty:pass
        if window.winfo_exists():window.after(150,drain)
    window.after(150,drain)
    window.open_saved=open_saved
    window.show_saved=open_saved
    return window

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--record');parser.add_argument('--image');parser.add_argument('--ground-truth');parser.add_argument('--network',choices=NETWORKS,default='ots');parser.add_argument('--view')
    args=parser.parse_args()
    if args.view:
        import tkinter as tk
        root=tk.Tk();root.withdraw();data=json.loads((Path(args.view)/'software_result.json').read_text(encoding='utf-8'))
        win=open_window(root,data['source'],data.get('ground_truth'),fpga_run=Path(data['linked_fpga_result']).parent if data.get('linked_fpga_result') else None)
        win.show_saved(args.view);win.protocol('WM_DELETE_WINDOW',root.destroy);root.mainloop()
    else:
        record=json.loads(Path(args.record).read_text(encoding='utf-8')) if args.record else None
        run,data=run_comparison(args.image or (record['source'] if record else ''),args.ground_truth or (record.get('ground_truth') if record else None),task=args.network,fpga_run=Path(args.record).parent if args.record else None,progress=lambda message:print(message,flush=True))
        print(json.dumps({'run':str(run),'methods':data['methods']},ensure_ascii=False,indent=2),flush=True)
