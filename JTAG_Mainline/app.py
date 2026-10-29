import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from PIL import Image,ImageTk,ImageOps
from pathlib import Path
import threading,queue,json,os,time
import backend
from evaluation import DATA_ROOT,auto_ground_truth,demo_pair,open_rgb,attach_quality,export_comparison,display_number
from project_info import optimization_text,hardware_details
from model import load_image

ROOT=Path(__file__).parent
PICTURES=[('图片','*.jpg *.jpeg *.JPG *.JPEG *.png *.bmp *.tif *.tiff'),('所有文件','*.*')]

class Demo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('去雾三图评测 · V3双核版 · 板上神经推理')
        self.geometry(f'{min(1300,self.winfo_screenwidth()-50)}x{min(820,self.winfo_screenheight()-100)}')
        self.minsize(1080,700)
        self.configure(bg='#eef2f6')
        self.messages=queue.Queue();self.ready=False;self.busy=False
        self.source=None;self.gt_source=None;self.images=[None,None,None]
        self.photos=[];self.last_run=None;self.last_result=None;self.operation_buttons=[]
        style=ttk.Style(self);style.theme_use('clam')
        style.configure('TFrame',background='#eef2f6')
        style.configure('TLabel',background='#eef2f6',font=('Microsoft YaHei UI',10))
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',21,'bold'),foreground='#142b45')
        style.configure('TButton',font=('Microsoft YaHei UI',10),padding=(10,6))
        style.configure('Treeview',font=('Microsoft YaHei UI',10),rowheight=24)
        style.configure('Treeview.Heading',font=('Microsoft YaHei UI',10,'bold'))
        top=ttk.Frame(self,padding=(18,10));top.pack(fill='x')
        ttk.Label(top,text='高清彩色去雾 · 三图评测',style='Title.TLabel').pack(side='left')
        ttk.Label(top,text='Z15 / XC7Z015  ·  100 MHz  ·  独立运行').pack(side='right')
        controls=ttk.Frame(self,padding=(18,0));controls.pack(fill='x')
        self.init_button=self.button(controls,'① 初始化板卡',self.initialize)
        self.select_button=self.button(controls,'② 选择有雾图',self.select_hazy)
        self.gt_button=self.button(controls,'选择无雾参考',self.select_gt)
        self.run_button=self.button(controls,'③ 去雾处理与评测',self.run)
        self.batch_button=self.button(controls,'批量去雾',lambda:__import__('batch_ui').open_batch_window(self))
        ttk.Label(controls,text='最长边').pack(side='left',padx=(8,3))
        self.edge=tk.StringVar(value='1280')
        self.edge_box=ttk.Combobox(controls,textvariable=self.edge,values=['640','1280','1920'],width=5,state='readonly')
        self.edge_box.pack(side='left')
        settings=ttk.Frame(self,padding=(18,5));settings.pack(fill='x')
        ttk.Label(settings,text='去雾强度').pack(side='left')
        self.strength=tk.IntVar(value=160)
        self.strength_scale=tk.Scale(settings,from_=0,to=243,orient='horizontal',variable=self.strength,length=145,bg='#eef2f6',highlightthickness=0)
        self.strength_scale.pack(side='left')
        ttk.Label(settings,text='亮度补偿').pack(side='left',padx=(10,0))
        self.lift=tk.IntVar(value=64)
        self.lift_scale=tk.Scale(settings,from_=0,to=128,orient='horizontal',variable=self.lift,length=145,bg='#eef2f6',highlightthickness=0)
        self.lift_scale.pack(side='left')
        ttk.Label(settings,text='透射率下限 /255').pack(side='left',padx=(10,4))
        self.tmin=tk.IntVar(value=96)
        self.tmin_box=ttk.Spinbox(settings,from_=32,to=255,textvariable=self.tmin,width=5)
        self.tmin_box.pack(side='left')
        self.run_cpu=tk.BooleanVar(value=False)
        self.cpu_check=ttk.Checkbutton(settings,text='ARM完整参考',variable=self.run_cpu)
        self.cpu_check.pack(side='left',padx=(12,4))
        ttk.Label(settings,text='重复次数').pack(side='left')
        self.repeats=tk.StringVar(value='1')
        self.repeat_box=ttk.Combobox(settings,textvariable=self.repeats,values=['1','3','5','10'],width=3,state='readonly')
        self.repeat_box.pack(side='left',padx=4)
        self.info=tk.StringVar(value='选择有雾图片后自动匹配数据集无雾参考；也可手动选择。')
        ttk.Label(self,textvariable=self.info,padding=(18,0)).pack(fill='x')
        holder=ttk.Frame(self,padding=(18,8));holder.pack(fill='both',expand=True)
        holder.rowconfigure(1,weight=1)
        self.canvases=[];self.captions=[]
        for column,title in enumerate(['无雾参考图（GT）','有雾输入图','去雾结果（执行方式见下方）']):
            holder.columnconfigure(column,weight=1,uniform='images')
            header=ttk.Frame(holder);header.grid(row=0,column=column,sticky='ew',padx=5,pady=(0,5))
            ttk.Label(header,text=title,font=('Microsoft YaHei UI',12,'bold')).pack(side='left')
            ttk.Button(header,text='100%',width=5,command=lambda col=column:self.full_view(col)).pack(side='right')
            if column==0:
                from detail_view import show_detail
                ttk.Button(header,text='三图同步放大',command=lambda:show_detail(self,self.images,['无雾参考','有雾输入','去雾结果'])).pack(side='right',padx=4)
            canvas=tk.Canvas(holder,bg='#172638',highlightthickness=0,height=180)
            canvas.grid(row=1,column=column,sticky='nsew',padx=5)
            canvas.bind('<Configure>',lambda _:self.render())
            canvas.bind('<Double-1>',lambda event,col=column:self.full_view(col))
            self.canvases.append(canvas)
            caption=tk.StringVar(value='尚未选择')
            self.captions.append(caption)
            ttk.Label(holder,textvariable=caption,wraplength=360).grid(row=2,column=column,sticky='nw',padx=5,pady=(4,0))
        notebook=ttk.Notebook(self);notebook.pack(fill='x',padx=18,pady=(0,5))
        metrics_tab=ttk.Frame(notebook,padding=8)
        notebook.add(metrics_tab,text='质量与性能指标')
        metrics_tab.columnconfigure(0,weight=1);metrics_tab.columnconfigure(1,weight=1)
        left=ttk.Frame(metrics_tab);left.grid(row=0,column=0,sticky='nsew',padx=(0,12))
        ttk.Label(left,text='相对无雾参考的画质（RGB）',font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w')
        self.quality_table=ttk.Treeview(left,columns=('name','hazy','fpga','delta'),show='headings',height=3)
        for name,label,width in [('name','指标',105),('hazy','有雾图',95),('fpga','去雾结果',95),('delta','输出−输入',95)]:
            self.quality_table.heading(name,text=label);self.quality_table.column(name,width=width,anchor='center',stretch=True)
        self.quality_table.pack(fill='x',pady=4)
        self.quality_note=tk.StringVar(value='有配对无雾图后才计算；PSNR/SSIM 越高、MAE 越低通常越好。')
        ttk.Label(left,textvariable=self.quality_note,wraplength=560).pack(anchor='w')
        right=ttk.Frame(metrics_tab);right.grid(row=0,column=1,sticky='nsew')
        ttk.Label(right,text='板上计算与数据传输',font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w')
        self.perf_table=ttk.Treeview(right,columns=('name','value','name2','value2'),show='headings',height=6)
        for name,label,width in [('name','项目',110),('value','实测',105),('name2','项目',110),('value2','实测',115)]:
            self.perf_table.heading(name,text=label);self.perf_table.column(name,width=width,anchor='center',stretch=True)
        self.perf_table.pack(fill='x',pady=4)
        params_tab=ttk.Frame(notebook,padding=8);notebook.add(params_tab,text='运行参数与口径')
        self.param_text=tk.Text(params_tab,height=8,wrap='word',font=('Microsoft YaHei UI',10),bg='white',relief='flat')
        self.param_text.pack(fill='x')
        optimization_tab=ttk.Frame(notebook,padding=8);notebook.add(optimization_tab,text='FPGA 做了哪些优化')
        optimization=tk.Text(optimization_tab,height=8,wrap='word',font=('Microsoft YaHei UI',10),bg='white',relief='flat')
        scrollbar=ttk.Scrollbar(optimization_tab,orient='vertical',command=optimization.yview)
        scrollbar.pack(side='right',fill='y');optimization.pack(fill='both',expand=True)
        optimization.configure(yscrollcommand=scrollbar.set)
        optimization.insert('1.0',optimization_text());optimization.configure(state='disabled')
        from selection_info import decision_text,explanation_window
        selection_tab=ttk.Frame(notebook,padding=8);notebook.add(selection_tab,text='研究目标与算法进展')
        selection_actions=ttk.Frame(selection_tab);selection_actions.pack(fill='x')
        ttk.Button(selection_actions,text='查看目标与候选结果',command=lambda:explanation_window(self)).pack(side='left',padx=(0,8))
        ttk.Button(selection_actions,text='研究阶段讲解稿',command=lambda:explanation_window(self,narration=True)).pack(side='left')
        selection_text=tk.Text(selection_tab,height=6,wrap='word',font=('Microsoft YaHei UI',10),bg='white',relief='flat')
        selection_bar=ttk.Scrollbar(selection_tab,command=selection_text.yview)
        selection_bar.pack(side='right',fill='y');selection_text.pack(fill='both',expand=True)
        selection_text.configure(yscrollcommand=selection_bar.set)
        selection_text.insert('1.0',decision_text());selection_text.configure(state='disabled')
        bottom=ttk.Frame(self,padding=(18,5));bottom.pack(fill='x')
        self.preset=tk.StringVar(value='板上优化V3（FPGA神经推理）')
        self.preset_box=ttk.Combobox(bottom,textvariable=self.preset,values=['板上优化V3（FPGA神经推理）','自适应增强（V2）','旧版柔和','旧版增强'],width=32,state='readonly')
        self.preset_box.pack(side='left',padx=(0,8))
        def apply_preset(event=None):
            strength,tmin,lift=(230,64,0) if self.preset.get()=='旧版增强' else (160,96,64)
            self.strength.set(strength);self.tmin.set(tmin);self.lift.set(lift);self.update_buttons()
        self.preset_box.bind('<<ComboboxSelected>>',apply_preset)
        self.status=tk.StringVar(value='尚未初始化板卡')
        ttk.Label(bottom,textvariable=self.status).pack(side='left')
        self.bar=ttk.Progressbar(bottom,mode='indeterminate',length=100);self.bar.pack(side='right')
        self.export_button=ttk.Button(bottom,text='导出三图与指标',command=self.export)
        self.export_button.pack(side='right',padx=8)
        self.software_button=ttk.Button(bottom,text='电脑 / FPGA同算法对比',command=self.software_comparison)
        self.software_button.pack(side='right',padx=8)
        self.operation_buttons.append(self.software_button)
        ttk.Button(bottom,text='结果目录',command=self.open_results).pack(side='right')
        self.set_quality(None);self.set_performance(None);self.set_parameters(None)
        self.update_buttons()
        self.after(100,self.drain)

    def button(self,parent,text,command):
        button=ttk.Button(parent,text=text,command=command)
        button.pack(side='left',padx=(0,6));self.operation_buttons.append(button)
        return button

    def worker(self,action):
        if self.busy:return
        self.busy=True;self.bar.start(12);self.update_buttons()
        def task():
            try:self.messages.put(('done',action()))
            except Exception as error:self.messages.put(('error',str(error)))
        threading.Thread(target=task,daemon=True).start()

    def progress(self,text):self.messages.put(('status',text))
    def initialize(self):
        def action():backend.initialize(self.progress);return ('initialized',)
        self.worker(action)

    def select_hazy(self):
        name=filedialog.askopenfilename(title='选择有雾图片',initialdir=str(DATA_ROOT/'natural_hazy'),filetypes=PICTURES)
        if name:self.set_source(name)

    def set_source(self,name):
        self.source=str(Path(name).resolve())
        gt=auto_ground_truth(name);self.gt_source=str(gt) if gt else None
        self.images=[open_rgb(gt) if gt else None,open_rgb(name),None]
        self.last_run=None;self.last_result=None
        self.info.set(f'{Path(name).name}  |  原始尺寸 {self.images[1].width}×{self.images[1].height} RGB  |  '+('已自动匹配数据集无雾图' if gt else '未找到无雾参考，可手动选择'))
        self.captions[0].set(Path(gt).name if gt else '未找到配对 GT，请选择无雾参考图')
        self.captions[1].set(Path(name).name);self.captions[2].set('等待 FPGA 实际处理与回读')
        self.set_quality(None);self.set_performance(None);self.set_parameters(None)
        self.render();self.update_buttons()

    def select_gt(self):
        if not self.source:
            messagebox.showinfo('先选择输入','请先选择有雾图片，再指定同一场景、同一视角且像素对齐的无雾参考图。');return
        initial=str(Path(self.gt_source).parent) if self.gt_source else str(DATA_ROOT/'natural_gt')
        name=filedialog.askopenfilename(title='选择同场景、像素对齐的无雾参考图',initialdir=initial,filetypes=PICTURES)
        if not name:return
        self.gt_source=str(Path(name).resolve());self.images[0]=open_rgb(name)
        self.captions[0].set(Path(name).name+'（手动选择）')
        self.render()
        if self.last_result and self.images[2] is not None:
            result=dict(self.last_result);run=self.last_run;hazy=self.images[1];output=self.images[2];gt=self.gt_source
            def action():
                clear=attach_quality(result,run,hazy,output,gt)
                (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
                export_comparison(run,result,hazy,output,clear)
                return ('quality',result,clear)
            self.worker(action)
        else:
            self.quality_note.set('已选择参考图；请确认同场景且像素对齐，完成 FPGA 处理后计算质量指标。')

    def load_example(self):
        hazy,gt=demo_pair()
        if not hazy:messagebox.showerror('样例不存在','成对样例目录不存在，请手动选择图片。');return
        self.set_source(hazy)

    def run(self):
        if not self.source:return
        try:
            tmin=int(self.tmin.get());repeats=int(self.repeats.get())
            if not 32<=tmin<=255:raise ValueError('透射率下限必须在 32..255 之间。')
        except (ValueError,tk.TclError) as error:
            messagebox.showerror('参数错误',str(error));return
        args=dict(source=self.source,max_edge=int(self.edge.get()),strength=self.strength.get(),t_min=tmin,lift=self.lift.get(),
                  repeats=repeats,run_cpu=self.run_cpu.get(),ground_truth=self.gt_source,progress=self.progress,algorithm="pl_v3" if self.preset.get().startswith("板上优化V3") else ("fused" if self.preset.get()=="自适应增强（V2）" else "baseline"))
        self.images[2]=None;self.captions[2].set('正在运行，等待新的板上输出');self.render()
        self.worker(lambda:('result',backend.run_image(**args)))

    def load_history(self):
        name=filedialog.askopenfilename(title='打开历史实测 result.json',initialdir=str(ROOT/'outputs'),filetypes=[('实测记录','*.json')])
        if name:self.load_result(name)

    def load_result(self,name):
        run=Path(name).parent
        result=json.loads(Path(name).read_text(encoding='utf-8'))
        if result.get('batch_compacted'):
            raise ValueError('这是精简批处理记录，请到批处理输出文件夹的images中查看结果；records与summary.csv保存指标。')
        if not (run/'fpga_output.png').is_file() or not (run/'input.png').is_file():
            raise ValueError('记录目录缺少实际 FPGA 输出或输入图片。')
        hazy=open_rgb(run/'input.png');output=open_rgb(run/('hybrid_output.png' if result.get('algorithm')=='fused-neural-v3-hybrid' else 'fpga_output.png'))
        self.source=result['source'];self.gt_source=result.get('ground_truth')
        if not self.gt_source:
            gt=auto_ground_truth(self.source);self.gt_source=str(gt) if gt else None
        clear=open_rgb(run/'ground_truth.png') if result.get('quality',{}).get('available') and (run/'ground_truth.png').exists() else None
        if clear is None and self.gt_source and Path(self.source).exists():
            clear=attach_quality(result,run,hazy,output,self.gt_source)
            (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        self.show_result(hazy,output,result,run,clear,history=True)

    def show_result(self,hazy,output,result,run,clear=None,history=False):
        if clear is None and result.get('quality',{}).get('available') and (run/'ground_truth.png').exists():
            clear=open_rgb(run/'ground_truth.png')
        self.images=[clear,hazy,output];self.last_run=run;self.last_result=result
        self.source=result['source'];self.gt_source=result.get('ground_truth')
        self.captions[0].set(Path(self.gt_source).name if self.gt_source else '缺少配对无雾参考，质量指标留空')
        self.captions[1].set(Path(self.source).name)
        self.captions[2].set(('优化V3 · FPGA基础＋电脑神经校正' if result.get('algorithm')=='fused-neural-v3-hybrid' else ('增强V2 · ' if result.get('algorithm')=='fused-guided-v2' else '旧版暗通道 · ')+'板上 DDR 实际回读')+(' · 历史记录' if history else ' · 本次运行'))
        if result.get('algorithm')=='fused-neural-v3-pl':self.captions[2].set(('板上V3 · FPGA统计＋神经推理' if result.get('pl_statistics') else '板上V3 · FPGA神经推理＋ARM统计')+(' · 历史记录' if history else ' · 本次运行'))
        if result.get('dual_arm'):
            self.captions[2].set('板上V3 · FPGA神经推理＋'+('ARM双核统计' if result.get('dual_active') else 'ARM单核统计（尺寸回退）')+(' · 历史记录' if history else ' · 本次运行'))
        self.info.set(f'{Path(self.source).name}  |  实际处理 {hazy.width}×{hazy.height} RGB  |  '+('历史实测，非本次运行' if history else '本次处理完成（执行位置见结果说明）'))
        self.set_quality(result.get('quality'));self.set_performance(result);self.set_parameters(result)
        self.status.set('已载入历史实测（非本次运行）' if history else '完成：板上基础去雾已验证；最终执行方式见图片说明')
        self.render();self.update_buttons()

    def set_quality(self,quality):
        for row in self.quality_table.get_children():self.quality_table.delete(row)
        fields=[('PSNR ↑ (dB)','psnr_db',2),('SSIM ↑','ssim',4),('MAE ↓','mae',3)]
        available=quality and quality.get('available')
        for label,key,precision in fields:
            before=quality['input'].get(key) if available else None
            after=quality.get('output',quality.get('fpga',{})).get(key) if available else None
            delta=quality['delta'].get(key) if available else None
            difference=f'{delta:+.{precision}f}' if delta is not None else '—'
            self.quality_table.insert('','end',values=(label,display_number(before,precision),display_number(after,precision),difference))
        if available:self.quality_note.set(f'相同处理尺寸 {quality["evaluation_size"][0]}×{quality["evaluation_size"][1]}；PSNR/SSIM 越高、MAE 越低通常越好；无雾图只用于评价。')
        else:self.quality_note.set(quality.get('reason') if quality else '请选择配对无雾图；没有 GT 时不计算 PSNR/SSIM/MAE。')

    def set_performance(self,result):
        for row in self.perf_table.get_children():self.perf_table.delete(row)
        def value(key,precision=2,suffix=''):
            return display_number(result.get(key),precision)+suffix if result and result.get(key) is not None else '—'
        fps=1000/result['fpga_ms'] if result and result.get('fpga_ms',0)>0 else None
        mismatch=str(result['host_channel_mismatches'])+' 通道' if result else '—'
        rows=[('ARM (O2)',value('arm_ms',2,' ms'),'FPGA 板上总计',value('fpga_ms',2,' ms')),
              ('开始至完成',value('fpga_start_done_ms',2,' ms'),'板上加速比',value('speedup',2,'×')),
              ('JTAG 上传',value('upload_ms',0,' ms'),'JTAG 回读',value('readback_ms',0,' ms')),
              ('全流程含传输',value('wall_seconds',2,' s'),'等效板上 FPS',display_number(fps,2)),
              ('算法参考校验',mismatch,'持续视频帧率','尚未实测')]
        if result and result.get('algorithm')=='fused-neural-v3-hybrid':
            rows[0]=('ARM基础DCP',value('arm_ms',2,' ms'),'FPGA基础DCP',value('fpga_ms',2,' ms'))
            rows[1]=('DCP核计算',value('fpga_start_done_ms',2,' ms'),'仅板上DCP比',value('speedup',2,'×'))
            rows[3]=('全流程含传输',value('wall_seconds',2,' s'),'算法计算合计',value('pipeline_compute_ms',2,' ms'))
            rows.append(('电脑辅助＋校正',value('host_post_ms',2,' ms'),'神经模型准备',value('host_model_load_ms',2,' ms')))
        if result and result.get('algorithm')=='fused-neural-v3-pl':
            rows=[('ARM完整参考',value('arm_ms',2,' ms'),'板卡算法总计',value('fpga_ms',2,' ms')),
                  ('FPGA暗通道',value('dcp_ms',2,' ms'),('统计＋神经阶段' if result.get('pl_statistics') else 'FPGA神经阶段'),value('neural_ms',2,' ms')),
                  (('ARM统计（旁路）' if result.get('pl_statistics') else '板上ARM统计'),value('arm_feature_ms',2,' ms'),'FPGA神经核',value('neural_kernel_ms',2,' ms')),
                  ('JTAG上传',value('upload_ms',0,' ms'),'JTAG回读',value('readback_ms',0,' ms')),
                  ('算法含帧参数',value('pipeline_compute_ms',2,' ms'),'完整参考校验',mismatch),
                  ('含传输及校验',value('wall_seconds',2,' s'),'ARM/板卡比',value('speedup',2,'×'))]
        if result and result.get('dual_arm'):
            rows.append(('CPU1启动等待',value('dual_startup_ms',2,' ms'),'双核启动握手',value('dual_bootstrap_ms',0,' ms')))
        for row in rows:self.perf_table.insert('','end',values=row)

    def set_parameters(self,result):
        if result:
            w,h=result['processing_size'];air=', '.join(str(x) for x in result['air'])
            text=(f'实际输入/输出：{w}×{h} RGB888（DDR 按 32 bit/像素打包）；15×15 局部最小值；PL 时钟 {result.get("fclk_mhz",100)} MHz。\n'
                  f'空气光 A：({air})，电脑端估计。去雾强度：{result["strength"]}/256 = {result["strength"]/256:.4f}。\n'
                  f'透射率下限：{result["t_min"]}/255 = {result["t_min"]/255:.4f}；亮度补偿：{result["lift"]}（0..128，多项式中间调提升）。\n'
                  f'预热 1 次，平均计时 {result["repeats"]} 次；ARM 参考的编译选项见本次记录；缓存开启。最大通道计算差异：{result["max_channel_error"]}。\n'
                  'FPGA 板上总计含缓存维护、寄存器控制、DDR 访问和完成等待；开始至完成不含前后缓存维护。加速比与当前同算法 ARM 参考比较。\n'
                  '全流程含 PC 输入准备、JTAG 上传/回读和算法校验；质量评价与对比图导出另在电脑端执行。等效板上 FPS 不代表持续视频。\n'
                  f'无雾参考：{result.get("ground_truth") or "未提供"}\n'
                  '无雾参考只参与显示与质量评价，不输入 FPGA，不用于自动调参。')
        else:
            text=('运行后显示实际分辨率、RGB 空气光、去雾强度、透射率下限、亮度补偿、重复次数和校验结果。\n'
                  '去雾强度越高，通常去雾越强，也更容易压暗暗部；透射率下限限制暗部放大；亮度补偿抬升中间调。\n'
                  'PSNR/SSIM 对应的是无雾参考；“算法参考校验”对应的是同一固定点算法的独立实现，两者含义不同。\n'
                  '缺少同场景、同视角、像素对齐的无雾图时，质量指标留空。')
        if result and result.get('algorithm')=='fused-guided-v2':
            text=('算法：双分支暗通道＋快速导向滤波＋范围约束；不是神经网络。\n'
                  f'强分支空气光：{result["strong_air"]}；强度230/256、下限38/255；融合权重{result["blend_q8"]}/256。\n'
                  '电脑只估计两组空气光与帧级融合权重；暗通道、导向滤波、双线性系数恢复、RGB恢复与融合均由FPGA执行。\n'
                  '下方强度/亮度参数描述柔和分支；自适应增强模式下不使用手动滑块。\n'+text)
        if result and result.get('algorithm')=='fused-neural-v3-hybrid':
            from hybrid_runtime_v3 import explain
            text=explain()+'\n\n'+text
        if result and result.get('algorithm')=='fused-neural-v3-pl':
            from pl_runtime_info import explain
            text=explain(result)+'\n\n'+text
        self.param_text.configure(state='normal');self.param_text.delete('1.0','end');self.param_text.insert('1.0',text);self.param_text.configure(state='disabled')

    def drain(self):
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=='status':self.status.set(payload)
                elif kind=='error':
                    self.busy=False;self.bar.stop();self.status.set('操作失败；请查看错误和结果日志')
                    messagebox.showerror('演示操作失败',payload);self.update_buttons()
                elif kind=='done':
                    self.busy=False;self.bar.stop()
                    if payload[0]=='initialized':
                        self.ready=True;self.status.set('硬件已就绪：100 MHz 彩色去雾模块')
                    elif payload[0]=='result':
                        image,output,result,run=payload[1];self.show_result(image,output,result,run)
                    elif payload[0]=='quality':
                        result,clear=payload[1:];self.last_result=result
                        if clear is not None:self.images[0]=clear
                        self.set_quality(result.get('quality'));self.set_parameters(result);self.render()
                        self.status.set('无雾参考与质量指标已更新；未重新运行 FPGA')
                    elif payload[0]=='export':
                        self.status.set('三图、指标和参数已导出到：'+str(payload[1]))
                    self.update_buttons()
        except queue.Empty:pass
        self.after(100,self.drain)

    def update_buttons(self):
        for button in self.operation_buttons:button.configure(state='disabled' if self.busy else 'normal')
        self.run_button.configure(state='normal' if self.ready and self.source and not self.busy else 'disabled')
        self.export_button.configure(state='normal' if self.last_result and not self.busy else 'disabled')
        self.software_button.configure(state='normal' if self.last_result and self.images[2] is not None and not self.busy else 'disabled')
        self.edge_box.configure(state='disabled' if self.busy else 'readonly')
        self.repeat_box.configure(state='disabled' if self.busy else 'readonly')
        self.preset_box.configure(state='disabled' if self.busy else 'readonly')
        self.cpu_check.configure(state='disabled' if self.busy else 'normal')
        for widget in [self.strength_scale,self.lift_scale,self.tmin_box]:
            widget.configure(state='disabled' if self.busy or self.preset.get() in ['自适应增强（V2）','板上优化V3（FPGA神经推理）'] else 'normal')

    def render(self):
        self.photos=[]
        for index,(canvas,image) in enumerate(zip(self.canvases,self.images)):
            canvas.delete('all');w=max(20,canvas.winfo_width());h=max(20,canvas.winfo_height())
            if image is None:
                text=['请选择配对无雾参考','请选择有雾输入','等待 FPGA 实际处理'][index]
                canvas.create_text(w//2,h//2,text=text,fill='#b8c9dc',font=('Microsoft YaHei UI',13));continue
            fitted=ImageOps.contain(image,(max(1,w-12),max(1,h-12)),Image.Resampling.LANCZOS)
            photo=ImageTk.PhotoImage(fitted);self.photos.append(photo);canvas.create_image(w//2,h//2,image=photo)

    def full_view(self,index):
        if index==0 and self.gt_source and Path(self.gt_source).exists():image=open_rgb(self.gt_source)
        elif index==1 and self.source and Path(self.source).exists():image=open_rgb(self.source)
        else:image=self.images[index]
        if image is None:return
        title=['无雾参考','有雾原始图片','去雾结果'][index]
        window=tk.Toplevel(self);window.title(f'{title} 100% · {image.width}×{image.height}');window.geometry('1100x760')
        canvas=tk.Canvas(window);xs=ttk.Scrollbar(window,orient='horizontal',command=canvas.xview);ys=ttk.Scrollbar(window,orient='vertical',command=canvas.yview)
        xs.pack(side='bottom',fill='x');ys.pack(side='right',fill='y');canvas.pack(fill='both',expand=True)
        canvas.configure(xscrollcommand=xs.set,yscrollcommand=ys.set,scrollregion=(0,0,image.width,image.height))
        window.photo=ImageTk.PhotoImage(image);canvas.create_image(0,0,anchor='nw',image=window.photo)

    def export(self):
        if not self.last_result:return
        result=dict(self.last_result);run=self.last_run;clear,hazy,output=self.images
        def action():
            path=export_comparison(run,result,hazy,output,clear if result.get('quality',{}).get('available') else None)
            (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            return ('export',path)
        self.worker(action)

    def software_comparison(self):
        if not self.last_result or self.images[2] is None:
            messagebox.showinfo('先完成FPGA处理','请先完成当前图片的FPGA处理，再用相同输入、尺寸和参数对比电脑算法。');return
        from same_algorithm_compare import open_window
        open_window(self,self.last_run,self.gt_source)

    def open_results(self):
        if self.last_run:os.startfile(self.last_run)
        elif (ROOT/'outputs').exists():os.startfile(ROOT/'outputs')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--result')
    arguments=parser.parse_args()
    app=Demo()
    if arguments.result:
        def restore():
            try:app.load_result(arguments.result)
            except Exception as error:messagebox.showerror('历史实测加载失败',str(error))
        app.after(250,restore)
    else:
        app.info.set('请选择有雾图片；有配对无雾参考时显示质量指标，也可直接批量处理。')
    app.mainloop()
