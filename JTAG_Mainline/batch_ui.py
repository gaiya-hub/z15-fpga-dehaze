"""Folder processing window with progress, stop-after-current, and verified resume."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import queue,threading,time,os

def open_batch_window(parent):
    if parent.busy:return
    existing=getattr(parent,'batch_window',None)
    if existing is not None and existing.winfo_exists():existing.lift();return
    from batch_dehaze import ROOT,run_batch
    from portable_settings import ffa_root
    win=tk.Toplevel(parent);parent.batch_window=win;win.title('文件夹批量去雾 · 板上神经V3 / V2');win.geometry('880x700')
    frame=ttk.Frame(win,padding=16);frame.pack(fill='both',expand=True);frame.columnconfigure(1,weight=1)
    source=tk.StringVar(value=str(ffa_root()/'dehaze_test/input_images'))
    output=tk.StringVar(value=str(ROOT/'batch_outputs'/time.strftime('%Y%m%d_%H%M%S')));gt=tk.StringVar()
    engine=tk.StringVar(value='真实 FPGA 参与处理（需要Z15）');algorithm=tk.StringVar(value='板上V3（FPGA神经推理）');edge=tk.StringVar(value='1280')
    recursive=tk.BooleanVar(value=True);arm=tk.BooleanVar(value=False);debug=tk.BooleanVar(value=False)
    options=[]
    def folder(row,label,variable):
        ttk.Label(frame,text=label).grid(row=row,column=0,sticky='w',pady=5)
        entry=ttk.Entry(frame,textvariable=variable);entry.grid(row=row,column=1,sticky='ew',padx=8);options.append(entry)
        button=ttk.Button(frame,text='选择',command=lambda:choose(variable));button.grid(row=row,column=2);options.append(button)
    def choose(variable):
        p=filedialog.askdirectory(parent=win,initialdir=variable.get() if Path(variable.get()).is_dir() else str(ROOT))
        if p:variable.set(p)
    folder(0,'有雾图片文件夹',source);folder(1,'输出文件夹',output);folder(2,'无雾参考（可选）',gt)
    ttk.Label(frame,text='执行模式').grid(row=3,column=0,sticky='w')
    modes=['电脑纯软件（同算法参考）','真实 FPGA 参与处理（需要Z15）']
    combo=ttk.Combobox(frame,textvariable=engine,values=modes,state='readonly',width=34);combo.grid(row=3,column=1,sticky='w',padx=8,pady=8);options.append(combo)
    algorithm_box=ttk.Combobox(frame,textvariable=algorithm,values=['板上V3（FPGA神经推理）','自适应增强V2'],state='readonly',width=28);algorithm_box.grid(row=3,column=2);options.append(algorithm_box)
    settings=ttk.Frame(frame);settings.grid(row=4,column=0,columnspan=3,sticky='ew',pady=5)
    ttk.Label(settings,text='最长边上限').pack(side='left');size=ttk.Combobox(settings,textvariable=edge,values=['640','1280','1920'],state='readonly',width=6);size.pack(side='left',padx=8);options.append(size)
    for text,var in [('包含子文件夹',recursive),('板上ARM基准',arm),('保留完整硬件调试文件',debug)]:
        c=ttk.Checkbutton(settings,text=text,variable=var);c.pack(side='left',padx=8);options.append(c)
    note=('每次自动处理一张，完成后接着下一张。电脑模式不使用FPGA；上板模式每张均实际上传、处理、回读并核对。\n'
          'V3暗通道、神经推理及保护均在FPGA，ARM准备局部统计；电脑模式全程不使用板卡。保持比例、不放大小图，当前最高1920×1280。\n'
          '输出独立PNG，不覆盖原图；有配对GT才计算PSNR/SSIM/MAE。恢复任务时选择原来的输出文件夹。')
    ttk.Label(frame,text=note,wraplength=830).grid(row=5,column=0,columnspan=3,sticky='w',pady=10)
    status=tk.StringVar(value='等待选择文件夹');ttk.Label(frame,textvariable=status,wraplength=830).grid(row=6,column=0,columnspan=3,sticky='w')
    bar=ttk.Progressbar(frame,mode='determinate');bar.grid(row=7,column=0,columnspan=3,sticky='ew',pady=10)
    log=tk.Text(frame,height=12,wrap='word',font=('Microsoft YaHei UI',9));log.grid(row=8,column=0,columnspan=3,sticky='nsew');frame.rowconfigure(8,weight=1)
    actions=ttk.Frame(frame);actions.grid(row=9,column=0,columnspan=3,sticky='ew',pady=12)
    state={'busy':False,'close_parent':False};events=queue.Queue();stop=threading.Event()
    old_close=parent.protocol('WM_DELETE_WINDOW')
    def set_busy(value):
        state['busy']=value;parent.busy=value
        for widget in options:widget.configure(state='disabled' if value else ('readonly' if isinstance(widget,ttk.Combobox) else 'normal'))
        start_button.configure(state='disabled' if value else 'normal');stop_button.configure(state='normal' if value else 'disabled')
        parent.update_buttons()
    def request_stop():stop.set();status.set('已请求停止：当前图片完成后停止，已完成结果保留。')
    def close_parent():
        if state['busy']:state['close_parent']=True;request_stop()
        elif old_close:parent.tk.call(old_close)
        else:parent.destroy()
    def close():
        if state['busy']:request_stop();return
        parent.protocol('WM_DELETE_WINDOW',old_close);win.destroy()
    win.protocol('WM_DELETE_WINDOW',close)
    def start():
        if state['busy']:return
        if parent.busy:messagebox.showinfo('正在处理','请等当前单图操作结束。',parent=win);return
        chosen='fpga' if engine.get()==modes[1] else 'pc'
        args=dict(input_dir=source.get(),output_dir=output.get(),engine=chosen,max_edge=int(edge.get()),ground_truth_dir=gt.get() or None,
                  algorithm='pl_v3' if algorithm.get().startswith('板上V3') else 'fused',recursive=recursive.get(),arm_benchmark=arm.get(),keep_debug=debug.get(),stop=stop,progress=lambda value:events.put(('progress',value)))
        stop.clear();state['engine']=chosen;state['prior_ready']=parent.ready;set_busy(True);parent.protocol('WM_DELETE_WINDOW',close_parent)
        def work():
            try:events.put(('done',run_batch(**args)))
            except Exception as error:events.put(('error',str(error)))
        threading.Thread(target=work,daemon=True).start()
    start_button=ttk.Button(actions,text='开始 / 继续批处理',command=start);start_button.pack(side='left')
    stop_button=ttk.Button(actions,text='完成当前张后停止',command=request_stop,state='disabled');stop_button.pack(side='left',padx=10)
    ttk.Button(actions,text='打开输出文件夹',command=lambda:os.startfile(output.get()) if Path(output.get()).is_dir() else None).pack(side='right')
    def drain():
        try:
            while True:
                kind,value=events.get_nowait()
                if kind=='progress':
                    bar.configure(maximum=max(1,value['total']),value=value['processed'])
                    status.set(f"{value['processed']}/{value['total']} · 新完成{value['succeeded']} · 已有跳过{value['skipped']} · 失败{value['failed']} · {value['elapsed_seconds']:.1f}秒")
                    log.insert('end',value['message']+'\n');log.see('end')
                    if int(log.index('end-1c').split('.')[0])>200:log.delete('1.0','50.0')
                else:
                    set_busy(False);parent.protocol('WM_DELETE_WINDOW',old_close)
                    if state.get('engine')=='fpga':parent.ready=kind=='done' and (value.get('board_initialized',False) or state.get('prior_ready',False)) and value['status']!='hardware_error';parent.update_buttons()
                    if kind=='error':status.set('未能完成：'+value);messagebox.showerror('批处理未完成',value,parent=win)
                    else:status.set(f"结束：新完成{value['succeeded']}，已有跳过{value['skipped']}，失败{value['failed']}。状态：{value['status']}。")
                    if state['close_parent']:parent.destroy();return
        except queue.Empty:pass
        if win.winfo_exists():win.after(100,drain)
    win.after(100,drain);return win
