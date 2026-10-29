"""SD task export/import and serial progress; the main comparison layout stays."""
from pathlib import Path
import queue,threading,time
def open_window(parent,initial_import=None):
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox
    from sd_tasks import export_task,import_task
    import sd_backend
    win=tk.Toplevel(parent);win.title('SD任务 · 串口进度与结果导入');win.geometry('900x660')
    area=ttk.Frame(win,padding=14);area.pack(fill='both',expand=True)
    ttk.Label(area,text='JTAG一次启动；图片从SD读取，结果写回SD；完整图片处理后导入。',wraplength=850).pack(anchor='w')
    events=queue.Queue();state={'busy':False,'running':False,'serial':None,'stop':threading.Event(),'records':[]}
    edge=tk.StringVar(value='640');repeats=tk.StringVar(value='1');port=tk.StringVar();status=tk.StringVar(value='选择导出SD任务、串口运行或导入已有结果。')
    line=ttk.Frame(area);line.pack(fill='x',pady=8)
    ttk.Label(line,text='最长边').pack(side='left');ttk.Combobox(line,textvariable=edge,values=['640','1280','1920'],width=7,state='readonly').pack(side='left',padx=6)
    ttk.Label(line,text='每图计时次数').pack(side='left');ttk.Combobox(line,textvariable=repeats,values=['1','3','5'],width=5,state='readonly').pack(side='left',padx=6)
    buttons=ttk.Frame(area);buttons.pack(fill='x',pady=6)
    serial_row=ttk.Frame(area);serial_row.pack(fill='x',pady=8)
    ttk.Label(serial_row,text='PS_UART串口（115200）').pack(side='left')
    ports=ttk.Combobox(serial_row,textvariable=port,width=16,state='readonly');ports.pack(side='left',padx=6)
    log=tk.Text(area,height=15,wrap='word');log.pack(fill='both',expand=True,pady=8)
    outputs=ttk.Combobox(area,state='readonly');outputs.pack(fill='x')
    ttk.Label(area,textvariable=status,wraplength=850).pack(anchor='w',pady=8)
    def write(text):log.insert('end',str(text)+'\n');log.see('end')
    def set_running(value):
        state['running']=value;parent.busy=value
        if value:parent.bar.start(12)
        else:parent.bar.stop()
        parent.update_buttons()
    def work(action):
        if state['busy']:return
        state['busy']=True
        def run():
            try:events.put(('done',action()))
            except Exception as error:events.put(('error',str(error)))
        threading.Thread(target=run,daemon=True).start()
    def progress(text):events.put(('log',text))
    def export():
        if state['running']:return
        source=filedialog.askdirectory(title='选择有雾图片文件夹',parent=win)
        if not source:return
        target=filedialog.askdirectory(title='选择SD卡盘符根目录（例如E盘，不要选择归档文件夹）',parent=win)
        if not target:return
        size=int(edge.get());count=int(repeats.get())
        work(lambda:('export',export_task(source,target,size,count,progress=progress)))
    def import_results():
        if state['running']:return
        source=filedialog.askdirectory(title='选择含DEHAZE的目录或DEHAZE目录',parent=win)
        if source:work(lambda:('import',import_task(source,verify=True,progress=progress)))
    def show_result(event=None):
        index=outputs.current()
        if index<0 or not state['records']:return
        from PIL import Image
        record=state['records'][index];run=Path(record['run']);result=record['result']
        with Image.open(run/'input.png') as i:hazy=i.copy()
        with Image.open(run/'fpga_output.png') as i:output=i.copy()
        clear=None
        if (run/'ground_truth.png').is_file():
            with Image.open(run/'ground_truth.png') as i:clear=i.copy()
        parent.source=result['source'];parent.show_result(hazy,output,result,run,clear,history=True)
        parent.captions[2].set('SD实测导入 · FPGA去雾＋'+('ARM双核统计' if result['dual_active'] else 'ARM单核回退'))
        parent.status.set('已导入SD板卡结果；本次未通过JTAG传输图片。')
    outputs.bind('<<ComboboxSelected>>',show_result)
    def refresh_ports():
        try:
            from serial.tools.list_ports import comports
            names=[p.device for p in comports()];ports['values']=names
            if port.get() not in names:port.set(names[0] if len(names)==1 else '')
        except ImportError:status.set('串口需要安装requirements-sd.txt中的pyserial；任务导出/导入仍可使用。')
    def connect_serial():
        if state['serial'] is not None:return
        if not port.get():
            messagebox.showinfo('选择串口','请选择与板卡PS_UART对应的COM口，可通过拔插USB线后刷新识别。',parent=win);return
        try:
            import serial
            state['serial']=serial.Serial(port.get(),115200,timeout=.2)
        except Exception as error:messagebox.showerror('串口连接',str(error),parent=win);return
        state['stop'].clear()
        def receive():
            buffer=b''
            try:
                while not state['stop'].is_set():
                    raw=state['serial'].readline()
                    buffer+=raw
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1);events.put(('serial',line.decode('utf-8',errors='replace').strip()))
                    if len(buffer)>8192:buffer=b''
            except Exception as error:events.put(('log','串口断开：'+str(error)+'；可用“查询完成状态”确认板卡是否已结束。'))
        threading.Thread(target=receive,daemon=True).start();status.set('串口已连接。')
    def start():
        if state['running'] or parent.busy:return
        if state['serial'] is None or not state['serial'].is_open:
            messagebox.showinfo('先连接串口','请先连接PS_UART串口，便于确认读写完成。',parent=win);return
        set_running(True)
        work(lambda:('started',sd_backend.start(progress)))
    def query():work(lambda:('query',sd_backend.query_status(progress)))
    for label,action in [('① 导出SD任务',export),('② 一次初始化并启动SD批处理',start),('③ 导入SD完整结果',import_results)]:ttk.Button(buttons,text=label,command=action).pack(side='left',padx=4)
    ttk.Button(serial_row,text='刷新',command=refresh_ports).pack(side='left')
    ttk.Button(serial_row,text='连接串口',command=connect_serial).pack(side='left',padx=6)
    ttk.Button(serial_row,text='查询完成状态',command=query).pack(side='left')
    def drain():
        try:
            while True:
                kind,value=events.get_nowait()
                if kind in ['serial','log']:
                    write(value);status.set(value)
                    if kind=='serial' and (value.startswith('SD_BATCH_DONE') or value.startswith('SD_ERROR')):
                        sd_backend.release();set_running(False)
                elif kind=='error':
                    state['busy']=False
                    if not sd_backend.reserved():set_running(False)
                    status.set(value);messagebox.showerror('SD任务',value,parent=win)
                else:
                    state['busy']=False;action,payload=value
                    if action=='export':status.set('已生成：'+str(payload[0])+'。将完整DEHAZE目录复制到卡根目录。')
                    elif action=='import':
                        root,report=payload;state['records']=report['records'];outputs['values']=[Path(r['result']['source']).name for r in report['records']]
                        status.set(f'已导入{len(report["records"])}张，缺少{len(report["missing_jobs"])}张结果；目录：{root}')
                        if report.get('batch_summary') and not report['batch_summary'].get('timing_valid',True):status.set('图片核验通过；整批计时未通过一致性检查，不能用于性能结论。')
                        if report.get('batch_summary') and report['batch_summary'].get('timing_valid'):
                            status.set(f'本批板上运行 {report["batch_summary"]["board_before_final_summary_ms"]/1000:.3f} s（含SD读写、启动和预热；不含最初JTAG配置及电脑准备/导入）')
                        if state['records']:outputs.current(0);show_result()
                    elif action=='query':
                        status.set(payload['description'])
                        if payload['finished']:set_running(False)
                    else:status.set('板卡正在从SD批量处理，请等待串口SD_BATCH_DONE后再取卡。' if state['running'] else 'SD任务已停止或完成，请查看串口记录后导入结果。')
        except queue.Empty:pass
        if win.winfo_exists():win.after(100,drain)
    def close():
        if state['running']:win.iconify();return
        state['stop'].set()
        if state['serial'] is not None:state['serial'].close()
        win.destroy()
    if initial_import is not None:events.put(('done',('import',initial_import)))
    win.protocol('WM_DELETE_WINDOW',close);win.after(100,drain);refresh_ports();return win
