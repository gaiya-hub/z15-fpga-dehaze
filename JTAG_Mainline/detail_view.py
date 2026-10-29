"""Synchronized, cursor-anchored photo viewer with bounded-memory viewport rendering."""
import math,time

class Viewport:
    def __init__(self,width,height):self.width=width;self.height=height;self.cx=width/2;self.cy=height/2;self.scale=1.0
    def at(self,x,y,w,h):return self.cx+(x-w/2)/self.scale,self.cy+(y-h/2)/self.scale
    def zoom(self,factor,x,y,w,h):
        px,py=self.at(x,y,w,h);self.scale=max(.02,min(16.0,self.scale*factor))
        self.cx=px-(x-w/2)/self.scale;self.cy=py-(y-h/2)/self.scale
    def pan(self,dx,dy):self.cx-=dx/self.scale;self.cy-=dy/self.scale
    def fit(self,w,h):self.scale=min(1.0,max(.02,min(w/self.width,h/self.height)));self.cx=self.width/2;self.cy=self.height/2
    def constrain(self,w,h):
        halfx=w/(2*self.scale);halfy=h/(2*self.scale)
        self.cx=self.width/2 if 2*halfx>=self.width else max(halfx,min(self.width-halfx,self.cx))
        self.cy=self.height/2 if 2*halfy>=self.height else max(halfy,min(self.height-halfy,self.cy))

def show_detail(parent,images,titles):
    import tkinter as tk
    from tkinter import ttk,messagebox
    from PIL import Image,ImageTk
    from windows_gestures import WindowsGestureHook
    if not images or not all(im is not None for im in images):
        messagebox.showinfo('先完成处理','请先准备完整的对比图像，再查看同步细节。',parent=parent);return
    if len({im.size for im in images})!=1:
        messagebox.showinfo('尺寸不一致','请先完成同尺寸处理，再查看同步细节。',parent=parent);return
    win=tk.Toplevel(parent);win.title('同步放大 · 滚轮缩放 / 触屏捏合');win.geometry('1400x820');win.minsize(900,620)
    view=Viewport(*images[0].size);win.viewport=view;win.photos=[];win.gesture_hooks=[]
    top=ttk.Frame(win,padding=10);top.pack(fill='x')
    ttk.Label(top,text='滚轮缩放 · 鼠标/单指拖动 · 双指捏合 · 双击适合窗口').pack(side='left')
    amount=tk.StringVar(value='100%');ttk.Label(top,textvariable=amount,width=8).pack(side='right')
    holder=ttk.Frame(win);holder.pack(fill='both',expand=True);holder.rowconfigure(1,weight=1)
    canvases=[];pending=[None];drag=[None]
    for index,title in enumerate(titles):
        holder.columnconfigure(index,weight=1,uniform='photo')
        ttk.Label(holder,text=title,anchor='center').grid(row=0,column=index,sticky='ew',pady=4)
        canvas=tk.Canvas(holder,bg='#172638',highlightthickness=0,cursor='hand2')
        canvas.grid(row=1,column=index,sticky='nsew',padx=3);canvases.append(canvas)
    win.canvases=canvases
    def render():
        pending[0]=None
        if not win.winfo_exists():return
        w=min(c.winfo_width() for c in canvases);h=min(c.winfo_height() for c in canvases);view.constrain(w,h)
        win.photos=[];amount.set(f'{view.scale*100:.0f}%')
        for canvas,image in zip(canvases,images):
            cw=max(1,canvas.winfo_width());ch=max(1,canvas.winfo_height());left=view.cx-cw/(2*view.scale);upper=view.cy-ch/(2*view.scale)
            if abs(view.scale-1)<1e-8:left=round(left);upper=round(upper)
            picture=image.transform((cw,ch),Image.Transform.AFFINE,(1/view.scale,0,left,0,1/view.scale,upper),
                                    resample=Image.Resampling.BICUBIC,fillcolor='#172638')
            photo=ImageTk.PhotoImage(picture);win.photos.append(photo);canvas.delete('all');canvas.create_image(0,0,anchor='nw',image=photo)
    def schedule():
        if pending[0] is None:pending[0]=win.after(16,render)
    def zoom(canvas,factor,x=None,y=None):
        cw=canvas.winfo_width();ch=canvas.winfo_height();view.zoom(factor,cw/2 if x is None else x,ch/2 if y is None else y,cw,ch);schedule()
    def fit(event=None):view.fit(min(c.winfo_width() for c in canvases),min(c.winfo_height() for c in canvases));schedule();return 'break'
    def actual():zoom(canvases[0],1/view.scale)
    def wheel(event):
        zoom(event.widget,math.exp(max(-8,min(8,event.delta/120))*.16),event.x,event.y);return 'break'
    def press(event):
        if time.monotonic()<getattr(event.widget,'_ignore_mouse_until',0):return 'break'
        drag[0]=(event.widget,event.x,event.y);return 'break'
    def motion(event):
        if drag[0] and time.monotonic()>=getattr(event.widget,'_ignore_mouse_until',0):
            _,x,y=drag[0];view.pan(event.x-x,event.y-y);drag[0]=(event.widget,event.x,event.y);schedule()
        return 'break'
    for canvas in canvases:
        canvas.bind('<MouseWheel>',wheel);canvas.bind('<Button-4>',lambda e:zoom(e.widget,1.17,e.x,e.y));canvas.bind('<Button-5>',lambda e:zoom(e.widget,1/1.17,e.x,e.y))
        canvas.bind('<ButtonPress-1>',press);canvas.bind('<B1-Motion>',motion);canvas.bind('<ButtonRelease-1>',lambda e:drag.__setitem__(0,None))
        canvas.bind('<Double-Button-1>',fit);canvas.bind('<Configure>',lambda e:schedule())
    for label,command in [('适合窗口',fit),('100%',actual),('＋',lambda:zoom(canvases[0],1.25)),('－',lambda:zoom(canvases[0],.8))]:
        ttk.Button(top,text=label,command=command,width=9 if len(label)>2 else 4).pack(side='right',padx=3)
    touch=tk.StringVar(value='正在检测触屏支持…');win.touch_status=touch;ttk.Label(win,textvariable=touch,padding=(10,6)).pack(fill='x')
    def install():
        for canvas in canvases:
            hook=WindowsGestureHook(canvas,lambda factor,x,y,c=canvas:zoom(c,factor,x,y),lambda dx,dy:(view.pan(dx,dy),schedule()))
            win.gesture_hooks.append(hook)
        points=max((h.touch_points for h in win.gesture_hooks),default=0)
        if all(h.enabled for h in win.gesture_hooks):
            touch.set('触屏捏合和平移已接入。' if points>=2 else '滚轮与拖动已启用；Windows当前未报告多点触屏设备。')
        else:touch.set('滚轮与拖动已启用；本机原生触屏手势接口未启用。')
        fit()
    win.after(50,install)
    def close():
        if pending[0] is not None:win.after_cancel(pending[0]);pending[0]=None
        for hook in win.gesture_hooks:hook.close()
        win.destroy()
    win.protocol('WM_DELETE_WINDOW',close);win.zoom_at=zoom;win.fit_images=fit;win.render_view=render
    return win
