"""Native WM_GESTURE adapter for our Tk image canvases; no global input hook.
See https://learn.microsoft.com/windows/win32/wintouch/wm-gesture
"""
import sys,time

class GestureSequence:
    def __init__(self,zoom,pan):self.zoom=zoom;self.pan=pan;self.distance=None;self.position=None
    def consume(self,kind,flags,x,y,argument):
        if kind==3:
            distance=float(argument & 0xffffffff)
            if flags&1 or self.distance is None:self.distance=distance
            elif distance>0 and self.distance>0:self.zoom(distance/self.distance,x,y);self.distance=distance
            if flags&4:self.distance=None
            return True
        if kind==4:
            if flags&1 or self.position is None:self.position=(x,y)
            else:
                self.pan(x-self.position[0],y-self.position[1]);self.position=(x,y)
            if flags&4:self.position=None
            return True
        return False

class WindowsGestureHook:
    def __init__(self,canvas,zoom,pan):
        self.canvas=canvas;self.enabled=False;self.error=None;self.old=None;self.callback=None;self.touch_points=0
        self.sequence=GestureSequence(zoom,pan)
        if sys.platform!='win32':return
        import ctypes as c
        from ctypes import wintypes as w
        self.c=c;self.user=c.WinDLL('user32',use_last_error=True)
        class POINTS(c.Structure):_fields_=[('x',c.c_short),('y',c.c_short)]
        class INFO(c.Structure):
            _fields_=[('cbSize',w.UINT),('dwFlags',w.DWORD),('dwID',w.DWORD),('hwndTarget',w.HWND),('ptsLocation',POINTS),
                      ('dwInstanceID',w.DWORD),('dwSequenceID',w.DWORD),('ullArguments',c.c_ulonglong),('cbExtraArgs',w.UINT)]
        class CONFIG(c.Structure):_fields_=[('dwID',w.DWORD),('dwWant',w.DWORD),('dwBlock',w.DWORD)]
        self.INFO=INFO
        self.user.GetGestureInfo.argtypes=[w.HANDLE,c.POINTER(INFO)];self.user.GetGestureInfo.restype=w.BOOL
        self.user.CloseGestureInfoHandle.argtypes=[w.HANDLE];self.user.CloseGestureInfoHandle.restype=w.BOOL
        self.user.SetGestureConfig.argtypes=[w.HWND,w.DWORD,w.UINT,c.POINTER(CONFIG),w.UINT];self.user.SetGestureConfig.restype=w.BOOL
        pointer_name='SetWindowLongPtrW' if c.sizeof(c.c_void_p)==8 else 'SetWindowLongW'
        self.set_proc=getattr(self.user,pointer_name);self.set_proc.argtypes=[w.HWND,c.c_int,c.c_void_p];self.set_proc.restype=c.c_void_p
        self.user.CallWindowProcW.argtypes=[c.c_void_p,w.HWND,w.UINT,w.WPARAM,w.LPARAM];self.user.CallWindowProcW.restype=c.c_ssize_t
        self.user.IsWindow.argtypes=[w.HWND];self.user.IsWindow.restype=w.BOOL
        self.POINT=w.POINT;self.user.ScreenToClient.argtypes=[w.HWND,c.POINTER(w.POINT)];self.user.ScreenToClient.restype=w.BOOL
        self.hwnd=int(canvas.winfo_id());self.touch_points=int(self.user.GetSystemMetrics(95))
        config=(CONFIG*2)(CONFIG(3,1,0),CONFIG(4,7,0x18))
        try:
            if not self.user.SetGestureConfig(self.hwnd,0,2,config,c.sizeof(CONFIG)):raise c.WinError(c.get_last_error())
            proc=c.WINFUNCTYPE(c.c_ssize_t,w.HWND,w.UINT,w.WPARAM,w.LPARAM)
            self.callback=proc(self._message)
            c.set_last_error(0);old=self.set_proc(self.hwnd,-4,c.cast(self.callback,c.c_void_p))
            if not old:raise c.WinError(c.get_last_error())
            self.old=old;self.enabled=True
            canvas.bind('<Destroy>',lambda event:self.close() if event.widget==canvas else None,add='+')
        except Exception as error:self.error=str(error)
    def _message(self,hwnd,message,wparam,lparam):
        if message==0x0119:
            info=self.INFO();info.cbSize=self.c.sizeof(info)
            if self.user.GetGestureInfo(lparam,self.c.byref(info)) and info.dwID in [3,4]:
                try:
                    self.canvas._ignore_mouse_until=time.monotonic()+.3
                    point=self.POINT(info.ptsLocation.x,info.ptsLocation.y)
                    self.user.ScreenToClient(hwnd,self.c.byref(point))
                    self.sequence.consume(int(info.dwID),int(info.dwFlags),point.x,point.y,int(info.ullArguments))
                except Exception as error:self.error=str(error)
                finally:self.user.CloseGestureInfoHandle(lparam)
                return 0
        # In particular, forward GID_BEGIN/GID_END to the original default handler.
        return self.user.CallWindowProcW(self.old,hwnd,message,wparam,lparam)
    def close(self):
        if self.enabled and self.old and self.user.IsWindow(self.hwnd):self.set_proc(self.hwnd,-4,self.old)
        self.enabled=False
