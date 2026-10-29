"""Deterministic engineering drawing, not an altered hardware photograph."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def draw_connections(root):
    directory=Path(root)/'docs';directory.mkdir(exist_ok=True)
    image=Image.new('RGB',(1920,1180),'#f2f5fa');d=ImageDraw.Draw(image)
    def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc',size)
    def text(x,y,value,size=26,fill='#142b45',bold=False):d.text((x,y),value,font=font(size,bold),fill=fill)
    def box(rect,fill,outline=None,radius=16,width=3):d.rounded_rectangle(rect,radius,fill=fill,outline=outline,width=width)
    def line(points,color,width=8,dashed=False):
        if not dashed:d.line(points,fill=color,width=width,joint='curve')
        else:
            for start,end in zip(points,points[1:]):
                dx=end[0]-start[0];dy=end[1]-start[1];length=max(1,(dx*dx+dy*dy)**.5)
                for distance in range(0,int(length),26):
                    a=distance/length;b=min(1,(distance+15)/length)
                    d.line((start[0]+dx*a,start[1]+dy*a,start[0]+dx*b,start[1]+dy*b),fill=color,width=width)
    text(55,30,'Z15 图片去雾演示：板子需要接哪些线',42,bold=True)
    text(58,92,'当前流程：电脑图片 → JTAG传入DDR → ARM / FPGA处理 → JTAG回读到电脑显示',27,fill='#4c6077')
    blue='#1769c7';red='#d54147';orange='#be7518';gray='#718196'
    # PC at left, board at right. Ports are arranged to match the board in the user's first overview photo.
    box((65,230,445,540),'#18364f');box((86,251,424,480),'#ffffff')
    text(124,310,'电脑演示程序',31,bold=True);text(117,369,'选择图片 / 三图对比',23)
    d.polygon([(170,541),(338,541),(376,578),(130,578)],fill='#748ba1')
    text(94,602,'USB①：接JTAG下载器',25,fill=blue,bold=True)
    text(94,644,'USB②：接PS_UART（可选）',24,fill=orange)
    # Main board.
    box((820,245,1780,825),'#203b37','#54756c',radius=24)
    text(850,657,'Z15 ZYNQ  ·  XC7Z015',33,fill='white',bold=True)
    box((1110,420,1495,648),'#17252d','#a0b6b5',radius=12)
    text(1185,454,'Zynq芯片',31,fill='white',bold=True)
    text(1140,515,'PS：ARM软件',27,fill='#cae3f2')
    text(1140,565,'PL：FPGA硬件',27,fill='#cae3f2')
    ports=[((865,252,960,315),'PL_UART'),((990,252,1100,315),'PS_UART'),((1130,252,1255,315),'USB_OTG'),((1285,252,1400,315),'USB HOST'),((1430,252,1545,315),'网口'),((1580,252,1700,315),'JTAG')]
    for rect,label in ports:
        box(rect,'#dce3e8',radius=7);text(rect[0]+8,rect[1]+22,label,18)
    box((1675,390,1775,465),'#dde4e7',radius=7);text(1686,411,'DC电源',20)
    box((1658,492,1765,555),'#596d64',radius=7);text(1672,507,'电源开关',19,fill='white')
    box((860,708,1048,778),'#a82b40',radius=8);text(878,716,'BOOT_CFG',23,fill='white',bold=True);text(880,747,'1=ON  2=ON',20,fill='white')
    for rect,label in [((1115,742,1260,800),'CAMERA'),((1295,742,1430,800),'HDMI'),((1465,742,1610,800),'TF卡')]:
        box(rect,'#6e8278',radius=7);text(rect[0]+13,rect[1]+18,label,20,fill='#d7e3dd')
    # JTAG: PC USB -> separate downloader -> board 14-pin header.
    box((490,365,745,485),'#ffffff',blue,width=4)
    text(523,382,'JTAG下载器',29,fill=blue,bold=True);text(524,433,'USB端       排线端',21,fill=blue)
    line([(444,420),(490,420)],blue)
    line([(745,419),(775,419),(775,185),(1640,185),(1640,252)],blue)
    d.ellipse((1628,240,1652,264),fill=blue)
    text(905,147,'① 必须：电脑USB → 下载器 → 板上JTAG',27,fill=blue,bold=True)
    text(501,510,'排线不是直接插电脑；',22,fill=blue)
    text(501,543,'红边仅作参考，以1脚标识为准。',22,fill=blue)
    # Power.
    box((1500,50,1860,128),'#fff0ef',red,width=3)
    text(1523,70,'② 必须：配套12V适配器',25,fill=red,bold=True)
    line([(1820,128),(1850,128),(1850,428),(1776,428)],red)
    d.ellipse((1763,416,1787,440),fill=red)
    text(1518,850,'接DC插座，再打开电源开关',25,fill=red,bold=True)
    # Optional USB serial, routed below PC then up to PS_UART. Dashed to distinguish from programming path.
    line([(445,470),(465,470),(465,862),(790,862),(790,216),(1045,216),(1045,252)],orange,6,True)
    d.ellipse((1035,242,1055,262),fill=orange)
    text(500,884,'③ 可选：电脑USB → PS_UART；仅用于串口日志',26,fill=orange,bold=True)
    # Bottom notes.
    box((60,958,1860,1135),'#ffffff','#d2dce8',radius=15,width=2)
    text(86,978,'当前图片演示不需要连接：摄像头、HDMI、网线、TF卡、USB HOST设备。',28,bold=True)
    text(86,1026,'断电后接线和设置拨码；JTAG模式为两个开关ON。排线1脚对应，图示不代替接口丝印。',25,fill='#4c6077')
    text(86,1070,'照片中摄像头已经插着也不代表程序在采集视频；当前图像数据通过JTAG传输。',25,fill='#4c6077')
    text(58,1150,'工程接线示意图，非开发板实拍；按Z15手册及已验证的软件传输路径绘制。',18,fill=gray)
    path=directory/'Z15_接线示意图.png';image.save(path)
    return path

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();print(draw_connections(args.root))
