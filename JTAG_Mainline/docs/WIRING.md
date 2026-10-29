# Z15接线

![JTAG基础接线示意](Z15_接线示意图.png)

## 电源与JTAG

电脑USB连接JTAG下载器，下载器排线连接板上JTAG接口，按接口1脚标识对应。配套12V适配器连接DC电源插座。断电时设置BOOT_CFG两个开关为ON以使用JTAG模式，核对连接后上电。

更换板上模块或插拔TF卡前断电，按厂家硬件手册核对插槽和电源位置。PS_UART、PL_UART和USB HOST不是JTAG接口。

## 可选串口

电脑USB另接PS_UART可观察日志。JTAG主线收发图像不依赖串口，不要求连接TF卡、HDMI或摄像头。SD数据通路见[SD扩展](../../SD_Extension/README.md)。

图为连接示意，实际接口方向按板上丝印与厂家手册确认。
