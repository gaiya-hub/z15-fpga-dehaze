# SD顺序批处理

本目录首选SD流程：导出任务到卡根目录 → 安全弹出并断电插卡 → JTAG每批初始化一次 → 双ARM与PL逐图计算 → 输出写回SD → 读卡器导入。详细步骤见[SD_TASKS.md](SD_TASKS.md)。

处理期间显示串口进度，完整图片在导入后显示。新批次前归档原DEHAZE，保持已有结果；当前SD任务采用重新准备新任务的方式运行，不使用JTAG批处理的停止续跑规则。

本目录保留JTAG相关公共代码，如使用launch.cmd或batch_dehaze.py，图像仍经JTAG传输。两种数据通路的计时分开解释，入口见[CURRENT_VERSION.md](CURRENT_VERSION.md)。
