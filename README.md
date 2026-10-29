# 雾天图像去雾算法加速器设计

基于正点原子Z15与AMD Zynq-7000的图像去雾软硬件协同原型。系统使用双分支暗通道、导向细化、Q14小型网络和输出保护，由双Cortex-A9与FPGA共同完成像素处理。

项目仓库：[https://github.com/gaiya-hub/z15-fpga-dehaze](https://github.com/gaiya-hub/z15-fpga-dehaze)。

## 选择运行方案

| 目录 | 数据通路与入口 | 操作说明 |
|---|---|---|
| [JTAG_Mainline](JTAG_Mainline/README.md) | JTAG加载与传图，双ARM统计和PL去雾，入口launch.cmd | [当前版本](JTAG_Mainline/docs/CURRENT_VERSION.md)、[构建](JTAG_Mainline/docs/BUILD.md)、[接线](JTAG_Mainline/docs/WIRING.md) |
| [SD_Extension](SD_Extension/README.md) | 每批一次JTAG初始化，SD读写图片、串口进度，入口launch_sd.cmd | [当前版本](SD_Extension/docs/CURRENT_VERSION.md)、[构建](SD_Extension/docs/BUILD.md)、[SD任务](SD_Extension/docs/SD_TASKS.md) |

两个目录是同一仓库内的两套构建配置，保留相同去雾算法和模型。选择对应目录完成构建、配置和运行，位流、固件与启动协议按方案配套使用。

## 执行分工

- 电脑：图片解码与缩放、空气光及帧参数准备、任务管理、显示和独立评价。
- Zynq ARM0与ARM1：15×15局部统计，使用私有16行环形缓存与NEON，在小图或不满足对齐条件时回退ARM0。
- FPGA PL：双分支暗通道、导向细化、32路共享MAC的41→32→24→4定点网络及输出保护。

无雾参考图仅用于离线训练和质量评价。电脑同算法对比采用本仓库Q14参考，使用相同图片、处理尺寸和参数。

## 构建和运行

工具配置为Vivado/Vitis 2024.2、Windows与Python 3.12，板卡为XC7Z015CLG485-2，使用1 GB DDR配置。各方案安装requirements-core.txt，SD方案另安装requirements-sd.txt。

源码包提供HLS、PS–PL集成、BSP生成与固件构建脚本。先按对应BUILD.md生成位流和固件，再启动界面。工具路径使用settings.example.json示例配置为本机settings.local.json。构建依赖由复现者安装，包内不附AMD工具、BSP或预编译BIT/ELF。

## 实验结果与计时

134对既有回归的定点参考平均SSIM为0.866651、PSNR为20.1765 dB、MAE为21.6563，板级抽样用于验证输出一致性。双核统计优化在三个尺寸上使完整板上计算耗时降低约2.6%至2.8%。SD方案已完成两张图片的实卡读写、输出CRC与逐像素核对。

算法计算、初始化、JTAG传输、SD读写和独立评价分别记录。性能结论对应所列样本与环境，完整任务速度按同一输入输出范围评价。当前SD流程以任务包为输入、处理后导入图片；启动镜像和连续视频为后续扩展方向。

## 文档导航

当前操作以各方案的README和[主线文档索引](JTAG_Mainline/docs/README.md)、[SD文档索引](SD_Extension/docs/README.md)为入口。早期V2、电脑后处理V3、单ARM PL V3与界面阶段记录已归入各自docs/history，原路径保留跳转说明。历史记录中的版本状态与实测数据按原阶段解释。

## 离线验证

在所选方案目录执行：

```powershell
python -B -m unittest test_evaluation test_backend_errors test_batch
```

SD方案另执行 `python -B -m unittest test_sd_export_preflight`。解析器中的模拟样本用于协议、CRC与异常恢复检查，板级结论采用实卡和实际回读记录。

## 许可与复现资料

自研源码及自训练小型模型参数采用[MIT许可](LICENSE)。第三方依赖见[JTAG说明](JTAG_Mainline/THIRD_PARTY.md)和[SD说明](SD_Extension/THIRD_PARTY.md)。数据集、厂家资料和外部预训练模型由使用者独立获取，不随源码发布。

[AI协作摘要](AI协作记录.md)记录实际任务分工，[工程流程](skill/README.md)提供构建与验证步骤，SUBMISSION_MANIFEST.json记录源码文件哈希。[发布状态与操作说明](JTAG_Mainline/docs/PUBLISH.md)区分仓库创建、本地整理和远程提交。


## 模型文件与字节校验

当前板上定点推理由fixed_v3.json、frozen_model.json及生成的HLS权重头文件定义。neural_v3/model.pt与research/neural_v3/model.pt保留自训练浮点检查点，供训练复现和历史CPU参考使用，其candidate.json按原阶段记录解释。运行当前Q14参考无需加载这些PyTorch检查点。

源码清单和运行版本标识采用SHA256校验原始文件字节。.gitattributes关闭Git自动换行转换，使源码从Git检出后与发布清单保持一致。运行算法、模型数值和固件构建脚本不因该打包设置改变。
