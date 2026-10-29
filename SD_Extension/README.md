# Z15 SD与串口批处理扩展

项目仓库：[https://github.com/gaiya-hub/z15-fpga-dehaze](https://github.com/gaiya-hub/z15-fpga-dehaze)。本目录是SD数据通路构建配置。

电脑准备任务后，板卡每批经JTAG初始化一次，从SD读取输入，双ARM统计与PL去雾计算后写回SD，串口显示进度，处理完成后由读卡器导入图片。

## 从这里开始

1. 阅读[当前版本与执行分工](docs/CURRENT_VERSION.md)。
2. 按[构建指南](docs/BUILD.md)安装依赖、配置工具并生成本目录的位流、固件与协议。
3. 按[接线指南](docs/WIRING.md)连接开发板。
4. 双击launch_sd.cmd，按[SD任务流程](docs/SD_TASKS.md)导出、运行和导入结果。

无雾参考只用于画质评价。SD方案的双核启动按批次复用，启动等待和数据读写与算法计算分别记录。

## 功能与验证

两图真实SD读写、CRC、卡内与串口计时以及最终输出逐像素核对已通过，结果见[SD实验记录](docs/SD_ACCEPTANCE.md)。

SD批处理使用电脑预处理的任务包，完整图片在处理完成并导入后显示。SD上电启动和网络回传属于后续扩展。

## 文档与许可

完整导航见[文档索引](docs/README.md)。早期实现保存在[历史目录](docs/history/README.md)，其中“默认”“当前”等表述只适用于原阶段。

自研代码与模型参数采用[MIT许可](LICENSE)，第三方范围见[THIRD_PARTY.md](THIRD_PARTY.md)。公开包不含数据集、AMD工具/BSP、BIT/ELF或本机settings.local.json。Python依赖见requirements-core.txt与requirements-sd.txt。
