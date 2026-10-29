# Z15 JTAG双核FPGA去雾主线

项目仓库：[https://github.com/gaiya-hub/z15-fpga-dehaze](https://github.com/gaiya-hub/z15-fpga-dehaze)。本目录是JTAG交互与顺序批处理构建配置。

电脑通过JTAG初始化和传图，ARM0/ARM1准备局部统计，PL执行暗通道、导向细化、Q14网络与输出保护，结果从板上DDR回读。

## 从这里开始

1. 阅读[当前版本与执行分工](docs/CURRENT_VERSION.md)。
2. 按[构建指南](docs/BUILD.md)安装依赖、配置工具并生成本目录的位流、固件与协议。
3. 按[接线指南](docs/WIRING.md)连接开发板。
4. 双击launch.cmd，初始化板卡，选择有雾图进行处理。

无雾参考只用于画质评价。JTAG主线按每图执行双核启动协议，启动等待和数据读写与算法计算分别记录。

## 功能与验证

单图与顺序批处理默认提供板上V3算法，GUI批处理执行模式默认选择真实FPGA。同算法电脑比较使用Q14参考。双核实现与范围见[双核说明](docs/DUAL_ARM.md)，验证数据见[版本说明](docs/CURRENT_VERSION.md)。

图像放大、质量指标和同算法对比针对当前处理记录。文件夹批处理按顺序逐张执行，停止与续跑见[批处理说明](docs/BATCH.md)。

## 文档与许可

完整导航见[文档索引](docs/README.md)。早期实现保存在[历史目录](docs/history/README.md)，其中“默认”“当前”等表述只适用于原阶段。

自研代码与模型参数采用[MIT许可](LICENSE)，第三方范围见[THIRD_PARTY.md](THIRD_PARTY.md)。公开包不含数据集、AMD工具/BSP、BIT/ELF或本机settings.local.json。运行模型不需要PyTorch，训练依赖见requirements-optional.txt。
