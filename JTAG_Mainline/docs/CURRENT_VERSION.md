# 当前版本与执行分工

文档状态：2026年10月29日本地整理。方案目录：JTAG_Mainline。仓库：https://github.com/gaiya-hub/z15-fpga-dehaze。源码远程提交由发布步骤单独执行。

## 配置与入口

| 项目 | 配置 |
|---|---|
| 板卡 | Z15 XC7Z015CLG485-2，1 GB DDR |
| 工具 | Vivado/Vitis 2024.2，Python 3.12 |
| 数据通路 | JTAG初始化、图片上传与结果回读 |
| 程序入口 | launch.cmd |
| 固件构建 | build_firmware.py |
| 固件协议 | board/firmware_protocol.json与demo.elf |

电脑准备帧参数，ARM0/ARM1计算19项局部特征，PL完成双分支DCP、导向细化、41→32→24→4的32路Q14网络和输出保护。小图和不满足对齐条件时回退ARM0。模型权重固定在PL ROM，变更权重需重新构建HLS与硬件。

无雾参考只参与训练或质量评价。当前方案的神经推理位于PL，早期“电脑神经后处理”的描述只对应history/HYBRID_V3.md阶段。

## 验证与计时范围

134对图像为既有离线回归，其中自然类56对、合成类78对，平均SSIM0.866651、PSNR20.1765 dB、MAE21.6563。板级抽样用于验证实现与定点参考的逐像素一致性。

双核相对单核的三个尺寸计算耗时降低约2.6%至2.8%，常用图、最大尺寸和边界回退均有抽样核对，见[双核说明](DUAL_ARM.md)。

算法合计包括PC帧参数和板上计算。初始化、CPU1启动等待、JTAG上传回读以及独立评价分别记录，性能比较使用相同输入、尺寸、算法与计时范围。

## 文档适用范围

[BUILD.md](BUILD.md)、[DUAL_ARM.md](DUAL_ARM.md)、[BATCH.md](BATCH.md)、[WIRING.md](WIRING.md)是当前操作说明。[history](history/README.md)保存早期阶段文档。DEPLOYMENT、VALIDATION等JSON是具体实验时刻的记录，不作为整个仓库当前功能开关。
