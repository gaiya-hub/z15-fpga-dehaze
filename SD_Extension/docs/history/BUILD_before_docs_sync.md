> 历史阶段文档：下文的“当前”“默认”“本轮”描述当时实现。当前运行入口和执行分工见[版本说明](../CURRENT_VERSION.md)，构建步骤见[构建指南](../BUILD.md)。

# 在自己的电脑复现

## 工具与本地配置

安装匹配器件的 Vivado/Vitis 2024.2 及 JTAG 驱动。Python 3.12 用于电脑界面与编排；Vitis 自带 Python 用于生成 BSP。
复制 settings.example.json 为 settings.local.json 并填写工具目录，或在 PowerShell 设置：

```powershell
$env:DEHAZE_VITIS_ROOT='D:\tools\Vitis\2024.2'
$env:DEHAZE_VIVADO_ROOT='D:\tools\Vivado\2024.2'
```

也支持官方 XILINX_VITIS / XILINX_VIVADO 环境变量与 PATH。

## 1. HLS

默认板上神经V3。先执行 `python -B tools/export_quantized_weights.py` 从Q14参数重建头文件。

```powershell
python -B build_hls.py --native-only
python -B build_hls.py
```

原生检查用同目录 hls/ 中的独立测试台。第二条综合、打包 HLS IP 至 hls_build/；ip/ 为生成目录，不提交厂家工具生成的完整IP树。

## 2. PS-PL 硬件系统

```powershell
python -B build_hardware.py
```

hardware/create_bd.tcl 从源码创建 PS、控制 AXI、3 个数据 AXI（输出与神经数据共享32位端口）、复位与100MHz时钟；不打开任何旧工程。
生成 board/design_1_wrapper.xsa、design_1_wrapper.bit、ps7_init.tcl。默认另建 build/vivado，不覆盖作者以前的 vivado/ 设计。

## 3. 从 XSA 生成新的 BSP

```powershell
& "$env:DEHAZE_VITIS_ROOT\bin\vitis.bat" -s create_bsp.py
python -B build_firmware.py
```

create_bsp.py 在 build/v时间/ 生成 ps7_cortexa9_0 / standalone 平台和探测用 hello_world 应用，取得 BSP 与链接脚本并写入本地配置。
这些是由本机 AMD 工具生成的构建依赖，既不冒充自研源码，也不打入 MIT 源码包。
已有 BSP 时也可设置 DEHAZE_BSP（含 Xilinx.spec、include、lib 的目录）与 DEHAZE_LINKER_SCRIPT（lscript.ld）。

## 4. 运行

安装 requirements-core.txt，按 WIRING.md 接线，launch.cmd 后点击初始化；选择图片再处理。
内存区域固定供独立裸机固件使用，当前不支持与Linux或其他DDR程序同时占用：输入0x10000000、输出0x11000000、中间图0x12000000、ARM参考0x13000000、邮箱0x14000000、神经特征0x15000000、控制寄存器0x40000000。

## 可选软件对照

安装 requirements-optional.txt，独立取得外部实现与权重，并配置 DEHAZE_FFA_ROOT / DEHAZE_DATASET_ROOT。FFA-Net根目录预期含 net/models/FFA.py、FFA_trained_models/；原自适应代码路径预期为 dehaze_test/adaptive_dehaze.py。
外部内容未配置时仍可运行同算法电脑参考。外部数据集、代码和权重应分别核对来源和使用条件。

Windows路径较长时，先设置 `$env:DEHAZE_BSP_WORKSPACE='D:\z15_bsp'`，指向尚不存在的短目录，再运行create_bsp.py。脚本采用短组件名并跳过JTAG流程不需要的FSBL；不会删除已有工作区。
Vitis批处理可能在子流程失败时仍返回零，请同时确认输出包含PORTABLE_BSP_READY，并检查固件构建完成。

V3需要1GB DDR配置。神经推理在PL，局部统计在ARM；内存与阶段说明见PL_V3.md。初始化是易失性JTAG下载，掉电需重新执行。

当前加速版使用32路Q14神经核与优化ARM统计，输入模型打包时验证19位激活范围。正式固件-O3、NEON，开方后整数修正；NN_CHECK_NEON诊断宏默认关闭。新参考不依赖原FFA代码。

双核版：build_firmware.py同时编译firmware/dual_entry.S，并从当前ELF生成board/firmware_protocol.json。ELF和启动信息必须配套部署，CPU1入口不允许手工沿用旧地址。详见DUAL_ARM.md。
