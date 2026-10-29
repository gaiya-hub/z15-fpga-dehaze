# 构建与运行

以下命令在当前方案目录执行。工具采用Windows下Vivado/Vitis 2024.2与Python 3.12，器件为XC7Z015CLG485-2，配置1 GB DDR。

## 依赖与本机配置

```powershell
python -m pip install -r requirements-core.txt
```

固定参数推理无需PyTorch，网络训练依赖另见requirements-optional.txt。

复制settings.example.json为settings.local.json，填写本机工具目录。也可以使用环境变量：

```powershell
$env:DEHAZE_VITIS_ROOT='D:\tools\Vitis\2024.2'
$env:DEHAZE_VIVADO_ROOT='D:\tools\Vivado\2024.2'
```

settings.local.json与python.local.cmd属于本机配置，不提交仓库。工具、BSP和生成二进制由本机构建，第三方范围见../THIRD_PARTY.md。

## 参数与HLS

```powershell
python -B tools/export_quantized_weights.py
python -B build_hls.py --native-only
python -B build_hls.py
```

参数导出生成Q14头文件并检查表示范围，native阶段核对独立参考，HLS阶段生成IP。模型权重采用固定ROM，修改权重后重新构建HLS与硬件。

## PS与PL硬件系统

```powershell
python -B build_hardware.py
```

hardware/create_bd.tcl生成PS、AXI控制与数据互连、复位和100 MHz时钟，三个逻辑数据主接口共享PS HP0。本目录生成JTAG交互主线硬件配置。

输出board/design_1_wrapper.xsa、design_1_wrapper.bit与ps7_init.tcl。继续检查布局布线资源和时序报告后，再使用对应XSA创建BSP。

## BSP与固件

```powershell
& "$env:DEHAZE_VITIS_ROOT\bin\vitis.bat" -s create_bsp.py
python -B build_firmware.py
```

create_bsp.py生成standalone BSP并将BSP与链接脚本保存至本机配置。长路径环境可将DEHAZE_BSP_WORKSPACE设为尚不存在的短路径工作目录。build_firmware.py同时编译main.c和dual_entry.S，输出board/demo.elf与firmware_protocol.json。

ELF与生成协议配套使用，CPU1入口从当前ELF提取。协议中的哈希以本次构建输出为准。

## 启动与离线测试

按接线说明连接后双击launch.cmd，初始化板卡，选择有雾图进行处理。初始化为易失性JTAG加载，重新上电后重新初始化。

```powershell
python -B -m unittest test_evaluation test_backend_errors test_batch
```

这些离线测试无需板卡。硬件运行另按CURRENT_VERSION.md检查实际输出和计时。

独立裸机固件使用固定DDR区域：输入0x10000000、输出0x11000000、中间图0x12000000、可选ARM参考0x13000000、邮箱0x14000000、神经特征0x15000000。运行时按该内存配置单独使用开发板。

执行分工与计时见[CURRENT_VERSION.md](CURRENT_VERSION.md)，连接见[WIRING.md](WIRING.md)。
