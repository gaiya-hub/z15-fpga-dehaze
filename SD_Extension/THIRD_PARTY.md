# 第三方内容与发布边界

MIT覆盖本包自研HLS/ARM/电脑界面、构建编排、测试和原创说明/接线图。

- AMD Vivado/Vitis、HLS头文件、PS/AXI IP、BSP库、生成驱动/初始化脚本及工具链是外部构建依赖，适用各自条款。本源码包不包含这些完整工具、生成IP源码树或BSP静态库。
- FPGA器件参数和AXI寄存器地址是工程配置/接口说明；自研寄存器头仅定义本内核使用的接口常量。
- Python第三方依赖：NumPy、Pillow、scikit-image、OpenCV、threadpoolctl；可选PyTorch，分别遵守其发行条款。
- FFA-Net方法来源：https://github.com/zhilin007/FFA-Net 。本地副本未发现LICENSE/COPYING文件，因此本包不附带其架构实现、训练代码或权重，不将其标为MIT代码。外部使用者需独立取得所需内容并核对使用条件。
- 原 adaptive_dehaze.py 是外部可选软件对照，不随本包发布。它执行传统DCP/导向滤波/手工增强，不能称为已训练神经网络。
- 原始照片、成对数据集、厂家资料盘与用户实拍均不打入公开源码包。作者本机的绝对路径、原始评测日志、签名、账号和授权书也不发布。

完整FFA-Net论文：Qin et al., FFA-Net: Feature Fusion Attention Network for Single Image Dehazing, AAAI 2020。
导向滤波方法：https://people.csail.mit.edu/kaiming/eccv10/index.html 。本工程的V2/V3已用自研定点HLS实现导向细化。

自训练小型V3网络参数与Q14导出参数随源码发布；其结构和训练脚本位于research/neural_v3。它不是FFA-Net或外部预训练FFA权重。
