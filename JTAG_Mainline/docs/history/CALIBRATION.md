> 历史阶段文档：下文的“当前”“默认”“本轮”描述当时实现。当前运行入口和执行分工见[版本说明](../CURRENT_VERSION.md)，构建步骤见[构建指南](../BUILD.md)。

# 复现融合控制器标定

控制器是22项图像统计量上的岭回归，只输出0..256的帧级融合权重；不是神经网络。运行去雾不读取GT。

准备自己的配对数据和JSON清单，执行：

```powershell
python -B tools/calibrate_fusion.py --manifest path/to/protocol.json --output quality_upgrade
```

清单结构示例（source/gt可相对清单目录）：

```json
{"manifest":[{"source":"hazy/scene1.png","gt":"clear/scene1.png","dataset":"outdoor","group":"scene1","split":"development"}]}
```

完整清单需要至少5个开发场景组，以及同数据组的holdout回归条目。相同场景组不能跨development与holdout。GT须与输入同场景、同视角、像素对齐且原始尺寸一致。工具拒绝场景组交叉；数据来源仍需使用者检查。

统一最长边640、不放大小图；分别试两种空气光估计，固定融合0.25/0.5/0.75及岭回归alpha1/10/100。每张开发图的最佳均方误差融合比例作为监督目标；按场景5折计算开发集外折预测，筛选各数据组SSIM不低于基线0.005的方案，再按各组等权PSNR最高选择。选择完成先写frozen_model.json，然后计算holdout回归。

本项目的frozen_model.json来自既有49张开发图；134张已查看图片只作回归，不能当作未见盲测。示例清单不含完整数据，源码包不分发第三方数据或FFA权重。

输出protocol.json、development_models.json、selection.json、frozen_model.json、regression.json和summary.json。更换默认模型前，应在新的未见场景验证，并重复同尺寸软件/ARM/FPGA一致性检查。标定工具不会自动替换演示模型。
