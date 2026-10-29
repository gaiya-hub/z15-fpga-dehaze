# 去雾质量优化候选源码（实验版）

这份代码是 DCP + 轻量神经纹理校正的 **电脑参考实现**，没有部署到 FPGA。
现有134张回归图平均 SSIM 为0.8667，目标0.90尚未达到，失真回归门槛也未全部通过。不能用于宣称当前演示界面已经获得这些提升。

`detail_dehaze.py`：19项局部特征与22项帧统计量输入32/24单元的ReLU网络，预测RGB残差和共享纹理增益。
`safe_detail_v3.py`：RGB共用的残差混合系数，限制暗部降低、亮部增加和每通道最大变化；推理不读取GT。
`grid_dehaze.py`：另一个在1/4尺度估计系数的候选；测试表现弱于上面的方案，未采用。

## 本地复现推理

安装 requirements.txt 后，在本目录执行：

    python -B infer_quality_candidate.py 图片路径 --output 输出.png --edge 640

可选 `--ground-truth 配对无雾图` 只在推理之后评价，去掉该参数不会改变输出像素。
该命令不连接板卡，也不显示FPGA耗时。只适合查看候选算法。

## 复现训练

数据不随包附带。环境变量 `DEHAZE_DATASET` 指向包含四个natural/synthetic图像目录的数据根目录。dataset_manifest.json 固定了开发/回归划分与原始文件哈希。

    python -B train_detail_v3.py --steps 4000
    python -B train_detail_v3.py --steps 2000 --all
    python -B select_safe_v3.py

第一条只在开发集内部选型，第二条使用49张开发图重训冻结。原V2帧控制器已用过全部49张开发图，所以内部验证不是整个组合系统完全未见数据的盲测。134张回归图未参与梯度训练；它们过去已经反复查看，只能称为回归集。继续筛选网络结构会适应该集合，最终仍需新的独立室外场景验证。

原始FFA-Net代码、FFA权重、AMD软件与生成IP、数据图片均未包含。代码许可证见LICENSE；本包未发布到GitHub。
