# 文件夹顺序批处理

启动launch.cmd，点击“批量去雾”，选择有雾输入文件夹和输入目录之外的输出目录。GUI默认执行模式为“真实FPGA参与处理”，算法为板上V3，任务逐张上传、处理和回读。

无雾参考可选，缺少匹配GT时仍可去雾，PSNR、SSIM和MAE留空。尺寸保持比例，最长边可选640、1280或1920，并限制最高1920×1280。

输出保存图像、summary.csv和逐图记录。停止操作在当前图片完成后生效，使用相同任务配置与输出目录可继续，程序检查源图、模型和结果哈希后跳过已完成项。

## 命令行

```powershell
python -B batch_dehaze.py --input "有雾图片目录" --output "独立结果目录" --engine fpga --algorithm pl_v3 --edge 1280
python -B batch_dehaze.py --input "有雾图片目录" --output "软件参考目录" --engine pc --algorithm pl_v3 --edge 1280
```

命令行省略--engine时默认为pc，硬件任务显式传入--engine fpga。GUI和命令行的默认执行模式分别按上述设置解释。

计时区分帧参数、板上计算、JTAG上传回读与整个任务。SD数据通路由[SD扩展目录](../../SD_Extension/README.md)提供。
