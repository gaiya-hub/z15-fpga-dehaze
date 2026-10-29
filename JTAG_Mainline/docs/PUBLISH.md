# 源码获取与发布说明

项目仓库：[https://github.com/gaiya-hub/z15-fpga-dehaze](https://github.com/gaiya-hub/z15-fpga-dehaze)。

通过仓库首页的Code菜单下载ZIP，或使用Git克隆完整仓库。根目录包含JTAG_Mainline、SD_Extension、README.md与LICENSE；两个方案的运行入口、构建配置与文档分别位于对应目录。

## 发布内容

公开源码包括自研C/C++、Python与Tcl、模型参数、构建脚本、操作说明、测试及MIT许可。数据集、AMD工具与BSP、厂家资料、签署材料和本机设置由使用者另行准备。

首次运行前，按照所选方案的BUILD.md生成配套位流与固件，并参照settings.example.json配置本机环境。JTAG主线与SD扩展的启动步骤以各自README及CURRENT_VERSION.md为准。

## 版本与文件核对

GitHub提交记录用于标识源码版本，根目录SUBMISSION_MANIFEST.json及各方案SOURCE_MANIFEST.json用于核对原始文件字节。下载或克隆时保留完整目录结构；正式源码RAR与公开仓库采用相同的发布文件。板级效果与性能结论以文档标明的算法版本、样本和计时范围为依据。
