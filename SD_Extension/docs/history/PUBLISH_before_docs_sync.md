> 历史阶段文档：下文的“当前”“默认”“本轮”描述当时实现。当前运行入口和执行分工见[版本说明](../CURRENT_VERSION.md)，构建步骤见[构建指南](../BUILD.md)。

# 创建真正的开源链接

当前只有本地源码目录与归档，尚未创建或推送GitHub仓库。
使用自己的GitHub账号创建空仓库后，在本目录执行：

```powershell
git add .
git commit -m "Publish reproducible Z15 dehaze baseline"
git remote add origin <你创建的仓库HTTPS地址>
git push -u origin main
```

提交前再次确认没有settings.local.json、python.local.cmd、数据、外部预训练权重、厂家资料或生成IP树。仓库中THIRD_PARTY.md与MIT许可证都要保留。只有推送成功之后才能在比赛表单填写该真实仓库链接。
