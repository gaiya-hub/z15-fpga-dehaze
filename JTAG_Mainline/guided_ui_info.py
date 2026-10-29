from pathlib import Path
import json
ROOT=Path(__file__).parent

def quality_text():
    report_root=ROOT if (ROOT/'quality_upgrade').exists() else ROOT.parent
    p=report_root/'quality_upgrade/summary.json'
    lines=['增强 V2：双分支暗通道＋快速导向透射率细化＋恢复范围约束＋自适应融合。',
           '选择理由：旧版柔和分支保留细节；较强分支去除浓雾；导向滤波细化边缘；帧级统计控制融合强度。彩色图像不通过低分辨率放大来冒充高清，只有透射率系数在1/4尺度处理。',
           '不是去雾神经网络。22项统计量的岭回归控制器只选一个融合权重。电脑估计空气光和帧参数；暗通道、导向滤波、系数插值、RGB恢复与融合由 FPGA PL 执行。',
           '控制器用49张开发图离线标定、按场景5折交叉验证后冻结；运行时不读取无雾图。134张以前已查看，只作回归。最终仍需新的未见室外场景。','']
    if p.exists():
        result=json.loads(p.read_text(encoding='utf-8'))
        lines.append('134张回归结果，最长边640，RGB同尺寸：')
        for name,r in result['regression'].items():
            lines.append(f'{name} {r["n"]}张：PSNR {r["baseline"]["psnr_db"]:.2f}→{r["fused"]["psnr_db"]:.2f} dB；SSIM {r["baseline"]["ssim"]:.4f}→{r["fused"]["ssim"]:.4f}。')
    lines+=['','两图组平均画质均改善，115/134张PSNR优于旧版；仍有天空噪声、薄雾过暗失败场景，3/134张相对输入下降超过3 dB。可以切换旧版核对。',
            'FFA-Net在既有natural公共16张上仍优于V2：23.30 vs20.69 dB；synthetic公共16张V2更好：16.39 vs11.23 dB。不能宣称全面超过神经网络。',
            '增强版增加了计算量。当前同算法ARM/FPGA实测决定其速度取舍；旧版约11倍加速不能移用给V2。JTAG传输另计，尚无持续视频帧率与板上功耗实测。',
            '完整画质报告：'+str(report_root/'quality_upgrade/画质优化报告.md')]
    board=report_root/'quality_upgrade/board_records.json'
    if board.exists():
        lines+=['','增强版真实板上记录（不含JTAG传输）：']
        for item in json.loads(board.read_text(encoding='utf-8')):
            r=item['result']
            if item['algorithm']=='fused':
                lines.append(f'{item["label"]} {r["processing_size"]}：ARM {r["arm_ms"]:.2f}ms / FPGA {r["fpga_ms"]:.2f}ms / {r["speedup"]:.2f}倍；差异{r["host_channel_mismatches"]}通道。')
    from pl_runtime_info import explain
    return explain()+'\n\n以下为板上基础V2的历史验证：\n'+ '\n'.join(lines)

def enhancement_optimizations():
    from pl_runtime_info import explain
    return explain()+'\n\n'+'\n'.join([
      '当前板上V3的硬件优化：',
      '神经网络采用Q14权重、32路并行乘加、分层复用和固定权重ROM；帧级全局贡献只算一次，逐像素计算局部贡献。推理、RGB恢复及共同混合比例保护均在PL；ARM局部统计使用共享RGB扫描、16行环形缓存、NEON RGB统计/特征打包及精确整数归一化；FPGA用特征预取FIFO→32路神经计算→独立结果写出流水。',
      '1. 15×15暗通道用可分离最小值、14行缓存和并行比较；两组空气光分先后两遍复用同一单元。',
      '2. 导向滤波的I/t块均值缩到1/4尺度，11×11两次盒式统计；行缓存复用邻域数据，读入、求和、系数计算分成DATAFLOW阶段。',
      '3. Q8归一化、Q12系数与恢复倒数，整数双线性插值；低分辨率系数恢复到原处理尺寸，RGB输出保持原处理尺寸。',
      '4. 帧级参数只在开始时传入。增强模式按行将RGB读入、暗通道读入、恢复融合和结果写出分成FIFO流水阶段；旧版保留独立快速恢复路径。',
      '5. 三个AXI主接口经HP0访问DDR，输出与神经数据共用32位接口，字节工作区独立；合并输出写进程以满足数据流约束，减少布线资源。行缓冲驻留BRAM，局部统计中间值改用16行环形缓存；最终19项特征仍由ARM按40字节/像素打包到DDR，FPGA突发预取并与计算、结果写出重叠。全PL统计候选因LUT超限未采用。',
      '6. 单像素暗通道流水的II不能代表全系统吞吐。导向计算、分支阶段、DDR等待和缓存维护均计入真实板上时间。',
      '7. 独立ARM参考使用完整低分辨率数组与滑动求和，HLS使用行缓存；两者与NumPy定点规范核对，板上再次逐像素检查。',
      '8. 无雾GT只用于PSNR/SSIM/MAE评价。控制器是开发集标定的线性统计模型，不是部署了FFA-Net。',
      '', '实现资源和时序以当前logs/utilization.rpt、timing_summary.rpt为准；不要沿用旧版资源占用和加速倍数。'])
