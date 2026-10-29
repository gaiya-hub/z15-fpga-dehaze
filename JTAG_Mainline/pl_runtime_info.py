from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
def explain(result=None):
    if result is None:
        result={'pl_statistics':False,'neural_engine_lanes':32}
    spatial=result.get('pl_statistics',False);lanes=result.get('neural_engine_lanes',16)
    location='FPGA PL局部统计通过FIFO直接连接神经核' if spatial else 'Z15 ARM局部统计；其耗时计入板卡总计'
    if result.get('dual_arm'):
        location=('Z15 ARM0/ARM1分行并行统计' if result.get('dual_active') else '当前尺寸采用ARM0单核回退')+'；统计、同步和缓存维护计入板卡算法合计'
    return (f'板上V3：双分支暗通道、导向细化、{lanes}路Q14神经推理及防失真保护。'+location+'。电脑只估计空气光、融合权重和帧统计，发送图像，回读最终图并核对；不在电脑上修正FPGA输出。\n'
            '电脑同算法以新Q14 Python参考完整处理；不调用FFA-Net-master/dehaze_test中的原算法。无雾GT只用于评价。\n'
            '板卡算法时间包含PL计算、DDR访问、必要缓存控制及ARM工作；参数估计加上板卡时间为方案计算时间。JTAG、文件读写、验证另计。没有把此结果当作持续视频帧率。\n'
            +('双核固件使用私有环形缓存与缓存行对齐分区；CPU1启动等待/握手单列，包含在全流程等待中，未计入稳定算法计算时间。\n' if result.get('dual_arm') else '')+
            '本轮采用ARM RGB共享扫描、16行环形缓存、NEON RGB统计/特征打包及精确整数归一化；FPGA采用固定权重ROM、特征预取、32路乘加和结果写出流水。本轮保持已验证的神经权重。134对既有回归平均SSIM为0.866651，目标0.90尚未达到；部分画面仍有退步。新版质量候选因增加失真未采用。')
