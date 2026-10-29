"""User-facing rationale backed by frozen protocol and measured records."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
SELECTION=ROOT/'selection'

def load(name):
    p=SELECTION/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def automatic_ready():
    release=load('release.json') or {}
    return bool(release.get('auto_enabled') and release.get('board_verified') and (SELECTION/'parameter_policy.json').exists())

def historical_decision_text():
    summary=load('summary.json');protocol=load('protocol.json');release=load('release.json') or {}
    lines=['首要目标：找到适合 FPGA 部署、去雾画质确有改善、延迟与资源可接受的通用室外算法。硬件加速是实现手段，当前定点暗通道仅为基线，最终算法尚未确定。','',
           '当前已上板基线：RGB 15×15 暗通道固定点恢复，电脑估计空气光，FPGA 完成像素恢复。画质改善仍不足，不能凭约11倍计算加速就认定应用算法合格。当前没有神经网络硬件加速或摄像头实时视频链路。','',
           '基线已经验证的内容（不等于最终算法选型理由）：',
           '1. 已在 XC7Z015 上完成真实实现，100 MHz 时序通过；每次 FPGA 回读均与同参数独立软件参考核对。',
           '2. ARM 与 FPGA 执行同一算法、同一图像和同一参数，能给出有明确含义的板上加速比。电脑与不同算法的速度差另外展示。',
           '3. 资源、延迟和残余画质问题都能从日志与图像查证；完整 FFA-Net 目前只有电脑 CPU 实测，不能用它的耗时冒充同算法硬件加速。',
           '', '另外两条路线的作用：',
           '• 你的 adaptive_dehaze.py 是暗通道、导向滤波、手工增强和融合，没有加载训练网络。它是软件画质对照。',
           '• FFA-Net 是真实神经网络，作为学习方法的画质对照；本项目目前未将其部署到 FPGA。',
           '• 真正的暗通道＋轻量网络需要另行完成训练、量化与上板验证。', '']
    if protocol:
        dev=sum(r['split']=='development' for r in protocol['manifest']);test=len(protocol['manifest'])-dev
        lines += [f'选型数据：共 {len(protocol["manifest"])} 对图，{dev} 张开发集、{test} 张保留验证集。按文件名前缀场景分组，同组不跨集合；之前查看的三张图只进入开发集。',
                  '扩大对比统一最长边 640，保持比例且不放大小图；演示仍可使用最高 1920×1280。640 结果不能证明全高清画质或实时视频帧率。',
                  '场景分组依据文件名，真实拍摄来源未验证；预训练模型是否见过这些图片未知，因此不称为与网络训练数据完全独立的测试。','']
    if summary:
        gate=summary['gate']
        lines += ['帧级自适应参数候选：只有开发集无雾图参与离线标定，运行时只读取有雾输入统计量，GT 不进入控制器或 FPGA。',
                  '上一轮参数试验的冻结条件（不等于最终应用验收）：保留集两组等权平均 PSNR 增益至少 0.2 dB；SSIM 降幅不超过 0.005；相对有雾输入下降超过 3 dB 的比例增加不超过 2 个百分点。',
                  f'实测：PSNR 变化 {gate["macro_psnr_gain"]:+.3f} dB；SSIM 变化 {gate["macro_ssim_change"]:+.4f}；严重退化比例变化 {gate["macro_severe_rate_increase"]*100:+.2f} 个百分点。',
                  '上一轮候选按旧规则的结果（保留历史记录）：'+('通过' if gate['passed'] else '旧规则未通过，不自动启用；继续作为研究候选，不据此结束算法研究。')]
        key='candidate' if release.get('auto_enabled') else 'current'
        lines+=['','当前基线在已查看的回归图集上的结果（640 基准）：']
        for group,label in [('natural','natural 文件夹组'),('synthetic','synthetic 文件夹组')]:
            s=summary['holdout'][group][key]
            lines.append(f'• {label} {s["n"]} 张：PSNR {s["mean_psnr_db"]:.2f} dB，SSIM {s["mean_ssim"]:.4f}，相对输入 PSNR 改善 {s["psnr_improved_fraction"]*100:.1f}%，下降超过 3 dB 占 {s["severe_degradation_fraction"]*100:.1f}%。')
        if summary.get('common_subset'):
            lines+=['','历史公共子集的多算法对比（电脑单次计算时间，不是持续帧率）：']
            common=summary['common_subset']['all']
            for method,label in [(key,'基线算法的软件参考'),('adaptive','你的自适应暗通道'),('ffa','FFA-Net OTS 原始输出')]:
                s=common[method]
                lines.append(f'• {label}：{s["n"]} 张，PSNR {s["mean_psnr_db"]:.2f} dB，SSIM {s["mean_ssim"]:.4f}，MAE {s["mean_mae"]:.2f}，电脑单次均值 {s["mean_single_pass_ms"]:.1f} ms。')
        else:lines.append('神经网络扩大对比尚未完成。')
    if release.get('board_records'):
        lines+=['','部署后的真实板上验证（不含 JTAG 上传回读）：']
        for item in release['board_records']:
            lines.append(f'• {item["size"][0]}×{item["size"][1]}：ARM {item["arm_ms"]:.2f} ms，FPGA 总计 {item["fpga_ms"]:.2f} ms，{item["speedup"]:.2f}×；同参数计算差异 {item["mismatches"]} 个通道。')
    complexity=load('network_complexity.json')
    if complexity:
        lines+=['','完整 FFA-Net 的实现规模：',
                f'• 实际参数 {complexity["parameters"]:,}，FP32 权重约 {complexity["fp32_weight_bytes"]/2**20:.2f} MiB；器件全部 BRAM 约 {complexity["device_bram_bytes"]/2**20:.3f} MiB。',
                f'• {complexity["input_size"][0]}×{complexity["input_size"][1]} 实际前向统计，仅卷积约 {complexity["conv_macs"]/1e9:.1f} GMAC。',
                '• 可利用 DDR 存权重，但仍需设计搬运与计算架构。这是复杂度依据，不是神经网络无法上板的证明，也不是硬件吞吐实测。']
    lines+=['','演示中必须讲清的边界：',
            '• 选择依据是可验证的工程取舍，不宣称所有图片画质最优；薄雾、天空、非均匀雾仍有失败情况。',
            '• 画质统计与板上计时采用各自注明的尺寸，不跨尺寸拼成一个速度结论。',
            '• 本次/历史结果明确标记；JTAG 传输耗时单列。等效板上 FPS 不能当作持续视频 FPS。',
            '• 未测板上功耗与整机能耗，不宣称节能倍数。','',
            '历史加速与选型记录：'+str(SELECTION/'算法选型与演示说明.md'),
            '冻结规则：'+str(SELECTION/'protocol.json')]
    research_path=ROOT/'quality_research'/'summary.json'
    lines+=['','本轮研究：通用室外场景，画质优先；原134张验证图已被查看，后续只作回归验证，最终需要新的未见场景。',
            '优先验证空气光估计、透射率细化、快速导向滤波与恢复范围约束；真正轻量神经网络为另一条待研究路线。']
    if research_path.exists():
        research=json.loads(research_path.read_text(encoding='utf-8'))
        lines.append('本轮状态：'+research.get('status','待验证'))
        for group,item in research.get('groups',{}).items():
            if group=='all':continue
            lines.append(f'• {group} {item["n"]} 张回归图：基线 PSNR/SSIM {item["baseline"]["psnr_db"]:.2f}/{item["baseline"]["ssim"]:.4f}，新软件候选 {item["candidate"]["psnr_db"]:.2f}/{item["candidate"]["ssim"]:.4f}；候选尚未上板。')
    lines+=['本轮报告：'+str(ROOT/'quality_research'/'候选筛查报告.md'),
            '研究目标：'+str(ROOT/'quality_research'/'实验目标与路线.md')]
    return '\n'.join(lines)

def decision_text():
    from guided_ui_info import quality_text
    return quality_text()

def explanation_window(parent,narration=False):
    import tkinter as tk
    from tkinter import ttk
    window=tk.Toplevel(parent);window.title('为什么选择这个方案 · 演示讲解依据');window.geometry('1080x780')
    frame=ttk.Frame(window,padding=15);frame.pack(fill='both',expand=True)
    text=tk.Text(frame,wrap='word',font=('Microsoft YaHei UI',12),padx=12,pady=12)
    bar=ttk.Scrollbar(frame,command=text.yview);bar.pack(side='right',fill='y');text.pack(fill='both',expand=True)
    content=decision_text()
    text.configure(yscrollcommand=bar.set);text.insert('1.0',content);text.configure(state='disabled')
    return window
