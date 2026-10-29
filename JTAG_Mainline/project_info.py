"""Explain the implemented optimizations and their actual build evidence."""
from pathlib import Path
import re
ROOT=Path(__file__).parent

def hardware_details():
    info={'clock_mhz':100,'window':'15×15','max_size':'1920×1280 RGB',
          'resources':{},'wns_ns':None,'whs_ns':None,'filter_ii':None}
    path=ROOT/'logs/utilization.rpt'
    if path.exists():
        for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
            parts=[p.strip() for p in line.split('|')]
            if len(parts)>=8 and parts[1] in ['Slice LUTs','Slice Registers','Block RAM Tile','DSPs'] and parts[1] not in info['resources']:
                try:info['resources'][parts[1]]={'used':float(parts[2]),'available':float(parts[5]),'percent':float(parts[6])}
                except ValueError:pass
    path=ROOT/'logs/timing_summary.rpt'
    if path.exists():
        text=path.read_text(encoding='utf-8',errors='replace')
        if 'Design Timing Summary' in text:
            for line in text.split('Design Timing Summary',1)[1].splitlines():
                parts=line.split()
                if len(parts)>=10:
                    try:info['wns_ns']=float(parts[0]);info['whs_ns']=float(parts[4]);break
                    except ValueError:pass
    report=ROOT/'hls_build/hls/syn/report/filter_dark_Pipeline_VITIS_LOOP_29_1_VITIS_LOOP_35_3_csynth.rpt'
    if report.exists():
        for line in report.read_text(errors='replace').splitlines():
            if '|- VITIS_LOOP_' in line:
                parts=[p.strip() for p in line.split('|')]
                try:info['filter_ii']=int(parts[5])
                except (ValueError,IndexError):pass
                break
    return info

def optimization_text():
    from guided_ui_info import enhancement_optimizations
    text=enhancement_optimizations();r=hardware_details()
    if r['wns_ns'] is not None:text+=f"\n\n当前100MHz实现：WNS {r['wns_ns']:.3f}ns / WHS {r['whs_ns']:.3f}ns。"
    for name,v in r['resources'].items():text+=f"\n{name}: {v['used']:g}/{v['available']:g} ({v['percent']:.2f}%)"
    return text
