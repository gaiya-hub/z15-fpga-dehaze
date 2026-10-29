"""Run with Vitis 2024.2: vitis -s create_bsp.py. Outputs remain under this project."""
from pathlib import Path
import json,time,os
import vitis
ROOT=Path(__file__).resolve().parent
xsa=ROOT/'board/design_1_wrapper.xsa'
if not xsa.is_file():raise FileNotFoundError('先执行 HLS 与硬件构建，生成 board/design_1_wrapper.xsa')
workspace=Path(os.environ.get('DEHAZE_BSP_WORKSPACE',str(ROOT/'build'/('v'+time.strftime('%H%M%S')))))
if workspace.exists():raise RuntimeError('BSP工作目录已存在，请换一个空目录：'+str(workspace))
client=vitis.create_client()
try:
    client.set_workspace(str(workspace))
    platform=client.create_platform_component(name='pf',hw_design=str(xsa),cpu='ps7_cortexa9_0',os='standalone',domain_name='sa9',no_boot_bsp=True,generate_dtb=False)
    platform.build()
    xpfm=client.find_platform_in_repos('pf')
    if not xpfm:raise RuntimeError('平台未生成有效xpfm，请检查BSP构建日志')
    app=client.create_app_component(name='probe',platform=xpfm,domain='sa9',template='hello_world')
    app.build()
    candidates=[p.parent for p in workspace.rglob('Xilinx.spec') if (p.parent/'include/xparameters.h').is_file()]
    preferred=[p for p in candidates if 'sa9' in p.parts]
    if preferred:candidates=preferred
    if not candidates:raise RuntimeError('平台构建后未找到 standalone BSP 根目录')
    linker=list((workspace/'probe').rglob('lscript.ld'))
    if not linker:raise RuntimeError('示例应用没有生成 lscript.ld')
    path=ROOT/'settings.local.json'
    settings=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    settings.update(bsp=str(candidates[0]),linker_script=str(linker[0]))
    path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PORTABLE_BSP_READY',settings['bsp'],settings['linker_script'],flush=True)
finally:vitis.dispose()
