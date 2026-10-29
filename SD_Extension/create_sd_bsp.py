from pathlib import Path
import json,os,sys
import vitis
ROOT=Path(__file__).resolve().parent
workspace=Path(os.environ.get('DEHAZE_SD_BSP_WORKSPACE',str(ROOT/'build'/'sd_bsp')))
if workspace.exists():raise RuntimeError('SD BSP directory already exists; inspect instead of overwrite')
client=vitis.create_client()
try:
    client.set_workspace(str(workspace))
    platform=client.create_platform_component(name='pf',hw_design=str(ROOT/'board/design_1_wrapper.xsa'),cpu='ps7_cortexa9_0',os='standalone',domain_name='sd9',no_boot_bsp=True,generate_dtb=False)
    domain=platform.get_domain(name='sd9');domain.set_lib(lib_name='xilffs')
    platform.build()
    xpfm=client.find_platform_in_repos('pf');assert xpfm
    app=client.create_app_component(name='probe',platform=xpfm,domain='sd9',template='hello_world');app.build()
    bsp=[p.parent for p in workspace.rglob('Xilinx.spec') if (p.parent/'include/ff.h').is_file() and (p.parent/'include/xparameters.h').is_file()]
    assert bsp,'No xilffs-enabled BSP generated'
    linker=next((workspace/'probe').rglob('lscript.ld'))
    settings=json.loads((ROOT/'settings.local.json').read_text(encoding='utf-8'));settings['sd_bsp']=str(bsp[0]);settings['sd_linker_script']=str(linker)
    (ROOT/'settings.local.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print('SD_BSP_READY',bsp[0],linker,flush=True)
finally:vitis.dispose()
