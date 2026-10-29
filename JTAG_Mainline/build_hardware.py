import os
from pathlib import Path
import zipfile,subprocess,shutil
from portable_settings import tool_executable
ROOT=Path(__file__).parent
(ROOT/'logs').mkdir(exist_ok=True)
IP=ROOT/'ip';IP.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'hls_build/dehaze_hd.zip') as z:
    for item in z.infolist():
        if not (IP/item.filename).resolve().is_relative_to(IP.resolve()):raise RuntimeError('Unsafe IP archive')
    z.extractall(IP)
cmd=['cmd.exe','/d','/c',str(tool_executable('vivado',required=True)),'-mode','batch','-notrace','-source',str(ROOT/'build_hardware.tcl'),'-log',str(ROOT/'logs/vivado.log'),'-nojournal']
with (ROOT/'logs/hardware_build.log').open('w',encoding='utf-8') as log:
    p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    for raw in p.stdout:
        line=raw.decode('utf-8',errors='replace');log.write(line);log.flush()
        if any(k in line for k in ['ERROR','CRITICAL','HD_','Complete','completed','Timing']):print(line.strip()[:600],flush=True)
    if p.wait():raise RuntimeError('Hardware build failed')
if os.environ.get('DEHAZE_VALIDATE_ONLY')=='1':
    print('BD_VALIDATION_ONLY: no implementation or new XSA exported');raise SystemExit(0)
with zipfile.ZipFile(ROOT/'board/design_1_wrapper.xsa') as z:
    for name in ['design_1_wrapper.bit','ps7_init.tcl']:(ROOT/'board'/name).write_bytes(z.read(name))
print('HD_BOARD_ARTIFACTS_READY',flush=True)
