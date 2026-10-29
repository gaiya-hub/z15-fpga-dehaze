from pathlib import Path
import subprocess,json,os
from portable_settings import tool_root,setting
ROOT=Path(__file__).resolve().parent
bsp_value=setting('bsp');linker_value=setting('linker_script')
if not bsp_value or not linker_value:
    raise SystemExit('请按 docs/BUILD.md 创建 standalone BSP，并设置 DEHAZE_BSP 和 DEHAZE_LINKER_SCRIPT。')
BSP=Path(bsp_value);LINKER=Path(linker_value)
for required in [BSP/'Xilinx.spec',BSP/'include/xparameters.h',LINKER]:
    if not required.is_file():raise SystemExit('缺少构建依赖：'+str(required))
GCC=tool_root('vitis')/'gnu/aarch32/nt/gcc-arm-none-eabi/bin/arm-none-eabi-gcc.exe'
args=[str(GCC),'-O3','-fno-math-errno','-g3','-DSDT','-mcpu=cortex-a9','-mfpu=neon','-mfloat-abi=hard',f'-specs={BSP}/Xilinx.spec',f'-I{BSP}/include',
      f'-I{ROOT}/firmware','-Wall','-Wextra','-D__FILENAME__=__FILE__',str(ROOT/'firmware/main.c'),str(ROOT/'firmware/dual_entry.S'),'-o',str(ROOT/'board/demo.elf'),
      '-Wl,-T,'+str(LINKER),f'-L{BSP}/lib','-Wl,--start-group','-lxilstandalone','-lxiltimer','-lxil','-lgcc','-lc','-Wl,--end-group']
if os.environ.get('NN_CHECK_NEON')=='1':args.insert(1,'-DNN_CHECK_NEON')
(ROOT/'board').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
(ROOT/'logs/firmware_command.json').write_text(json.dumps(args,indent=2),encoding='utf-8')
p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
(ROOT/'logs/firmware.log').write_text(p.stdout+p.stderr,encoding='utf-8');print(p.stdout+p.stderr)
if p.returncode:raise SystemExit('Firmware build failed')
# The launch address is resolved from this exact ELF, never copied from a trial.
import hashlib
nm=GCC.with_name('arm-none-eabi-nm.exe')
symbols=subprocess.check_output([str(nm),'-n',str(ROOT/'board/demo.elf')],text=True)
entry=next(int(line.split()[0],16) for line in symbols.splitlines() if line.endswith(' dual_cpu1_entry'))
protocol={'version':1,'cpu1_entry':entry,'descriptor_address':0x1f000000,
          'elf_sha256':hashlib.sha256((ROOT/'board/demo.elf').read_bytes()).hexdigest()}
(ROOT/'board/firmware_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print('HD_DUAL_FIRMWARE_READY',hex(entry))
