from pathlib import Path
import hashlib,json,subprocess,zlib
from portable_settings import setting,tool_root
ROOT=Path(__file__).resolve().parent
BSP=Path(setting('sd_bsp','missing'));LINKER=Path(setting('sd_linker_script','missing'))
for name in ['Xilinx.spec','include/ff.h','include/xparameters.h']:
    if not (BSP/name).is_file():raise SystemExit('先创建SD BSP，缺少：'+str(BSP/name))
if 'XPAR_XSDPS_NUM_INSTANCES' not in (BSP/'include/xparameters.h').read_text():raise SystemExit('SD BSP没有SD控制器，检查XSA。')
files=sorted((ROOT/'firmware').glob('*'))
build_id=zlib.crc32(b''.join(p.read_bytes() for p in files if p.is_file()))&0xffffffff
gcc=tool_root('vitis')/'gnu/aarch32/nt/gcc-arm-none-eabi/bin/arm-none-eabi-gcc.exe'
args=[str(gcc),'-O3','-fno-math-errno','-g3','-DSDT','-mcpu=cortex-a9','-mfpu=neon','-mfloat-abi=hard',f'-DSD_BUILD_ID={build_id}U',
 f'-specs={BSP}/Xilinx.spec',f'-I{BSP}/include',f'-I{ROOT}/firmware','-Wall','-Wextra',str(ROOT/'firmware/sd_main.c'),str(ROOT/'firmware/dual_entry.S'),
 '-o',str(ROOT/'board/sd_batch.elf'),'-Wl,-T,'+str(LINKER),f'-L{BSP}/lib','-Wl,--start-group','-lxilffs','-lxilstandalone','-lxiltimer','-lxil','-lgcc','-lc','-Wl,--end-group']
p=subprocess.run(args,capture_output=True,text=True,cwd=ROOT)
(ROOT/'logs/sd_firmware.log').write_text(p.stdout+p.stderr,encoding='utf-8');print(p.stdout+p.stderr)
if p.returncode:raise SystemExit('SD firmware build failed')
symbols=subprocess.check_output([str(gcc.with_name('arm-none-eabi-nm.exe')),'-n',str(ROOT/'board/sd_batch.elf')],text=True)
entry=next(int(line.split()[0],16) for line in symbols.splitlines() if line.endswith(' dual_cpu1_entry'))
protocol={'version':1,'cpu1_entry':entry,'build_id':build_id,
 'elf_sha256':hashlib.sha256((ROOT/'board/sd_batch.elf').read_bytes()).hexdigest(),
 'bit_sha256':hashlib.sha256((ROOT/'board/design_1_wrapper.bit').read_bytes()).hexdigest()}
(ROOT/'board/sd_firmware_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
(ROOT/'logs/sd_firmware_command.json').write_text(json.dumps(args,indent=2),encoding='utf-8')
print('SD_FIRMWARE_READY',hex(entry),hex(build_id))
