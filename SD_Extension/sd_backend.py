"""One initial JTAG load; no image upload/readback in the SD processing path."""
from pathlib import Path
import hashlib,json,re
import backend
ROOT=Path(__file__).resolve().parent
_handle=None
def reserved():return _handle is not None
def release():
    global _handle
    if _handle is not None:
        import msvcrt
        _handle.seek(0);msvcrt.locking(_handle.fileno(),msvcrt.LK_UNLCK,1);_handle.close();_handle=None
def reserve():
    global _handle
    if _handle is not None:raise RuntimeError('本窗口已有SD任务运行。')
    import msvcrt
    from portable_settings import setting
    p=Path(setting('device_lock',str(ROOT/'device.lock')))
    handle=p.open('a+b')
    if handle.seek(0,2)==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:handle.close();raise RuntimeError('板卡正被其他演示或测试占用。')
    _handle=handle
def start(progress=lambda _:None):
    protocol=json.loads((ROOT/'board/sd_firmware_protocol.json').read_text(encoding='utf-8'))
    for name,key in [('sd_batch.elf','elf_sha256'),('design_1_wrapper.bit','bit_sha256')]:
        if hashlib.sha256((ROOT/'board'/name).read_bytes()).hexdigest()!=protocol[key]:raise RuntimeError('SD硬件/固件与启动信息不匹配，重新构建后再运行。')
    reserve()
    from dual_arm import reset_worker_commands
    commands=['connect -url tcp:127.0.0.1:3121','targets -set -filter {name =~ "APU*"}',
      'puts "PHASE: JTAG target verified; loading hardware"','rst -system','after 1000',
      'fpga -file '+backend.path_tcl(ROOT/'board/design_1_wrapper.bit'),
      'targets -set -filter {name =~ "ARM*#0"}','catch {stop}',
      'source '+backend.path_tcl(ROOT/'board/ps7_init.tcl'),'ps7_init','ps7_post_config',*reset_worker_commands(),
      'targets -set -filter {name =~ "ARM*#0"}','catch {stop}','rst -processor',
      'dow '+backend.path_tcl(ROOT/'board/sd_batch.elf'),
      'mwr 0x14000000 0','mwr 0x1f000000 0','mwr 0x1f000004 0',
      'puts "PHASE: SD mount and task validation"','con','targets -set -filter {name =~ "APU*"}',
      'set deadline [expr {[clock milliseconds]+120000}]','set ready 0','set status 0',
      'while {[clock milliseconds]<$deadline} {','after 100',
      'set status [lindex [mrd -address-space AP0 -value 0x14000000] 0]',
      'set ready [lindex [mrd -address-space AP0 -value 0x1f000000] 0]',
      'if {$ready==0x4455414c || $status!=0} {break}','}',
      'puts "SD_START_STATUS: [format 0x%08x $status]"',
      'if {$ready!=0x4455414c} {disconnect; exit}',
      'targets -set -filter {name =~ "ARM*#1"}',f'rwr pc {protocol["cpu1_entry"]:#x}','con',
      'puts "SD_BATCH_STARTED_NO_IMAGE_JTAG_TRANSFER"','disconnect','exit']
    p=ROOT/'outputs/start_sd.tcl';p.parent.mkdir(exist_ok=True);p.write_text('\n'.join(commands)+'\n',encoding='utf-8')
    try:text=backend._invoke_unlocked(p,progress,timeout=180)
    except Exception:
        if not p.with_suffix('.log').exists() or 'SD_BATCH_STARTED_NO_IMAGE_JTAG_TRANSFER' not in p.with_suffix('.log').read_text(encoding='utf-8'):release()
        raise
    if 'SD_BATCH_STARTED_NO_IMAGE_JTAG_TRANSFER' not in text:
        status=re.search(r'SD_START_STATUS: (0x[0-9a-fA-F]+)',text)
        detail=status.group(1) if status else 'timeout'
        reasons={'0x53440001':'SD卡挂载失败，请检查卡已插入且为支持的FAT文件系统；程序没有格式化卡。',
                 '0x53440002':'SD卡任务索引缺失或校验失败，请先导出DEHAZE任务。',
                 '0x53440003':'首张任务校验或已有结果检查失败，请查看串口错误码。'}
        release();raise RuntimeError(reasons.get(detail,'SD启动未完成：'+detail))
    progress('SD批处理已启动：图片在卡内读写，请通过串口查看完成状态。')
    return {'status':'started','firmware':protocol,'log':str(p.with_suffix('.log'))}
def query_status(progress=lambda _:None):
    p=ROOT/'outputs/query_sd.tcl';p.write_text('connect -url tcp:127.0.0.1:3121\ntargets -set -filter {name =~ "APU*"}\nputs "SD_STATE: [mrd -address-space AP0 -value 0x14000000]"\ndisconnect\nexit\n',encoding='utf-8')
    text=(backend._invoke_unlocked if reserved() else backend.invoke)(p,progress,timeout=30)
    m=re.search(r'SD_STATE: (\S+)',text)
    if not m:raise RuntimeError('没有读到SD状态。')
    code=int(m.group(1),0);finished=code in [0x53444f4b,0x53440001,0x53440002,0x53440003]
    if finished:release()
    description={0x53444f4b:'SD批处理已完成并关闭文件，可以断电后取卡。',0x53440001:'SD卡挂载失败；板卡已停止。',
                 0x53440002:'SD任务索引无效；板卡已停止。',0x53440003:'SD任务处理出错；板卡已停止，请查看串口错误码。'}.get(code,'尚未确认结束，状态：'+hex(code))
    return {'status':hex(code),'finished':finished,'description':description}
