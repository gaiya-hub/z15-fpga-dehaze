"""CPU1 launch protocol for the single-ELF Zynq-7000 AMP firmware."""
from pathlib import Path
import hashlib,json

def load_protocol(root):
    root=Path(root);path=root/'board/firmware_protocol.json'
    if not path.is_file():raise RuntimeError('缺少双核固件启动信息，请重新运行 build_firmware.py。')
    data=json.loads(path.read_text(encoding='utf-8'))
    digest=hashlib.sha256((root/'board/demo.elf').read_bytes()).hexdigest()
    if data.get('version')!=1 or data.get('elf_sha256')!=digest:
        raise RuntimeError('双核启动信息与固件不匹配，请重新构建固件后初始化板卡。')
    entry=data.get('cpu1_entry')
    if not isinstance(entry,int) or not 0x100000<=entry<0x10000000 or entry%4:
        raise RuntimeError('第二核入口地址无效，请重新构建固件。')
    return data

def reset_worker_commands():
    return ['targets -set -filter {name =~ "ARM*#1"}','catch {stop}','rst -processor']

def start_commands(protocol):
    return [
      'mwr 0x1f000000 0','mwr 0x1f000004 0',
      'set dual_boot_begin [clock milliseconds]','con',
      'targets -set -filter {name =~ "APU*"}',
      'set boot_deadline [expr {[clock milliseconds]+5000}]','set boot_ready 0',
      'while {[clock milliseconds]<$boot_deadline} {','after 20',
      'set boot_ready [lindex [mrd -address-space AP0 -value 0x1f000000] 0]',
      'if {$boot_ready==0x4455414c} {break}','}',
      'if {$boot_ready!=0x4455414c} {disconnect; error "CPU0 dual boot descriptor timeout; initialize board and retry"}',
      'targets -set -filter {name =~ "ARM*#1"}',f'rwr pc {protocol["cpu1_entry"]:#x}','con',
      'targets -set -filter {name =~ "APU*"}',
      'set worker_deadline [expr {[clock milliseconds]+3000}]','set worker_ready 0',
      'while {[clock milliseconds]<$worker_deadline} {','after 10',
      'set worker_ready [lindex [mrd -address-space AP0 -value 0x1f000004] 0]',
      'if {$worker_ready==0x43505531} {break}','}',
      'if {$worker_ready!=0x43505531} {targets -set -filter {name =~ "ARM*#1"}; catch {stop}; disconnect; error "CPU1 worker ready timeout; initialize board and retry"}',
      'puts "DUAL_BOOTSTRAP_MS: [expr {[clock milliseconds]-$dual_boot_begin}]"',
      'puts "DUAL_WORKER_STARTED"']

def diagnostics(words,frequency,repeats):
    if words[46]!=0x4455414c or words[47]!=1 or words[36]!=1:
        raise RuntimeError('未收到匹配的双核固件标记，请重新构建并初始化板卡。')
    active=bool(words[37])
    if active and (words[38]<=0 or words[39]!=repeats+1):
        raise RuntimeError('第二核作业数或图像划分校验失败，不能作为有效双核处理结果。')
    return {'dual_arm':True,'dual_active':active,'arm_cores_used':2 if active else 1,
      'cpu1_core_id':words[36],'dual_split_y':words[38],
      'cpu1_completed_jobs':words[39] if active else 0,
      'dual_startup_ms':(words[40]+(words[43]<<32))/frequency*1000,
      'cpu1_last_feature_ms':(words[41]+(words[42]<<32))/frequency*1000 if active else None,
      'firmware_protocol_version':words[47],
      'local_feature_location':'Zynq ARM0 + ARM1' if active else 'Zynq ARM0 (alignment/small-image fallback)'}
