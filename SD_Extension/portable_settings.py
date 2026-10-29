"""Portable configuration. Local machine paths stay out of source releases."""
from pathlib import Path
import json,os,shutil
ROOT=Path(__file__).resolve().parent
LOCAL=ROOT/'settings.local.json'
def setting(name,default=None):
    value=os.environ.get('DEHAZE_'+name.upper())
    if value:return value
    if LOCAL.exists():
        values=json.loads(LOCAL.read_text(encoding='utf-8'))
        if values.get(name):return values[name]
    return default

def tool_root(product,required=True):
    value=setting(product+'_root') or os.environ.get('XILINX_'+product.upper())
    if value:
        root=Path(value).expanduser()
        if root.is_dir():return root
        if required:raise FileNotFoundError(f'{product} 目录不存在：{root}')
    name='vivado.bat' if product=='vivado' else 'v++.bat'
    executable=shutil.which(name)
    if executable:return Path(executable).resolve().parent.parent
    if required:raise FileNotFoundError(f'请设置 DEHAZE_{product.upper()}_ROOT，或在 settings.local.json 配置 {product}_root。')
    return None

def tool_executable(name,required=False):
    product,relative={'xsdb':('vitis','bin/xsdb.bat'),'hw_server':('vivado','bin/hw_server.bat'),
                      'vivado':('vivado','bin/vivado.bat')}[name]
    root=tool_root(product,required=required)
    candidate=root/relative if root else Path(shutil.which(name+'.bat') or name+'.bat')
    if required and not candidate.is_file():raise FileNotFoundError(str(candidate))
    return candidate

def ffa_root():
    value=setting('ffa_root')
    if value:return Path(value).expanduser()
    for parent in [ROOT.parent,ROOT.parent.parent]:
        candidate=parent/'FFA-Net-master'
        if candidate.is_dir():return candidate
    return ROOT/'external/FFA-Net-master'

def dataset_root():
    value=setting('dataset_root')
    return Path(value).expanduser() if value else ffa_root()/'数据集/data'
