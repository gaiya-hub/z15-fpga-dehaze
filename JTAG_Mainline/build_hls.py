from pathlib import Path
import subprocess,os,sys
ROOT=Path(__file__).parent
HLS=ROOT/'hls'
LOG=ROOT/'logs';LOG.mkdir(exist_ok=True)
from portable_settings import tool_root
VITIS=tool_root('vitis')
env=os.environ.copy();env['PATH']=str(VITIS/'tps/mingw/10.0.0/win64.o/nt/bin')+os.pathsep+env['PATH']
def run(name,args,environment=None):
    print('START',name,flush=True)
    with (LOG/(name+'.log')).open('w',encoding='utf-8') as log:
        p=subprocess.Popen([str(x) for x in args],cwd=HLS,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        for raw in p.stdout:
            line=raw.decode('utf-8',errors='replace');log.write(line);log.flush()
            if any(k in line for k in ['ERROR','WARNING','PASS','completed','Completed','FAIL','II Violation']):print(line.strip()[:600],flush=True)
        rc=p.wait()
    if rc:raise RuntimeError(name+' failed; inspect '+str(LOG/(name+'.log')))
    print('END',name,flush=True)
if '--resume' not in sys.argv:
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/generate_neural_fixtures.py')],check=True)
    run('native_compile',[VITIS/'tps/mingw/10.0.0/win64.o/nt/bin/g++.exe','-O2','-std=c++14','-Wno-unknown-pragmas',
                          '-I'+str(VITIS/'include'),'-static-libstdc++','-static-libgcc',HLS/'dehaze_hd.cpp',HLS/'dehaze_hd_tb.cpp','-o',ROOT/'hd_native.exe'],env)
    run('native_test',[ROOT/'hd_native.exe'],env)
if '--native-only' in sys.argv:raise SystemExit(0)
run('hls_synth',['cmd.exe','/d','/c',VITIS/'bin/v++.bat','--compile','--mode','hls','--config',HLS/'hls_config.cfg','--work_dir',ROOT/'hls_build'])
run('hls_package',['cmd.exe','/d','/c',VITIS/'bin/vitis-run.bat','--mode','hls','--package','--config',HLS/'hls_config.cfg','--work_dir',ROOT/'hls_build'])
print('HD_HLS_COMPLETE',flush=True)
