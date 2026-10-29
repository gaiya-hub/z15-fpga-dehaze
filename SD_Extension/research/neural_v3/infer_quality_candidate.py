"""Run the experimental CPU candidate. This command does not operate an FPGA."""
from pathlib import Path
import sys,json,time,hashlib,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
import torch
from model import load_image
from detail_dehaze import DetailDehazer
from safe_detail_v3 import protect
from evaluation import reference_image,image_scores
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path);parser.add_argument('--edge',type=int,default=640)
    args=parser.parse_args()
    if not 32<=args.edge<=1920:parser.error('--edge must be 32..1920')
    source=args.image.resolve();destination=args.output.resolve()
    if source==destination:parser.error('Output must not overwrite input')
    if args.ground_truth and args.ground_truth.resolve()==destination:parser.error('Output must not overwrite the clear reference')
    torch.set_num_threads(min(6,torch.get_num_threads()))
    state=json.loads((ROOT/'candidate.json').read_text(encoding='utf-8'))
    checkpoint=ROOT/'model.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=state['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
    image,original=load_image(source,args.edge)
    model=DetailDehazer(checkpoint,json.loads((ROOT/'frozen_model.json').read_text(encoding='utf-8')))
    begin=time.perf_counter();candidate,base=model.process(image);output=protect(base,candidate,state['guard']);ms=(time.perf_counter()-begin)*1000
    destination.parent.mkdir(parents=True,exist_ok=True);output.save(destination)
    data={'status':'experimental CPU candidate, not board deployed','hardware_used':False,'target_ssim_achieved':False,
          'source':str(source),'output':str(destination),'original_size':list(original),'processing_size':list(image.size),'cpu_single_run_ms':ms,
          'timing_scope':'single run after model load; DCP parameters, DCP pixels, neural correction and guard; excludes file I/O; not a benchmark speedup',
          'checkpoint_sha256':state['checkpoint_sha256'],'output_rgb_sha256':hashlib.sha256(output.tobytes()).hexdigest(),'ground_truth_used_at_inference':False}
    clear,reason=reference_image(source,args.ground_truth,image.size)
    data['quality_reason']=reason;data['scores']=image_scores(clear,output) if clear else None
    destination.with_suffix(destination.suffix+'.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False))
if __name__=='__main__':main()
