"""Paired clear-image evaluation. Clear images never enter the FPGA algorithm."""
from pathlib import Path
import hashlib, math
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

from portable_settings import dataset_root
DATA_ROOT=dataset_root()
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def open_rgb(path):
    with Image.open(path) as loaded:
        return ImageOps.exif_transpose(loaded).convert('RGB')

def auto_ground_truth(source):
    """Match only explicit GT directory conventions; never use restored outputs."""
    source=Path(source)
    parent=source.parent
    name=parent.name.lower()
    if name=='natural_hazy':
        folders=[parent.with_name('natural_gt')];stems=[source.stem]
    elif name=='synthetic_hazy':
        folders=[parent.with_name('synthetic_gt')]
        stems=[source.stem[:-5] if source.stem.lower().endswith('_hazy') else source.stem]
    elif name=='hazy':
        folders=[parent.with_name('clear'),parent.with_name('gt')]
        stems=[source.stem,source.stem.split('_')[0]]
    else:
        return None
    for stem in dict.fromkeys(stems):
        matches=[]
        for folder in folders:
            if folder.is_dir():
                matches.extend(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.lower()==stem.lower() and p.resolve()!=source.resolve())
        if len(matches)==1:return matches[0]
        if len(matches)>1:return None
    return None

def demo_pair():
    hazy=DATA_ROOT/'natural_hazy/0001_0.8_0.2.jpg'
    gt=auto_ground_truth(hazy)
    return (hazy,gt) if hazy.is_file() and gt else (None,None)

def reference_image(source,ground_truth,processing_size):
    """Require identical original geometry; apply the same resampling as input."""
    if not ground_truth:return None,'缺少同场景无雾参考图，质量指标暂不可用。'
    ground_truth=Path(ground_truth)
    if not ground_truth.is_file():return None,'无雾参考图文件不存在。'
    if ground_truth.resolve()==Path(source).resolve():
        return None,'参考图与有雾输入是同一个文件，请选择对应的无雾图。'
    reference=open_rgb(ground_truth)
    with Image.open(source) as loaded:
        original_size=ImageOps.exif_transpose(loaded).size
    if reference.size!=original_size:
        return None,f'原始尺寸不一致：无雾图 {reference.width}×{reference.height}，有雾图 {original_size[0]}×{original_size[1]}；请使用像素对齐的成对图片，程序不拉伸配准。'
    if reference.size!=tuple(processing_size):
        reference=reference.resize(tuple(processing_size),Image.Resampling.LANCZOS)
    return reference,None

def image_scores(clear,image):
    a=np.asarray(clear,dtype=np.float64)
    b=np.asarray(image,dtype=np.float64)
    difference=a-b
    mse=float(np.mean(difference*difference))
    psnr='Infinity' if mse==0.0 else float(10*math.log10(255.0*255.0/mse))
    ssim=None
    if min(clear.size)>=11:
        ssim=float(structural_similarity(np.asarray(clear),np.asarray(image),data_range=255,
                   channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
    return {'psnr_db':psnr,'ssim':ssim,'mae':float(np.mean(np.abs(difference))),'mse':mse}

def evaluate_quality(source,input_image,output_image,ground_truth):
    clear,reason=reference_image(source,ground_truth,input_image.size)
    metadata={'available':False,'ground_truth':str(Path(ground_truth).resolve()) if ground_truth else None,
              'reason':reason,'evaluation_size':list(input_image.size),
              'method':'RGB uint8, data_range=255; SSIM Gaussian 11x11, sigma=1.5, population covariance; no automatic registration'}
    if clear is None:return metadata,None
    if output_image.size!=input_image.size:
        metadata['reason']='FPGA 输出尺寸与输入不一致，禁止计算质量指标。'
        return metadata,None
    before=image_scores(clear,input_image)
    after=image_scores(clear,output_image)
    delta={}
    for key in ('psnr_db','ssim','mae'):
        x,y=before[key],after[key]
        delta[key]=float(y-x) if isinstance(x,(int,float)) and isinstance(y,(int,float)) else None
    metadata.update(available=True,reason=None,input=before,fpga=after,delta=delta,
                    ground_truth_sha256=hashlib.sha256(Path(ground_truth).read_bytes()).hexdigest(),
                    pairing='dataset filename' if auto_ground_truth(source)==Path(ground_truth) else 'user-selected; user must confirm same scene and pixel alignment')
    if min(clear.size)<11:metadata['ssim_note']='SSIM 要求宽高至少 11；PSNR 与 MAE 仍可计算。'
    return metadata,clear


def attach_quality(result,run,input_image,output_image,ground_truth):
    """Attach paired-image metrics without altering any hardware measurements."""
    run=Path(run)
    quality,clear=evaluate_quality(result['source'],input_image,output_image,ground_truth)
    result['ground_truth']=str(Path(ground_truth).resolve()) if ground_truth else None
    result['quality']=quality
    if clear is not None:clear.save(run/'ground_truth.png')
    return clear

def display_number(value,precision=3):
    if value is None:return '—'
    if value=='Infinity':return '∞'
    return f'{value:.{precision}f}'

def export_comparison(run,result,input_image,output_image,clear=None):
    """Export the actual processed pixel size, rather than a tiny UI screenshot."""
    from PIL import ImageDraw,ImageFont
    run=Path(run);w,h=input_image.size
    if clear is None and result.get('quality',{}).get('available') and (run/'ground_truth.png').exists():
        clear=open_rgb(run/'ground_truth.png')
    font_size=max(16,min(34,w//24))
    font_path='C:/Windows/Fonts/msyh.ttc'
    font=ImageFont.truetype(font_path,font_size)
    small=ImageFont.truetype(font_path,max(14,font_size-3))
    margin=20;head=font_size*3+22;foot=font_size*4+30
    canvas=Image.new('RGB',(w*3+margin*4,h+head+foot),'#eef2f6')
    draw=ImageDraw.Draw(canvas)
    quality=result.get('quality',{})
    titles=['数据集无雾参考','有雾输入','FPGA 实际回读']
    for column,image in enumerate([clear,input_image,output_image]):
        x=margin+(w+margin)*column
        draw.text((x,12),titles[column],font=font,fill='#142b45')
        if column and quality.get('available'):
            scores=quality['input' if column==1 else 'fpga']
            draw.text((x,font_size+20),f'PSNR {display_number(scores["psnr_db"],2)} dB  SSIM {display_number(scores["ssim"],4)}',font=small,fill='#42556b')
        if image is None:
            draw.rectangle((x,head,x+w,head+h),fill='#d9e1eb')
            draw.text((x+12,head+24),'缺少配对参考图',font=small,fill='#42556b')
        else:
            if image.size!=(w,h):raise ValueError('导出图片必须采用相同评价尺寸。')
            canvas.paste(image,(x,head))
    y=head+h+15
    arm=display_number(result.get('arm_ms'),2)
    fpga=display_number(result.get('fpga_ms'),2)
    speed=display_number(result.get('speedup'),2)
    draw.text((margin,y),f'实际像素 {w}×{h} RGB   ARM {arm} ms   FPGA {fpga} ms   板上加速比 {speed}×',font=small,fill='#142b45')
    draw.text((margin,y+font_size+9),'计时不含 JTAG 传输；无雾参考只用于评价，不参与去雾计算。',font=small,fill='#42556b')
    draw.text((margin,y+(font_size+9)*2),'PSNR/SSIM 是画质评价；与独立算法参考零误差是计算正确性。',font=small,fill='#42556b')
    canvas.save(run/'three_view_comparison.png')
    return run/'three_view_comparison.png'
