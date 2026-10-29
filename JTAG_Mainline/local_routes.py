"""Adapter to the user's classical adaptive pipeline; original sources stay unchanged."""
from pathlib import Path
import importlib.util,sys,hashlib
import numpy as np
from PIL import Image
from portable_settings import ffa_root
SOURCE=ffa_root()/'dehaze_test/adaptive_dehaze.py'
_processor=None

def adaptive_image(image):
    global _processor
    if _processor is None:
        previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
        try:
            spec=importlib.util.spec_from_file_location('_demo_user_adaptive',SOURCE)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:sys.dont_write_bytecode=previous
        _processor=module.AdaptiveDehazeProcessor(module.Config())
    output=_processor.process_single_image(np.asarray(image))['dehazed']
    if not np.isfinite(output).all():raise ValueError('自适应暗通道产生无效像素')
    return Image.fromarray((np.clip(output,0,1)*255).astype(np.uint8))

def adaptive_metadata():
    return {'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'trained_neural_network':False,'processing':'DCP + guided filter + handcrafted attention-like enhancement + adaptive fusion + gamma',
            'resizing':'none; batch entry point and its 800-pixel limit are bypassed'}
