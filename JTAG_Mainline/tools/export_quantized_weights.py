"""Regenerate the two C headers from the published Q14 JSON; no training runtime needed."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from quantized_v3 import QuantizedV3
QuantizedV3(ROOT).packed_network()
model=json.loads((ROOT/'fixed_v3.json').read_text(encoding='utf-8'))
w=np.asarray(model['weights'],dtype='<i2');b=np.asarray(model['bias'],dtype='<i4')
tag=int(hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()[:8],16)
if tag!=model['tag']:raise ValueError('Weights/tag mismatch')
bank_w=np.zeros((16,5,64),np.int16);bank_b=np.zeros((16,5),np.int32)
for row in range(60):
 virtual=row if row<56 else row+8
 bank_w[virtual%16,virtual//16]=w[row];bank_b[virtual%16,virtual//16]=b[row]
def literal(a):
 if isinstance(a,list):return '{'+','.join(literal(v) for v in a)+'}'
 return str(int(a))
header='#pragma once\n#include <stdint.h>\n#define NN_TAG '+hex(tag)+'U\n#define NN_Q 14\n#define NN_ONE 16384\n'
for typ,name,value in [('short','NN_W',w.tolist()),('int','NN_B',b.tolist()),('unsigned short','NN_NORM',model['pixel_normalization']),('unsigned','NN_RECIP',model['guard_inverse_q24'])]:
 shape=np.shape(value);header+=f'static const {typ} {name}'+''.join(f'[{n}]' for n in shape)+'='+literal(value)+';\n'
cpu_header=header
for typ,name,value in [('short','NN_BANK_W',bank_w.tolist()),('int','NN_BANK_B',bank_b.tolist())]:
 header+=f'static const {typ} {name}'+''.join(f'[{n}]' for n in np.shape(value))+'='+literal(value)+';\n'
(ROOT/'hls/neural_weights.h').write_text(header,encoding='utf-8')
(ROOT/'firmware/neural_weights.h').write_text(cpu_header,encoding='utf-8')
print('Q14_HEADERS_READY',hex(tag))

compact_w=np.zeros((32,192),np.int16);compact_b=np.zeros((32,3),np.int32)
compact_w[:,:64]=w[:32];compact_b[:,0]=b[:32];compact_w[:24,64:128]=w[32:56];compact_b[:24,1]=b[32:56]
for row in range(4):
 compact_b[8*row,2]=b[56+row]
 for lane in range(8):compact_w[8*row+lane,128:131]=w[56+row,lane:24:8]
header+='\nstatic const short NN_COMPACT_W[32][192]='+literal(compact_w.tolist())+';\n'
header+='static const ap_int<18> NN_COMPACT_B[32][3]='+literal(compact_b.tolist())+';\n'
(ROOT/'hls/neural_weights.h').write_text(header,encoding='utf-8')
