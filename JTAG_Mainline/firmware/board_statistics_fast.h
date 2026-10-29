/* Exact Q14 statistics: 16-row ring, shared RGB scan, 32-bit bounds. At 640 width the intermediate is 200 KiB, fitting Cortex-A9 L2. */
#pragma once
#include <stdint.h>
#include <math.h>
#define FAST_MAX_W 1920
#define FAST_MAX_H 1280
typedef struct { uint32_t orig[3]; uint16_t base[3]; uint16_t padding; } nn_fast_horizontal;
static nn_fast_horizontal nn_fast_rows[FAST_MAX_W*16];
static uint32_t nn_fast_sum[3][FAST_MAX_W],nn_fast_square[3][FAST_MAX_W],nn_fast_base[3][FAST_MAX_W];
static uint16_t nn_fast_xout[FAST_MAX_W],nn_fast_xin[FAST_MAX_W];
static int nn_fast_yout[FAST_MAX_H],nn_fast_yin[FAST_MAX_H];
static __attribute__((unused)) uint32_t nn_fast_sqrt(uint32_t value) {
    uint32_t r=(uint32_t)__builtin_sqrt((double)value);
    while(r<65535U && (r+1U)*(r+1U)<=value)r++;
    while(r*r>value)r--;
    return r;
}
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
static inline __attribute__((always_inline)) void nn_fast_roots(const uint32_t values[4],uint32_t roots[4]) {
#ifdef __ARM_NEON
    uint32x4_t v=vld1q_u32(values);
    float32x4_t x=vcvtq_f32_u32(vmaxq_u32(v,vdupq_n_u32(1)));
    float32x4_t inverse=vrsqrteq_f32(x);
    inverse=vmulq_f32(inverse,vrsqrtsq_f32(vmulq_f32(x,inverse),inverse));
    inverse=vmulq_f32(inverse,vrsqrtsq_f32(vmulq_f32(x,inverse),inverse));
    vst1q_u32(roots,vcvtq_u32_f32(vmulq_f32(x,inverse)));
    for(int c=0;c<4;c++) {
        uint32_t a=roots[c]>65535U?65535U:roots[c],n=values[c];
        while(a<65535U && (a+1U)*(a+1U)<=n)a++;
        while(a*a>n)a--;
        roots[c]=a;
    }
#else
    for(int c=0;c<4;c++)roots[c]=nn_fast_sqrt(values[c]);
#endif
}
#ifdef NN_CHECK_NEON
static int nn_check_neon(void) {
    uint32_t state=8,values[4],roots[4];
    for(int group=0;group<25000;group++) {
        for(int c=0;c<4;c++){state=state*1664525U+1013904223U;values[c]=state;}
        if(group==0){values[0]=0;values[1]=1;values[2]=4294967295U;values[3]=4294836225U;}
        nn_fast_roots(values,roots);
        for(int c=0;c<4;c++)if(roots[c]!=nn_isqrt(values[c]))return 1;
    }
    return 0;
}
#endif
static void nn_fast_make_row(const uint32_t *image,const uint32_t *base,int w,int y,const int initial_x[15]) {
        const uint32_t *in=image+y*w,*bs=base+y*w;nn_fast_horizontal *row=nn_fast_rows+(y&15)*w;
        uint32_t sum[3]={0,0,0},square[3]={0,0,0},base_sum[3]={0,0,0};
        for(int k=0;k<15;k++){uint32_t p=in[initial_x[k]],b=bs[initial_x[k]];for(int c=0;c<3;c++){unsigned v=(p>>(8*c))&255;sum[c]+=v;square[c]+=v*v;base_sum[c]+=(b>>(8*c))&255;}}
        for(int x=0;x<w;x++) {
            uint32_t pa=in[nn_fast_xout[x]],pb=in[nn_fast_xin[x]],ba=bs[nn_fast_xout[x]],bb=bs[nn_fast_xin[x]];
            _Pragma("GCC unroll 3")
            for(int c=0;c<3;c++) {
                row[x].orig[c]=sum[c]|(square[c]<<12);row[x].base[c]=(uint16_t)base_sum[c];
                unsigned a=(pa>>(8*c))&255,b=(pb>>(8*c))&255;
                sum[c]=sum[c]-a+b;square[c]=square[c]-a*a+b*b;
                base_sum[c]=base_sum[c]-((ba>>(8*c))&255)+((bb>>(8*c))&255);
            }
        }
}
static __attribute__((unused)) void nn_prepare_fast_scalar(const uint32_t *image,const uint32_t *base,uint32_t *nnmem,int w,int h) {
    int count=w*h;int initial_x[15],initial_y[15];
    for(int k=0;k<15;k++){initial_x[k]=nn_reflect(k-7,w);initial_y[k]=nn_reflect(k-7,h)*w;}
    for(int x=0;x<w;x++){nn_fast_xout[x]=(uint16_t)nn_reflect(x-7,w);nn_fast_xin[x]=(uint16_t)nn_reflect(x+8,w);}
    for(int y=0;y<h;y++){nn_fast_yout[y]=nn_reflect(y-7,h)*w;nn_fast_yin[y]=nn_reflect(y+8,h)*w;}
    int prepared=-1,initial_end=h-1<7?h-1:7;
    while(prepared<initial_end){prepared++;nn_fast_make_row(image,base,w,prepared,initial_x);}
    for(int x=0;x<w;x++) {
        for(int c=0;c<3;c++){nn_fast_sum[c][x]=0;nn_fast_square[c][x]=0;nn_fast_base[c][x]=0;}
        for(int k=0;k<15;k++)_Pragma("GCC unroll 3")
            for(int c=0;c<3;c++) {
            uint32_t p=nn_fast_rows[initial_y[k]+x].orig[c];
            nn_fast_sum[c][x]+=p&4095U;nn_fast_square[c][x]+=p>>12;nn_fast_base[c][x]+=nn_fast_rows[initial_y[k]+x].base[c];
        }
    }
    for(int y=0;y<h;y++) {
        int next_y=nn_fast_yin[y]/w;
        while(prepared<next_y){prepared++;nn_fast_make_row(image,base,w,prepared,initial_x);}
        const nn_fast_horizontal *old=nn_fast_rows+((nn_fast_yout[y]/w)&15)*w,*next=nn_fast_rows+(next_y&15)*w;
        for(int x=0;x<w;x++) {
            int index=y*w+x;uint32_t p=image[index],b=base[index],s=nnmem[256+index];
            unsigned m[3],bm[3],sd[3];uint32_t variances[4],roots[4];
            for(int c=0;c<3;c++){uint32_t total=nn_fast_sum[c][x];variances[c]=nn_fast_square[c][x]*225U-total*total;}
            variances[3]=1;nn_fast_roots(variances,roots);
            _Pragma("GCC unroll 3")
            for(int c=0;c<3;c++) {
                uint32_t total=nn_fast_sum[c][x];
                m[c]=(total*16384U+28687U)/57375U;
                bm[c]=(nn_fast_base[c][x]*16384U+28687U)/57375U;
                /* Both products are <= 3,291,890,625, so unsigned 32-bit is exact. */

                sd[c]=(roots[c]*16384U+28687U)/57375U;
                uint32_t a=old[x].orig[c],z=next[x].orig[c];
                nn_fast_sum[c][x]=total-(a&4095U)+(z&4095U);
                nn_fast_square[c][x]=nn_fast_square[c][x]-(a>>12)+(z>>12);
                nn_fast_base[c][x]=nn_fast_base[c][x]-old[x].base[c]+next[x].base[c];
            }
            uint32_t *q=nnmem+256+count+10*index;
            q[0]=NN_NORM[p&255]|((uint32_t)NN_NORM[(p>>8)&255]<<16);
            q[1]=NN_NORM[(p>>16)&255]|((uint32_t)NN_NORM[b&255]<<16);
            q[2]=NN_NORM[(b>>8)&255]|((uint32_t)NN_NORM[(b>>16)&255]<<16);
            q[3]=NN_NORM[s&255]|((uint32_t)NN_NORM[(s>>8)&255]<<16);
            q[4]=NN_NORM[(s>>16)&255]|(m[0]<<16);
            q[5]=m[1]|(m[2]<<16);q[6]=bm[0]|(bm[1]<<16);q[7]=bm[2]|(sd[0]<<16);q[8]=sd[1]|(sd[2]<<16);
            q[9]=NN_NORM[s>>24];
            nnmem[256+index]=b; /* Retain exact raw V2 only after its auxiliary features are consumed. */
        }
    }
}

#include "board_statistics_vector.h"
