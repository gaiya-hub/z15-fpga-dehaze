#pragma once
typedef struct {
    nn_vector_row rows[16*FAST_MAX_W];
    uint32_t sum[FAST_MAX_W][4],square[FAST_MAX_W][4],base[FAST_MAX_W][4];
    uint16_t xout[FAST_MAX_W],xin[FAST_MAX_W];
} dual_context;
static dual_context dual_contexts[2] __attribute__((aligned(64)));
static void dual_make_row(const uint32_t *image,const uint32_t *base,int w,int y,const int initial_x[15],dual_context *ctx) {
    const uint32_t *in=image+y*w,*bs=base+y*w;nn_vector_row *row=ctx->rows+(y&15)*w;
    uint32x4_t sum=vdupq_n_u32(0),square=sum,base_sum=sum;
    for(int k=0;k<15;k++) {
        uint32x4_t a=nn_vec_rgba(in[initial_x[k]]),b=nn_vec_rgba(bs[initial_x[k]]);
        sum=vaddq_u32(sum,a);square=vaddq_u32(square,vmulq_u32(a,a));base_sum=vaddq_u32(base_sum,b);
    }
    for(int x=0;x<w;x++) {
        vst1q_u32(row[x].orig,vorrq_u32(sum,vshlq_n_u32(square,12)));
        vst1_u16(row[x].base,vmovn_u32(base_sum));
        uint32x4_t a=nn_vec_rgba(in[ctx->xout[x]]),b=nn_vec_rgba(in[ctx->xin[x]]);
        uint32x4_t ba=nn_vec_rgba(bs[ctx->xout[x]]),bb=nn_vec_rgba(bs[ctx->xin[x]]);
        sum=vaddq_u32(vsubq_u32(sum,a),b);
        square=vaddq_u32(vsubq_u32(square,vmulq_u32(a,a)),vmulq_u32(b,b));
        base_sum=vaddq_u32(vsubq_u32(base_sum,ba),bb);
    }
}
static void dual_range(const uint32_t *image,const uint32_t *base,uint32_t *nnmem,int w,int h,int y_begin,int y_end,dual_context *ctx) {
    int n=w*h,initial_x[15],initial_y[15];
    for(int k=0;k<15;k++){initial_x[k]=nn_reflect(k-7,w);initial_y[k]=nn_reflect(y_begin+k-7,h);}
    for(int x=0;x<w;x++){ctx->xout[x]=(uint16_t)nn_reflect(x-7,w);ctx->xin[x]=(uint16_t)nn_reflect(x+8,w);}
    int first=h-1,initial_end=0;for(int k=0;k<15;k++){if(initial_y[k]<first)first=initial_y[k];if(initial_y[k]>initial_end)initial_end=initial_y[k];}
    int prepared=first-1;
    while(prepared<initial_end){prepared++;dual_make_row(image,base,w,prepared,initial_x,ctx);}
    for(int x=0;x<w;x++) {
        uint32x4_t sum=vdupq_n_u32(0),square=sum,b=sum;
        for(int k=0;k<15;k++) {
            const nn_vector_row *r=ctx->rows+(initial_y[k]&15)*w+x;uint32x4_t p=vld1q_u32(r->orig);
            sum=vaddq_u32(sum,vandq_u32(p,vdupq_n_u32(4095)));square=vaddq_u32(square,vshrq_n_u32(p,12));b=vaddq_u32(b,vmovl_u16(vld1_u16(r->base)));
        }
        vst1q_u32(ctx->sum[x],sum);vst1q_u32(ctx->square[x],square);vst1q_u32(ctx->base[x],b);
    }
    for(int y=y_begin;y<y_end;y++) {
        int next_y=nn_reflect(y+8,h);while(prepared<next_y){prepared++;dual_make_row(image,base,w,prepared,initial_x,ctx);}
        const nn_vector_row *old=ctx->rows+(nn_reflect(y-7,h)&15)*w,*next=ctx->rows+(next_y&15)*w;
        for(int x=0;x<w;x++) {
            int index=y*w+x;uint32_t p=image[index],b=base[index],s=nnmem[256+index];
            uint16_t fields[20] __attribute__((aligned(16)));
            vst1_u16(fields,vmovn_u32(nn_vec_pixel_norm(nn_vec_rgba(p))));
            vst1_u16(fields+3,vmovn_u32(nn_vec_pixel_norm(nn_vec_rgba(b))));
            uint32x4_t strong=nn_vec_pixel_norm(nn_vec_rgba(s));
            vst1_u16(fields+6,vmovn_u32(strong));fields[19]=(uint16_t)vgetq_lane_u32(strong,3);
            uint32x4_t sums=vld1q_u32(ctx->sum[x]),squares=vld1q_u32(ctx->square[x]),bases=vld1q_u32(ctx->base[x]);
            uint32_t variance[4],roots[4];
            vst1q_u32(variance,vsubq_u32(vmulq_u32(squares,vdupq_n_u32(225)),vmulq_u32(sums,sums)));
            vst1_u16(fields+9,vmovn_u32(nn_vec_norm(sums)));vst1_u16(fields+12,vmovn_u32(nn_vec_norm(bases)));
            uint32x4_t a=vld1q_u32(old[x].orig),z=vld1q_u32(next[x].orig);
            sums=vaddq_u32(vsubq_u32(sums,vandq_u32(a,vdupq_n_u32(4095))),vandq_u32(z,vdupq_n_u32(4095)));
            squares=vaddq_u32(vsubq_u32(squares,vshrq_n_u32(a,12)),vshrq_n_u32(z,12));
            bases=vaddq_u32(vsubq_u32(bases,vmovl_u16(vld1_u16(old[x].base))),vmovl_u16(vld1_u16(next[x].base)));
            vst1q_u32(ctx->sum[x],sums);vst1q_u32(ctx->square[x],squares);vst1q_u32(ctx->base[x],bases);
            nn_fast_roots(variance,roots);vst1_u16(fields+15,vmovn_u32(nn_vec_norm(vld1q_u32(roots))));
            fields[18]=fields[19];fields[19]=0;
            uint16_t *q=(uint16_t*)(nnmem+256+n+10*index);
            vst1q_u16(q,vld1q_u16(fields));vst1q_u16(q+8,vld1q_u16(fields+8));vst1_u16(q+16,vld1_u16(fields+16));
            nnmem[256+index]=b;
        }
    }
}
