#pragma once
#ifdef __ARM_NEON
typedef struct { uint32_t orig[4]; uint16_t base[4]; } nn_vector_row;
static nn_vector_row nn_vec_rows[16*FAST_MAX_W] __attribute__((aligned(16)));
static uint32_t nn_vec_sum[FAST_MAX_W][4] __attribute__((aligned(16)));
static uint32_t nn_vec_square[FAST_MAX_W][4] __attribute__((aligned(16)));
static uint32_t nn_vec_base[FAST_MAX_W][4] __attribute__((aligned(16)));
static inline uint32x4_t nn_vec_rgba(uint32_t pixel) {
    return vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(pixel)))));
}
static inline uint32x4_t nn_vec_norm(uint32x4_t total) {
    const uint32_t magic=(uint32_t)(((1ULL<<46)+57374)/57375);
    uint32x4_t n=vaddq_u32(vshlq_n_u32(total,14),vdupq_n_u32(28687));
    uint32x2_t lo=vshrn_n_u64(vmull_u32(vget_low_u32(n),vdup_n_u32(magic)),32);
    uint32x2_t hi=vshrn_n_u64(vmull_u32(vget_high_u32(n),vdup_n_u32(magic)),32);
    return vshrq_n_u32(vcombine_u32(lo,hi),14);
}
static inline uint32x4_t nn_vec_pixel_norm(uint32x4_t v) {
    return vaddq_u32(vshlq_n_u32(v,6),vshrq_n_u32(vaddq_u32(v,vdupq_n_u32(2)),2));
}
static void nn_vec_make_row(const uint32_t *image,const uint32_t *base,int w,int y,const int initial_x[15]) {
    const uint32_t *in=image+y*w,*bs=base+y*w;nn_vector_row *row=nn_vec_rows+(y&15)*w;
    uint32x4_t sum=vdupq_n_u32(0),square=sum,base_sum=sum;
    for(int k=0;k<15;k++) {
        uint32x4_t a=nn_vec_rgba(in[initial_x[k]]),b=nn_vec_rgba(bs[initial_x[k]]);
        sum=vaddq_u32(sum,a);square=vaddq_u32(square,vmulq_u32(a,a));base_sum=vaddq_u32(base_sum,b);
    }
    for(int x=0;x<w;x++) {
        vst1q_u32(row[x].orig,vorrq_u32(sum,vshlq_n_u32(square,12)));
        vst1_u16(row[x].base,vmovn_u32(base_sum));
        uint32x4_t a=nn_vec_rgba(in[nn_fast_xout[x]]),b=nn_vec_rgba(in[nn_fast_xin[x]]);
        uint32x4_t ba=nn_vec_rgba(bs[nn_fast_xout[x]]),bb=nn_vec_rgba(bs[nn_fast_xin[x]]);
        sum=vaddq_u32(vsubq_u32(sum,a),b);
        square=vaddq_u32(vsubq_u32(square,vmulq_u32(a,a)),vmulq_u32(b,b));
        base_sum=vaddq_u32(vsubq_u32(base_sum,ba),bb);
    }
}
static void nn_prepare_fast(const uint32_t *image,const uint32_t *base,uint32_t *nnmem,int w,int h) {
    int n=w*h,initial_x[15],initial_y[15];
    for(int k=0;k<15;k++){initial_x[k]=nn_reflect(k-7,w);initial_y[k]=nn_reflect(k-7,h);}
    for(int x=0;x<w;x++){nn_fast_xout[x]=(uint16_t)nn_reflect(x-7,w);nn_fast_xin[x]=(uint16_t)nn_reflect(x+8,w);}
    for(int y=0;y<h;y++){nn_fast_yout[y]=nn_reflect(y-7,h);nn_fast_yin[y]=nn_reflect(y+8,h);}
    int prepared=-1,initial_end=h-1<7?h-1:7;
    while(prepared<initial_end){prepared++;nn_vec_make_row(image,base,w,prepared,initial_x);}
    for(int x=0;x<w;x++) {
        uint32x4_t sum=vdupq_n_u32(0),square=sum,b=sum;
        for(int k=0;k<15;k++) {
            const nn_vector_row *r=nn_vec_rows+(initial_y[k]&15)*w+x;uint32x4_t p=vld1q_u32(r->orig);
            sum=vaddq_u32(sum,vandq_u32(p,vdupq_n_u32(4095)));square=vaddq_u32(square,vshrq_n_u32(p,12));b=vaddq_u32(b,vmovl_u16(vld1_u16(r->base)));
        }
        vst1q_u32(nn_vec_sum[x],sum);vst1q_u32(nn_vec_square[x],square);vst1q_u32(nn_vec_base[x],b);
    }
    for(int y=0;y<h;y++) {
        int next_y=nn_fast_yin[y];while(prepared<next_y){prepared++;nn_vec_make_row(image,base,w,prepared,initial_x);}
        const nn_vector_row *old=nn_vec_rows+(nn_fast_yout[y]&15)*w,*next=nn_vec_rows+(next_y&15)*w;
        for(int x=0;x<w;x++) {
            int index=y*w+x;uint32_t p=image[index],b=base[index],s=nnmem[256+index];
            uint16_t fields[20] __attribute__((aligned(16)));
            vst1_u16(fields,vmovn_u32(nn_vec_pixel_norm(nn_vec_rgba(p))));
            vst1_u16(fields+3,vmovn_u32(nn_vec_pixel_norm(nn_vec_rgba(b))));
            uint32x4_t strong=nn_vec_pixel_norm(nn_vec_rgba(s));
            vst1_u16(fields+6,vmovn_u32(strong));fields[19]=(uint16_t)vgetq_lane_u32(strong,3);
            uint32x4_t sums=vld1q_u32(nn_vec_sum[x]),squares=vld1q_u32(nn_vec_square[x]),bases=vld1q_u32(nn_vec_base[x]);
            uint32_t variance[4],roots[4];
            vst1q_u32(variance,vsubq_u32(vmulq_u32(squares,vdupq_n_u32(225)),vmulq_u32(sums,sums)));
            vst1_u16(fields+9,vmovn_u32(nn_vec_norm(sums)));vst1_u16(fields+12,vmovn_u32(nn_vec_norm(bases)));
            uint32x4_t a=vld1q_u32(old[x].orig),z=vld1q_u32(next[x].orig);
            sums=vaddq_u32(vsubq_u32(sums,vandq_u32(a,vdupq_n_u32(4095))),vandq_u32(z,vdupq_n_u32(4095)));
            squares=vaddq_u32(vsubq_u32(squares,vshrq_n_u32(a,12)),vshrq_n_u32(z,12));
            bases=vaddq_u32(vsubq_u32(bases,vmovl_u16(vld1_u16(old[x].base))),vmovl_u16(vld1_u16(next[x].base)));
            vst1q_u32(nn_vec_sum[x],sums);vst1q_u32(nn_vec_square[x],squares);vst1q_u32(nn_vec_base[x],bases);
            nn_fast_roots(variance,roots);vst1_u16(fields+15,vmovn_u32(nn_vec_norm(vld1q_u32(roots))));
            fields[18]=fields[19];fields[19]=0;
            uint16_t *q=(uint16_t*)(nnmem+256+n+10*index);
            vst1q_u16(q,vld1q_u16(fields));vst1q_u16(q+8,vld1q_u16(fields+8));vst1_u16(q+16,vld1_u16(fields+16));
            nnmem[256+index]=b;
        }
    }
}
#ifdef NN_CHECK_NEON
static int nn_check_vector_statistics(void) {
    for(uint32_t i=0;i<57376;i+=4) {
        uint32_t v[4]={i,i+1,i+2,i+3},got[4];vst1q_u32(got,nn_vec_norm(vld1q_u32(v)));
        for(int c=0;c<4;c++)if(got[c]!=(v[c]*16384U+28687U)/57375U)return 1;
    }
    for(uint32_t i=0;i<256;i+=4) {
        uint32_t v[4]={i,i+1,i+2,i+3},got[4];vst1q_u32(got,nn_vec_pixel_norm(vld1q_u32(v)));
        for(int c=0;c<4;c++)if(got[c]!=NN_NORM[v[c]])return 1;
    }
    return 0;
}
#endif
#else
static void nn_prepare_fast(const uint32_t *image,const uint32_t *base,uint32_t *nnmem,int w,int h) {nn_prepare_fast_scalar(image,base,nnmem,w,h);}
#endif
