#include "xparameters.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xiltimer.h"
#include "dehaze_regs.h"
#include <string.h>
#define MAX_N (1920U*1280U)
#define INPUT 0x10000000U
#define OUTPUT 0x11000000U
#define WORK 0x12000000U
#define CPU_OUT 0x13000000U
#define MAILBOX 0x14000000U
#define ACCEL 0x40000000U
#define NNMEM 0x15000000U
#define CACHE_BYTES(n) (((unsigned)(n)+31U)&~31U)
#include "board_neural_v3.h"
#include "board_statistics_fast.h"
#include "dual_statistics.h"
#include "dual_runtime.h"
static uint32_t nn_temporary[MAX_N];
static u64 phase_dcp=0,phase_features=0,phase_neural=0,phase_neural_kernel=0;
#define READY 0x600D600DU
#define BAD 0xDEAD0001U
static unsigned char norm[MAX_N],horizontal[MAX_N],dark[MAX_N];
static int width,height,ar,ag,ab,omega,tmin,lift;
static int bound(int n,int lo,int hi){return n<lo?lo:(n>hi?hi:n);}
static void reg32(unsigned offset,u32 value){Xil_Out32(ACCEL+offset,value);}
static void pointer(unsigned offset,u32 value){reg32(offset,value);reg32(offset+4,0);}
static void mailbox64(unsigned index,u64 value){Xil_Out32(MAILBOX+index*4,(u32)value);Xil_Out32(MAILBOX+index*4+4,(u32)(value>>32));}
static void done(u32 status){Xil_Out32(MAILBOX,status);Xil_DCacheFlushRange(MAILBOX,256);}

/* Independent CPU reference, clipped separable 15x15 minima; compiled at -O2. */
__attribute__((noinline)) static void reference(void) {
    const u32 *in=(const u32 *)INPUT;u32 *out=(u32 *)CPU_OUT;
    int air[3]={ar,ag,ab},gain[3]={(65280+ar/2)/ar,(65280+ag/2)/ag,(65280+ab/2)/ab};
    unsigned short inverse[256];
    for(int d=0;d<256;d++){int t=255-((omega*d+127)>>8);if(t<tmin)t=tmin;inverse[d]=(1044480+t/2)/t;}
    for(int i=0;i<width*height;i++) {
        u32 p=in[i];int value=255;
        for(int c=0;c<3;c++){int q=bound((int)((((p>>(8*c))&255U)*gain[c]+128)>>8),0,255);if(q<value)value=q;}
        norm[i]=value;
    }
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        int value=255,lo=x>7?x-7:0,hi=x+7<width?x+7:width-1;
        for(int xx=lo;xx<=hi;xx++)if(norm[y*width+xx]<value)value=norm[y*width+xx];
        horizontal[y*width+x]=value;
    }
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        int value=255,lo=y>7?y-7:0,hi=y+7<height?y+7:height-1;
        for(int yy=lo;yy<=hi;yy++)if(horizontal[yy*width+x]<value)value=horizontal[yy*width+x];
        dark[y*width+x]=value;
    }
    for(int i=0;i<width*height;i++) {
        u32 p=in[i],q=0xff000000U;int inv=inverse[dark[i]];
        for(int c=0;c<3;c++) {
            int n=((int)((p>>(8*c))&255U)-air[c])*inv;
            int d=n>=0?(n+2048)/4096:-((-n+2048)/4096);
            int v=bound(air[c]+d,0,255);
            v=bound(v+(v*(255-v)*lift+32768)/65536,0,255);
            q|=(u32)v<<(8*c);
        }
        out[i]=q;
    }
    __asm__ volatile("" ::: "memory");
}
/* Independent ARM implementation: clipped separable rolling box sums.
 * HLS uses line buffers; this reference uses complete low-resolution arrays. */
#define LOW_N (480*320)
static int low_i[LOW_N],low_t[LOW_N],ca[LOW_N],cb[LOW_N],ma[LOW_N],mb[LOW_N];
static int sums[4][LOW_N],temporary[LOW_N];
static int sr,sg,sb,blend,mode,lw,lh,pl_statistics;
static long long round_signed(long long v,int bits){return v<0?-((-v+(1LL<<(bits-1)))>>bits):((v+(1LL<<(bits-1)))>>bits);}
static int box_field(int kind,int stage,int i){
    if(stage)return kind==0?ca[i]:cb[i];
    if(kind==0)return low_i[i];
    if(kind==1)return low_t[i];
    return kind==2?low_i[i]*low_i[i]:low_i[i]*low_t[i];
}
static void cpu_box(int kind,int stage) {
    for(int y=0;y<lh;y++) {
        int total=0;for(int x=0;x<lw && x<=5;x++)total+=box_field(kind,stage,y*lw+x);
        for(int x=0;x<lw;x++) {
            temporary[y*lw+x]=total;
            if(x>=5)total-=box_field(kind,stage,y*lw+x-5);
            if(x+6<lw)total+=box_field(kind,stage,y*lw+x+6);
        }
    }
    for(int x=0;x<lw;x++) {
        int total=0;for(int y=0;y<lh && y<=5;y++)total+=temporary[y*lw+x];
        for(int y=0;y<lh;y++) {
            sums[kind][y*lw+x]=total;
            if(y>=5)total-=temporary[(y-5)*lw+x];
            if(y+6<lh)total+=temporary[(y+6)*lw+x];
        }
    }
}
__attribute__((noinline)) static void fused_reference(void) {
    reference(); /* The complete weak branch is stored in CPU_OUT. */
    const u32 *input=(const u32*)INPUT;u32 *output=(u32*)CPU_OUT;
    int air[3]={sr,sg,sb},gain[3]={(65280+sr/2)/sr,(65280+sg/2)/sg,(65280+sb/2)/sb};
    int upper[3]={(65280+(255-sr)/2)/(255-sr),(65280+(255-sg)/2)/(255-sg),(65280+(255-sb)/2)/(255-sb)};
    for(int i=0;i<width*height;i++) {
        int value=255;u32 p=input[i];
        for(int c=0;c<3;c++){int v=bound((((p>>(8*c))&255)*gain[c]+128)>>8,0,255);if(v<value)value=v;}
        norm[i]=value;
    }
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        int v=255;for(int xx=bound(x-7,0,width-1);xx<=bound(x+7,0,width-1);xx++)if(norm[y*width+xx]<v)v=norm[y*width+xx];horizontal[y*width+x]=v;
    }
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        int v=255;for(int yy=bound(y-7,0,height-1);yy<=bound(y+7,0,height-1);yy++)if(horizontal[yy*width+x]<v)v=horizontal[yy*width+x];dark[y*width+x]=v;
    }
    lw=(width+3)/4;lh=(height+3)/4;
    for(int by=0;by<lh;by++)for(int bx=0;bx<lw;bx++) {
        int si=0,st=0,n=0;
        for(int y=by*4;y<height && y<by*4+4;y++)for(int x=bx*4;x<width && x<bx*4+4;x++) {
            int index=y*width+x;u32 p=input[index];
            si+=(77*(p&255)+150*((p>>8)&255)+29*((p>>16)&255)+128)>>8;
            st+=255-((230*dark[index]+127)>>8);n++;
        }
        low_i[by*lw+bx]=(si+n/2)/n;low_t[by*lw+bx]=(st+n/2)/n;
    }
    for(int c=0;c<4;c++)cpu_box(c,0);
    for(int y=0;y<lh;y++)for(int x=0;x<lw;x++) {
        int index=y*lw+x,n=(bound(x+5,0,lw-1)-bound(x-5,0,lw-1)+1)*(bound(y+5,0,lh-1)-bound(y-5,0,lh-1)+1);
        long long variance=(long long)sums[2][index]*n-(long long)sums[0][index]*sums[0][index];if(variance<0)variance=0;
        long long covariance=(long long)sums[3][index]*n-(long long)sums[0][index]*sums[1][index];
        ca[index]=(int)(covariance*4096/(variance+65LL*n*n));
        cb[index]=round_signed((((long long)sums[1][index]<<12)-(long long)ca[index]*sums[0][index])*(((1<<20)+n/2)/n),20);
    }
    for(int c=0;c<2;c++)cpu_box(c,1);
    for(int y=0;y<lh;y++)for(int x=0;x<lw;x++) {
        int index=y*lw+x,n=(bound(x+5,0,lw-1)-bound(x-5,0,lw-1)+1)*(bound(y+5,0,lh-1)-bound(y-5,0,lh-1)+1);
        int inverse=((1<<20)+n/2)/n;ma[index]=round_signed((long long)sums[0][index]*inverse,20);mb[index]=round_signed((long long)sums[1][index]*inverse,20);
    }
    unsigned short inverse[256];for(int t=38;t<256;t++)inverse[t]=(1044480+t/2)/t;
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        int py=(y*2-3)*32,ry=py<0?-((-py+255)/256):py/256,fy=py-ry*256;
        int px=(x*2-3)*32,rx=px<0?-((-px+255)/256):px/256,fx=px-rx*256;
        int x0=bound(rx,0,lw-1),x1=bound(rx+1,0,lw-1),y0=bound(ry,0,lh-1),y1=bound(ry+1,0,lh-1);
        long long coefficient[2];
        for(int c=0;c<2;c++) {
            int *a=c==0?ma:mb;
            long long top=(long long)a[y0*lw+x0]*(256-fx)+(long long)a[y0*lw+x1]*fx;
            long long bottom=(long long)a[y1*lw+x0]*(256-fx)+(long long)a[y1*lw+x1]*fx;
            coefficient[c]=round_signed(top*(256-fy)+bottom*fy,16);
        }
        int index=y*width+x;u32 p=input[index],q=0xff000000,weak=output[index];
        int gray=(77*(p&255)+150*((p>>8)&255)+29*((p>>16)&255)+128)>>8;
        int t=bound(round_signed(coefficient[0]*gray+coefficient[1],12),38,255);
        for(int c=0;c<3;c++) {
            int v=(p>>(8*c))&255,lower=255-((v*gain[c]+128)>>8),high=((v-air[c])*upper[c]+128)>>8;
            if(lower>t)t=lower;
            if(high>t)t=high;
        }
        t=bound(t,38,255);u32 neural_aux=(u32)t<<24;
        for(int c=0;c<3;c++) {
            int product=((int)((p>>(8*c))&255)-air[c])*inverse[t];
            int v=bound(air[c]+round_signed(product,12),0,255);neural_aux|=(u32)v<<(8*c);
            int fused=(((weak>>(8*c))&255)*(256-blend)+v*blend+128)>>8;q|=(u32)fused<<(8*c);
        }
        output[index]=q;if(mode==2)((u32*)NNMEM)[256+index]=neural_aux;
    }
    __asm__ volatile("" ::: "memory");
}

static void full_reference(void){fused_reference();nn_prepare((const u32*)INPUT,(const u32*)CPU_OUT,(u32*)NNMEM,width,height,nn_temporary);nn_cpu_infer((u32*)CPU_OUT,(const u32*)NNMEM,width*height);}

static int fpga_stage(int stage,u64 *total,u64 *kernel) {
    XTime a,b,c,d,now;
    XTime_GetTime(&a);
    /* INPUT, OUTPUT and WORK are never written by the CPU in production. */
    if(stage==3)Xil_DCacheFlushRange(NNMEM,4U*(256U+11U*width*height));
    else Xil_DCacheFlushRange(NNMEM,160);
    pointer(0x9c,NNMEM);
    pointer(XDEHAZE_HD_CONTROL_ADDR_IMG_DATA,INPUT);
    pointer(XDEHAZE_HD_CONTROL_ADDR_OUT_R_DATA,OUTPUT);
    pointer(XDEHAZE_HD_CONTROL_ADDR_WORK_DATA,WORK);
    reg32(XDEHAZE_HD_CONTROL_ADDR_WIDTH_DATA,width);
    reg32(XDEHAZE_HD_CONTROL_ADDR_HEIGHT_DATA,height);
    reg32(XDEHAZE_HD_CONTROL_ADDR_AIR_R_DATA,ar);
    reg32(XDEHAZE_HD_CONTROL_ADDR_AIR_G_DATA,ag);
    reg32(XDEHAZE_HD_CONTROL_ADDR_AIR_B_DATA,ab);
    reg32(XDEHAZE_HD_CONTROL_ADDR_STRENGTH_R_DATA,omega);
    reg32(XDEHAZE_HD_CONTROL_ADDR_T_MIN_DATA,tmin);
    reg32(XDEHAZE_HD_CONTROL_ADDR_LIFT_DATA,lift);
    reg32(0x74,stage);reg32(0x7c,sr);reg32(0x84,sg);reg32(0x8c,sb);reg32(0x94,blend);
    XTime_GetTime(&b);reg32(0,1);
    while(!(Xil_In32(ACCEL)&2U)) {
        XTime_GetTime(&now);
        if(now-b>120ULL*COUNTS_PER_SECOND)return 1;
    }
    XTime_GetTime(&c);
    Xil_DCacheInvalidateRange(OUTPUT,CACHE_BYTES(width*height*4));
    if(stage==2)Xil_DCacheInvalidateRange(NNMEM+1024,CACHE_BYTES(width*height*4));
    if(stage==3 || stage==4)Xil_DCacheInvalidateRange(NNMEM,160);
    XTime_GetTime(&d);*total=d-a;*kernel=c-b;return 0;
}
static int fpga(u64 *total,u64 *kernel){
    if(mode!=2)return fpga_stage(mode,total,kernel);
    XTime start,prepare_begin,prepare_end,finish;u64 dt,dk,nt,nk;
    XTime_GetTime(&start);
    if(fpga_stage(2,&dt,&dk))return 1;
    XTime_GetTime(&prepare_begin);
    if(!pl_statistics && dual_prepare((const u32*)INPUT,(const u32*)OUTPUT,(u32*)NNMEM,width,height))return 3;
    ((u32*)NNMEM)[23]=0;((u32*)NNMEM)[24]=0;
    XTime_GetTime(&prepare_end);
    if(fpga_stage(pl_statistics?4:3,&nt,&nk))return 1;
    if(((volatile u32*)NNMEM)[23]!=NN_TAG || ((volatile u32*)NNMEM)[24]!=0x504c5633)return 2;
    XTime_GetTime(&finish);*total=finish-start;*kernel=dk+nk;
    phase_dcp+=dt;phase_features+=pl_statistics?0:prepare_end-prepare_begin;phase_neural+=nt;phase_neural_kernel+=nk;
    return 0;
}
int main(void) {
    XTime a,b;u64 total=0,kernel=0,cpu=0,t,k;u32 errors=0;
    Xil_ICacheEnable();Xil_DCacheEnable();Xil_DCacheInvalidateRange(MAILBOX,256);
    width=Xil_In32(MAILBOX+4);height=Xil_In32(MAILBOX+8);
    ar=Xil_In32(MAILBOX+12);ag=Xil_In32(MAILBOX+16);ab=Xil_In32(MAILBOX+20);
    omega=Xil_In32(MAILBOX+24);tmin=Xil_In32(MAILBOX+28);lift=Xil_In32(MAILBOX+32);
    u32 repeats=Xil_In32(MAILBOX+36),run_cpu=Xil_In32(MAILBOX+40);
    sr=Xil_In32(MAILBOX+44);sg=Xil_In32(MAILBOX+48);sb=Xil_In32(MAILBOX+52);blend=Xil_In32(MAILBOX+56);mode=Xil_In32(MAILBOX+60);
    pl_statistics=mode==2?Xil_In32(MAILBOX+35*4):0;
    if(pl_statistics>1){done(BAD);return 1;}
    if(mode<0||mode>2||(mode && (sr<64||sr>250||sg<64||sg>250||sb<64||sb>250||blend<0||blend>256))){done(BAD);return 1;}
    if(mode){omega=160;tmin=96;lift=64;}
    if(width<1||width>1920||height<1||height>1280||ar<1||ar>255||ag<1||ag>255||ab<1||ab>255||omega<0||omega>243||tmin<32||tmin>255||lift<0||lift>128||repeats<1||repeats>10) {
        xil_printf("HD_FAIL invalid parameters\r\n");done(BAD);return 1;
    }
    if(mode==2)Xil_DCacheInvalidateRange(NNMEM,CACHE_BYTES(4U*(256U+11U*width*height)));
    Xil_DCacheInvalidateRange(INPUT,CACHE_BYTES(width*height*4));
    for(unsigned i=36;i<48;i++)Xil_Out32(MAILBOX+i*4,0);
    if(mode==2 && !pl_statistics && dual_setup()){done(BAD+16);return 1;}
    xil_printf("HD_RUN %dx%d RGB repeats=%d\r\n",width,height,repeats);
    if(run_cpu) {
        if(mode==2)full_reference();else if(mode)fused_reference();else reference();XTime_GetTime(&a);
        for(u32 r=0;r<repeats;r++){if(mode==2)full_reference();else if(mode)fused_reference();else reference();}
        XTime_GetTime(&b);cpu=(b-a)/repeats;
    }
    if(run_cpu && mode==2)Xil_DCacheFlushRange(NNMEM,4U*(256U+11U*width*height));
#ifdef NN_CHECK_NEON
    if(nn_check_neon() || nn_check_vector_statistics()){done(BAD+8);return 1;}
    Xil_Out32(MAILBOX+39*4,0x4e454f4e);
#endif
    if(fpga(&t,&k)){done(BAD+1);return 1;}
    phase_dcp=phase_features=phase_neural=phase_neural_kernel=0;
    for(u32 r=0;r<repeats;r++){if(fpga(&t,&k)){done(BAD+1);return 1;}total+=t;kernel+=k;}
    if(run_cpu)for(int i=0;i<width*height;i++)if(((u32 *)OUTPUT)[i]!=((u32 *)CPU_OUT)[i])errors++;
    mailbox64(25,phase_dcp/repeats);mailbox64(27,phase_features/repeats);mailbox64(29,phase_neural/repeats);mailbox64(32,phase_neural_kernel/repeats);
    Xil_Out32(MAILBOX+31*4,mode==2?NN_TAG:0);Xil_Out32(MAILBOX+34*4,mode==2?0x504c5633:0);
    mailbox64(16,cpu);mailbox64(18,total/repeats);mailbox64(20,kernel/repeats);
    Xil_Out32(MAILBOX+88,errors);Xil_Out32(MAILBOX+92,COUNTS_PER_SECOND);Xil_Out32(MAILBOX+96,width*height);
    if(run_cpu)Xil_DCacheFlushRange(CPU_OUT,width*height*4);
    xil_printf("HD_DONE pixels=%d mismatches=%d cpu_ticks=%d fpga_ticks=%d\r\n",width*height,errors,(u32)cpu,(u32)(total/repeats));
    done(errors?BAD+2:READY);
    for(;;)__asm__ volatile("wfi");
}
