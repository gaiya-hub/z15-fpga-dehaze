/* Board ARM local statistics and independent integer CPU neural reference.
 * Production inference MAC and range guard are in FPGA PL. */
#pragma once
#include <stdint.h>
#include "neural_weights.h"
static int nn_reflect(int x,int n){if(n<=1)return 0;while(x<0||x>=n)x=x<0?-x:2*n-2-x;return x;}
static uint32_t nn_isqrt(uint32_t value){uint32_t r=0,bit=1U<<30;while(bit>value)bit>>=2;while(bit){if(value>=r+bit){value-=r+bit;r=(r>>1)+bit;}else r>>=1;bit>>=2;}return r;}
static void nn_field(uint32_t *nnmem,int count,int index,int field,unsigned value){
    int address=256+count+10*index+field/2;unsigned shift=(field&1)*16,mask=65535U<<shift;
    nnmem[address]=(nnmem[address]&~mask)|((value&65535U)<<shift);
}
static void nn_prepare(const uint32_t *image,const uint32_t *base,uint32_t *nnmem,int w,int h,uint32_t *temporary){
    int count=w*h;
    for(int i=0;i<count;i++){
        for(int j=0;j<10;j++)nnmem[256+count+10*i+j]=0;
        uint32_t p=image[i],b=base[i],s=nnmem[256+i];
        for(int c=0;c<3;c++){nn_field(nnmem,count,i,c,NN_NORM[(p>>(c*8))&255]);nn_field(nnmem,count,i,3+c,NN_NORM[(b>>(c*8))&255]);nn_field(nnmem,count,i,6+c,NN_NORM[(s>>(c*8))&255]);}
        nn_field(nnmem,count,i,18,NN_NORM[s>>24]);
    }
    for(int pass=0;pass<6;pass++){
        const uint32_t *source=pass<3?image:base;int channel=pass%3,shift=channel*8;
        for(int y=0;y<h;y++){
            uint32_t sum=0,square=0;
            for(int dx=-7;dx<=7;dx++){unsigned v=(source[y*w+nn_reflect(dx,w)]>>shift)&255;sum+=v;square+=v*v;}
            for(int x=0;x<w;x++){
                temporary[y*w+x]=sum|(square<<12);
                unsigned a=(source[y*w+nn_reflect(x-7,w)]>>shift)&255,b=(source[y*w+nn_reflect(x+8,w)]>>shift)&255;
                sum=sum-a+b;square=square-a*a+b*b;
            }
        }
        for(int x=0;x<w;x++){
            uint32_t sum=0,square=0;
            for(int dy=-7;dy<=7;dy++){uint32_t v=temporary[nn_reflect(dy,h)*w+x];sum+=v&4095;square+=v>>12;}
            for(int y=0;y<h;y++){
                int index=y*w+x;unsigned mean=((uint64_t)sum*16384+28687)/57375;
                nn_field(nnmem,count,index,(pass<3?9:12)+channel,mean);
                if(pass<3){uint32_t variance=(uint32_t)((uint64_t)square*225-(uint64_t)sum*sum);unsigned sd=((uint64_t)nn_isqrt(variance)*16384+28687)/57375;nn_field(nnmem,count,index,15+channel,sd);}
                uint32_t a=temporary[nn_reflect(y-7,h)*w+x],b=temporary[nn_reflect(y+8,h)*w+x];sum=sum-(a&4095)+(b&4095);square=square-(a>>12)+(b>>12);
            }
        }
    }
}
static int64_t nn_cpu_round(int64_t x,int bits){return x<0?-((-x+(1LL<<(bits-1)))>>bits):((x+(1LL<<(bits-1)))>>bits);}
static void nn_cpu_layer(const int32_t *input,int32_t *output,const int32_t *frame,int start,int rows,int cols,int offset,int bias_mode,int relu){
    for(int r=0;r<rows;r++){int64_t sum=(int64_t)(bias_mode?frame[r]:NN_B[start+r])*16384;for(int c=0;c<cols;c++)sum+=(int64_t)input[c]*NN_W[start+r][offset+c];int v=(int)nn_cpu_round(sum,14);output[r]=relu&&v<0?0:v;}
}
static int nn_clip(int n,int low,int high){return n<low?low:(n>high?high:n);}
static void nn_cpu_infer(uint32_t *out,const uint32_t *nnmem,int count){
    int32_t input[64],temp[64],frame[32];for(int j=0;j<22;j++)input[j]=(int32_t)nnmem[j];nn_cpu_layer(input,temp,frame,0,32,22,19,0,0);for(int j=0;j<32;j++)frame[j]=temp[j];
    for(int i=0;i<count;i++){
        for(int j=0;j<10;j++){uint32_t word=nnmem[256+count+10*i+j];input[2*j]=word&65535;input[2*j+1]=word>>16;}
        int base[3],detail[3];for(int c=0;c<3;c++){base[c]=input[3+c];detail[c]=nnmem[35]?input[6+c]-(int)nnmem[32+c]:input[c]-input[9+c];}
        nn_cpu_layer(input,temp,frame,0,32,19,0,1,1);for(int j=0;j<32;j++)input[j]=temp[j];nn_cpu_layer(input,temp,frame,32,24,32,0,0,1);for(int j=0;j<24;j++)input[j]=temp[j];nn_cpu_layer(input,temp,frame,56,4,24,0,0,0);
        int gain=nn_clip(temp[3],-8192,65536),base8[3],delta[3];unsigned alpha=65536;
        for(int c=0;c<3;c++){int v=nn_clip(base[c]+nn_clip(temp[c],-6144,6144)+(int)nn_cpu_round((int64_t)gain*detail[c],14),0,16384);int candidate=(v*255+8192)>>14;base8[c]=(base[c]*255+8192)>>14;delta[c]=candidate-base8[c];unsigned allowance=delta[c]<0?3*base8[c]:3*(255-base8[c]);if(allowance>256)allowance=256;unsigned ratio=delta[c]==0?65536:(unsigned)(((uint64_t)allowance*NN_RECIP[delta[c]<0?-delta[c]:delta[c]]+512)>>10);if(ratio<alpha)alpha=ratio;}
        unsigned weight=(alpha*3+2)>>2;uint32_t p=0xff000000U;
        for(int c=0;c<3;c++){int64_t value=((int64_t)base8[c]<<16)+(int64_t)delta[c]*weight;int floor=value>>16,rem=value&65535;int v=floor+(rem>32768||(rem==32768&&(floor&1)));p|=(uint32_t)nn_clip(v,0,255)<<(8*c);}out[i]=p;
    }
}
