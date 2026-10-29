// Shared 32-lane Q14 engine; final four outputs reuse all lanes across input columns.
#include "neural_weights.h"
static long long nn_round(long long x,int bits) {
#pragma HLS INLINE
    return x<0?-((-x+(1LL<<(bits-1)))>>bits):((x+(1LL<<(bits-1)))>>bits);
}
typedef ap_uint<320> nn_feature_word;
static void nn_engine(const ap_int<19> input[32],ap_int<19> output[32],const ap_int<19> frame[32],
                      const short weights[32][192],const ap_int<18> biases[32][3],int mode) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input complete
#pragma HLS ARRAY_PARTITION variable=output complete
#pragma HLS ARRAY_PARTITION variable=frame complete
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=biases complete
    ap_int<40> accum[32];
#pragma HLS ARRAY_PARTITION variable=accum complete
    int group=mode<2?0:(mode==2?1:2);
    for(int lane=0;lane<32;lane++) {
#pragma HLS UNROLL
        int bias=mode==1?(int)frame[lane]:(int)biases[lane][group];
        accum[lane]=(ap_int<40>)bias*16384;
    }
    int columns=mode==0?22:(mode==1?19:(mode==2?32:3)),offset=mode==0?19:0;
    for(int col=0;col<columns;col++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=3 max=32
        for(int lane=0;lane<32;lane++) {
#pragma HLS UNROLL
            ap_int<19> value=input[col];
            if(mode==3) {int k=lane&7;value=col==0?input[k]:(col==1?input[8+k]:input[16+k]);}
            ap_int<16> weight=weights[lane][group*64+offset+col];
            accum[lane]+=value*weight;
        }
    }
    if(mode==3) {
        for(int row=0;row<4;row++) {
#pragma HLS UNROLL
            ap_int<44> a0=(ap_int<44>)accum[8*row]+accum[8*row+1],a1=(ap_int<44>)accum[8*row+2]+accum[8*row+3];
            ap_int<44> a2=(ap_int<44>)accum[8*row+4]+accum[8*row+5],a3=(ap_int<44>)accum[8*row+6]+accum[8*row+7];
            ap_int<44> sum=(a0+a1)+(a2+a3);int value=(int)nn_round(sum.to_int64(),14);output[row]=row==3?bounded(value,-8192,65536):bounded(value,-6144,6144);
        }
    } else {
        for(int row=0;row<32;row++) {
#pragma HLS UNROLL
            int value=(int)nn_round(accum[row].to_int64(),14);output[row]=mode && value<0?0:value;
        }
    }
}
static int nn_mul_q14(int a,int b) {
#pragma HLS INLINE off
    ap_int<33> product=(ap_int<18>)a*(ap_int<16>)b;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
    return (int)nn_round(product.to_int64(),14);
}
static unsigned nn_ratio(unsigned allowance,unsigned difference) {
#pragma HLS INLINE off
    if(difference==0)return 65536;
    ap_uint<34> product=(ap_uint<9>)allowance*(ap_uint<25>)NN_RECIP[difference];
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
    return (unsigned)((product+512)>>10);
}
static unsigned char nn_output_channel(int base,int delta,unsigned weight) {
#pragma HLS INLINE off
    ap_int<27> product=(ap_int<10>)delta*(ap_uint<17>)weight;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
    long long value=((long long)base<<16)+product.to_int64();
    int floor=(int)(value>>16),rem=(int)(value&65535);
    int rounded=floor+(rem>32768 || (rem==32768 && (floor&1)));
    return (unsigned char)bounded(rounded,0,255);
}

static void nn_infer_features(hls::stream<nn_feature_word>& input,hls::stream<rgb_pixel>& result,hls::stream<rgb_pixel>& original_base,int count,
                               const ap_int<19> globals[32],const short weights[32][192],const ap_int<18> biases[32][3]) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=nn_engine limit=1
#pragma HLS ARRAY_PARTITION variable=globals complete
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=biases complete
    ap_int<19> a[32],b[32],frame[32];
#pragma HLS ARRAY_PARTITION variable=a complete
#pragma HLS ARRAY_PARTITION variable=b complete
#pragma HLS ARRAY_PARTITION variable=frame complete
    for(int j=0;j<22;j++)a[j]=globals[j];
    nn_engine(a,b,frame,weights,biases,0);
    for(int j=0;j<32;j++)frame[j]=b[j];
    for(int index=0;index<count;index++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2457600
        nn_feature_word features=input.read();
        for(int j=0;j<19;j++) {
#pragma HLS UNROLL
            a[j]=(unsigned)features.range(16*j+15,16*j);
        }
        int base[3],detail[3];
#pragma HLS ARRAY_PARTITION variable=base complete
#pragma HLS ARRAY_PARTITION variable=detail complete
        for(int c=0;c<3;c++){base[c]=a[3+c];detail[c]=globals[25]?a[6+c]-globals[22+c]:a[c]-a[9+c];}
        nn_engine(a,b,frame,weights,biases,1);
        nn_engine(b,a,frame,weights,biases,2);
        nn_engine(a,b,frame,weights,biases,3);
        int gain=bounded(b[3],-8192,65536),delta[3],base8[3];unsigned alpha=65536;
#pragma HLS ARRAY_PARTITION variable=delta complete
#pragma HLS ARRAY_PARTITION variable=base8 complete
        for(int c=0;c<3;c++) {
#pragma HLS UNROLL
            int value=bounded(base[c]+bounded(b[c],-6144,6144)+nn_mul_q14(gain,detail[c]),0,16384);
            int candidate=(value*255+8192)>>14;base8[c]=(base[c]*255+8192)>>14;delta[c]=candidate-base8[c];
            int allowance=delta[c]<0?3*base8[c]:3*(255-base8[c]);if(allowance>256)allowance=256;
            unsigned ratio=nn_ratio(allowance,delta[c]<0?-delta[c]:delta[c]);if(ratio<alpha)alpha=ratio;
        }
        unsigned weight=(alpha*3+2)>>2;rgb_pixel output=0xff000000;
        for(int c=0;c<3;c++) {
#pragma HLS UNROLL
            output.range(c*8+7,c*8)=nn_output_channel(base8[c],delta[c],weight);
        }
        result.write(output);
        original_base.write(0xff000000U|((unsigned)base8[0])|((unsigned)base8[1]<<8)|((unsigned)base8[2]<<16));
    }
}
static void nn_read_features(const rgb_pixel *nnmem,int count,hls::stream<nn_feature_word>& output) {
    for(int i=0;i<count;i++) {
        nn_feature_word word=0;
        for(int j=0;j<10;j++) {
#pragma HLS PIPELINE II=1
            word.range(32*j+31,32*j)=nnmem[256+count+10*i+j];
        }
        output.write(word);
    }
}
static int nn_address(int y,int w,int x) {
#pragma HLS INLINE
    ap_uint<22> product=(ap_uint<11>)y*(ap_uint<11>)w;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
    return product.to_uint()+x;
}
static void nn_write_results(hls::stream<rgb_pixel>& result,hls::stream<rgb_pixel>& baseline,rgb_pixel *out,rgb_pixel *memory,int count) {
#pragma HLS INLINE off
    for(int i=0;i<count;i++) {
#pragma HLS PIPELINE II=2
        out[i]=result.read();memory[256+i]=baseline.read();
    }
}
static void nn_combined_dataflow(const rgb_pixel *image,rgb_pixel *out,rgb_pixel *memory,int w,int h,int count,int spatial,
                                const ap_int<19> globals[32],const short weights[32][192],const ap_int<18> biases[32][3]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS STABLE variable=weights
#pragma HLS STABLE variable=biases
#pragma HLS STABLE variable=globals
    hls::stream<nn_feature_word> fields("prefetched_features");
#pragma HLS STREAM variable=fields depth=16
#pragma HLS BIND_STORAGE variable=fields type=fifo impl=lutram
    hls::stream<rgb_pixel> result("neural_results"),baseline("raw_v2_audit");
#pragma HLS STREAM variable=result depth=32
#pragma HLS STREAM variable=baseline depth=32
#pragma HLS BIND_STORAGE variable=result type=fifo impl=lutram
#pragma HLS BIND_STORAGE variable=baseline type=fifo impl=lutram
    nn_read_features(memory,count,fields);
    nn_infer_features(fields,result,baseline,count,globals,weights,biases);
    nn_write_results(result,baseline,out,memory,count);
}
static void neural_correct(const rgb_pixel *image,rgb_pixel *out,rgb_pixel *nnmem,int width,int height,int spatial) {
    int count=nn_address(height,width,0);
#pragma HLS INLINE off
    unsigned tag=(unsigned)nnmem[22];
    if(tag!=NN_TAG){nnmem[23]=0;nnmem[24]=0xdead0003;return;}
    ap_int<19> globals[32];
#pragma HLS ARRAY_PARTITION variable=globals complete
    for(int j=0;j<22;j++)globals[j]=(int)(unsigned)nnmem[j];
    for(int j=0;j<4;j++)globals[22+j]=(int)(unsigned)nnmem[32+j];
    if(spatial){nnmem[23]=0;nnmem[24]=0xdead0004;return;}
    nn_combined_dataflow(image,out,nnmem,width,height,count,spatial,globals,NN_COMPACT_W,NN_COMPACT_B);
    nnmem[23]=tag;nnmem[24]=0x504c5633;
}
