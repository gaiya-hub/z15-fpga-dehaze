#include "dehaze_hd.h"
#include <hls_stream.h>

static int bounded(int value,int lo,int hi) {
#pragma HLS INLINE
    return value<lo?lo:(value>hi?hi:value);
}
static void load_rgb(const rgb_pixel *img,hls::stream<rgb_pixel>& pixels,int count) {
    for(int i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=2457600
        pixels.write(img[i]);
    }
}
static void store_dark(hls::stream<unsigned char>& dark,unsigned char *work,int count) {
    for(int i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=2457600
        work[i]=dark.read();
    }
}
static void filter_dark(hls::stream<rgb_pixel>& pixels,hls::stream<unsigned char>& dark,
                        int width,int height,int gain_r,int gain_g,int gain_b) {
    unsigned char lines[HD_K-1][HD_MAX_W];
#pragma HLS ARRAY_PARTITION variable=lines complete dim=1
#pragma HLS BIND_STORAGE variable=lines type=ram_s2p impl=bram
    unsigned char shift[HD_K];
#pragma HLS ARRAY_PARTITION variable=shift complete
    for(int y=0;y<height+HD_K-1;y++) {
#pragma HLS LOOP_TRIPCOUNT min=15 max=1294
        for(int j=0;j<HD_K;j++) {
#pragma HLS UNROLL
            shift[j]=255;
        }
        for(int x=0;x<width+HD_K-1;x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=15 max=1934
            unsigned char value=255;
            if(y>=HD_R && y<height+HD_R && x>=HD_R && x<width+HD_R) {
                rgb_pixel p=pixels.read();
                int r=bounded(((int)p.range(7,0)*gain_r+128)>>8,0,255);
                int g=bounded(((int)p.range(15,8)*gain_g+128)>>8,0,255);
                int b=bounded(((int)p.range(23,16)*gain_b+128)>>8,0,255);
                value=(unsigned char)(r<g?(r<b?r:b):(g<b?g:b));
            }
            for(int j=0;j<HD_K-1;j++) {
#pragma HLS UNROLL
                shift[j]=shift[j+1];
            }
            shift[HD_K-1]=value;
            unsigned char hmin=255;
            for(int j=0;j<HD_K;j++) {
#pragma HLS UNROLL
                if(shift[j]<hmin)hmin=shift[j];
            }
            if(x>=HD_K-1) {
                const int col=x-(HD_K-1);
                unsigned char vmin=hmin;
                unsigned char old[HD_K-1];
#pragma HLS ARRAY_PARTITION variable=old complete
                for(int j=0;j<HD_K-1;j++) {
#pragma HLS UNROLL
                    old[j]=lines[j][col];
                    if(old[j]<vmin)vmin=old[j];
                }
                for(int j=0;j<HD_K-2;j++) {
#pragma HLS UNROLL
                    lines[j][col]=old[j+1];
                }
                lines[HD_K-2][col]=hmin;
                // At this row every old line entry has been initialized by this frame.
                if(y>=HD_K-1)dark.write(vmin);
            }
        }
    }
}
static void make_dark(const rgb_pixel *img,unsigned char *work,int width,int height,int count,
                      int gain_r,int gain_g,int gain_b) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<rgb_pixel> pixels("dark_rgb");
    hls::stream<unsigned char> dark("dark_result");
#pragma HLS STREAM variable=pixels depth=128
#pragma HLS STREAM variable=dark depth=128
    load_rgb(img,pixels,count);
    filter_dark(pixels,dark,width,height,gain_r,gain_g,gain_b);
    store_dark(dark,work,count);
}
static void load_dark(const unsigned char *work,hls::stream<unsigned char>& dark,int count) {
    for(int i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=2457600
        dark.write(work[i]);
    }
}
static unsigned char recover_channel(unsigned char pixel,int air,unsigned short inv,int lift) {
#pragma HLS INLINE
    int product=((int)pixel-air)*(int)inv;
    int delta=product>=0?((product+2048)>>12):-((-product+2048)>>12);
    int result=bounded(air+delta,0,255);
    // Monotonic bounded brightness lift, with fixed black and white points.
    result+=((result*(255-result)*lift+32768)>>16);
    return (unsigned char)bounded(result,0,255);
}
static void recover_pixels(hls::stream<rgb_pixel>& pixels,hls::stream<unsigned char>& dark,
                            hls::stream<rgb_pixel>& output,int count,int ar,int ag,int ab,
                            int strength,int t_min,int lift) {
    unsigned short inverse[256];
    for(int d=0;d<256;d++) {
#pragma HLS PIPELINE II=1
        int t=255-((strength*d+127)>>8);
        if(t<t_min)t=t_min;
        inverse[d]=(unsigned short)((255*4096+t/2)/t);
    }
    for(int i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=2457600
        rgb_pixel p=pixels.read();
        unsigned short inv=inverse[dark.read()];
        rgb_pixel q=0xff000000;
        q.range(7,0)=recover_channel((unsigned char)p.range(7,0),ar,inv,lift);
        q.range(15,8)=recover_channel((unsigned char)p.range(15,8),ag,inv,lift);
        q.range(23,16)=recover_channel((unsigned char)p.range(23,16),ab,inv,lift);
        output.write(q);
    }
}
static void store_rgb(hls::stream<rgb_pixel>& pixels,rgb_pixel *out,int count) {
    for(int i=0;i<count;i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=2457600
        out[i]=pixels.read();
    }
}
static void make_result(const rgb_pixel *img,rgb_pixel *out,const unsigned char *work,
                        int count,int ar,int ag,int ab,int strength,int t_min,int lift) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<rgb_pixel> pixels("recover_rgb"),output("recover_output");
    hls::stream<unsigned char> dark("recover_dark");
#pragma HLS STREAM variable=pixels depth=128
#pragma HLS STREAM variable=output depth=128
#pragma HLS STREAM variable=dark depth=128
    load_rgb(img,pixels,count);
    load_dark(work,dark,count);
    recover_pixels(pixels,dark,output,count,ar,ag,ab,strength,t_min,lift);
    store_rgb(output,out,count);
}
// Integer guided transmission refinement. Work DDR: dark0, dark1, low I/t, a/b, mean a/b.
static long long signed_round(long long value,int bits) {
#pragma HLS INLINE
    return value<0?-((-value+(1LL<<(bits-1)))>>bits):((value+(1LL<<(bits-1)))>>bits);
}
static int read_i32(const unsigned char *memory,int address) {
#pragma HLS INLINE off
#pragma HLS LATENCY min=8
    unsigned value=0;
    for(int k=0;k<4;k++)value|=(unsigned)memory[address+k]<<(k*8);
    return (int)value;
}
static void write_i32(unsigned char *memory,int address,int value) {
#pragma HLS INLINE
    for(int k=0;k<4;k++)memory[address+k]=(unsigned char)((unsigned)value>>(k*8));
}
static void reduce_image(const rgb_pixel *img,unsigned char *work,int width,int height,int count) {
    int sums_i[HD_MAX_W/4],sums_t[HD_MAX_W/4];
#pragma HLS BIND_STORAGE variable=sums_i type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=sums_t type=ram_s2p impl=bram
    int lw=(width+3)/4,lh=(height+3)/4;
    for(int by=0;by<lh;by++) {
        for(int x=0;x<lw;x++) {
#pragma HLS PIPELINE II=1
            sums_i[x]=0;sums_t[x]=0;
        }
        for(int dy=0;dy<4;dy++)if(by*4+dy<height) {
            for(int x=0;x<width;x++) {
#pragma HLS PIPELINE II=2
                int i=(by*4+dy)*width+x;
                rgb_pixel p=img[i];
                int gray=(77*(int)p.range(7,0)+150*(int)p.range(15,8)+29*(int)p.range(23,16)+128)>>8;
                int t=255-((230*(int)work[count+i]+127)>>8);
                sums_i[x/4]+=gray;sums_t[x/4]+=t;
            }
        }
        for(int bx=0;bx<lw;bx++) {
#pragma HLS PIPELINE II=4
            int nx=width-bx*4;if(nx>4)nx=4;
            int ny=height-by*4;if(ny>4)ny=4;int n=nx*ny;
            int address=2*count+2*(by*lw+bx);
            work[address]=(sums_i[bx]+n/2)/n;work[address+1]=(sums_t[bx]+n/2)/n;
        }
    }
}
static int sum11(int value[11]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=value complete
#pragma HLS LATENCY min=5
    int a0=value[0]+value[1],a1=value[2]+value[3],a2=value[4]+value[5],a3=value[6]+value[7],a4=value[8]+value[9];
    int b0=a0+a1,b1=a2+a3,b2=a4+value[10];
    int c0=b0+b1,result=c0+b2;
    return result;
}
static int divide_coefficient(ap_int<44> numerator,ap_uint<32> denominator) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    ap_uint<44> magnitude=numerator<0?(ap_uint<44>)(-numerator):(ap_uint<44>)numerator;
    ap_uint<44> quotient=0;ap_uint<33> remainder=0;
    for(int bit=43;bit>=0;bit--) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL off
        remainder=(remainder<<1)|(ap_uint<1>)magnitude[bit];
        if(remainder>=denominator){remainder-=denominator;quotient[bit]=1;}
    }
    return numerator<0?-quotient.to_int():quotient.to_int();
}
static void fit_coefficients(int si,int st,int sii,int sit,int n,unsigned inverse,int &a,int &b) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off
    ap_uint<15> i=si,t=st;ap_uint<23> ii=sii,it=sit;ap_uint<7> count=n;
    ap_int<32> variance=(ap_int<32>)(ii*count)-(ap_int<32>)(i*i);
    if(variance<0)variance=0;
    ap_int<32> covariance=(ap_int<32>)(it*count)-(ap_int<32>)(i*t);
    ap_int<44> numerator=(ap_int<44>)covariance*4096;
    ap_int<32> denominator=variance+65*n*n;
    ap_int<16> slope=divide_coefficient(numerator,(ap_uint<32>)denominator);
    a=slope.to_int();
    b=(int)signed_round((((long long)st<<12)-(long long)a*si)*inverse,20);
}
typedef ap_int<128> box_word;
static void load_box_fields(const unsigned char *work,int offset,int width,int height,int stage,hls::stream<box_word>& data) {
    for(int index=0;index<width*height;index++) {
#pragma HLS PIPELINE II=16
        box_word word=0;
        if(stage==0) {
            int i=work[offset+2*index],t=work[offset+2*index+1];
            word.range(31,0)=i;word.range(63,32)=t;word.range(95,64)=i*i;word.range(127,96)=i*t;
        } else {
            word.range(31,0)=read_i32(work,offset+8*index);
            word.range(63,32)=read_i32(work,offset+8*index+4);
        }
        data.write(word);
    }
}
static void sum_box_fields(hls::stream<box_word>& data,hls::stream<box_word>& output,int width,int height) {
    int lines[4][10][HD_MAX_W/4];
#pragma HLS ARRAY_PARTITION variable=lines complete dim=1
#pragma HLS ARRAY_PARTITION variable=lines complete dim=2
#pragma HLS BIND_STORAGE variable=lines type=ram_s2p impl=bram
    int shift[4][11];
#pragma HLS ARRAY_PARTITION variable=shift complete
    for(int y=0;y<height+10;y++) {
        for(int c=0;c<4;c++) {
#pragma HLS UNROLL
            for(int k=0;k<11;k++) {
#pragma HLS UNROLL
                shift[c][k]=0;
            }
        }
        for(int x=0;x<width+10;x++) {
#pragma HLS PIPELINE II=1
            box_word word=0,result=0;
            if(y>=5 && y<height+5 && x>=5 && x<width+5)word=data.read();
            for(int c=0;c<4;c++) {
#pragma HLS UNROLL
                for(int k=0;k<10;k++) {
#pragma HLS UNROLL
                    shift[c][k]=shift[c][k+1];
                }
                ap_int<32> field=word.range(c*32+31,c*32);shift[c][10]=field.to_int();
                int horizontal=sum11(shift[c]);
                if(x>=10) {
                    int col=x-10,values[11];
#pragma HLS ARRAY_PARTITION variable=values complete
                    values[10]=horizontal;
                    for(int k=0;k<10;k++) {
#pragma HLS UNROLL
                        int previous=lines[c][k][col];values[k]=previous;
                        if(k>0)lines[c][k-1][col]=previous;
                    }
                    lines[c][9][col]=horizontal;result.range(c*32+31,c*32)=sum11(values);
                }
            }
            if(y>=10 && x>=10)output.write(result);
        }
    }
}
static void store_box_coefficients(hls::stream<box_word>& data,unsigned char *work,int offset,int width,int height,int stage) {
    unsigned inverse[122];
    for(int n=1;n<=121;n++)inverse[n]=((1<<20)+n/2)/n;
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
#pragma HLS PIPELINE off
        box_word word=data.read();
        ap_int<32> v0=word.range(31,0),v1=word.range(63,32),v2=word.range(95,64),v3=word.range(127,96);
        int nx=bounded(x+5,0,width-1)-bounded(x-5,0,width-1)+1;
        int ny=bounded(y+5,0,height-1)-bounded(y-5,0,height-1)+1,n=nx*ny,a,b;
        if(stage==0)fit_coefficients(v0.to_int(),v1.to_int(),v2.to_int(),v3.to_int(),n,inverse[n],a,b);
        else {
            a=(int)signed_round((long long)v0.to_int()*inverse[n],20);
            b=(int)signed_round((long long)v1.to_int()*inverse[n],20);
        }
        int address=offset+8*(y*width+x);write_i32(work,address,a);write_i32(work,address+4,b);
    }
}
static void guided_boxes(unsigned char *work,int input_offset,int output_offset,int width,int height,int stage) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<box_word> data("box_fields"),sums("box_sums");
#pragma HLS STREAM variable=data depth=32
#pragma HLS STREAM variable=sums depth=32
    load_box_fields(work,input_offset,width,height,stage,data);
    sum_box_fields(data,sums,width,height);
    store_box_coefficients(sums,work,output_offset,width,height,stage);
}

static void recover_row(hls::stream<rgb_pixel>& pixels,hls::stream<unsigned char>& dark,
                         hls::stream<rgb_pixel>& output,hls::stream<rgb_pixel>& auxiliary,const int cache[2][2][HD_MAX_W/4],
                         const unsigned short inverse[256],const unsigned short weak_inverse[256],
                         int width,int y,int ar,int ag,int ab,int sr,int sg,int sb,
                         int weight,const int gain[3],const int upper[3],int lift) {
#pragma HLS ARRAY_PARTITION variable=cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=cache complete dim=2
#pragma HLS ARRAY_PARTITION variable=gain complete
#pragma HLS ARRAY_PARTITION variable=upper complete
    int air[3]={sr,sg,sb};
#pragma HLS ARRAY_PARTITION variable=air complete
    int lw=(width+3)/4,py=(y*2-3)*32;
    int ry=py<0?-((-py+255)/256):py/256,fy=py-ry*256;
    for(int x=0;x<width;x++) {
#pragma HLS PIPELINE II=1
        rgb_pixel p=pixels.read(),q=0xff000000;
        unsigned short wi=weak_inverse[dark.read()];
        int posx=(x*2-3)*32,rawx=posx<0?-((-posx+255)/256):posx/256;
        int fx=posx-rawx*256,x0=bounded(rawx,0,lw-1),x1=bounded(rawx+1,0,lw-1);
        ap_uint<9> fxx=fx,ixx=256-fx,fyy=fy,iyy=256-fy;
        ap_int<25> at=(ap_int<16>)cache[0][0][x0]*ixx+(ap_int<16>)cache[0][0][x1]*fxx;
        ap_int<25> a_bottom=(ap_int<16>)cache[1][0][x0]*ixx+(ap_int<16>)cache[1][0][x1]*fxx;
        ap_int<35> av=(ap_int<35>)at*iyy+(ap_int<35>)a_bottom*fyy;
        ap_int<34> bt=(ap_int<25>)cache[0][1][x0]*ixx+(ap_int<25>)cache[0][1][x1]*fxx;
        ap_int<34> bb=(ap_int<25>)cache[1][1][x0]*ixx+(ap_int<25>)cache[1][1][x1]*fxx;
        ap_int<44> bv=(ap_int<44>)bt*iyy+(ap_int<44>)bb*fyy;
        ap_int<16> coeff_a=(int)signed_round(av.to_int64(),16);
        ap_int<25> coeff_b=(int)signed_round(bv.to_int64(),16);
        int pixel[3]={(int)p.range(7,0),(int)p.range(15,8),(int)p.range(23,16)};
#pragma HLS ARRAY_PARTITION variable=pixel complete
        int gray=(77*pixel[0]+150*pixel[1]+29*pixel[2]+128)>>8;
        ap_int<26> qt=coeff_a*(ap_uint<8>)gray+coeff_b;
        ap_uint<8> t=bounded((int)signed_round(qt.to_int64(),12),38,255);
        for(int c=0;c<3;c++) {
#pragma HLS UNROLL
            int lower=255-((pixel[c]*gain[c]+128)>>8);
            int high=((pixel[c]-air[c])*upper[c]+128)>>8;
            if(lower>(int)t)t=lower;if(high>(int)t)t=high;
        }
        unsigned short si=inverse[bounded(t,38,255)];
        unsigned char r=recover_channel(pixel[0],ar,wi,lift),g=recover_channel(pixel[1],ag,wi,lift),b=recover_channel(pixel[2],ab,wi,lift);
        rgb_pixel aux=0;aux.range(7,0)=recover_channel(pixel[0],sr,si,0);aux.range(15,8)=recover_channel(pixel[1],sg,si,0);aux.range(23,16)=recover_channel(pixel[2],sb,si,0);aux.range(31,24)=bounded(t,38,255);
        q.range(7,0)=((int)r*(256-weight)+(int)aux.range(7,0)*weight+128)>>8;
        q.range(15,8)=((int)g*(256-weight)+(int)aux.range(15,8)*weight+128)>>8;
        q.range(23,16)=((int)b*(256-weight)+(int)aux.range(23,16)*weight+128)>>8;
        output.write(q);auxiliary.write(aux);
    }
}
static void store_auxiliary(hls::stream<rgb_pixel>& input,rgb_pixel *nnmem,int count,int enabled){
    for(int i=0;i<count;i++){
#pragma HLS PIPELINE II=1
        rgb_pixel p=input.read();if(enabled)nnmem[i]=p;
    }
}
static void store_output_pair(hls::stream<rgb_pixel>& output,hls::stream<rgb_pixel>& auxiliary,
                              rgb_pixel *out,rgb_pixel *nnmem,int width,int export_aux) {
    for(int i=0;i<width;i++) {
#pragma HLS PIPELINE II=2
        rgb_pixel pixel=output.read(),aux=auxiliary.read();
        out[i]=pixel;
        if(export_aux)nnmem[i]=aux;
    }
}
static void result_row(const rgb_pixel *img,rgb_pixel *out,const unsigned char *work,
                        const int cache[2][2][HD_MAX_W/4],const unsigned short inverse[256],
                        const unsigned short weak_inverse[256],int width,int y,
                        int ar,int ag,int ab,int sr,int sg,int sb,int weight,
                        const int gain[3],const int upper[3],int lift,rgb_pixel *nnmem,int export_aux) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS STABLE variable=cache
#pragma HLS STABLE variable=inverse
#pragma HLS STABLE variable=weak_inverse
#pragma HLS STABLE variable=gain
#pragma HLS STABLE variable=upper
    hls::stream<rgb_pixel> pixels("row_rgb"),output("row_output"),auxiliary("row_auxiliary");
#pragma HLS STREAM variable=auxiliary depth=64
    hls::stream<unsigned char> dark("row_dark");
#pragma HLS STREAM variable=pixels depth=64
#pragma HLS STREAM variable=output depth=64
#pragma HLS STREAM variable=dark depth=64
    load_rgb(img,pixels,width);load_dark(work,dark,width);
    recover_row(pixels,dark,output,auxiliary,cache,inverse,weak_inverse,width,y,ar,ag,ab,sr,sg,sb,weight,gain,upper,lift);
    store_output_pair(output,auxiliary,out,nnmem,width,export_aux);
}
static void fused_recover(const rgb_pixel *img,rgb_pixel *out,const unsigned char *work,
                          int width,int height,int ar,int ag,int ab,int sr,int sg,int sb,int weight,
                          int enhanced,int omega,int floor,int lift,rgb_pixel *nnmem) {
    int gain[3]={(65280+sr/2)/sr,(65280+sg/2)/sg,(65280+sb/2)/sb};
    int upper[3]={(65280+(255-sr)/2)/(255-sr),(65280+(255-sg)/2)/(255-sg),(65280+(255-sb)/2)/(255-sb)};
#pragma HLS ARRAY_PARTITION variable=gain complete
#pragma HLS ARRAY_PARTITION variable=upper complete
    unsigned short inverse[256],weak_inverse[256];
    for(int t=1;t<256;t++) {
#pragma HLS PIPELINE II=1
        inverse[t]=(1044480+t/2)/t;
        int wt=255-((omega*t+127)>>8);if(wt<floor)wt=floor;
        weak_inverse[t]=(1044480+wt/2)/wt;
    }
    weak_inverse[0]=4096;
    int cache[2][2][HD_MAX_W/4];
#pragma HLS ARRAY_PARTITION variable=cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=cache complete dim=2
#pragma HLS BIND_STORAGE variable=cache type=ram_2p impl=bram
    int lw=(width+3)/4,lh=(height+3)/4;
    ap_uint<22> count=(ap_uint<11>)width*(ap_uint<11>)height;
#pragma HLS BIND_OP variable=count op=mul impl=fabric
    ap_uint<18> low_count=(ap_uint<9>)lw*(ap_uint<9>)lh;
#pragma HLS BIND_OP variable=low_count op=mul impl=fabric
    int coefficients=2*(int)count+10*(int)low_count,previous=-2;
    for(int y=0;y<height;y++) {
        int posy=(y*2-3)*32,rawy=posy<0?-((-posy+255)/256):posy/256;
        int y0=bounded(rawy,0,lh-1),y1=bounded(rawy+1,0,lh-1);
        if(rawy!=previous) {
            for(int xx=0;xx<lw;xx++) {
#pragma HLS PIPELINE II=16
                for(int row=0;row<2;row++)for(int c=0;c<2;c++)
                    cache[row][c][xx]=read_i32(work,coefficients+8*((row==0?y0:y1)*lw+xx)+4*c);
            }
            previous=rawy;
        }
        result_row(img+y*width,out+y*width,work+y*width,cache,inverse,weak_inverse,
                   width,y,ar,ag,ab,sr,sg,sb,weight,gain,upper,lift,nnmem?nnmem+256+y*width:nnmem,enhanced==2);
    }
}

#include "neural_pl_v3.inc"

void dehaze_hd(const rgb_pixel *img,rgb_pixel *out,unsigned char *work,
               int width,int height,int air_r,int air_g,int air_b,
               int strength,int t_min,int lift,int mode,int strong_r,int strong_g,int strong_b,int blend,rgb_pixel *nnmem) {
#pragma HLS INTERFACE m_axi port=img offset=slave bundle=gmem0 depth=2457600 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=2457600 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=work offset=slave bundle=gmem2 depth=8388608 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=nnmem offset=slave bundle=gmem1 depth=27037024 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=nnmem bundle=control offset=0x9c
#pragma HLS INTERFACE s_axilite port=img bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=work bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=air_r bundle=control
#pragma HLS INTERFACE s_axilite port=air_g bundle=control
#pragma HLS INTERFACE s_axilite port=air_b bundle=control
#pragma HLS INTERFACE s_axilite port=strength bundle=control
#pragma HLS INTERFACE s_axilite port=t_min bundle=control
#pragma HLS INTERFACE s_axilite port=lift bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=strong_r bundle=control
#pragma HLS INTERFACE s_axilite port=strong_g bundle=control
#pragma HLS INTERFACE s_axilite port=strong_b bundle=control
#pragma HLS INTERFACE s_axilite port=blend bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ALLOCATION function instances=make_dark limit=1
#pragma HLS ALLOCATION function instances=guided_boxes limit=1
    if(width<1 || height<1 || width>HD_MAX_W || height>HD_MAX_H)return;
    if(mode==3 || mode==4){neural_correct(img,out,nnmem,width,height,mode==4);return;}
    int ar=bounded(air_r,1,255),ag=bounded(air_g,1,255),ab=bounded(air_b,1,255);
    int omega=bounded(strength,0,243),floor=bounded(t_min,32,255),brightness=bounded(lift,0,128);
    int gr=(255*256+ar/2)/ar,gg=(255*256+ag/2)/ag,gb=(255*256+ab/2)/ab;
    int count=(ap_uint<11>)width*(ap_uint<11>)height;
    make_dark(img,work,width,height,count,gr,gg,gb);

    if(mode==0)make_result(img,out,work,count,ar,ag,ab,omega,floor,brightness);
    else {
        int sr=bounded(strong_r,64,250),sg=bounded(strong_g,64,250),sb=bounded(strong_b,64,250);
        make_dark(img,work+count,width,height,count,(65280+sr/2)/sr,(65280+sg/2)/sg,(65280+sb/2)/sb);
        reduce_image(img,work,width,height,count);
        int lw=(width+3)/4,lh=(height+3)/4;
        ap_uint<18> L=(ap_uint<9>)lw*(ap_uint<9>)lh;
#pragma HLS BIND_OP variable=L op=mul impl=fabric
        guided_boxes(work,2*count,2*count+2*L,lw,lh,0);
        guided_boxes(work,2*count+2*L,2*count+10*L,lw,lh,1);
    }
    if(mode!=0)fused_recover(img,out,work,width,height,ar,ag,ab,bounded(strong_r,64,250),bounded(strong_g,64,250),bounded(strong_b,64,250),bounded(blend,0,256),mode,omega,floor,brightness,nnmem);
}
