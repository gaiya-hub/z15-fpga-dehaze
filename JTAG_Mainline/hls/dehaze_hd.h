#ifndef DEHAZE_HD_H
#define DEHAZE_HD_H
#include <ap_int.h>
static const int HD_MAX_W=1920;
static const int HD_MAX_H=1280;
static const int HD_MAX_N=HD_MAX_W*HD_MAX_H;
static const int HD_K=15;
static const int HD_R=HD_K/2;
typedef ap_uint<32> rgb_pixel;
void dehaze_hd(const rgb_pixel *img,rgb_pixel *out,unsigned char *work,
               int width,int height,int air_r,int air_g,int air_b,
               int strength,int t_min,int lift,int mode,int strong_r,int strong_g,int strong_b,int blend,rgb_pixel *nnmem=nullptr);
#endif
