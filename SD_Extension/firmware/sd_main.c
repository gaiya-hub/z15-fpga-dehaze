/* SD-data/JTAG-once entry. Never formats a card and never overwrites outputs. */
#define main jtag_main_unused
#include "main.c"
#undef main
#include "ff.h"
#include "sd_protocol.h"
#include "sleep.h"
#include <stdio.h>
static FATFS sd_fs;
static FIL sd_file;
static uint32_t sd_index[16],sd_header[16],sd_regs[16],sd_params[40],sd_result[SD_RESULT_WORDS];
static uint32_t sd_batch_summary[64];
static int sd_exact_read(const char *path,void *buffer,unsigned bytes){
    UINT got=0;FRESULT r=f_open(&sd_file,path,FA_READ);if(r!=FR_OK)return 100+(int)r;
    if(f_size(&sd_file)!=bytes){f_close(&sd_file);return 190;}
    r=f_read(&sd_file,buffer,bytes,&got);FRESULT end=f_close(&sd_file);
    return r==FR_OK && end==FR_OK && got==bytes?0:191;
}
static int sd_new_file(const char *path,const void *buffer,unsigned bytes){
    UINT sent=0;FRESULT r=f_open(&sd_file,path,FA_WRITE|FA_CREATE_NEW);if(r!=FR_OK)return 200+(int)r;
    r=f_write(&sd_file,buffer,bytes,&sent);FRESULT sync=f_sync(&sd_file),end=f_close(&sd_file);
    return r==FR_OK && sync==FR_OK && end==FR_OK && sent==bytes?0:291;
}
static int sd_read_job(unsigned index){
    char name[48];UINT got;snprintf(name,sizeof name,"0:/DEHAZE/IN/I%07u.BIN",index);
    FRESULT r=f_open(&sd_file,name,FA_READ);if(r!=FR_OK)return 300+(int)r;
    r=f_read(&sd_file,sd_header,sizeof sd_header,&got);
    if(r!=FR_OK || got!=sizeof sd_header || sd_header[0]!=SD_INPUT_MAGIC || sd_header[1]!=SD_VERSION || sd_header[15]!=sd_crc(sd_header,60))goto invalid;
    unsigned w=sd_header[2],h=sd_header[3],n=sd_header[4];
    if(w<1 || w>1920 || h<1 || h>1280 || n!=w*h || sd_header[8]!=NN_TAG || sd_header[11]!=index || sd_header[12]!=n*4 || sd_header[13]!=288+n*4 || f_size(&sd_file)!=288+n*4)goto invalid;
    r=f_read(&sd_file,sd_regs,sizeof sd_regs,&got);if(r!=FR_OK || got!=sizeof sd_regs || sd_crc(sd_regs,sizeof sd_regs)!=sd_header[7])goto invalid;
    r=f_read(&sd_file,sd_params,sizeof sd_params,&got);if(r!=FR_OK || got!=sizeof sd_params || sd_crc(sd_params,sizeof sd_params)!=sd_header[6])goto invalid;
    if(sd_regs[1]!=w || sd_regs[2]!=h || sd_regs[6]!=160 || sd_regs[7]!=96 || sd_regs[8]!=64 || sd_regs[9]<1 || sd_regs[9]>10 || sd_regs[10]!=0 || sd_regs[15]!=2 || sd_regs[9]!=sd_index[4] || sd_params[22]!=NN_TAG)goto invalid;
    for(int c=0;c<3;c++)if(sd_regs[3+c]<1 || sd_regs[3+c]>255 || sd_regs[11+c]<64 || sd_regs[11+c]>250)goto invalid;
    if(sd_regs[14]>256 || sd_params[35]!=0)goto invalid;
    r=f_read(&sd_file,(void*)INPUT,n*4,&got);if(r!=FR_OK || got!=n*4)goto invalid;
    if(f_close(&sd_file)!=FR_OK)return 392;
    if(sd_crc((const void*)INPUT,n*4)!=sd_header[5])return 393;
    width=w;height=h;ar=sd_regs[3];ag=sd_regs[4];ab=sd_regs[5];omega=160;tmin=96;lift=64;
    sr=sd_regs[11];sg=sd_regs[12];sb=sd_regs[13];blend=sd_regs[14];mode=2;pl_statistics=0;
    for(int i=0;i<16;i++)Xil_Out32(MAILBOX+i*4,sd_regs[i]);
    memcpy((void*)NNMEM,sd_params,sizeof sd_params);
    /* FatFs may copy an unaligned first sector using CPU stores. */
    Xil_DCacheFlushRange(INPUT,CACHE_BYTES(n*4));
    return 0;
invalid:f_close(&sd_file);return 391;
}
static void sd_ticks(unsigned slot,uint64_t value){sd_result[slot]=(uint32_t)value;sd_result[slot+1]=(uint32_t)(value>>32);}
static int sd_run_job(unsigned index,int first){
    XTime begin,read_end,compute_begin,compute_end,crc_begin,crc_end,write_begin,write_end;
    XTime_GetTime(&begin);int e=sd_read_job(index);XTime_GetTime(&read_end);if(e)return e;
    xil_printf("SD_JOB_BEGIN index=%u width=%u height=%u\r\n",index,width,height);
    char image_path[48],record_path[48];FILINFO info;
    snprintf(image_path,sizeof image_path,"0:/DEHAZE/OUT/O%07u.BIN",index);
    snprintf(record_path,sizeof record_path,"0:/DEHAZE/OUT/R%07u.BIN",index);
    if(f_stat(image_path,&info)!=FR_NO_FILE || f_stat(record_path,&info)!=FR_NO_FILE)return 401;
    if(first){
        for(unsigned i=36;i<48;i++)Xil_Out32(MAILBOX+i*4,0);
        if(dual_setup())return 402;
    }
    uint32_t previous_jobs=DUAL[3],repeats=sd_regs[9];u64 total=0,kernel=0,t,k;
    XTime_GetTime(&compute_begin);
    if(first && fpga(&t,&k))return 403;
    phase_dcp=phase_features=phase_neural=phase_neural_kernel=0;
    for(unsigned j=0;j<repeats;j++){if(fpga(&t,&k))return 404;total+=t;kernel+=k;}
    XTime_GetTime(&compute_end);
    mailbox64(25,phase_dcp/repeats);mailbox64(27,phase_features/repeats);mailbox64(29,phase_neural/repeats);mailbox64(32,phase_neural_kernel/repeats);
    mailbox64(16,0);mailbox64(18,total/repeats);mailbox64(20,kernel/repeats);
    Xil_Out32(MAILBOX+31*4,NN_TAG);Xil_Out32(MAILBOX+34*4,0x504c5633);
    Xil_Out32(MAILBOX+22*4,0);Xil_Out32(MAILBOX+23*4,COUNTS_PER_SECOND);Xil_Out32(MAILBOX+24*4,width*height);
    XTime_GetTime(&crc_begin);uint32_t output_crc=sd_crc((const void*)OUTPUT,width*height*4);XTime_GetTime(&crc_end);
    XTime_GetTime(&write_begin);e=sd_new_file(image_path,(const void*)OUTPUT,width*height*4);XTime_GetTime(&write_end);if(e)return e;
    memset(sd_result,0,sizeof sd_result);
    sd_result[0]=SD_RESULT_MAGIC;sd_result[1]=SD_VERSION;sd_result[2]=0;sd_result[3]=index;sd_result[4]=width;sd_result[5]=height;sd_result[6]=width*height;
    sd_result[7]=NN_TAG;sd_result[8]=sd_header[5];sd_result[9]=sd_header[6];sd_result[10]=sd_header[7];sd_result[11]=output_crc;
    sd_result[12]=COUNTS_PER_SECOND;sd_result[13]=repeats;sd_result[14]=SD_BUILD_ID;
    for(unsigned i=0;i<48;i++)sd_result[16+i]=Xil_In32(MAILBOX+i*4);
    sd_result[16]=READY;
    sd_ticks(64,read_end-begin);sd_ticks(66,write_end-write_begin);sd_ticks(68,crc_end-crc_begin);sd_ticks(70,write_end-begin);sd_ticks(72,compute_end-compute_begin);
    sd_result[74]=first?1:0;sd_result[75]=DUAL[3]-previous_jobs;sd_result[76]=DUAL[3];sd_result[77]=sd_index[15];
    sd_result[127]=sd_crc(sd_result,sizeof sd_result-4);
    e=sd_new_file(record_path,sd_result,sizeof sd_result);if(e)return e;
    xil_printf("SD_JOB_DONE index=%u compute_ticks=%u read_ticks=%u write_ticks=%u frequency=%u job_ticks_lo=%u job_ticks_hi=%u\r\n",index,(u32)(total/repeats),(u32)(read_end-begin),(u32)(write_end-write_begin),(u32)COUNTS_PER_SECOND,sd_result[70],sd_result[71]);
    return 0;
}
int main(void){
    Xil_ICacheEnable();Xil_DCacheEnable();sd_crc_init();
    /* xilffs uses usleep. Its first call resets the SDK global timer;
     * establish that timer epoch BEFORE any batch timestamp is sampled. */
    usleep(1);
    Xil_Out32(MAILBOX,0);xil_printf("SD_BATCH_V1 build=%08x\r\n",SD_BUILD_ID);
    XTime batch_begin,batch_end;XTime_GetTime(&batch_begin);
    FRESULT mounted=f_mount(&sd_fs,"0:/",1);
    if(mounted!=FR_OK){xil_printf("SD_ERROR mount=%u NO_FORMAT_ATTEMPTED\r\n",mounted);done(SD_MOUNT_ERROR);goto idle;}
    int e=sd_exact_read("0:/DEHAZE/INDEX.BIN",sd_index,sizeof sd_index);
    if(e || sd_index[0]!=SD_INDEX_MAGIC || sd_index[1]!=SD_VERSION || sd_index[2]<1 || sd_index[2]>100000 || sd_index[3]!=NN_TAG || sd_index[4]<1 || sd_index[4]>10 || sd_index[15]!=sd_crc(sd_index,60)){
        xil_printf("SD_ERROR index=%u\r\n",e);done(SD_TASK_ERROR);goto unmount;
    }
    xil_printf("SD_BATCH_BEGIN jobs=%u repeats=%u\r\n",sd_index[2],sd_index[4]);
    FRESULT dir=f_mkdir("0:/DEHAZE/OUT");if(dir!=FR_OK && dir!=FR_EXIST){done(SD_TASK_ERROR);goto unmount;}
    for(unsigned i=1;i<=sd_index[2];i++){
        e=sd_run_job(i,i==1);
        if(e){xil_printf("SD_ERROR job=%u code=%u\r\n",i,e);done(SD_JOB_ERROR);goto unmount;}
    }
    XTime_GetTime(&batch_end);memset(sd_batch_summary,0,sizeof sd_batch_summary);
    sd_batch_summary[0]=SD_BATCH_MAGIC;sd_batch_summary[1]=SD_VERSION;sd_batch_summary[2]=sd_index[2];sd_batch_summary[3]=0;
    sd_batch_summary[4]=NN_TAG;sd_batch_summary[5]=COUNTS_PER_SECOND;sd_batch_summary[6]=SD_BUILD_ID;sd_batch_summary[7]=sd_index[15];
    sd_batch_summary[8]=(uint32_t)(batch_end-batch_begin);sd_batch_summary[9]=(uint32_t)((batch_end-batch_begin)>>32);
    sd_batch_summary[63]=sd_crc(sd_batch_summary,sizeof sd_batch_summary-4);
    e=sd_new_file("0:/DEHAZE/OUT/BATCH.BIN",sd_batch_summary,sizeof sd_batch_summary);
    if(e){xil_printf("SD_ERROR batch_summary=%u\r\n",e);done(SD_JOB_ERROR);goto unmount;}
    f_mount(0,"0:/",0);xil_printf("SD_BATCH_DONE jobs=%u batch_ticks_lo=%u batch_ticks_hi=%u frequency=%u\r\n",sd_index[2],sd_batch_summary[8],sd_batch_summary[9],sd_batch_summary[5]);done(SD_BATCH_DONE);goto idle;
unmount:f_mount(0,"0:/",0);
idle:for(;;)__asm__ volatile("wfi");
}
