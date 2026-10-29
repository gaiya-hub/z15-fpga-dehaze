#pragma once
#include "xil_mmu.h"
/* One ELF, two private statistics contexts/stacks. CPU1 never reinitializes L2/BSS. */
#define DUAL ((volatile uint32_t*)0x1f000000U)
static unsigned char dual_stack[65536] __attribute__((aligned(64),used));
static void dual_barrier(void){__asm__ volatile("dsb sy" ::: "memory");}
static void dual_l1(uintptr_t address,unsigned length,int clean){
    uintptr_t end=address+length;
    for(address&=~31U;address<end;address+=32){
        if(clean)__asm__ volatile("mcr p15,0,%0,c7,c10,1"::"r"(address):"memory");
        else __asm__ volatile("mcr p15,0,%0,c7,c6,1"::"r"(address):"memory");
    }
    dual_barrier();
}
void dual_worker_main(void) __attribute__((used,noinline));
void dual_worker_main(void){
    uint32_t id;__asm__ volatile("mrc p15,0,%0,c0,c0,5":"=r"(id));
    DUAL[20]=id&3;DUAL[1]=0x43505531;dual_barrier();
    uint32_t previous=0;
    for(;;){
        while(DUAL[2]==previous)__asm__ volatile("wfe" ::: "memory");
        dual_barrier();uint32_t seq=DUAL[2];int w=DUAL[4],h=DUAL[5],split=DUAL[6],n=w*h,start=split*w;
        dual_l1(INPUT,n*4,0);dual_l1(OUTPUT,n*4,0);dual_l1(NNMEM+1024+start*4,(n-start)*4,0);
        XTime a,b;XTime_GetTime(&a);
        dual_range((const u32*)INPUT,(const u32*)OUTPUT,(u32*)NNMEM,w,h,split,h,&dual_contexts[1]);
        dual_l1(NNMEM+1024+start*4,(n-start)*4,1);
        dual_l1(NNMEM+1024+n*4+start*40,(n-start)*40,1);
        XTime_GetTime(&b);DUAL[21]=(uint32_t)(b-a);DUAL[22]=(uint32_t)((b-a)>>32);
        previous=seq;dual_barrier();DUAL[3]=seq;dual_barrier();__asm__ volatile("sev");
    }
}
#define DUAL_READ(slot,reg) __asm__ volatile("mrc p15,0,%0," reg:"=r"(value));DUAL[slot]=value
static int dual_setup(void){
    Xil_SetTlbAttributes(0x1f000000,NORM_NONCACHE);
    for(int i=0;i<32;i++)DUAL[i]=0;
    uint32_t value;
    DUAL_READ(8,"c2,c0,0");DUAL_READ(9,"c2,c0,1");DUAL_READ(10,"c2,c0,2");DUAL_READ(11,"c3,c0,0");
    DUAL_READ(12,"c1,c0,0");DUAL_READ(13,"c1,c0,1");DUAL_READ(14,"c1,c0,2");DUAL_READ(15,"c12,c0,0");
    __asm__ volatile("vmrs %0,fpscr":"=r"(value));DUAL[16]=value;
    DUAL[17]=(uintptr_t)(dual_stack+sizeof dual_stack);
    /* Reject unanticipated CPU0 non-SMP startup instead of silently racing caches. */
    if(!(DUAL[13]&64))return 1;
    dual_barrier();DUAL[0]=0x4455414c;dual_barrier();
    XTime a,b;XTime_GetTime(&a);
    while(DUAL[1]!=0x43505531){XTime_GetTime(&b);if(b-a>10ULL*COUNTS_PER_SECOND)return 2;}
    XTime_GetTime(&b);Xil_Out32(MAILBOX+36*4,DUAL[20]);Xil_Out32(MAILBOX+40*4,(uint32_t)(b-a));
    Xil_Out32(MAILBOX+43*4,(uint32_t)((b-a)>>32));
    Xil_Out32(MAILBOX+46*4,0x4455414c);Xil_Out32(MAILBOX+47*4,1);
    return DUAL[20]!=1;
}
static int dual_prepare(const u32 *image,const u32 *base,u32 *mem,int w,int h){
    int split=0;
    if((w*h)%8==0 && h>=16){
        for(int delta=0;delta<h/2;delta++){
            int row=h/2-delta;
            if(row>0 && (row*w)%8==0){split=row;break;}
            row=h/2+delta;
            if(row<h && (row*w)%8==0){split=row;break;}
        }
    }
    Xil_Out32(MAILBOX+37*4,split!=0);Xil_Out32(MAILBOX+38*4,split);
    if(!split){nn_prepare_fast(image,base,mem,w,h);return 0;}
    DUAL[4]=w;DUAL[5]=h;DUAL[6]=split;dual_barrier();uint32_t seq=DUAL[2]+1;DUAL[2]=seq;dual_barrier();__asm__ volatile("sev");
    dual_range(image,base,mem,w,h,0,split,&dual_contexts[0]);
    XTime a,b;XTime_GetTime(&a);
    while(DUAL[3]!=seq){XTime_GetTime(&b);if(b-a>10ULL*COUNTS_PER_SECOND)return 1;}
    dual_barrier();Xil_Out32(MAILBOX+39*4,seq);Xil_Out32(MAILBOX+41*4,DUAL[21]);Xil_Out32(MAILBOX+42*4,DUAL[22]);
    return 0;
}
