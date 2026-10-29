.syntax unified
.arm
.fpu neon
.text
.align 5
.global dual_cpu1_entry
.type dual_cpu1_entry,%function
dual_cpu1_entry:
    cpsid if
    cps #0x13
    ldr r4,=0x1f000000
    ldr sp,[r4,#68]
    ldr r0,[r4,#56]
    mcr p15,0,r0,c1,c0,2
    isb
    mov r0,#0x40000000
    vmsr fpexc,r0
    ldr r0,[r4,#64]
    vmsr fpscr,r0
    bl Xil_L1DCacheInvalidate
    mov r0,#0
    mcr p15,0,r0,c7,c5,0
    mcr p15,0,r0,c7,c5,6
    mcr p15,0,r0,c8,c7,0
    ldr r0,[r4,#32]
    mcr p15,0,r0,c2,c0,0
    ldr r0,[r4,#36]
    mcr p15,0,r0,c2,c0,1
    ldr r0,[r4,#40]
    mcr p15,0,r0,c2,c0,2
    ldr r0,[r4,#44]
    mcr p15,0,r0,c3,c0,0
    ldr r0,[r4,#60]
    mcr p15,0,r0,c12,c0,0
    dsb sy
    ldr r0,[r4,#48]
    mcr p15,0,r0,c1,c0,0
    isb
    ldr r0,[r4,#52]
    mcr p15,0,r0,c1,c0,1
    isb
    bl dual_worker_main
1:  wfe
    b 1b
.size dual_cpu1_entry,.-dual_cpu1_entry
