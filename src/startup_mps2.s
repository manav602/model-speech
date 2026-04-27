/* Minimal Cortex-M4 startup for mps2-an386. Vector table + Reset_Handler that
 * copies .data, zeros .bss, sets up semihosting, and jumps to main. */
    .syntax unified
    .cpu cortex-m4
    .fpu softvfp
    .thumb

    .section .vectors, "a", %progbits
    .type vectors, %object
vectors:
    .word _estack                 /* MSP */
    .word Reset_Handler
    .word Default_Handler         /* NMI */
    .word Default_Handler         /* HardFault */
    .word Default_Handler         /* MemManage */
    .word Default_Handler         /* BusFault */
    .word Default_Handler         /* UsageFault */
    .word 0                       /* reserved */
    .word 0
    .word 0
    .word 0
    .word Default_Handler         /* SVC */
    .word Default_Handler         /* DebugMon */
    .word 0
    .word Default_Handler         /* PendSV */
    .word Default_Handler         /* SysTick */
    .size vectors, .-vectors

    .text
    .thumb_func
    .global Reset_Handler
    .type Reset_Handler, %function
Reset_Handler:
    /* Copy .data from FLASH to SRAM */
    ldr   r0, =_sdata
    ldr   r1, =_edata
    ldr   r2, =_sidata
    movs  r3, #0
.copy_loop:
    cmp   r0, r1
    bge   .zero_bss
    ldr   r4, [r2, r3]
    str   r4, [r0, r3]
    adds  r3, #4
    adds  r0, #4
    b     .copy_loop

.zero_bss:
    ldr   r0, =_sbss
    ldr   r1, =_ebss
    movs  r2, #0
.bss_loop:
    cmp   r0, r1
    bge   .call_main
    str   r2, [r0]
    adds  r0, #4
    b     .bss_loop

.call_main:
    bl    main
    /* if main returns, semihosting exit (handled in main.c on real hw) */
    b     .

    .thumb_func
    .global Default_Handler
    .type Default_Handler, %function
Default_Handler:
    b     .
    .size Default_Handler, .-Default_Handler
