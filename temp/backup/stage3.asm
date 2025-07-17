; ==============================================================================
; BOOTLOADER_STAGE3.ASM - Long Mode (64-bit) Stage
; ==============================================================================
; Được nạp tại 0x9000 bởi Stage 2
; Nhiệm vụ: Chạy trong long mode 64-bit và khởi tạo kernel
; ==============================================================================

[BITS 64]                   ; Chế độ 64-bit long mode
[ORG 0x9000]               ; Địa chỉ Stage 3

; Segment selectors từ Stage 2
CODE_SEG64 equ 0x18        ; Code segment 64-bit từ GDT
DATA_SEG64 equ 0x20        ; Data segment 64-bit từ GDT

; ==============================================================================
; Long Mode Entry Point
; ==============================================================================
long_mode_entry:
    ; Thiết lập segments cho long mode
    mov ax, DATA_SEG64      ; Use 64-bit data segment
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov rsp, 0x90000        ; Thiết lập stack 64-bit

    ; Vẽ màn hình xanh để chứng minh long mode hoạt động
    mov rdi, 0xB8000        ; VGA text buffer
    mov rcx, 80 * 25        ; 80x25 screen
    mov ax, 0x2020          ; Blue background với space character (0x20 = blue bg, 0x20 = space)

.fill_screen:
    mov [rdi], ax
    add rdi, 2
    loop .fill_screen

    ; In thông báo thành công long mode
    mov rdi, 0xB8000 + (80 * 2 * 2) ; Dòng thứ 2
    mov rsi, msg_long_mode
    mov ah, 0x0F            ; Bright white on black
.print_loop:
    lodsb
    test al, al
    jz .print_done
    mov [rdi], ax
    add rdi, 2
    jmp .print_loop
.print_done:

    ; In thông báo version và CPU info
    mov rdi, 0xB8000 + (80 * 2 * 4) ; Dòng thứ 4
    mov rsi, msg_version
    mov ah, 0x0A            ; Bright green on black
.print_version:
    lodsb
    test al, al
    jz .version_done
    mov [rdi], ax
    add rdi, 2
    jmp .print_version
.version_done:

    ; Hiển thị thông tin CPU
    call display_cpu_info

    ; Khởi tạo kernel space (placeholder)
    call init_kernel_space

    ; Infinite loop - thành công long mode!
    jmp halt_system

; ==============================================================================
; Kernel Space Initialization
; ==============================================================================
init_kernel_space:
    ; Placeholder cho kernel initialization
    mov rdi, 0xB8000 + (80 * 2 * 6) ; Dòng thứ 6
    mov rsi, msg_kernel_init
    mov ah, 0x0C            ; Bright red on black
.print_init:
    lodsb
    test al, al
    jz .init_done
    mov [rdi], ax
    add rdi, 2
    jmp .print_init
.init_done:
    ret

; ==============================================================================
; CPU Information Display
; ==============================================================================
display_cpu_info:
    ; Hiển thị thông tin CPU đơn giản
    mov rdi, 0xB8000 + (80 * 2 * 5) ; Dòng thứ 5
    mov rsi, msg_cpu_info
    mov ah, 0x0E            ; Bright yellow on black
.print_cpu:
    lodsb
    test al, al
    jz .cpu_done
    mov [rdi], ax
    add rdi, 2
    jmp .print_cpu
.cpu_done:
    ret

; ==============================================================================
; System Halt
; ==============================================================================
halt_system:
    cli
    hlt
    jmp halt_system

; ==============================================================================
; Data Section
; ==============================================================================
msg_long_mode:       db "LONG MODE ACTIVATED SUCCESSFULLY!", 0
msg_version:         db "AI-OS Stage 3 v1.0 - 64-bit Kernel", 0
msg_cpu_info:        db "CPU: x86_64 Long Mode Ready", 0
msg_kernel_init:     db "Kernel Space Initialized", 0

; Padding để đảm bảo kích thước chính xác
times 2048 - ($ - $$) db 0
