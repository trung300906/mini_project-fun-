; ==============================================================================
; BS2_DEBUG.ASM - Debug version without VBE for testing protected mode
; ==============================================================================
; Được nạp tại 0x8000 bởi Stage 1
; Nhiệm vụ: Chuyển từ real mode -> protected mode (bỏ qua long mode và VBE)
; ==============================================================================

[BITS 16]                   ; Chế độ 16-bit real mode
[ORG 0x8000]               ; Địa chỉ Stage 2

; Điểm bắt đầu
stage2_start:
    cli                     ; Tắt ngắt
    mov si, msg_stage2_entry
    call print_string       ; In thông báo khởi động

    ; Kích hoạt A20
    call enable_a20
    mov si, msg_a20_enabled
    call print_string

    ; Nạp GDT
    lgdt [gdt_descriptor]
    mov si, msg_gdt_loaded
    call print_string

    ; Chuyển sang protected mode
    mov eax, cr0
    or eax, 0x1             ; Bật bit PE
    mov cr0, eax
    jmp CODE_SEG:protected_mode_entry

; Hàm kích hoạt A20
enable_a20:
    in al, 0x92
    test al, 0x02
    jnz .done
    or al, 0x02
    out 0x92, al
.done:
    ret

; Hàm in chuỗi (real mode)
print_string:
    mov ah, 0x0E
.loop:
    lodsb
    test al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    ret

; ==============================================================================
; Protected Mode (32-bit)
; ==============================================================================
[BITS 32]
protected_mode_entry:
    cli                     ; Ensure interrupts are disabled
    mov ax, DATA_SEG
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov esp, 0x90000        ; Thiết lập stack
    mov ss, ax

    ; Vẽ pattern đơn giản lên VGA buffer để test protected mode
    mov edi, 0xB8000        ; VGA text buffer
    mov ecx, 80 * 25        ; 80x25 screen
    mov eax, 0x1F201F20     ; 2 ký tự cùng lúc (tối ưu tốc độ)

fill_screen:
    mov [edi], eax
    add edi, 4
    loop fill_screen

    ; In thông báo thành công protected mode
    mov esi, msg_protected_mode
    call print_string_pm

    ; Delay một chút để có thể thấy thông báo PM32
    mov ecx, 0x10000000
.delay_loop:
    loop .delay_loop

    ; Kiểm tra hỗ trợ long mode
    call check_long_mode
    cmp eax, 1
    jne no_long_mode

    ; Hiển thị thông báo chuẩn bị chuyển sang long mode
    mov esi, msg_entering_long_mode
    call print_string_pm_line3

    ; Thiết lập paging cho long mode
    call setup_paging

    ; Hiển thị thông báo paging đã setup
    mov esi, msg_paging_setup
    call print_string_pm_line4

    ; Bật PAE (Physical Address Extension)
    mov eax, cr4
    or eax, (1 << 5)        ; Bật bit PAE
    mov cr4, eax

    ; Bật long mode trong EFER register
    mov ecx, 0xC0000080     ; EFER register
    rdmsr
    or eax, (1 << 8)        ; Bật bit LME (Long Mode Enable)
    wrmsr

    ; Bật paging
    mov eax, cr0
    or eax, (1 << 31)       ; Bật bit PG (Paging)
    mov cr0, eax

    ; Hiển thị thông báo trước khi jump
    mov esi, msg_jumping_long
    call print_string_pm_line5

    ; Delay trước khi chuyển sang long mode để có thể thấy thông báo
    mov ecx, 0x5000000
.final_delay:
    loop .final_delay

    ; Nhảy sang long mode (Stage 3)
    jmp CODE_SEG64:0x9000

no_long_mode:
    ; Hiển thị thông báo lỗi
    mov esi, msg_no_long_mode
    call print_string_pm
    jmp halt_system

; Kiểm tra CPU có hỗ trợ long mode không
check_long_mode:
    ; Kiểm tra CPUID có hỗ trợ extended functions
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_support
    
    ; Kiểm tra long mode support
    mov eax, 0x80000001
    cpuid
    test edx, (1 << 29)     ; Bit 29 = Long Mode support
    jz .no_support
    
    mov eax, 1              ; Long mode supported
    ret
.no_support:
    mov eax, 0              ; Long mode not supported
    ret

; Thiết lập paging cho long mode
setup_paging:
    ; Xóa vùng nhớ cho page tables (đơn giản hóa)
    mov edi, 0x1000
    mov ecx, 0x1000         ; Xóa 4KB thay vì 12KB
    xor eax, eax
    rep stosd

    ; Thiết lập PML4 (Page Map Level 4) - chỉ entry đầu tiên
    mov edi, 0x1000
    mov eax, 0x2003         ; PDP tại 0x2000, present + writable
    stosd
    
    ; Thiết lập PDP (Page Directory Pointer) - chỉ entry đầu tiên
    mov edi, 0x2000
    mov eax, 0x3003         ; PD tại 0x3000, present + writable
    stosd
    
    ; Thiết lập PD (Page Directory) - identity mapping với 2MB pages
    mov edi, 0x3000
    mov eax, 0x00000083     ; 2MB page, present + writable + page size
    mov ecx, 4              ; Chỉ map 4 entries = 8MB (đủ cho bootloader)
.map_pages:
    stosd
    add edi, 4              ; Bỏ qua high 32-bit
    add eax, 0x200000       ; Tăng 2MB
    loop .map_pages

    ; Đặt CR3 trỏ đến PML4
    mov eax, 0x1000
    mov cr3, eax
    ret

; In chuỗi trong protected mode
print_string_pm:
    mov edi, 0xB8000 + (80 * 2 * 2) ; Dòng thứ 2
    mov ah, 0x0F            ; Màu trắng sáng trên đen
.loop:
    lodsb
    test al, al
    jz .done
    mov [edi], ax
    add edi, 2
    jmp .loop
.done:
    ret

; In chuỗi trong protected mode dòng 3
print_string_pm_line3:
    mov edi, 0xB8000 + (80 * 2 * 3) ; Dòng thứ 3
    mov ah, 0x0E            ; Màu vàng trên đen
.loop:
    lodsb
    test al, al
    jz .done
    mov [edi], ax
    add edi, 2
    jmp .loop
.done:
    ret

; In chuỗi trong protected mode dòng 4
print_string_pm_line4:
    mov edi, 0xB8000 + (80 * 2 * 4) ; Dòng thứ 4
    mov ah, 0x0C            ; Màu đỏ sáng trên đen
.loop:
    lodsb
    test al, al
    jz .done
    mov [edi], ax
    add edi, 2
    jmp .loop
.done:
    ret

; In chuỗi trong protected mode dòng 5
print_string_pm_line5:
    mov edi, 0xB8000 + (80 * 2 * 5) ; Dòng thứ 5
    mov ah, 0x0D            ; Màu tím trên đen
.loop:
    lodsb
    test al, al
    jz .done
    mov [edi], ax
    add edi, 2
    jmp .loop
.done:
    ret

halt_system:
    cli
    hlt
    jmp halt_system

; ==============================================================================
; GDT
; ==============================================================================
gdt_start:
gdt_null:
    dd 0x0
    dd 0x0
gdt_code:                   ; Code 32-bit
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10011010b            ; Present, ring 0, code, executable
    db 11001111b            ; Granularity, 32-bit
    db 0x0
gdt_data:                   ; Data
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10010010b            ; Present, ring 0, data, writable
    db 11001111b
    db 0x0
gdt_code64:                 ; Code 64-bit
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10011010b            ; Present, ring 0, code, executable
    db 10101111b            ; Granularity, 64-bit (L=1, D=0)
    db 0x0
gdt_data64:                 ; Data 64-bit
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10010010b            ; Present, ring 0, data, writable
    db 00101111b            ; Granularity: G=1, D/B=0, L=0
    db 0x0
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

CODE_SEG equ gdt_code - gdt_start
DATA_SEG equ gdt_data - gdt_start
CODE_SEG64 equ gdt_code64 - gdt_start
DATA_SEG64 equ gdt_data64 - gdt_start

; Dữ liệu
msg_stage2_entry:    db "Stage2 Debug: Loaded", 0x0D, 0x0A, 0
msg_a20_enabled:     db "A20: Enabled", 0x0D, 0x0A, 0
msg_gdt_loaded:      db "GDT: Loaded", 0x0D, 0x0A, 0
msg_protected_mode:  db "PM32: SUCCESS!", 0
msg_entering_long_mode: db "Entering Long Mode...", 0
msg_paging_setup:    db "Paging Setup OK", 0
msg_jumping_long:    db "Jumping to Stage 3", 0
msg_no_long_mode:    db "No Long Mode!", 0

; Padding
times 2048 - ($ - $$) db 0
