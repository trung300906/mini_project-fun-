; ==============================================================================
; BOOTLOADER.ASM - Stage 1 Bootloader
; ==============================================================================
; Đây là stage 1 bootloader được BIOS load vào 0x7C00
; Nhiệm vụ: Setup môi trường, debug output, load stage 2
; ==============================================================================

[BITS 16]                   ; 16-bit real mode
[ORG 0x7C00]               ; BIOS loads bootloader at 0x7C00

; ==============================================================================
; 1️⃣ ENTRY POINT - Điểm bắt đầu từ BIOS
; ==============================================================================
start:
    cli                     ; Disable interrupts during setup
    
    ; Debug: In ra "B" để biết bootloader đã được load
    mov al, 'B'
    call print_char
    
    jmp setup_environment

; ==============================================================================
; 2️⃣ SETUP ENVIRONMENT - Tạo môi trường ổn định
; ==============================================================================
setup_environment:
    ; Reset segment registers
    xor ax, ax              ; ax = 0
    mov ds, ax              ; Data segment = 0
    mov es, ax              ; Extra segment = 0
    mov fs, ax              ; FS = 0
    mov gs, ax              ; GS = 0
    
    ; Setup stack
    mov ss, ax              ; Stack segment = 0
    mov sp, 0x7C00          ; Stack pointer just below bootloader
    
    sti                     ; Re-enable interrupts
    
    ; Debug: In ra "Boot OK" để xác minh môi trường đã setup
    mov si, msg_boot_ok
    call print_string
    
    jmp prepare_stage2

; ==============================================================================
; 3️⃣ DEBUG OUTPUT FUNCTIONS
; ==============================================================================
print_char:
    ; Input: AL = character to print
    mov ah, 0x0E            ; BIOS teletype function
    mov bh, 0x00            ; Page number
    mov bl, 0x07            ; Text attribute (white on black)
    int 0x10                ; Call BIOS video interrupt
    ret

print_string:
    ; Input: SI = pointer to null-terminated string
    mov ah, 0x0E            ; BIOS teletype function
    mov bh, 0x00            ; Page number
    mov bl, 0x07            ; Text attribute
.loop:
    lodsb                   ; Load byte from [SI] to AL, increment SI
    test al, al             ; Check if null terminator
    jz .done
    int 0x10                ; Print character
    jmp .loop
.done:
    ret

; ==============================================================================
; 4️⃣ LOAD STAGE 2 FROM DISK
; ==============================================================================
prepare_stage2:
    ; In ra thông báo đang chuẩn bị load stage 2
    mov si, msg_loading_stage2
    call print_string
    
    ; Reset disk system
    mov ah, 0x00            ; BIOS disk reset function
    mov dl, 0x80            ; Drive number (0x00 = floppy drive, 0x80 = first hard disk drive)
    int 0x13                ; Call BIOS disk interrupt
    jc disk_error           ; Jump if carry flag set (error)
    
    ; Load Stage 2 from disk
    mov ah, 0x02            ; BIOS read sector function
    mov al, 4               ; Read 4 sectors (2048 bytes for stage 2)
    mov ch, 0               ; Cylinder 0
    mov cl, 2               ; Sector 2 (sector 1 is bootloader)
    mov dh, 0               ; Head 0
    mov dl, 0x80            ; Drive number (0x00 = floppy drive, 0x80 = first hard disk drive)
    mov bx, 0x8000          ; Load stage 2 at 0x8000
    int 0x13                ; Call BIOS disk interrupt
    jc disk_error           ; Jump if carry flag set (error)
    cmp al, 4               ; Check if we read exactly 4 sectors
    jne disk_error          ; If not, it's an error
    
    ; Verify stage 2 loaded
    mov si, msg_stage2_loaded
    call print_string
    
    ; Load Stage 3 from disk
    mov si, msg_loading_stage3
    call print_string
    
    mov ah, 0x02            ; BIOS read sector function
    mov al, 4               ; Read 4 sectors (2048 bytes for stage 3)
    mov ch, 0               ; Cylinder 0
    mov cl, 6               ; Sector 6 (sectors 1-4 used by stage 1&2, stage 3 starts at sector 5)
    mov dh, 0               ; Head 0
    mov dl, 0x80            ; Drive number
    mov bx, 0x9000          ; Load stage 3 at 0x9000
    int 0x13                ; Call BIOS disk interrupt
    jc disk_error           ; Jump if carry flag set (error)
    cmp al, 4               ; Check if we read exactly 4 sectors
    jne disk_error          ; If not, it's an error
    
    ; Verify stage 3 loaded
    mov si, msg_stage3_loaded
    call print_string
    
    ; Jump to stage 2
    jmp 0x8000

; ==============================================================================
; ERROR HANDLERS
; ==============================================================================
disk_error:
    mov si, msg_disk_error
    call print_string
    jmp halt_system

halt_system:
    mov si, msg_halt
    call print_string
    cli                     ; Disable interrupts
    hlt                     ; Halt processor
    jmp $                   ; Infinite loop just in case

; ==============================================================================
; DATA SECTION
; ==============================================================================
msg_boot_ok:        db 'oot OK', 0x0D, 0x0A, 0    ; "Boot OK" with newline
msg_loading_stage2: db 'Loading Stage 2...', 0x0D, 0x0A, 0
msg_stage2_loaded:  db 'Stage 2 loaded', 0x0D, 0x0A, 0
msg_loading_stage3: db 'Loading Stage 3...', 0x0D, 0x0A, 0
msg_stage3_loaded:  db 'Stage 3 loaded, jumping...', 0x0D, 0x0A, 0
msg_disk_error:     db 'Disk error!', 0x0D, 0x0A, 0
msg_halt:           db 'System halted.', 0x0D, 0x0A, 0

; ==============================================================================
; BOOT SECTOR SIGNATURE
; ==============================================================================
times 510-($-$$) db 0      ; Pad with zeros to 510 bytes
dw 0xAA55                  ; Boot signature (magic number)

; ==============================================================================
; END OF STAGE 1 BOOTLOADER
; ==============================================================================