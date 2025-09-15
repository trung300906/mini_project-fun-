import random

def generate_vietnamese_phone_numbers(count=100):
    # Các đầu số phổ biến của các nhà mạng Việt Nam
    prefixes = [
        '032', '033', '034', '035', '036', '037', '038', '039',  # Viettel
        '070', '076', '077', '078', '079',  # MobiFone
        '081', '082', '083', '084', '085',  # Vinaphone
        '056', '058', '059',  # Vietnamobile
        '052', '055',  # Vietnammobile (khác)
        '099',  # G-Mobile
        '087', '088', '089',  # Vinaphone (thêm)
        '090', '093', '089',  # MobiFone (thêm)
        '091', '094',  # Vinaphone (thêm)
        '092', '0186', '0188'  # Vietnamobile (thêm)
    ]
    
    phone_numbers = []
    
    for _ in range(count):
        prefix = random.choice(prefixes)
        # Tạo 7 số ngẫu nhiên cho phần đuôi
        suffix = ''.join(random.choices('0123456789', k=10-len(prefix)))
        phone_number = prefix + suffix
        phone_numbers.append(phone_number)
    
    return phone_numbers

# Tạo 100 số điện thoại
phone_numbers = generate_vietnamese_phone_numbers(100)

# Lưu vào file checkso.txt
with open("/mnt/mydatadrive/CODE/src/check_phone_numbers/legacy/checkso.txt", "w", encoding="utf-8") as f:
    # Ghi mỗi số trên một dòng
    for phone in phone_numbers:
        f.write(phone + "\n")

print("✅ Đã tạo file checkso.txt với 100 số điện thoại Việt Nam")