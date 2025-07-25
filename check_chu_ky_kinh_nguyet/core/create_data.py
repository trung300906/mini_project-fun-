import os
import json
from datetime import datetime
import core.parse_data as parse_data
# Đường dẫn đến thư mục chứa file data (ngang cấp với core/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'user_data.json')

def ensure_data_file_exists_and_save(entry=None):
    """Đảm bảo thư mục và file tồn tại. Nếu có `entry`, thêm vào JSON."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump([], f, indent=2)
        print("✅ Created user_data.json at", DATA_FILE)
    else:
        print("📂 Data file already exists.")

    if entry:
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

        # 🔥 Chỉ chuyển sang chuỗi nếu là datetime
        if isinstance(entry["start_date"], datetime):
            entry["start_date"] = entry["start_date"].strftime("%Y-%m-%d")
        if isinstance(entry["end_date"], datetime):
            entry["end_date"] = entry["end_date"].strftime("%Y-%m-%d")

        entry["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data.append(entry)

        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

        print("✅ New entry saved.")


def create_data_folder(dict_data: dict) -> None:
    """Tạo một thư mục mới cho dữ liệu."""
    ensure_data_file_exists_and_save(entry=dict_data)