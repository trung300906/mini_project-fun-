import os
import json
import datetime as dt

HISTORY_FILE = "/mnt/mydatadrive/CODE/src/AI-assistant/database/history.json"

def save_to_history(user_input, model_response):
    # Nếu file chưa tồn tại, tạo file rỗng
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
        print(f"Created new history file: {HISTORY_FILE}")

    # Đọc lịch sử cũ
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    # Thêm lượt hội thoại mới
    history.append({
        "user": user_input,
        "assistant": model_response,
        "date": dt.datetime.now().isoformat()
    })

    # Ghi lại vào file
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        print(f"Cleared history file: {HISTORY_FILE}")
    else:
        print("No history file to clear.")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"No history file found at {HISTORY_FILE}.")
        return []

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    
    return history

def exists_history() -> bool:
    return os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0

def convert_history_to_content(history):
    content = ""
    for entry in history:
        user_input = entry.get("user", "")
        model_response = entry.get("assistant", "")
        date = entry.get("date", "")
        content += f"User: {user_input}\nAssistant: {model_response}\nDate: {date}\n\n"
    return content.strip()