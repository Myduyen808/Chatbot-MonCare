from langchain_core.messages import HumanMessage, AIMessage
from utils import get_timestamp, get_keyword_name

import yaml
import json
import os

with open("history_config.yml", "r", encoding="utf-8") as f:
    history_config = yaml.safe_load(f)

history_folder_path = history_config.get("history_path", "./data_store/history/")
os.makedirs(history_folder_path, exist_ok=True)

def get_chat_messages(history_obj, k=5):
    messages = history_obj.messages[-(k * 2):]
    langchain_messages = []
    for msg in messages:
        if msg.get("type") == "human":
            langchain_messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("type") == "ai":
            langchain_messages.append(AIMessage(content=msg.get("content", "")))
    return langchain_messages

def get_list_names():
    list_names = []
    folder_path = history_config["history_path"]
    if not os.path.exists(folder_path):
        return list_names

    for filepath in os.listdir(folder_path):
        if filepath.endswith(".json"):
            full_path = os.path.join(folder_path, filepath)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    save_pattern = json.load(f)
                    history_name = save_pattern.get("history_name")
                    if history_name:
                        list_names.append(history_name)
            except Exception:
                continue
    return list_names

def get_history_id(history_name):
    folder_path = history_config["history_path"]
    if not os.path.exists(folder_path):
        return None

    for filepath in os.listdir(folder_path):
        if filepath.endswith(".json"):
            full_path = os.path.join(folder_path, filepath)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    save_pattern = json.load(f)
                    if save_pattern.get("history_name") == history_name:
                        return save_pattern.get("history_id")
            except Exception:
                continue
    return None

class CustomHistory:
    def __init__(self):
        self.messages = []
        self.history_name = "New Session"
        self.history_id = get_timestamp()

    def update_name(self):
        if len(self.messages) > 0 and self.history_name == "New Session":
            first_user_msg = ""
            for msg in self.messages:
                if msg.get("type") == "human":
                    first_user_msg = msg.get("content", "")
                    break

            if not first_user_msg:
                return

            new_name = get_keyword_name(first_user_msg)
            list_names = get_list_names()

            if new_name in list_names:
                i = 2
                while True:
                    name = f"{new_name} #{i}"
                    if name not in list_names:
                        self.history_name = name
                        break
                    i += 1
            else:
                self.history_name = new_name

    def add_a_conversation(self, user_message: str, ai_message: str):
        self.messages.append({"type": "human", "content": user_message})
        self.messages.append({"type": "ai", "content": ai_message})
        self.update_name()
        self.save()

    def load(self, history_id):
        filepath = os.path.join(history_config["history_path"], f"{history_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                save_pattern = json.load(f)
                self.messages = save_pattern.get("history_conversations", [])
                self.history_name = save_pattern.get("history_name", "New Session")
                self.history_id = save_pattern.get("history_id", history_id)
        else:
            raise ValueError("Không tồn tại lịch sử trò chuyện này!")

    def save(self):
        save_pattern = {
            "history_name": self.history_name,
            "history_id": self.history_id,
            "history_conversations": self.messages
        }
        filepath = os.path.join(history_config["history_path"], f"{self.history_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_pattern, f, indent=4, ensure_ascii=False)