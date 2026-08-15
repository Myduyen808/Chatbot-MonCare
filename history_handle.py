from langchain_core.messages import HumanMessage, AIMessage
from utils import get_timestamp, get_keyword_name

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
    """Không cung cấp danh sách phiên lưu lâu dài."""
    return []

def get_history_id(history_name):
    """Không ánh xạ tên phiên sang dữ liệu trên đĩa."""
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
            self.history_name = new_name

    def add_a_conversation(self, user_message: str, ai_message: str):
        """Giữ hội thoại trong bộ nhớ của phiên Streamlit hiện tại.

        Hàm không ghi nội dung xuống đĩa. Danh sách ``messages`` vẫn được
        dùng đầy đủ bởi ACM và phần hiển thị hội thoại trong phiên.
        """
        self.messages.append({"type": "human", "content": user_message})
        self.messages.append({"type": "ai", "content": ai_message})
        self.update_name()

    def load(self, history_id):
        raise RuntimeError("MomCare không lưu lịch sử hội thoại lâu dài.")

    def save(self):
        """Giữ tương thích API nhưng không ghi dữ liệu xuống đĩa."""
        return None
