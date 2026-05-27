# test_quick.py — thêm debug context
from dotenv import load_dotenv
load_dotenv()
from vectordb import smart_retrieve

q = "Trẻ sơ sinh được định nghĩa là trẻ trong độ tuổi nào?"
docs = smart_retrieve(q, None, k=3)

context = "\n\n".join([d.page_content for d in docs])
print(f"=== CONTEXT ===")
print(context[:500])
print(f"=== ĐỘ DÀI CONTEXT: {len(context)} ký tự ===")