# test_retrieval.py
from dotenv import load_dotenv
load_dotenv()
from vectordb import smart_retrieve

q = "Trẻ sơ sinh được định nghĩa là trẻ trong độ tuổi nào?"
docs = smart_retrieve(q, None, k=3)

print(f"Tìm được {len(docs)} docs:")
for i, doc in enumerate(docs):
    print(f"\n[{i+1}] source: {doc.metadata.get('source', 'N/A')}")
    print(f"content: {doc.page_content[:300]}")