from vectordb import smart_retrieve

docs = smart_retrieve("Trong ngày đầu sau sinh, cần giữ ấm cho trẻ bằng cách nào?", None, 3)

for i, d in enumerate(docs):
    source = d.metadata.get("source")
    print(f"[{i+1}] {source}")
    print(d.page_content[:200])
    print()