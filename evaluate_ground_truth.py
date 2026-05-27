from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from llm_chain import RAGChain
from ragas.metrics import BleuScore, RougeScore
from ragas import evaluate
from datasets import Dataset

# ===== ĐỌC GROUND TRUTH =====
df = pd.read_csv('data_store/csv/qa_me_bim_sau_sinh_200.csv', encoding='utf-8')

# Lấy 30 câu đầu để test nhanh
df_sample = df.head(30)

print(f"Tổng câu Ground Truth: {len(df)}")
print(f"Số câu test: {len(df_sample)}")
print("Đang chạy RAG chain...\n")

# ===== CHẠY HỆ THỐNG =====
chain = RAGChain(k=5)
questions = []
answers = []
references = []
contexts = []

for i, row in df_sample.iterrows():
    question = row['Câu hỏi']
    reference = row['Trả lời']
    source = row['Nguồn']
    
    print(f"[{i+1}/30] {question[:50]}...")
    
    try:
        result = chain.invoke({"question": question, "history": []})
        answer = result["answer"]
        docs = result["docs"]
        context = [d.page_content for d in docs] if docs else [""]
    except Exception as e:
        print(f"  Lỗi: {e}")
        answer = ""
        context = [""]
    
    questions.append(question)
    answers.append(answer)
    references.append(reference)
    contexts.append(context)

# ===== ĐÁNH GIÁ =====
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "reference": references,
    "contexts": contexts,
})

print("\nĐang tính BLEU và ROUGE...")
results = evaluate(dataset, metrics=[BleuScore(), RougeScore()])
print("\n" + "=" * 60)
print("KẾT QUẢ ĐÁNH GIÁ GROUND TRUTH")
print("=" * 60)
print(results)

# ===== SO SÁNH CHI TIẾT =====
print("\nCHI TIẾT 5 CÂU ĐẦU:")
for i in range(5):
    print(f"\nCâu {i+1}: {questions[i]}")
    print(f"Nguồn  : {df_sample.iloc[i]['Nguồn']}")
    print(f"Chuẩn  : {references[i][:100]}...")
    print(f"AI trả : {answers[i][:100]}...")