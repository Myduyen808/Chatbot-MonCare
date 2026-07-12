import os
import time
import random
import json
import re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from vectordb import load_vector_db, smart_retrieve
from llm_chain import (
    _adaptive_hybrid_search, 
    get_reranker, 
    generate_multi_queries,
    rewrite_and_detect_intent,
    check_input_guardrails_with_llm,
    MENTAL_HEALTH_RESPONSE
)

ALL_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]

client = Groq(api_key=random.choice(ALL_KEYS))
MODEL_NAME = "llama-3.1-8b-instant"
K = 5

# ==========================================
# 1. NẠP DỮ LIỆU (Giữ nguyên)
# ==========================================
random.seed(42)
PATH_KB1 = "KB1_Medical_Standard.xlsx"
PATH_KB2 = "KB2_Mom_Style.xlsx"
PATH_KB3 = "KB3_Information_Noise.xlsx"
PATH_VIMEDAQA = "data_store/excel/Bo_De_Me_Va_Be.xlsx"

def load_kb_questions(file_path, num_samples=25):
    try:
        df = pd.read_excel(file_path)
        col_name = 'Câu hỏi người dùng (Input)'
        if col_name in df.columns: questions = df[col_name].dropna().astype(str).tolist()
        else: questions = df.iloc[:, 2].dropna().astype(str).tolist()
        questions = [q for q in questions if len(q) > 10]
        if len(questions) > num_samples: questions = random.sample(questions, num_samples)
        return questions
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc {file_path}: {e}")
        return []

DATASETS = {
    "KB1_Standard": load_kb_questions(PATH_KB1, 25),
    "KB2_TeenCode": load_kb_questions(PATH_KB2, 25),
    "KB3_Noise": load_kb_questions(PATH_KB3, 25),
    "ViMedAQA": []
}
try:
    df_vimed = pd.read_excel(PATH_VIMEDAQA)
    vimed_q = df_vimed["question"].dropna().astype(str).tolist() if 'question' in df_vimed.columns else df_vimed.iloc[:, 2].dropna().astype(str).tolist()
    DATASETS["ViMedAQA"] = [q for q in vimed_q if len(q) > 10][:25]
except: pass

print("📊 Đã nạp danh sách câu hỏi:", {k: len(v) for k, v in DATASETS.items()})

def generate_answer(question, docs):
    if not docs: return "Tôi chưa tìm thấy thông tin này trong tài liệu."
    context = "\n\n".join([f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
    try:
        res = client.chat.completions.create(messages=[{"role": "user", "content": f"Bạn là chuyên gia y tế MomCare. Trả lời CHỈ dựa trên tài liệu.\n{context}\nCÂU HỎI: {question}\nTRẢ LỜI:"}], model=MODEL_NAME, temperature=0.1)
        return res.choices[0].message.content
    except: return "Lỗi gọi LLM sinh câu trả lời."

# ==========================================
# CÁC PHƯƠNG PHÁP RAG (Giữ nguyên logic của em)
# ==========================================
def method_1_dense_only(q): return load_vector_db().similarity_search(q, k=K)
def method_2_bm25_only(q):
    from llm_chain import _get_production_hybrid_retriever
    cache = _get_production_hybrid_retriever()
    tokens = re.findall(r'[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*', q.lower())
    scores = cache["bm25"].get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K]
    return [cache["valid_docs"][i] for i in top_idx if i < len(cache["valid_docs"])]
def method_3_hybrid(q): return _adaptive_hybrid_search(q, k=K)
def method_4_hybrid_rerank(q):
    docs = _adaptive_hybrid_search(q, k=K*2)
    if not docs: return []
    reranker = get_reranker()
    ranked = sorted(zip(reranker.predict([(q, d.page_content) for d in docs]), docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:K]]
def method_5_hybrid_rerank_mq(question):
    primary_docs = _adaptive_hybrid_search(question, k=K*2)
    try:
        extra_queries = generate_multi_queries(question, n=2)
        time.sleep(1)
    except Exception: extra_queries = []
    all_docs = list(primary_docs)
    seen = {str(d.page_content)[:200] for d in primary_docs}
    for q in extra_queries:
        try:
            retrieved = _adaptive_hybrid_search(q, k=K)
            time.sleep(1)
        except Exception: continue
        for d in retrieved:
            key = str(d.page_content)[:200]
            if key not in seen: seen.add(key); all_docs.append(d)
    if not all_docs: return primary_docs[:K]
    try:
        reranker = get_reranker()
        scores = reranker.predict([(question, d.page_content) for d in all_docs])
        ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:K]]
    except Exception: return all_docs[:K]
def method_6_full_system(q):
    blocked = check_input_guardrails_with_llm(q)
    if blocked: return None, blocked
    enriched_q, intent = rewrite_and_detect_intent(q, [])
    if intent in ["BLOCKED", "SMALLTALK"]: return ([], enriched_q) if intent == "SMALLTALK" else (None, MENTAL_HEALTH_RESPONSE)
    primary = _adaptive_hybrid_search(enriched_q, k=K)
    queries = [enriched_q] + (generate_multi_queries(enriched_q, n=2) if len(q.split()) <= 5 else [])
    all_docs, seen = list(primary), {str(d.page_content)[:200] for d in primary}
    for query in queries:
        for d in smart_retrieve(query, None, K):
            key = str(d.page_content)[:200]
            if key not in seen: seen.add(key); all_docs.append(d)
    if len(all_docs) > K:
        ranked = sorted(zip(get_reranker().predict([(enriched_q, d.page_content) for d in all_docs]), all_docs), key=lambda x: x[0], reverse=True)
        docs = [d for _, d in ranked[:K]]
    else: docs = all_docs[:K]
    return docs, enriched_q

# ==========================================
# VÒNG LẶP CHẠY THU THẬP KẾT QUẢ THÔ
# ==========================================
def run_rag_collection():
    methods = {
        "1. Dense Only": method_1_dense_only, "2. BM25 Only": method_2_bm25_only,
        "3. Hybrid": method_3_hybrid, "4. Hybrid + Rerank": method_4_hybrid_rerank,
        "5. Hybrid + Rerank + MQ": method_5_hybrid_rerank_mq, "6. Full System": method_6_full_system
    }

    for dataset_name, questions in DATASETS.items():
        if not questions: continue
        
        raw_file = f"raw_rag_{dataset_name}.json"
        
        # Đọc checkpoint cũ nếu có chạy dở
        if os.path.exists(raw_file):
            with open(raw_file, "r", encoding="utf-8") as f: results = json.load(f)
        else:
            results = {name: {"questions": [], "answers": [], "contexts": [], "latency": []} for name in methods}

        done_count = len(results["1. Dense Only"]["questions"])
        print(f"\n🚀 Đang chạy RAG trên tập: {dataset_name} (Đã xong: {done_count}/{len(questions)})")

        for q_idx, question in enumerate(questions):
            if q_idx < done_count: continue
            
            print(f" -> Câu [{q_idx+1}/{len(questions)}]: {question[:40]}...", end=" | ", flush=True)
            
            for method_name, method_func in methods.items():
                start_time = time.time()
                try:
                    if method_name == "6. Full System":
                        result = method_func(question)
                        if result is None:
                            docs, answer = [], "Xin lỗi, MomCare không thể hỗ trợ yêu cầu này."
                        elif isinstance(result, tuple):
                            docs, enriched_q = result
                            if docs is None: docs = []
                            answer = enriched_q if (not docs and isinstance(enriched_q, str)) else generate_answer(enriched_q, docs)
                        else:
                            docs, answer = [], str(result)
                    else:
                        docs = method_func(question)
                        if not docs: docs = []
                        answer = generate_answer(question, docs)
                except Exception as e:
                    print(f"\n❌ Lỗi RAG tại [{method_name}]: {e}")
                    docs, answer = [], "Lỗi hệ thống trong luồng xử lý RAG."

                contexts_str = [d.page_content for d in docs] if docs else ["Không tìm thấy tài liệu phù hợp."]
                
                results[method_name]["questions"].append(question)
                results[method_name]["answers"].append(answer)
                results[method_name]["contexts"].append(contexts_str)
                results[method_name]["latency"].append(time.time() - start_time)
                time.sleep(1) # Nghỉ ngắn chống nghẽn
                
            print("Xong")
            
            # Lưu checkpoint thô liên tục
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        print(f"✅ Đã lưu xong dữ liệu thô cho tập {dataset_name} vào file: {raw_file}")

if __name__ == "__main__":
    run_rag_collection()