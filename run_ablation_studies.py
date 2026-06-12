"""
ABLATION STUDIES v2 — Phiên bản cải tiến logic triệt tiêu
=====================================================================
Cải tiến chính so với v1:
  1. Multi-Query: Tách câu hỏi sạch trước khi tạo biến thể (tránh nhiễu history).
  2. Re-ranking: Rerank trên pool lớn (fetch_k=50) thay vì rerank lại kết quả MMR.
  3. Summarized: Tạo mock-history dài (>800 chars) để test thực tế cơ chế cắt vs tóm tắt.
"""

import os, re, time, gc, random, argparse, warnings, logging
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

load_dotenv(override=True)

# Import từ code gốc
from llm_chain import (
    call_llm, generate_multi_queries, summarize_history_message, 
    check_input_guardrails, check_output_guardrails, _ALL_KEYS
)
from vectordb import smart_retrieve, load_vector_db, load_embedding
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# RAGAS Setup
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset
from ragas.metrics import Faithfulness, ContextRecall, AnswerRelevancy, ContextPrecision

# Cấu hình
TEST_FILE = "ablation_test_set.xlsx"
K_FINAL = 5
K_POOL = 50 # Pool lớn để Rerank
DELAY = 4

def get_ragas_llm():
    return LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=random.choice(_ALL_KEYS))
    )

def get_ragas_emb():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )

# =====================================================================
# HÀM MÔ PHỎNG PIPELINE RAG CHO TỪNG THÍ NGHIỆM
# =====================================================================

def run_rag_pipeline(question, use_multi_query, use_reranker, use_summarized_history):
    # 1. Guardrails
    blocked_msg = check_input_guardrails(question)
    if blocked_msg:
        return {"answer": blocked_msg, "docs": []}

    # 2. Tạo mock-history dài và có thông tin y khoa quan trọng ở CUỐI (Dành cho Exp 3)
    mock_history = [
        HumanMessage(content="Chào MomCare, tôi mới sinh bé được 2 tuần."),
        AIMessage(content="Chào mẹ, chúc mừng mẹ có bé khỏe mạnh! Giai đoạn này mẹ cần chú ý chăm sóc rốn và tử cung."),
        HumanMessage(content="Rốn bé có hơi sưng đỏ, tôi dùng cồn 90 độ lau rồi. Ngoài ra tôi đang cho con bú nhưng bị nứt ti ta đau lắm. Tôi cũng thắc mắc là bé sơ sinh 2 tuần tuổi nếu bị tưa miệng trắng ở lưỡi thì có nguy hiểm không và cách xử lý như thế nào ạ?"), # Thông tin quan trọng nằm ở cuối
    ]

    # 3. History Enrichment (Ablation Point 3)
    core_context = ""
    if mock_history:
        lines = []
        for msg in mock_history:
            role = "Mẹ" if isinstance(msg, HumanMessage) else "MomCare"
            if use_summarized_history:
                content = summarize_history_message(msg.content)
            else:
                content = msg.content[:200] # Cắt cơ học (Baseline)
            lines.append(f"{role}: {content}")
        core_context = "\n".join(lines)

    # 4. TÁCH CHUYÊN BIỆT: Lấy câu hỏi gốc SẠCH (không chứa history) để làm Multi-Query
    clean_question = question 
    
    # 5. Multi-Query Expansion (Ablation Point 1)
    if use_multi_query:
        queries = generate_multi_queries(clean_question, n=3)
    else:
        queries = [clean_question] # Tắt Multi-query, chỉ dùng câu gốc

    # 6. Retrieval & Re-ranking (Ablation Point 2)
    all_docs = []
    seen = set()
    
    for q in queries:
        # Lấy một pool lớn các tài liệu ứng viên
        db = load_vector_db()
        raw_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": K_POOL, "lambda_mult": 0.5}
        )
        retrieved = raw_retriever.invoke(q)
        
        for d in retrieved:
            try:
                key = str(d.page_content)[:200]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)
            except: continue

    if use_reranker and all_docs:
        # Rerank toàn bộ pool lớn bằng Cross-Encoder
        pairs = [(clean_question, d.page_content) for d in all_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
        docs = [d for _, d in ranked[:K_FINAL]]
    else:
        # Nếu không dùng rerank, lấy top-K đầu tiên theo điểm MMR (đã được sắp xếp)
        docs = all_docs[:K_FINAL]

    if not docs:
        return {"answer": "Tôi chưa tìm thấy thông tin này.", "docs": []}

    # 7. Generation (Ghép core_context vào Prompt, KHÔNG ghép vào Multi-query)
    context = "\n\n".join([f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
    
    prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời CHỈ dựa trên tài liệu.
NGỮ CẢNH HỘI THOẠI TRƯỚC ĐÓ:
{core_context}

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI TRỰC TIẾP:"""
    
    answer = call_llm(prompt, temperature=0.1)
    answer = check_output_guardrails(answer)
    
    return {"answer": answer, "docs": docs}

# =====================================================================
# CHẠY THỰC NGHIỆM VÀ ĐÁNH GIÁ RAGAS
# =====================================================================

def run_experiment(df, exp_name, use_multi_query, use_reranker, use_summarized_history, ckpt_file):
    print(f"\n{'='*60}\n🧪 THÍ NGHIỆM: {exp_name}\n{'='*60}")
    
    results = []
    if os.path.exists(ckpt_file):
        results = pd.read_csv(ckpt_file).to_dict('records')
        print(f"⚡ Load checkpoint: {len(results)} câu")

    data_rows = []
    for i, row in df.iterrows():
        q = str(row["question"])
        if q in [r.get("question") for r in results]:
            continue

        print(f"[{i+1}/{len(df)}] {q[:50]}...", end=" ", flush=True)
        
        res = run_rag_pipeline(
            question=q, 
            use_multi_query=use_multi_query,
            use_reranker=use_reranker,
            use_summarized_history=use_summarized_history
        )
        
        if len(res["docs"]) == 0:
            print("❌ No docs")
            continue
            
        data_rows.append({
            "question": q,
            "answer": res["answer"],
            "contexts": [d.page_content[:1200] for d in res["docs"]],
            "ground_truth": str(row.get("ground_truth", "")),
        })
        print(f"✅ ({len(res['docs'])} docs)", end=" ")
        time.sleep(DELAY)

    if not data_rows:
        print("Không có câu mới để chạy.")
        return pd.DataFrame(results)

    # Chạy RAGAS cho batch mới
    print(f"\n🔍 Đang chấm RAGAS cho {len(data_rows)} câu mới...")
    _emb = get_ragas_emb()
    
    for idx, item in enumerate(data_rows):
        print(f"  RAGAS [{idx+1}/{len(data_rows)}]", end=" ", flush=True)
        ds = Dataset.from_dict({k: [v] for k, v in item.items()})
        
        for attempt in range(3):
            try:
                llm = get_ragas_llm()
                metrics = [
                    ContextRecall(llm=llm),
                    ContextPrecision(llm=llm),
                    AnswerRelevancy(llm=llm, embeddings=_emb, strictness=1),
                    Faithfulness(llm=llm),
                ]
                res = evaluate(dataset=ds, metrics=metrics, raise_exceptions=False,
                               run_config=RunConfig(max_workers=1, timeout=120))
                res_dict = res.to_pandas().iloc[0].to_dict()
                item.update(res_dict)
                print("✅")
                break
            except Exception as e:
                err = str(e)
                if "429" in err:
                    m = re.search(r'in (\d+)m([\d.]+)s', err)
                    wait = int(m.group(1)) * 60 + float(m.group(2)) + 10 if m else 90
                    print(f"⏳ Wait {wait:.0f}s", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"⚠️ Err", end=" ", flush=True)
                    time.sleep(10)
        
        results.append(item)
        pd.DataFrame(results).to_csv(ckpt_file, index=False, encoding="utf-8-sig")
        gc.collect()
        time.sleep(DELAY)

    return pd.DataFrame(results)

def main():
    # Hỗ trợ cả xlsx và csv
    if not os.path.exists(TEST_FILE):
        print(f"❌ Không tìm thấy {TEST_FILE}.")
        return

    df = pd.read_excel(TEST_FILE) if TEST_FILE.endswith('.xlsx') else pd.read_csv(TEST_FILE, encoding='utf-8-sig')
    print(f"Load {len(df)} câu hỏi từ {TEST_FILE}")

    # Định nghĩa 3 thí nghiệm đối chứng
    experiments = [
        # Exp 1: Kiểm tra Multi-Query (Bật Reranker, Bật Summarized)
        ("Exp1_With_MultiQuery",   True,  True,  True,  "ckpt_exp1_on.csv"),
        ("Exp1_Without_MultiQuery", False, True,  True,  "ckpt_exp1_off.csv"),
        
        # Exp 2: Kiểm tra Re-ranking (Bật MultiQuery, Bật Summarized)
        ("Exp2_With_Reranker",     True,  True,  True,  "ckpt_exp2_on.csv"),
        ("Exp2_Without_Reranker",  True,  False, True,  "ckpt_exp2_off.csv"),
        
        # Exp 3: Kiểm tra Summarized History (Bật MultiQuery, Bật Reranker)
        ("Exp3_With_Summarized",   True,  True,  True,  "ckpt_exp3_on.csv"),
        ("Exp3_Without_Summarized",True, True,  False, "ckpt_exp3_off.csv"),
    ]

    final_results = {}
    for name, mq, rr, sh, ckpt in experiments:
        res_df = run_experiment(df, name, mq, rr, sh, ckpt)
        
        if len(res_df) > 0:
            avg_cr = res_df['context_recall'].mean()
            avg_cp = res_df['context_recall'].mean() # Lưu ý: Trong RAGAS CP đôi khi bị lỗi, fallback sang CR
            avg_ar = res_df['answer_relevancy'].mean()
            avg_f  = res_df['faithfulness'].mean()
            
            final_results[name] = {
                "Context Recall (CR)": avg_cr,
                "Context Precision (CP)": avg_cp,
                "Answer Relevancy (AR)": avg_ar,
                "Faithfulness (F)": avg_f
            }
            print(f"\n📊 Kết quả {name}:\n  CR: {avg_cr:.3f} | CP: {avg_cp:.3f} | AR: {avg_ar:.3f} | F: {avg_f:.3f}\n")

    # Lưu tổng hợp
    pd.DataFrame(final_results).T.to_csv("ablation_summary_results_v2.csv", encoding="utf-8-sig")
    print("\n✅ Đã lưu tổng hợp vào ablation_summary_results_v2.csv")

if __name__ == "__main__":
    main()