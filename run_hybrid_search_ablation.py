"""
THỰC NGHIỆM SO SÁNH: VECTOR SEARCH (FAISS) vs HYBRID SEARCH (FAISS + BM25)
=======================================================================
Mục tiêu: Đo lường hiệu năng khi bổ sung Keyword-based Search (BM25) 
vào Vector Search thuần túy (FAISS) để định vị chính xác các thực thể 
y khoa đặc thù (con số, tên thuốc, tỷ lệ phần trăm) mà Embedding làm mờ đi.

Cách dùng:
    python run_hybrid_search_ablation.py
"""

import os, re, time, gc, warnings, logging, math
import pandas as pd
from dotenv import load_dotenv
# --- SỬA LỖI TYPO: Đổi từ retrieversers thành retrievers chuẩn LangChain ---
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

load_dotenv(override=True)

# Import hệ thống hiện tại
from llm_chain import RAGChain, check_input_guardrails, check_output_guardrails, _ALL_KEYS
from vectordb import load_vector_db, load_embedding, clean_chunk_text

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
TEST_FILE = "ablation_test_set.xlsx" # Dùng lại tập 30 câu ở cùng thư mục
K_FINAL = 5
DELAY = 4

def get_ragas_llm():
    import random
    return LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=random.choice(_ALL_KEYS))
    )

def get_ragas_emb():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )

# =====================================================================
# HÀM TẠO RETRIEVER THUẦN TÚY (CHỈ DÙNG FAISS - BASELINE HIỆN TẠI)
# =====================================================================
def get_vector_only_retriever():
    db = load_vector_db()
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": K_FINAL, 
            "fetch_k": 30, 
            "lambda_mult": 0.7
        }
    )

# =====================================================================
# HÀM TẠO RETRIEVER LAI (HYBRID: FAISS + BM25) - ĐÃ TỐI ƯU METADATA
# =====================================================================
def get_hybrid_retriever():
    db = load_vector_db()
    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": K_FINAL, 
            "fetch_k": 30, 
            "lambda_mult": 0.7
        }
    )
    
    # Khởi tạo BM25 Retriever từ các chunk văn bản đã load vào DB
    docs = db.docstore._dict.values()
    
    # --- CẢI TIẾN SỬA LỖI: Trích xuất Document giữ nguyên vẹn Metadata y khoa ---
    valid_docs = []
    for doc in docs:
        clean_t = clean_chunk_text(doc.page_content)
        if len(clean_t) > 50:
            valid_docs.append(Document(page_content=clean_t, metadata=doc.metadata))
            
    # Tạo BM25 Retriever trực tiếp từ đối tượng Documents
    bm25_retriever = BM25Retriever.from_documents(documents=valid_docs)
    bm25_retriever.k = K_FINAL
    
    # Kết hợp 2 retriever (Vector weight = 0.7, Keyword weight = 0.3)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever

# =====================================================================
# HÀM CHẠY RAG PIPELINE THỰC TẾ ĐỂ XÁC LẬP RETRIEVER
# =====================================================================
def run_rag_with_custom_retriever(question, retriever):
    # 1. Guardrails
    blocked_msg = check_input_guardrails(question)
    if blocked_msg:
        return {"answer": blocked_msg, "docs": []}

    # 2. Lấy tài liệu bằng Retriever được truyền vào
    docs = retriever.invoke(question)
    
    if not docs:
        return {"answer": "Tôi chưa tìm thấy thông tin này.", "docs": []}

    # 3. Tạo Context và Prompt (Giữ nguyên logic gốc)
    context = "\n\n".join([f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
    
    prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời câu hỏi CHỈ dựa trên tài liệu.
TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

TRẢ LỜI TRỰC TIẾP:"""
    
    # 4. Gọi LLM sinh câu trả lời
    from llm_chain import call_llm
    answer = call_llm(prompt, temperature=0.1)
    answer = check_output_guardrails(answer)
    
    return {"answer": answer, "docs": docs}

# =====================================================================
# CHẠY THỰC NGHIỆM CHÍNH
# =====================================================================
def main():
    if not os.path.exists(TEST_FILE):
        print(f"❌ Không tìm thấy {TEST_FILE}.")
        return

    # --- SỬA LỖI CRITICAL: Xóa bỏ tham số encoding không hợp lệ của read_excel ---
    df = pd.read_excel(TEST_FILE)
    print(f"Load {len(df)} câu hỏi từ {TEST_FILE}\n")

    experiments = [
        ("Baseline_VectorOnly", get_vector_only_retriever, "ckpt_hybrid_vector.csv"),
        ("Hybrid_Vector_BM25", get_hybrid_retriever, "ckpt_hybrid_ensemble.csv"),
    ]

    final_results = {}
    _emb = get_ragas_emb()

    for name, retriever_fn, ckpt_file in experiments:
        print(f"\n{'='*60}\n🧪 THÍ NGHIỆM: {name}\n{'='*60}")
        
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
            
            # Lấy retriever thông qua hàm
            current_retriever = retriever_fn()
            
            res = run_rag_with_custom_retriever(q, current_retriever)
            
            if len(res["docs"]) == 0:
                print("❌ No docs")
                continue
                
            data_rows.append({
                "question": q,
                "answer": res["answer"],
                "contexts": [d.page_content[:1200] for d in res["docs"]],
                "ground_truth": str(row.get("ground_truth", "")),
            })
            print(f"✅", end=" ")
            time.sleep(DELAY)

        if not data_rows:
            print("Không có câu mới để chạy.")
            continue

        print(f"\n🔍 Đang chấm RAGAS cho {len(data_rows)} câu mới...")
        
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

        # Tính toán trung bình và hiển thị
        res_df = pd.DataFrame(results)
        if len(res_df) > 0:
            # Sửa tên cột theo cấu trúc chuẩn của RAGAS trả về trong Dataframe
            cr_col = 'context_recall' if 'context_recall' in res_df.columns else 'context_recall'
            cp_col = 'context_precision' if 'context_precision' in res_df.columns else 'context_precision'
            ar_col = 'answer_relevancy' if 'answer_relevancy' in res_df.columns else 'answer_relevancy'
            f_col = 'faithfulness' if 'faithfulness' in res_df.columns else 'faithfulness'

            final_results[name] = {
                "Context Recall (CR)": res_df[cr_col].mean(),
                "Context Precision (CP)": res_df[cp_col].mean(),
                "Answer Relevancy (AR)": res_df[ar_col].mean(),
                "Faithfulness (F)": res_df[f_col].mean()
            }
            print(f"\n📊 Kết quả trung bình cho {name}:")
            for k, v in final_results[name].items():
                print(f"  {k}: {v:.3f}")

    # Lưu tổng hợp cuối cùng
    if final_results:
        pd.DataFrame(final_results).T.to_csv("hybrid_search_results.csv", encoding="utf-8-sig")
        print("\n✅ Đã lưu tổng hợp vào hybrid_search_results.csv")

if __name__ == "__main__":
    main()