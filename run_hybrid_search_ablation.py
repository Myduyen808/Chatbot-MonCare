"""
THỰC NGHIỆM SO SÁNH: VECTOR SEARCH (FAISS) vs HYBRID SEARCH (FAISS + BM25)
=======================================================================
Cách dùng:
    python run_hybrid_search_ablation.py
"""

import os, re, time, gc, warnings, logging, math, collections
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

load_dotenv(override=True)

# Import hệ thống hiện tại
from llm_chain import RAGChain, check_input_guardrails, check_output_guardrails, call_llm, _ALL_KEYS
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
TEST_FILE = "ablation_test_set.xlsx"
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
# HÀM TÍNH TOÁN TỪ KHÓA CHUẨN CHO BM25
# =====================================================================
def tokenize_vietnamese(text):
    """Tách từ cơ bản cho BM25, hỗ trợ tiếng Việt"""
    text = text.lower()
    return re.findall(r'[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*', text)

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
# HÀM TẠO RETRIEVER LAI (HYBRID: FAISS + BM25 - FIX LỖI MAPPING)
# =====================================================================
def get_hybrid_retriever():
    db = load_vector_db()
    
    # 1. Lấy toàn bộ documents từ FAISS docstore
    all_ids = list(db.index_to_docstore_id.values())
    all_docs = [db.docstore.search(doc_id) for doc_id in all_ids]
    
    # 2. Làm sạch văn bản và tạo bộ Corpus cho BM25
    corpus = []
    valid_docs = [] # Lưu danh sách doc thực tế tương ứng với vị trí trong corpus
    
    for doc in all_docs:
        if doc is None:
            continue
        clean_t = clean_chunk_text(doc.page_content)
        if len(clean_t) > 50:
            corpus.append(tokenize_vietnamese(clean_t))
            valid_docs.append(doc)

    # 3. Khởi tạo BM25 Index
    print("\n[Hệ thống] Đang khởi tạo BM25 Index cho tài liệu y tế...", end=" ", flush=True)
    bm25 = BM25Okapi(corpus)
    print("Hoàn tất!")
    
    # Tạo một bảng băm để tìm kiếm vị trí của văn bản gốc nhanh hơn khi score
    doc_to_index = {doc.page_content: idx for idx, doc in enumerate(valid_docs)}
    
    # 4. Hàm lai ghép tìm kiếm chính xác (FIX LỖI ĐƯỜ CĂU ĐIỂU ĐIỂM ĐIỂM TỪNG THÊM ĐIỂM)
    def hybrid_search(query):
        # Bước 1: Lấy pool 20 tài liệu ứng viên từ Vector DB (MMR)
        vector_docs = db.similarity_search(query, k=20, fetch_k=40, lambda_mult=0.5)
        
        # Bước 2: Tính điểm BM25 cho toàn bộ kho
        query_tokens = tokenize_vietnamese(query)
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Chuẩn hóa điểm BM25 về khoảng [0, 1] để cộng điểm không bị lệch pha toán học
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        if max_bm25 == 0: max_bm25 = 1.0
        
        # Bước 3: Tính điểm kết hợp (BM25 CHỈ ĐÓNG ĐIỂM TĂNG ĐIỂM THÊM, KHÔNG THAY THAY ĐIỂM GHIẢM ĐIỂM)
        combined_scores = []
        for i, vec_doc in enumerate(vector_docs):
            # Điểm Vector dựa trên thứ hạng (Reciprocal Rank)
            vector_score = 1.0 / (i + 1)
            
            # Tra cứu vị trí chính xác của vec_doc trong kho BM25 thông qua bảng băm
            bm25_idx = doc_to_index.get(vec_doc.page_content, -1)
            bm25_score = (bm25_scores[bm25_idx] / max_bm25) if bm25_idx != -1 else 0.0
            
            # Công thức lai ghép: 70% ý nghĩa Vector + 30% Từ khóa BM25
            final_score = (0.7 * vector_score) + (0.3 * bm25_score)
            combined_scores.append((final_score, vec_doc))
            
        # Sắp xếp lại theo điểm kết hợp giảm dần
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Trả về đúng top K_FINAL y tế chuẩn
        return [doc for _, doc in combined_scores[:K_FINAL]]

    return hybrid_search

# =====================================================================
# HÀM CHẠY RAG PIPELINE THỰC TẾ
# =====================================================================
def run_rag_with_custom_retriever(question, retriever_fn):
    # 1. Guardrails
    blocked_msg = check_input_guardrails(question)
    if blocked_msg:
        return {"answer": blocked_msg, "docs": []}

    # 2. Lấy tài liệu bằng hàm Retriever được truyền vào
    if callable(retriever_fn):
        docs = retriever_fn(question)
    else:
        docs = retriever_fn.invoke(question)
    
    if not docs:
        return {"answer": "Tôi chưa tìm thấy thông tin này.", "docs": []}

    # 3. Tạo Context và Prompt
    context = "\n\n".join([f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
    
    prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời câu hỏi CHỈ dựa trên tài liệu.
TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

TRẢ LỜI TRỰC TIẾP:"""
    
    # 4. Gọi LLM sinh câu trả lời
    answer = call_llm(prompt, temperature=0.1)
    answer = check_output_guardrails(answer)
    
    return {"answer": answer, "docs": docs}

# =====================================================================
# CHẠY THỰC NGHIỆM CHÍNH
# =====================================================================
def main():
    if not os.path.exists(TEST_FILE):
        print(f"❌ Không tìm thấy file kiểm thử {TEST_FILE}.")
        return

    df = pd.read_excel(TEST_FILE)
    print(f"🎯 Load thành công {len(df)} câu hỏi kiểm thử từ {TEST_FILE}\n")

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
            results = pd.read_csv(ckpt_file, encoding='utf-8-sig', encoding_errors='ignore').to_dict('records')
            print(f"⚡ Load thành công checkpoint cũ: {len(results)} câu")

        # Khởi tạo retriever một lần duy nhất cho mỗi thí nghiệm để tối ưu RAM
        current_retriever = retriever_fn()

        data_rows = []
        for i, row in df.iterrows():
            q = str(row["question"])
            if q in [str(r.get("question", "")) for r in results]:
                continue

            print(f"[{i+1}/{len(df)}] {q[:50]}...", end=" ", flush=True)
            
            res = run_rag_with_custom_retriever(q, current_retriever)
            
            if len(res["docs"]) == 0:
                print("❌ Không lấy được tài liệu")
                continue
                
            data_rows.append({
                "question": q,
                "回答": res["answer"], 
                "contexts": [d.page_content[:1200] for d in res["docs"]],
                "ground_truth": str(row.get("ground_truth", "")),
            })
            print(f"✅", end=" ")
            time.sleep(DELAY)

        if not data_rows and len(results) == 0:
            print("Không có dữ liệu câu hỏi mới.")
            continue

        if data_rows:
            print(f"\n🔍 Đang chấm điểm RAGAS cho {len(data_rows)} câu mới phát sinh...")
            for idx, item in enumerate(data_rows):
                print(f"  Chấm RAGAS [{idx+1}/{len(data_rows)}]", end=" ", flush=True)
                
                if "回答" in item:
                    item["answer"] = item.pop("回答")
                    
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
                            print(f"⏳ Hết lượt API, chờ {wait:.0f}s", end=" ", flush=True)
                            time.sleep(wait)
                        else:
                            print(f"⚠️ Lỗi kết nối, thử lại", end=" ", flush=True)
                            time.sleep(10)
                
                results.append(item)
                pd.DataFrame(results).to_csv(ckpt_file, index=False, encoding="utf-8-sig")
                gc.collect()
                time.sleep(DELAY)

        # Tính toán kết quả trung bình
        res_df = pd.DataFrame(results)
        if len(res_df) > 0:
            cr_col = 'context_recall' if 'context_recall' in res_df.columns else 'context_recall'
            cp_col = 'context_precision' if 'context_precision' in res_df.columns else 'context_precision'
            ar_col = 'answer_relevancy' if 'answer_relevancy' in res_df.columns else 'answer_relevancy'
            f_col = 'faithfulness' if 'faithfulness' in res_df.columns else 'faithfulness'

            # --- SỬA LỖI ĐÚNG CHUẨN TOÁN HỌC VÀ TÊN BIẾN ---
            final_results[name] = {
                "Context Recall (CR)": res_df[cr_col].mean(),
                "Context Precision (CP)": res_df[cp_col].mean(),
                "Answer Relevancy (AR)": res_df[ar_col].mean(),
                "Faithfulness (F)": res_df[f_col].mean()
            }
            print(f"\n📊 BẢNG ĐIỂM TRUNG BÌNH [{name}]:")
            for k, v in final_results[name].items():
                print(f"  {k}: {v:.3f}")

    # --- SỬA LỖI CHÍNH TẢ ENCODING LÚC LƯU FILE ---
    if final_results:
        pd.DataFrame(final_results).T.to_csv("hybrid_search_results.csv", encoding="utf-8-sig")
        print("\n🎉 XONG RỒI! Đã lưu bảng tổng hợp đối chứng vào file: hybrid_search_results.csv")

if __name__ == "__main__":
    main()