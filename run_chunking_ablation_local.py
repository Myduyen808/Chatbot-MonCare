"""
KHẢO SÁT ĐỘ NHẠY THAM SỐ CHUNKING QUANH CẤU HÌNH TRIỂN KHAI CUỐI

- So sánh 3 cấu hình: 1200/240, 1500/300 và 1800/360.
- Dùng cùng tập câu hỏi giữa các cấu hình.
- Rebuild VectorDB cho từng cấu hình.
- Chấm RAGAS và lưu GIÁ TRỊ TRUNG BÌNH của toàn bộ mẫu hợp lệ.

Lưu ý:
- vectordb.py hiện có hard limit MAX_CHUNK_CHARS = 1800.
- Vì vậy không dùng 2000/400, 3000/600, 4000/800 trong khảo sát này,
  tránh trường hợp các cấu hình thực tế đều bị ép xuống 1800 ký tự.
"""
import os, gc, time, random, warnings, logging
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
load_dotenv(override=True)

# ══════════════════════════════════════════════════════
# 1. CẤU HÌNH THỰC NGHIỆM CỤC BỘ QUANH 1800/360
# ══════════════════════════════════════════════════════
CONFIGS = [
    {"chunk_size": 1200, "chunk_overlap": 240, "name": "CS1200_OV240"},
    {"chunk_size": 1500, "chunk_overlap": 300, "name": "CS1500_OV300"},
    {"chunk_size": 1800, "chunk_overlap": 360, "name": "CS1800_OV360_FINAL"},
]

# Đường dẫn file test (Lấy 20 câu đại diện cho KB1 để chạy nhanh)
INPUT_FILE = 'KB1_Medical_Standard.xlsx'
TEST_SAMPLE_SIZE = 10 
OUTPUT_CSV = 'experiments/results_chunking_local_ablation.csv'

print(f"🚀 Sẽ thử nghiệm {len(CONFIGS)} cấu hình, mỗi cấu hình {TEST_SAMPLE_SIZE} câu.")

# ══════════════════════════════════════════════════════
# 2. HÀM TIỆN ÍCH XÓA CACHE DB
# ══════════════════════════════════════════════════════
def clear_vector_db_cache():
    """Xóa cả cache FAISS và cache BM25/Hybrid sau mỗi lần rebuild."""
    import vectordb
    import llm_chain

    if hasattr(vectordb, "reset_vector_db_cache"):
        vectordb.reset_vector_db_cache()
    elif hasattr(vectordb, "_vector_db_cache"):
        vectordb._vector_db_cache = None

    if hasattr(llm_chain, "reset_retrieval_caches"):
        llm_chain.reset_retrieval_caches()
    elif hasattr(llm_chain, "_hybrid_retriever_cache"):
        llm_chain._hybrid_retriever_cache = {
            "bm25": None,
            "valid_docs": None,
            "doc_to_index": None,
        }

    print("🗑️ Đã xóa cache FAISS và BM25/Hybrid.")

# ══════════════════════════════════════════════════════
# 3. CORE: HÀM CHẠY 1 THỰC NGHIỆM
# ══════════════════════════════════════════════════════
def run_single_experiment(chunk_size, chunk_overlap, config_name):
    from vectordb import create_vectordb_with_file
    from llm_chain import RAGChain
    
    print(f"\n{'='*60}")
    print(f"🧬 BẮT ĐẦU: {config_name}")
    print(f"{'='*60}")
    start_time = time.time()

    # B1: Tạo lại DB
    print("⏳ Đang chia nhỏ và tạo FAISS DB mới...")
    create_vectordb_with_file(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    clear_vector_db_cache()
    
    # B2: Nạp RAG chain mới
    chain = RAGChain(k=5)
    
    # B3: Chạy RAG lấy kết quả
    df_input = pd.read_excel(INPUT_FILE).sample(n=TEST_SAMPLE_SIZE, random_state=42)
    rag_results = []
    
    for idx, (_, row) in enumerate(df_input.iterrows()):
        q = str(row['Câu hỏi người dùng (Input)'])
        ref = str(row['Phản hồi kỳ vọng (Expected Output)'])
        print(f"  [{idx+1}/{TEST_SAMPLE_SIZE}] RAG: {q[:50]}...", end=" ", flush=True)
        
        try:
            res = chain.invoke({"question": q, "history": []})
            ans = res.get("answer", "")
            docs = res.get("docs", [])
            
            if not ans or len(ans) < 10 or "không thể kết nối" in ans.lower():
                print("❌ Lỗi trả lời")
                continue
                
            # Dùng đúng các context cuối cùng mà RAGChain trả về.
            # Không cắt mỗi document theo chunk_size như file cũ vì việc đó
            # làm context chấm RAGAS khác với context hệ thống thực tế.
            contexts = [str(d.page_content) for d in docs] if docs else [""]

            rag_results.append({
                "question": q,
                "answer": ans,
                "contexts": contexts,
                "reference": ref,
                "doc_count": len(docs),
                "context_chars": sum(len(c) for c in contexts),
            })
            print("✅")
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            
        time.sleep(1) # Nghỉ nhẹ cho API Groq
        
    if not rag_results:
        print("❌ Không có kết quả RAG hợp lệ!")
        return None

    # B4: Chạy RAGAS
    print(f"\n📊 Đang chấm điểm RAGAS ({len(rag_results)} câu)...")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, ContextRecall, AnswerRelevancy, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.run_config import RunConfig
    
    # Setup Judge
    judge_keys = [k for k in [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ] if k]
    if not judge_keys:
        raise RuntimeError("Không tìm thấy GROQ API key cho RAGAS judge.")
    judge_key = random.choice(judge_keys)
    judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=judge_key))
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={'normalize_embeddings': True} # Ép chuẩn hóa vector, giúp câu tiếng Việt có điểm相似 độ cao hơn
        )
    )

    ds = Dataset.from_dict({
        "question": [r["question"] for r in rag_results],
        "answer": [r["answer"] for r in rag_results],
        "contexts": [r["contexts"] for r in rag_results],
        "reference": [r["reference"] for r in rag_results],
    })

    try:
        metrics = [
            Faithfulness(llm=judge_llm),
            ContextRecall(llm=judge_llm),
            AnswerRelevancy(llm=judge_llm, embeddings=judge_emb, strictness=1),
            ContextPrecision(llm=judge_llm),
        ]
        
        res = evaluate(ds, metrics=metrics, run_config=RunConfig(max_workers=1, timeout=600))
        score_df = res.to_pandas()

        # File cũ dùng .iloc[0], tức chỉ lấy điểm của câu đầu tiên.
        # Ablation phải lấy trung bình trên toàn bộ mẫu hợp lệ.
        def metric_mean(column_name):
            if column_name not in score_df.columns:
                return None
            return pd.to_numeric(score_df[column_name], errors="coerce").mean()

        result = {
            "Config": config_name,
            "Chunk_Size": chunk_size,
            "Overlap": chunk_overlap,
            "Valid_Samples": len(rag_results),
            "Faithfulness": metric_mean("faithfulness"),
            "Context_Recall": metric_mean("context_recall"),
            "Answer_Relevancy": metric_mean("answer_relevancy"),
            "Context_Precision": metric_mean("context_precision"),
            "Avg_Docs": round(sum(r["doc_count"] for r in rag_results) / len(rag_results), 3),
            "Avg_Context_Chars": round(sum(r["context_chars"] for r in rag_results) / len(rag_results), 2),
            "Time_Minutes": round((time.time() - start_time) / 60, 2)
        }
        print(f"✅ Xong {config_name}!")
        return result
        
    except Exception as e:
        print(f"❌ Lỗi RAGAS ở {config_name}: {str(e)[:100]}")
        return None

# ══════════════════════════════════════════════════════
# 4. MAIN: CHẠY TỰ ĐỘNG & LƯU NGAY LẬP TỨC
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)
    
    # KIỂM TRA XEM CÓ KẾT QUẢ CŨ KHÔNG (Để chạy tiếp nếu bị dở dang)
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        all_results = df_existing.to_dict('records')
        print(f"📂 Đã nạp {len(all_results)} kết quả cũ. Sẽ chạy tiếp từ chỗ dừng.")
    else:
        all_results = []
    
    for cfg in CONFIGS:
        # >>> BỎ QUA NẾU ĐÃ CHẠY XONG <<<
        if any(r.get("Config") == cfg["name"] for r in all_results):
            print(f"\n⏭️ Bỏ qua: {cfg['name']} (Đã có kết quả sẵn)")
            continue

        result = run_single_experiment(cfg["chunk_size"], cfg["chunk_overlap"], cfg["name"])
        
        if result:
            all_results.append(result)
            
            # >>> SỬA LỖI CHÍNH: LƯU FILE CSV NGAY LẬP TỨC <<<
            df_temp = pd.DataFrame(all_results)
            df_temp.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"💾 [QUAN TRỌNG] Đã lưu kết quả tạm thời vào {OUTPUT_CSV}")
        else:
            print(f"⚠️ Cấu hình {cfg['name']} thất bại. Đã bỏ qua.")
            
        gc.collect()
        time.sleep(15) # Nghỉ giữa các cấu hình để giải phóng RAM và giảm rate-limit
        
    print("\n" + "🎉" * 20)
    print("🏆 TỔNG KẾT KHẢO SÁT CHUNKING QUANH 1800/360")
    print(pd.DataFrame(all_results).to_string(index=False))
    print(f"\n📁 Đã lưu kết quả CUỐI CÙNG tại: {OUTPUT_CSV}")