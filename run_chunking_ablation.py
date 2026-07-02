"""
CHẠY THỰC NGHIỆM BIẾN THIÊN THAM SỐ CHUNKING (ABLATION STUDY)
- Tự động xoá DB cũ -> Chia chunk mới -> Đánh giá RAGAS -> Lưu kết quả
- Thử nghiệm 5 cấu hình (bao gồm cả Chunk > 2000)
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
# 1. CẤU HÌNH THỰC NGHIỆM (THEO YÊU CẦU THẦY)
# ══════════════════════════════════════════════════════
CONFIGS = [
    #{"chunk_size": 512,  "chunk_overlap": 100, "name": "CS512_OV100"},
    #{"chunk_size": 1000, "chunk_overlap": 200, "name": "CS1000_OV200"},
    {"chunk_size": 2000, "chunk_overlap": 400, "name": "CS2000_OV400 (Baseline)"},
    {"chunk_size": 3000, "chunk_overlap": 600, "name": "CS3000_OV600"},
    {"chunk_size": 4000, "chunk_overlap": 800, "name": "CS4000_OV800"},
]

# Đường dẫn file test (Lấy 20 câu đại diện cho KB1 để chạy nhanh)
INPUT_FILE = 'KB1_Medical_Standard.xlsx'
TEST_SAMPLE_SIZE = 10 
OUTPUT_CSV = 'experiments/results_chunking_ablation.csv'

print(f"🚀 Sẽ thử nghiệm {len(CONFIGS)} cấu hình, mỗi cấu hình {TEST_SAMPLE_SIZE} câu.")

# ══════════════════════════════════════════════════════
# 2. HÀM TIỆN ÍCH XÓA CACHE DB
# ══════════════════════════════════════════════════════
def clear_vector_db_cache():
    import vectordb
    if hasattr(vectordb, '_vector_db_cache'):
        vectordb._vector_db_cache = None
    if hasattr(vectordb, '_hybrid_retriever_cache'):
        vectordb._hybrid_retriever_cache = {"bm25": None, "valid_docs": None, "doc_to_index": None}
    print("🗑️ Đã xóa cache DB.")

# ══════════════════════════════════════════════════════
# 3. CORE: HÀM CHẠY 1 THỰC NGHIỆM
# ══════════════════════════════════════════════════════
def run_single_experiment(chunk_size, chunk_overlap, config_name):
    import importlib
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
                
            # QUAN TRỌNG: Giới hạn context gửi cho RAGAS để không bị tràn RAM
            # Khi chunk = 4000, gộp 5 đoạn sẽ là 20,000 từ -> RAGAS sẽ sập
            max_char_per_doc = min(1200, chunk_size // 2) 
            contexts = [d.page_content[:max_char_per_doc] for d in docs] if docs else [""]
            
            rag_results.append({
                "question": q,
                "answer": ans,
                "contexts": contexts,
                "reference": ref
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
    judge_key = random.choice([k for k in [
        os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")
    ] if k])
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
        scores = res.to_pandas().iloc[0].to_dict()
        
        result = {
            "Config": config_name,
            "Chunk_Size": chunk_size,
            "Overlap": chunk_overlap,
            "Faithfulness": scores.get("faithfulness"),
            "Context_Recall": scores.get("context_recall"),
            "Answer_Relevancy": scores.get("answer_relevancy"),
            "Context_Precision": scores.get("context_precision"),
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
            df_temp.to_csv(OUTPUT_CSV, index=False)
            print(f"💾 [QUAN TRỌNG] Đã lưu kết quả tạm thời vào {OUTPUT_CSV}")
        else:
            print(f"⚠️ Cấu hình {cfg['name']} thất bại. Đã bỏ qua.")
            
        gc.collect()
        time.sleep(30) # Nghỉ 30s giữa các vòng để giải phóng RAM và reset API limit
        
    print("\n" + "🎉" * 20)
    print("🏆 TỔNG KẾT THỰC NGHIỆM CHUNKING")
    print(pd.DataFrame(all_results).to_string(index=False))
    print(f"\n📁 Đã lưu kết quả CUỐI CÙNG tại: {OUTPUT_CSV}")