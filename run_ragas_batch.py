"""
Chạy RAGAS đánh giá từng batch 100 câu (4 Metrics Version)
========================================
Tương thích ragas v0.4+
Cách dùng:
  python run_ragas_batch.py --batch 1 --kb kb1
  python run_ragas_batch.py --batch 2 --kb kb1
  ...
"""

import argparse
import gc
import time
import os
import random
import re
import warnings
import logging
import torch
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

load_dotenv(override=True)

# ── Parse argument ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--kb",    type=str, default="kb1")
args = parser.parse_args()

BATCH_NUM   = args.batch
KB_NAME     = args.kb
INPUT_FILE  = f"{KB_NAME}_batch_{BATCH_NUM}.csv"
OUTPUT_FILE = f"result_{KB_NAME}_batch_{BATCH_NUM}.csv" # File output cuối cùng

print(f"\n{'='*60}")
print(f"  RAGAS 4-Metrics Evaluation — {KB_NAME.upper()} Batch {BATCH_NUM}")
print(f"  Input : {INPUT_FILE}")
print(f"  Output: {OUTPUT_FILE}")
print(f"{'='*60}\n")

# ── Load dữ liệu ───────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
print(f"Đã load {len(df)} câu hỏi từ {INPUT_FILE}")

# ── Setup Keys ─────────────────────────────────────────────────────────────
JUDGE_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]

if not JUDGE_KEYS:
    raise ValueError("❌ Không có GROQ_API_KEY trong .env!")
print(f"✅ Có {len(JUDGE_KEYS)} Groq key(s)")

# ── Import RAGAS & Metrics ─────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset

from ragas.metrics import (
    Faithfulness,       
    ContextRecall,      
    AnswerRelevancy,    
    ContextPrecision,   
)

METRIC_NAMES = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']

ERROR_PATTERNS = [
    "lỗi llm", "error code", "rate limit", "429",
    "không thể kết nối", "không tìm thấy thông tin trong tài liệu"
]

def is_valid_answer(ans: str) -> bool:
    if not ans or not ans.strip(): return False
    if any(p in ans.lower() for p in ERROR_PATTERNS[:4]): return False
    if len(ans.strip()) < 5: return False
    return True

def get_judge_llm():
    return LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=random.choice(JUDGE_KEYS))
    )

def get_judge_embeddings():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )

print("✅ Metrics: Faithfulness | ContextRecall | AnswerRelevancy | ContextPrecision")

# ── Import RAGChain ────────────────────────────────────────────────────────
from llm_chain import RAGChain
chain = RAGChain(k=5) # Đổi thành k=5 cho đồng bộ

# ── CHECKPOINT & RAG LOOP ──────────────────────────────────────────────────
results_map   = {}
CHECKPOINT_FILE = f"ragas_checkpoint_{KB_NAME}_batch_{BATCH_NUM}.csv"

if os.path.exists(CHECKPOINT_FILE):
    ckpt_df = pd.read_csv(CHECKPOINT_FILE, encoding='utf-8-sig')
    results_map = {r['question']: r for r in ckpt_df.to_dict('records')}
    print(f"⚡ Checkpoint: đã có {len(results_map)} câu — sẽ skip câu đã có điểm")

data_rows = []
for i, row in df.iterrows():
    q  = str(row["question"])
    gt = str(row.get("ground_truth", ""))

    # Skip nếu đã chấm đủ 4 metrics
    if q in results_map and all(pd.notna(results_map[q].get(m)) for m in METRIC_NAMES):
        continue

    print(f"\n{'─'*60}")
    print(f"  [{i+1:>3}/{len(df)}] {q[:60]}...")

    try:
        res    = chain.invoke({"question": q, "history": []})
        answer = res.get("answer", "")
        docs   = res.get("docs",   [])
        
        # ── DEBUG IN DOCS (Giúp bạn xem Vectordb lấy đúng chưa) ──
        print(f"\n  📄 RETRIEVED DOCS ({len(docs)} đoạn):")
        for j, doc in enumerate(docs):
            src     = doc.metadata.get('source', 'N/A')
            preview = doc.page_content[:120].replace('\n', ' ')
            print(f"    [{j+1}] {src} -> {preview}...")

        if not is_valid_answer(answer):
            print(f"  ⚠️ Answer rỗng — dùng fallback")
            answer = "Không có câu trả lời"  # fallback thay vì skip

        print(f"\n  🤖 ANSWER: {answer[:150]}...")
        
        data_rows.append({
            "question":     q,
            "answer":       answer,
            "contexts":     [d.page_content[:1200] for d in docs] if docs else [""],
            "ground_truth": gt,
        })
    except Exception as e:
        print(f"  ❌ Lỗi RAG câu {i+1}: {str(e)[:80]}")

    time.sleep(3)

if not data_rows and not results_map:
    print("❌ Không có câu nào hợp lệ để đánh giá!")
    exit(1)

print(f"\n✅ RAG xong: {len(data_rows)} câu mới cần chấm")

# ── RAGAS EVALUATE TỪNG CÂU ───────────────────────────────────────────────
print("\n🔍 Bắt đầu chạy RAGAS 4-metrics evaluation...")
_embeddings = get_judge_embeddings()

def eval_one(item, max_retry=4):
    ds = Dataset.from_dict({
        "question":     [item["question"]],
        "answer":       [item["answer"]],
        "contexts":     [item["contexts"]],
        "ground_truth": [item["ground_truth"]],
    })
    
    for attempt in range(max_retry):
        try:
            llm = get_judge_llm()
            metrics = [
                Faithfulness(llm=llm),
                ContextRecall(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=_embeddings, strictness=1),
                ContextPrecision(llm=llm),
            ]
            
            res = evaluate(
                dataset=ds, 
                metrics=metrics,
                raise_exceptions=False,
                run_config=RunConfig(max_workers=1, timeout=300, max_retries=2)
            )
            return res.to_pandas().iloc[0].to_dict()

        except Exception as e:
            err = str(e)
            if "429" in err:
                m    = re.search(r'in (\d+)m([\d.]+)s', err)
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 15 if m else 90 * (attempt + 1)
                print(f"\n     ⏳ Rate limit — chờ {wait:.0f}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"\n     ⚠️ Lỗi: {err[:60]}", end=" ", flush=True)
                time.sleep(15)
    return None

for idx, item in enumerate(data_rows):
    q_text = item['question']
    print(f"\nEval [{idx+1}/{len(data_rows)}] {q_text[:45]}...", end=" ", flush=True)

    res_dict = eval_one(item)

    if res_dict:
        item.update(res_dict)
        f   = item.get('faithfulness',      float('nan'))
        cr  = item.get('context_recall',    float('nan'))
        ar  = item.get('answer_relevancy',  float('nan'))
        cp  = item.get('context_precision', float('nan'))

        def fmt(v): return f"{v:.2f}" if pd.notna(v) else "nan"
        print(f"✅ F={fmt(f)} | CR={fmt(cr)} | AR={fmt(ar)} | CP={fmt(cp)}")
    else:
        print("❌ Bỏ qua (hết retry)")
        for m in METRIC_NAMES: item.setdefault(m, None)

    results_map[q_text] = item
    
    # Lưu checkpoint từng câu
    pd.DataFrame(list(results_map.values())).to_csv(CHECKPOINT_FILE, index=False, encoding='utf-8-sig')
    
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    time.sleep(10) # Nghỉ 10s giữa các câu để không bị ban

# ── LƯU KẾT QUẢ CUỐI CÙNG ───────────────────────────────────────────────
final_df = pd.DataFrame(list(results_map.values()))
keep     = ['question', 'answer', 'contexts', 'ground_truth'] + METRIC_NAMES
final_df = final_df[[c for c in keep if c in final_df.columns]]
num_cols = [c for c in METRIC_NAMES if c in final_df.columns]
final_df[num_cols] = final_df[num_cols].apply(pd.to_numeric, errors='coerce').round(3)

final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# === BỔ SUNG: LỌC CÁC CÂU BỊ CHẶN/KHÔNG TÌM THẤY TRƯỚC KHI TÍNH MEAN ===
# Các pattern này tương ứng với guardrails hoặc fallback rỗng
INVALID_ANSWER_PATTERNS = [
    "không tìm thấy thông tin", "không thể hỗ trợ yêu cầu này", 
    "momcare không thể", "đưa bé đến cơ sở y tế để được thăm khám",
    "hệ thống ai đang quá tải", "không có câu trả lời"
]

# Tạo mask: True nếu câu trả lời HỢP LỆ (không bị chặn/không rỗng)
def is_valid_for_ragas(answer):
    if not answer or not isinstance(answer, str):
        return False
    ans_lower = answer.lower().strip()
    if len(ans_lower) < 15: return False
    for pattern in INVALID_ANSWER_PATTERNS:
        if pattern in ans_lower:
            return False
    return True

final_df['is_valid_rag'] = final_df['answer'].apply(is_valid_for_ragas)
valid_rag_df = final_df[final_df['is_valid_rag'] == True]
invalid_rag_df = final_df[final_df['is_valid_rag'] == False]

# Tính điểm CHO TẤT CẢ (Original)
f_avg  = final_df['faithfulness'].mean()      if 'faithfulness'      in final_df.columns else float("nan")
cr_avg = final_df['context_recall'].mean()    if 'context_recall'    in final_df.columns else float("nan")
ar_avg = final_df['answer_relevancy'].mean()  if 'answer_relevancy'  in final_df.columns else float("nan")
cp_avg = final_df['context_precision'].mean() if 'context_precision' in final_df.columns else float("nan")

# Tính điểm CHỈ CHO CÂU HỢP LỆ (Adjusted - Dùng để báo cáo)
f_adj  = valid_rag_df['faithfulness'].mean()      if 'faithfulness'      in valid_rag_df.columns and len(valid_rag_df) > 0 else float("nan")
cr_adj = valid_rag_df['context_recall'].mean()    if 'context_recall'    in valid_rag_df.columns and len(valid_rag_df) > 0 else float("nan")
ar_adj = valid_rag_df['answer_relevancy'].mean()  if 'answer_relevancy'  in valid_rag_df.columns and len(valid_rag_df) > 0 else float("nan")
cp_adj = valid_rag_df['context_precision'].mean() if 'context_precision' in valid_rag_df.columns and len(valid_rag_df) > 0 else float("nan")

def fmt(v): return f"{v:.3f}" if pd.notna(v) else "N/A"

print(f"\n{'='*60}")
print(f"  KẾT QUẢ BATCH {BATCH_NUM} — {KB_NAME.upper()} (4 Metrics)")
print(f"{'='*60}")
print(f"  Tổng số câu: {len(final_df)}")
print(f"  Câu hợp lệ (RAG thành công): {len(valid_rag_df)} ({len(valid_rag_df)/len(final_df)*100:.1f}%)")
print(f"  Câu bị chặn/thất bại: {len(invalid_rag_df)} ({len(invalid_rag_df)/len(final_df)*100:.1f}%)")

print(f"\n  ── ĐIỂM TOÀN BỘ (Gồm câu thất bại) ──")
print(f"    Faithfulness      : {fmt(f_avg)}")
print(f"    Context Recall    : {fmt(cr_avg)}")
print(f"    Answer Relevancy  : {fmt(ar_avg)}")
print(f"    Context Precision : {fmt(cp_avg)}")

print(f"\n  ── ĐIỂM ĐIỀU CHỈNH (Chỉ tính câu RAG hợp lệ) <- DÙNG CHO BÁO CÁO ──")
print(f"    Faithfulness      : {fmt(f_adj)}")
print(f"    Context Recall    : {fmt(cr_adj)}")
print(f"    Answer Relevancy  : {fmt(ar_adj)}")
print(f"    Context Precision : {fmt(cp_adj)}")
print(f"  Lưu tại             : {OUTPUT_FILE}")