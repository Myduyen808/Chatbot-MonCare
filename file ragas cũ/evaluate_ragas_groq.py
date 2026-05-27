"""
RAGAS Evaluate - Full Fixed Version
- Tách key: llm_chain dùng GROQ_API_KEY, judge dùng GROQ_API_KEY_1/2/3
- Eval từng câu → không mất dữ liệu nếu lỗi
- Checkpoint → chạy lại tiếp tục từ chỗ dừng
- Lọc answer lỗi kỹ hơn (bao gồm "Lỗi LLM")
- In docs + answer để debug như cũ
"""
import os, gc, time, random, warnings, logging, re
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

load_dotenv(override=True)

# ══════════════════════════════════════════════════════
# 1. SETUP KEYS
# ══════════════════════════════════════════════════════
JUDGE_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]

if not JUDGE_KEYS:
    main_key = os.getenv("GROQ_API_KEY")
    if not main_key:
        raise ValueError("❌ Thiếu GROQ_API_KEY trong .env!")
    JUDGE_KEYS = [main_key]
    print("⚠️  Chỉ có 1 key — dễ bị rate limit. Nên tạo thêm tại console.groq.com")
else:
    print(f"✅ Có {len(JUDGE_KEYS)} judge key(s)")

from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ContextRecall
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset
from langchain_groq import ChatGroq

def get_judge_llm():
    key = random.choice(JUDGE_KEYS)
    return LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=key)
    )

METRIC_NAMES = ['faithfulness', 'context_recall']

# Các từ khóa nhận biết answer bị lỗi
ERROR_PATTERNS = [
    "lỗi llm", "error code", "rate limit", "429",
    "không thể kết nối", "không tìm thấy thông tin trong tài liệu"
]

def is_valid_answer(ans: str) -> bool:
    if not ans or not ans.strip():
        return False
    ans_lower = ans.lower()
    # Lọc câu trả lời là thông báo lỗi
    if any(p in ans_lower for p in ERROR_PATTERNS[:4]):  # chỉ lọc lỗi kỹ thuật
        return False
    if len(ans.strip()) < 10:
        return False
    return True

print("✅ Metrics: Faithfulness + ContextRecall")
print("=" * 60)

# ══════════════════════════════════════════════════════
# 2. CHECKPOINT
# ══════════════════════════════════════════════════════
CHECKPOINT_FILE = 'ragas_checkpoint.csv'

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE, encoding='utf-8-sig')
        print(f"📂 Checkpoint: đã có {len(df)} câu — sẽ skip câu đã có điểm")
        return df
    return pd.DataFrame()

def save_checkpoint(results_map: dict):
    pd.DataFrame(list(results_map.values())).to_csv(
        CHECKPOINT_FILE, index=False, encoding='utf-8-sig'
    )

checkpoint_df = load_checkpoint()
results_map   = {r['question']: r for r in checkpoint_df.to_dict('records')} \
                if not checkpoint_df.empty else {}

# ══════════════════════════════════════════════════════
# 3. CHẠY RAG
# ══════════════════════════════════════════════════════
from llm_chain import RAGChain

df_input = pd.read_excel('KB1_Medical_Standard.xlsx').head(30)
chain = RAGChain(k=3)

print(f"\n--- 🧠 B1: Chạy RAG ({len(df_input)} câu) ---\n")
data_rows = []

for idx, (i, row) in enumerate(df_input.iterrows()):
    q   = str(row['Câu hỏi người dùng (Input)'])
    ref = str(row['Phản hồi kỳ vọng (Expected Output)'])

    # Skip nếu đã có điểm đầy đủ trong checkpoint
    if q in results_map:
        item = results_map[q]
        if all(pd.notna(item.get(m)) for m in METRIC_NAMES):
            print(f"  ⏭️  [{idx+1}/{len(df_input)}] Skip: {q[:45]}...")
            continue

    print(f"{'─'*60}")
    print(f"  [{idx+1}/{len(df_input)}] {q[:60]}...")

    try:
        res  = chain.invoke({"question": q, "history": []})
        ans  = res.get("answer", "")
        docs = res.get("docs", [])

        # ── In docs để debug ──────────────────────────────
        print(f"\n  📄 RETRIEVED DOCS ({len(docs)} đoạn):")
        for j, doc in enumerate(docs):
            src     = doc.metadata.get('source', 'N/A')
            preview = doc.page_content[:120].replace('\n', ' ')
            print(f"    [{j+1}] {src}")
            print(f"         → {preview}...")

        # ── Kiểm tra retrieval ────────────────────────────
        all_ctx  = " ".join([d.page_content for d in docs]).lower()
        kw_check = any(kw in all_ctx for kw in q.lower().split()[:3])
        print(f"\n  🔍 Retrieval: {'✅ Relevant' if kw_check else '⚠️ Có thể không liên quan'}")


        # ── Lọc answer lỗi ────────────────────────────────
        if not is_valid_answer(ans):
            raise ValueError(f"Answer không hợp lệ: {ans[:60]}")
        
         # ── In answer để debug ──
        print(f"\n  🤖 ANSWER: {ans[:200]}...")
        print(f"  🔍 Retrieval: {'✅ Relevant' if kw_check else '⚠️ Không liên quan'}")


        data_rows.append({
            "question":  q,
            "answer":    ans,
            "contexts":  [d.page_content[:400] for d in docs] if docs else [""],
            "reference": ref,
        })
        print(f"  ✅ RAG OK — {len(docs)} docs")

    except Exception as e:
        print(f"  ❌ Lỗi: {str(e)[:80]}")

    gc.collect()
    time.sleep(1)

valid_count    = len(data_rows)
skip_count     = len([q for q in results_map
                      if all(pd.notna(results_map[q].get(m)) for m in METRIC_NAMES)])
print(f"\n{'='*60}")
print(f"✅ RAG xong: {valid_count} câu mới | {skip_count} câu đã có điểm từ checkpoint")

if not data_rows and not results_map:
    print("❌ Không có câu nào hợp lệ. Kiểm tra lại llm_chain hoặc key Groq.")
    exit(1)

# ══════════════════════════════════════════════════════
# 4. RAGAS EVALUATE
# ══════════════════════════════════════════════════════
print("\n--- 📊 B2: RAGAS Evaluate ---")

def eval_one(item, max_retry=4):
    ds = Dataset.from_dict({
        "question":  [item["question"]],
        "answer":    [item["answer"]],
        "contexts":  [item["contexts"]],
        "reference": [item["reference"]],
    })
    for attempt in range(max_retry):
        try:
            metrics = [
                Faithfulness(llm=get_judge_llm()),
                ContextRecall(llm=get_judge_llm()),
            ]
            res = evaluate(
                ds, metrics=metrics,
                run_config=RunConfig(max_workers=1, timeout=600)
            )
            return res.to_pandas().iloc[0].to_dict()

        except Exception as e:
            err = str(e)
            if "429" in err:
                m    = re.search(r'in (\d+)m([\d.]+)s', err)
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 15 if m else 90 * (attempt + 1)
                print(f"\n     ⏳ Rate limit — chờ {wait:.0f}s (lần {attempt+1}/{max_retry})...")
                time.sleep(wait)
            elif "timeout" in err.lower():
                wait = 40 * (attempt + 1)
                print(f"\n     ⏳ Timeout — chờ {wait}s (lần {attempt+1}/{max_retry})...")
                time.sleep(wait)
            else:
                print(f"\n     ⚠️  Lỗi: {err[:80]}")
                time.sleep(15)
    return None

for idx, item in enumerate(data_rows):
    q_text = item['question']
    print(f"\nEval [{idx+1}/{len(data_rows)}] {q_text[:50]}...", end=" ", flush=True)

    res_dict = eval_one(item)

    if res_dict:
        item.update(res_dict)
        f  = item.get('faithfulness',    float('nan'))
        cr = item.get('context_recall',  float('nan'))
        f_str  = f"{f:.3f}"  if pd.notna(f)  else "nan"
        cr_str = f"{cr:.3f}" if pd.notna(cr) else "nan"
        print(f"✅ faith={f_str} | recall={cr_str}")
    else:
        print("❌ Bỏ qua (hết retry)")
        for m in METRIC_NAMES:
            item.setdefault(m, None)

    results_map[q_text] = item
    save_checkpoint(results_map)
    gc.collect()

    if (idx + 1) % 5 == 0:
        print(f"   💤 Nghỉ 20s sau {idx+1} câu...")
        time.sleep(20)
    else:
        time.sleep(8)

# ══════════════════════════════════════════════════════
# 5. BÁO CÁO CUỐI
# ══════════════════════════════════════════════════════
final_df = pd.DataFrame(list(results_map.values()))
keep     = ['question', 'answer'] + METRIC_NAMES
final_df = final_df[[c for c in keep if c in final_df.columns]]
num_cols = [c for c in METRIC_NAMES if c in final_df.columns]
final_df[num_cols] = final_df[num_cols].apply(pd.to_numeric, errors='coerce').round(3)
final_df = final_df.drop_duplicates(subset=['question'], keep='last')

final_df.to_csv('ragas_pro_report.csv',    index=False, encoding='utf-8-sig')
final_df.to_excel('ragas_pro_report.xlsx', index=False)

print("\n" + "🏆" * 25)
print(f"🎯 KẾT QUẢ RAGAS ({len(final_df)} câu)")
print(final_df[['question'] + num_cols].to_string(index=False))

print("\n📈 TRUNG BÌNH:")
for col in num_cols:
    avg = final_df[col].mean()
    n   = final_df[col].notna().sum()
    nan = final_df[col].isna().sum()
    if pd.notna(avg):
        flag = "✅" if avg >= 0.7 else "⚠️  Cần cải thiện"
        print(f"  {col:20s}: {avg:.3f}  ({n} hợp lệ / {nan} NaN)  {flag}")
    else:
        print(f"  {col:20s}: N/A")

print(f"\n✅ Lưu: ragas_pro_report.csv | .xlsx")
print("🏆" * 25)