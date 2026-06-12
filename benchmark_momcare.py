"""
BENCHMARK MOMCARE — Chạy 500 câu/ngày, tích lũy đến hết 4,497 câu
====================================================================
Cách dùng hàng ngày:
    python benchmark_momcare.py            <- tự động chạy 500 câu tiếp theo
    python benchmark_momcare.py --report   <- in báo cáo cuối (khi đã xong hết)
    python benchmark_momcare.py --status   <- xem đã chạy được bao nhiêu câu
    python benchmark_momcare.py --n 200    <- chạy 200 câu (tuỳ chỉnh)

Kết quả tích lũy:
    benchmark_checkpoint_all.csv   <- tất cả câu đã chạy (không xoá)
    benchmark_summary.csv          <- bảng so sánh với bài báo
    benchmark_report.txt           <- báo cáo cho đồ án
"""

import os, argparse, gc, time, math, warnings, logging
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# ════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ════════════════════════════════════════════════════════════════

INPUT_EXCEL     = "Bo_De_Me_Va_Be.xlsx"
CHECKPOINT_FILE = "benchmark_checkpoint_all.csv"
OUTPUT_SUMMARY  = "benchmark_summary.csv"
OUTPUT_REPORT   = "benchmark_report.txt"

CAUHOI_MOI_NGAY = 500
SLEEP_BETWEEN   = 3
TOPIC_MAP = {0: "Body Part", 1: "Disease", 2: "Drug", 3: "Medicine"}

VIMEDAQA_BASELINE = {
    "Llama3-7B"     : {"bert": 71.36, "bleu": 25.33, "meteor": 67.97, "rouge_l": 55.52, "avg": 55.05},
    "Llama2-7B"     : {"bert": 41.65, "bleu":  6.93, "meteor": 24.36, "rouge_l": 24.34, "avg": 24.32},
    "Gemma-2B"      : {"bert": 64.28, "bleu": 32.04, "meteor": 53.48, "rouge_l": 53.57, "avg": 50.84},
    "Gemma-7B"      : {"bert": 68.49, "bleu": 31.17, "meteor": 63.52, "rouge_l": 57.03, "avg": 55.05},
    "PhoGPT-4B"     : {"bert": 68.94, "bleu": 21.06, "meteor": 59.76, "rouge_l": 50.75, "avg": 50.13},
    "VinaLlama-7B"  : {"bert": 72.47, "bleu": 31.70, "meteor": 64.29, "rouge_l": 59.08, "avg": 56.89},
    "VinaLlama-2.7B": {"bert": 70.09, "bleu": 26.07, "meteor": 59.77, "rouge_l": 54.96, "avg": 52.72},
    "ViGPT"         : {"bert": 59.07, "bleu": 10.94, "meteor": 44.39, "rouge_l": 34.27, "avg": 37.17},
}
BEST_MODEL    = "VinaLlama-7B"
BEST_BASELINE = VIMEDAQA_BASELINE[BEST_MODEL]

# ════════════════════════════════════════════════════════════════
# PARSE ARGUMENT
# ════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--report", action="store_true")
parser.add_argument("--status", action="store_true")
parser.add_argument("--n", type=int, default=None)
args = parser.parse_args()

if args.n:
    CAUHOI_MOI_NGAY = args.n

# ════════════════════════════════════════════════════════════════
# LOAD DỮ LIỆU & CHECKPOINT
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  BENCHMARK MOMCARE RAG vs ViMedAQA")
print("="*60)

df_all = pd.read_excel(INPUT_EXCEL)
df_all.columns = [str(c).strip().lower() for c in df_all.columns]
TOTAL = len(df_all)
print(f"\n  Tong cau hoi     : {TOTAL}")

if os.path.exists(CHECKPOINT_FILE):
    df_done = pd.read_csv(CHECKPOINT_FILE, encoding="utf-8-sig")
    done_questions = set(df_done["question"].tolist())
else:
    df_done = pd.DataFrame()
    done_questions = set()

done_count = len(done_questions)
remain     = TOTAL - done_count

print(f"  Da chay xong    : {done_count} cau")
print(f"  Con lai         : {remain} cau")
print(f"  Hom nay se chay : {min(CAUHOI_MOI_NGAY, remain)} cau")
print(f"  Can them        : ~{math.ceil(remain/CAUHOI_MOI_NGAY)} ngay nua\n")

# ── Chế độ --status ──
if args.status:
    if done_count > 0:
        print("  Diem trung binh hien tai:")
        for m in ["rouge_l", "bleu", "meteor", "bertscore"]:
            if m in df_done.columns:
                v = pd.to_numeric(df_done[m], errors="coerce").mean()
                print(f"    {m:<12}: {v:.3f}" if pd.notna(v) else f"    {m:<12}: N/A")
    exit(0)

# ════════════════════════════════════════════════════════════════
# KHỞI TẠO METRICS
# ════════════════════════════════════════════════════════════════

if not args.report:
    print("  Khoi tao metrics...")

    try:
        from rouge_score import rouge_scorer as _rs
        _rouge = _rs.RougeScorer(["rougeL"], use_stemmer=False)
        HAS_ROUGE = True; print("    OK ROUGE-L")
    except ImportError:
        HAS_ROUGE = False; print("    MISSING rouge-score  =>  pip install rouge-score")

    try:
        from bert_score import score as _bert_fn
        HAS_BERT = True; print("    OK BERTScore")
    except ImportError:
        HAS_BERT = False; print("    MISSING bert-score   =>  pip install bert-score")

    try:
        import nltk
        nltk.download("punkt",   quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.translate.bleu_score   import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        HAS_BLEU = HAS_METEOR = True; print("    OK BLEU + METEOR")
    except Exception:
        HAS_BLEU = HAS_METEOR = False; print("    MISSING nltk  =>  pip install nltk")

    try:
        from llm_chain import RAGChain
        chain  = RAGChain(k=5)
        RAG_OK = True; print("    OK RAGChain\n")
    except Exception as e:
        RAG_OK = False; print(f"    FAIL RAGChain: {e}\n")

# ════════════════════════════════════════════════════════════════
# HÀM TÍNH METRICS
# ════════════════════════════════════════════════════════════════

def calc_rouge(pred, ref):
    if not HAS_ROUGE or not pred or not ref: return float("nan")
    try: return round(_rouge.score(ref, pred)["rougeL"].fmeasure * 100, 3)
    except: return float("nan")

def calc_bleu(pred, ref):
    if not HAS_BLEU or not pred or not ref: return float("nan")
    try:
        sf = SmoothingFunction().method1
        return round(sentence_bleu([ref.split()], pred.split(), smoothing_function=sf) * 100, 3)
    except: return float("nan")

def calc_meteor(pred, ref):
    if not HAS_METEOR or not pred or not ref: return float("nan")
    try: return round(meteor_score([ref.split()], pred.split()) * 100, 3)
    except: return float("nan")

def calc_bertscore_batch(preds, refs):
    if not HAS_BERT: return [float("nan")] * len(preds)
    try:
        _, _, F = _bert_fn(preds, refs, lang="vi",
                           model_type="bert-base-multilingual-cased",
                           verbose=False)
        return [round(f.item() * 100, 3) for f in F]
    except Exception as e:
        print(f"  BERTScore loi: {e}")
        return [float("nan")] * len(preds)

# ════════════════════════════════════════════════════════════════
# CHẠY RAG HÔM NAY (500 câu tiếp theo)
# ════════════════════════════════════════════════════════════════

if not args.report:
    df_pending = df_all[~df_all["question"].isin(done_questions)].reset_index(drop=True)

    if len(df_pending) == 0:
        print("Da chay xong toan bo! Dung --report de tao bao cao.")
        exit(0)

    df_today = df_pending.head(CAUHOI_MOI_NGAY).reset_index(drop=True)
    print(f"  Chay {len(df_today)} cau (#{done_count+1} -> #{done_count+len(df_today)})\n")

    new_rows = []
    for i, row in df_today.iterrows():
        q  = str(row["question"])
        gt = str(row["answer"])
        t  = TOPIC_MAP.get(int(row["topic"]) if str(row["topic"]).isdigit() else -1, "Unknown")

        print(f"  [{done_count+i+1:>4}/{TOTAL}] ({t[:8]:<8}) {q[:50]}...", end=" ")

        if RAG_OK:
            try:
                res = chain.invoke({"question": q, "history": []})
                ans = res.get("answer", "").strip()
                if not ans or len(ans) < 5:
                    ans = "Khong tim thay thong tin."
            except Exception as e:
                ans = "Loi he thong."
                print(f"\n    LOI: {str(e)[:50]}", end=" ")
        else:
            ans = "RAG_NOT_AVAILABLE"

        r  = calc_rouge(ans, gt)
        bl = calc_bleu(ans, gt)
        me = calc_meteor(ans, gt)

        r_str  = f"{r:.2f}"  if pd.notna(r)  else "nan"
        bl_str = f"{bl:.2f}" if pd.notna(bl) else "nan"
        print(f"R={r_str} B={bl_str}")

        new_rows.append({
            "question"      : q,
            "ground_truth"  : gt,
            "momcare_answer": ans,
            "topic"         : t,
            "rouge_l"       : r,
            "bleu"          : bl,
            "meteor"        : me,
            "bertscore"     : float("nan"),
        })

        time.sleep(SLEEP_BETWEEN)

    # BERTScore cả batch một lần
    print("\n  Tinh BERTScore cho batch hom nay...")
    bs = calc_bertscore_batch(
        [r["momcare_answer"] for r in new_rows],
        [r["ground_truth"]   for r in new_rows]
    )
    for r, v in zip(new_rows, bs):
        r["bertscore"] = v

    # Ghi vào checkpoint tích lũy
    df_new  = pd.DataFrame(new_rows)
    df_done = pd.concat([df_done, df_new], ignore_index=True)
    df_done.to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  Da luu checkpoint: {len(df_done)}/{TOTAL} cau")

    gc.collect()

# ════════════════════════════════════════════════════════════════
# TỔNG HỢP BÁO CÁO (chạy mỗi ngày)
# ════════════════════════════════════════════════════════════════

if args.report:
    if not os.path.exists(CHECKPOINT_FILE):
        print("Chua co du lieu. Chay it nhat 1 ngay truoc.")
        exit(1)
    df_done = pd.read_csv(CHECKPOINT_FILE, encoding="utf-8-sig")

df_res = df_done.copy()
for col in ["rouge_l", "bleu", "meteor", "bertscore"]:
    if col in df_res.columns:
        df_res[col] = pd.to_numeric(df_res[col], errors="coerce")

n_done = len(df_res)
pct    = n_done / TOTAL * 100

rg  = df_res["rouge_l"].mean()
bl  = df_res["bleu"].mean()
me  = df_res["meteor"].mean()
bs_ = df_res["bertscore"].mean()
av  = pd.Series([rg, bl, me, bs_]).mean()

def fmt(v): return f"{v:.2f}" if pd.notna(v) else "N/A"
def dlt(mv, bv):
    if pd.isna(mv): return "N/A"
    d = mv - bv
    return f"{'UP' if d >= 0 else 'DN'}{abs(d):.2f}"

# CSV so sánh
rows = []
for m, sc in VIMEDAQA_BASELINE.items():
    rows.append({"Model": f"{m} (bai bao)",
                 "ROUGE-L": sc["rouge_l"], "BLEU": sc["bleu"],
                 "METEOR": sc["meteor"],   "BERTScore": sc["bert"],
                 "Average": sc["avg"],     "Ghi chu": "Baseline ViMedAQA"})
rows.append({"Model": f"MomCare RAG (n={n_done}/{TOTAL})",
             "ROUGE-L": fmt(rg), "BLEU": fmt(bl),
             "METEOR": fmt(me),  "BERTScore": fmt(bs_),
             "Average": fmt(av), "Ghi chu": f"{pct:.1f}% hoan thanh"})
pd.DataFrame(rows).to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

# Báo cáo text
best = BEST_BASELINE
lines = [
    "="*60,
    f"  BENCHMARK MOMCARE RAG - Tien do {pct:.1f}%",
    "="*60,
    f"  Du lieu    : Bo_De_Me_Va_Be.xlsx",
    f"  Da chay    : {n_done} / {TOTAL} cau hoi",
    f"  Con lai    : {TOTAL-n_done} cau (~{math.ceil((TOTAL-n_done)/CAUHOI_MOI_NGAY)} ngay)",
    "",
    "-"*60,
    "  KET QUA MOMCARE RAG",
    "-"*60,
    f"  ROUGE-L   : {fmt(rg):<7}  (vs {BEST_MODEL}: {best['rouge_l']}  => {dlt(rg,  best['rouge_l'])})",
    f"  BLEU      : {fmt(bl):<7}  (vs {BEST_MODEL}: {best['bleu']}  => {dlt(bl,  best['bleu'])})",
    f"  METEOR    : {fmt(me):<7}  (vs {BEST_MODEL}: {best['meteor']}  => {dlt(me,  best['meteor'])})",
    f"  BERTScore : {fmt(bs_):<7}  (vs {BEST_MODEL}: {best['bert']}  => {dlt(bs_, best['bert'])})",
    f"  Average   : {fmt(av):<7}  (vs {BEST_MODEL}: {best['avg']}  => {dlt(av,  best['avg'])})",
    "",
    "-"*60,
    "  BANG SO SANH DAY DU",
    "-"*60,
]
for r in rows:
    mark = " <--" if "MomCare" in str(r["Model"]) else ""
    lines.append(
        f"  {str(r['Model'])[:30]:<31}"
        f" R={str(r['ROUGE-L']):>5}"
        f" B={str(r['BLEU']):>5}"
        f" M={str(r['METEOR']):>5}"
        f" BS={str(r['BERTScore']):>5}"
        f" Avg={str(r['Average']):>5}"
        f"{mark}"
    )

# Phân tích theo topic
lines += ["", "-"*60, "  PHAN TICH THEO CHU DE", "-"*60]
if "topic" in df_res.columns:
    for tn in TOPIC_MAP.values():
        sub = df_res[df_res["topic"] == tn]
        if len(sub) == 0: continue
        r2 = pd.to_numeric(sub["rouge_l"],   errors="coerce").mean()
        b2 = pd.to_numeric(sub["bertscore"], errors="coerce").mean()
        lines.append(f"  {tn:<12} ({len(sub):>4} cau)  ROUGE={fmt(r2)}  BERT={fmt(b2)}")

# Nhận xét tự động
improve = sum(1 for mv, bv in [
    (rg, best["rouge_l"]), (bl, best["bleu"]),
    (me, best["meteor"]),  (bs_, best["bert"])
] if pd.notna(mv) and mv >= bv)

lines += ["", "-"*60, "  NHAN XET CHO DO AN", "-"*60]
if improve >= 3:
    lines += [f"  MomCare RAG vuot troi {BEST_MODEL} tren {improve}/4 chi so.",
              "  => RAG + du lieu chuyen biet giup tra loi bam sat tai lieu y khoa."]
elif improve >= 1:
    lines += [f"  MomCare cai thien {improve}/4 chi so.",
              "  => BERTScore la chi so dang tin cay hon cho he thong RAG."]
else:
    lines += ["  Diem tuong duong hoac thap hon.",
              "  => Chi so n-gram khong do duoc chat luong cau tra loi tu do."]

lines += ["", f"  Nguon baseline: Table 1, ViMedAQA (ACL SRW 2024)", "="*60, ""]
report = "\n".join(lines)
print("\n" + report)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print(f"  Luu: {OUTPUT_SUMMARY}")
print(f"  Luu: {OUTPUT_REPORT}")

if TOTAL - n_done > 0:
    print(f"\n  Ngay mai: python benchmark_momcare.py")
    print(f"  Con {TOTAL-n_done} cau, can ~{math.ceil((TOTAL-n_done)/CAUHOI_MOI_NGAY)} ngay nua.")
else:
    print("\n  Da xong toan bo 4,497 cau! Bao cao cuoi da luu.")
