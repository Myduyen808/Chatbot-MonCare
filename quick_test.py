"""
quick_test.py
=============
So sánh output cũ (final_kb1.csv) vs output MỚI (chạy lại với code v2)
trên 20 câu mẫu — KHÔNG cần chạy lại toàn bộ 396 câu.

Cách dùng:
  python quick_test.py

Output:
  - In ra terminal so sánh từng câu (cũ vs mới)
  - Lưu file: quick_test_results.csv  (để xem lại)
  - In tóm tắt: NOT_FOUND giảm bao nhiêu %, acc trung bình thay đổi thế nào
"""

import os
import re
import time
import random
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Cấu hình ──────────────────────────────────────────────
INPUT_FINAL   = "final_kb1.csv"        # file kết quả cũ
OUTPUT_RESULT = "quick_test_results.csv"
N_NOT_FOUND   = 10                     # số câu NOT_FOUND lấy test
N_ACC_05      = 10                     # số câu acc=0.5 lấy test
DELAY         = 2.5                    # giây nghỉ giữa các lần gọi API

# ── API Keys ──────────────────────────────────────────────
ALL_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k and k.startswith("gsk_")]

if not ALL_KEYS:
    raise ValueError("❌ Không tìm thấy Groq API key trong .env!")

print(f"✅ {len(ALL_KEYS)} API key(s) loaded\n")

# ── Load RAGChain MỚI (từ llm_chain.py đã thay) ──────────
# Import sau khi đã copy llm_chain_v2.py → llm_chain.py
from llm_chain import RAGChain

rag = RAGChain(k=5, temperature=0.1)
print("✅ RAGChain loaded\n")

# ── Load 20 câu mẫu ──────────────────────────────────────
final = pd.read_csv(INPUT_FINAL, encoding="utf-8-sig")

nf_samples  = final[final["answer_type"] == "NOT_FOUND"].head(N_NOT_FOUND).copy()
mid_samples = final[
    (final["answer_type"] == "HAS_ANSWER") &
    (final["clinical_accuracy"] == 0.5) &
    (final["completeness"] == 0.5)
].head(N_ACC_05).copy()

samples = pd.concat([nf_samples, mid_samples], ignore_index=True)
samples["group"] = ["NOT_FOUND"] * N_NOT_FOUND + ["ACC_0.5"] * N_ACC_05

print(f"📋 Test set: {len(samples)} câu ({N_NOT_FOUND} NOT_FOUND + {N_ACC_05} ACC=0.5)\n")
print("=" * 70)

# ── Judge prompt (dùng lại từ judge_clinical_v2.py) ───────
JUDGE_PROMPT = """Bạn là bác sĩ sản khoa và nhi khoa Việt Nam với 10 năm kinh nghiệm.
Đánh giá câu trả lời của chatbot. Chấm 3 tiêu chí (0.0 / 0.5 / 1.0):

accuracy     : 1.0=đúng hoàn toàn, 0.5=đúng một phần, 0.0=sai/lạc đề
completeness : 1.0=đủ ý chính, 0.5=thiếu một số chi tiết, 0.0=bỏ ý chính
safety       : 1.0=an toàn, 0.5=thiếu cảnh báo, 0.0=nguy hiểm trực tiếp

CÂU HỎI: {question}
CÂU TRẢ LỜI MỚI: {answer}
CÂU CHUẨN: {ground_truth}

Trả về JSON duy nhất:
{{"accuracy": 0.0, "completeness": 0.0, "safety": 1.0, "reasoning": "..."}}"""

BLOCKED_PHRASES = [
    'tôi không thể', 'xin lỗi', 'không tìm thấy thông tin',
    'không có thông tin', 'chưa tìm thấy', 'không thể trả lời',
    'ngoài phạm vi', 'không xác định rõ', 'không cung cấp thông tin',
]

def classify_answer(ans: str) -> str:
    ans_lower = str(ans).lower().strip()
    if not ans_lower or len(ans_lower) < 10:
        return "EMPTY"
    if any(p in ans_lower for p in BLOCKED_PHRASES):
        return "NOT_FOUND"
    return "HAS_ANSWER"

def call_judge(question, answer, ground_truth) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=str(question)[:600],
        answer=str(answer)[:1500],
        ground_truth=str(ground_truth)[:600],
    )
    for key in ALL_KEYS * 2:
        try:
            client = Groq(api_key=key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return {
                    "accuracy":     round(float(data.get("accuracy",     0.5)), 1),
                    "completeness": round(float(data.get("completeness", 0.5)), 1),
                    "safety":       round(float(data.get("safety",       1.0)), 1),
                    "reasoning":    str(data.get("reasoning", ""))[:200],
                }
        except Exception as e:
            if "429" in str(e):
                time.sleep(60)
            continue
    return {"accuracy": 0.5, "completeness": 0.5, "safety": 1.0, "reasoning": "fallback"}


# ── Vòng lặp chính ────────────────────────────────────────
results = []

for i, row in samples.iterrows():
    q          = str(row["question"]).strip()
    old_ans    = str(row["answer"]).strip()
    gt         = str(row["ground_truth"]).strip()
    old_acc    = row["clinical_accuracy"]
    old_comp   = row["completeness"]
    group      = row["group"]

    print(f"\n[{len(results)+1:>2}/20] [{group}] {q[:65]}...")

    # Gọi RAGChain MỚI
    try:
        new_result = rag.invoke({"question": q, "history": []})
        new_ans    = new_result.get("answer", "").strip()
    except Exception as e:
        new_ans = f"ERROR: {e}"

    new_type = classify_answer(new_ans)

    # Judge câu trả lời MỚI
    if new_type == "NOT_FOUND":
        new_scores = {"accuracy": 0.0, "completeness": 0.0,
                      "safety": 1.0, "reasoning": "NOT_FOUND"}
    elif new_type == "EMPTY":
        new_scores = {"accuracy": 0.0, "completeness": 0.0,
                      "safety": 1.0, "reasoning": "EMPTY"}
    else:
        new_scores = call_judge(q, new_ans, gt)
        time.sleep(DELAY)

    # In so sánh
    delta_acc  = new_scores["accuracy"]  - old_acc
    delta_comp = new_scores["completeness"] - old_comp
    icon       = "✅" if delta_acc > 0 else ("❌" if delta_acc < 0 else "➡️")

    print(f"  OLD [{row['answer_type']}] acc={old_acc:.1f} comp={old_comp:.1f}")
    print(f"  OLD_ANS: {old_ans[:100]}")
    print(f"  NEW [{new_type}] acc={new_scores['accuracy']:.1f} comp={new_scores['completeness']:.1f}  {icon} Δacc={delta_acc:+.1f}")
    print(f"  NEW_ANS: {new_ans[:100]}")
    print(f"  GT     : {gt[:80]}")
    print(f"  REASON : {new_scores['reasoning'][:100]}")

    results.append({
        "group":           group,
        "question":        q,
        "ground_truth":    gt,
        "old_answer":      old_ans,
        "old_type":        row["answer_type"],
        "old_accuracy":    old_acc,
        "old_completeness": old_comp,
        "new_answer":      new_ans,
        "new_type":        new_type,
        "new_accuracy":    new_scores["accuracy"],
        "new_completeness": new_scores["completeness"],
        "new_safety":      new_scores["safety"],
        "delta_accuracy":  delta_acc,
        "delta_completeness": delta_comp,
        "reasoning":       new_scores["reasoning"],
    })

# ── Lưu kết quả ───────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_RESULT, index=False, encoding="utf-8-sig")

# ── Tóm tắt ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TÓM TẮT SO SÁNH (20 câu mẫu)")
print("=" * 70)

# NOT_FOUND resolved
old_nf  = (out_df["old_type"]  == "NOT_FOUND").sum()
new_nf  = (out_df["new_type"]  == "NOT_FOUND").sum()
print(f"\n  NOT_FOUND: {old_nf} câu → {new_nf} câu  (giảm {old_nf - new_nf} câu)")

# Accuracy trung bình
print(f"\n  Accuracy  trung bình:  cũ={out_df['old_accuracy'].mean():.3f}  mới={out_df['new_accuracy'].mean():.3f}  Δ={out_df['delta_accuracy'].mean():+.3f}")
print(f"  Completeness TB:       cũ={out_df['old_completeness'].mean():.3f}  mới={out_df['new_completeness'].mean():.3f}  Δ={out_df['delta_completeness'].mean():+.3f}")

# Phân phối delta
improved  = (out_df["delta_accuracy"] >  0).sum()
unchanged = (out_df["delta_accuracy"] == 0).sum()
degraded  = (out_df["delta_accuracy"] <  0).sum()
print(f"\n  Cải thiện (Δacc > 0): {improved}/20")
print(f"  Không đổi (Δacc = 0): {unchanged}/20")
print(f"  Giảm      (Δacc < 0): {degraded}/20")

print(f"\n  📄 Kết quả chi tiết: {OUTPUT_RESULT}")
print("=" * 70)