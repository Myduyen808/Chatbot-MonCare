import os
import time
import random
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Thu thập các API Key hợp lệ
ALL_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]

MODEL_NAME = "llama-3.1-8b-instant"

def get_rotated_client():
    if not ALL_KEYS:
        return Groq()
    selected_key = random.choice(ALL_KEYS)
    return Groq(api_key=selected_key)

def evaluate_ragas_robust(question, answer, contexts):
    """Hàm chấm điểm siêu bền bỉ: Fix lỗi vòng lặp vô hạn 429 và tối ưu fallback"""
    context_str = "\n---\n".join([f"TÀI LIỆU {i+1}: {c}" for i, c in enumerate(contexts)])
    
    prompt = f"""You are an automated RAGAS evaluation system. Analyze the provided text and score the following 4 metrics from 0.0 to 1.0.

METRICS:
1. faithfulness: Is every detail in the ANSWER completely derivable from the provided CONTEXTS?
2. context_recall: Do the CONTEXTS contain enough information to fully answer the QUESTION?
3. answer_relevancy: Does the ANSWER directly and explicitly address the QUESTION?
4. context_precision: What proportion of the retrieved CONTEXTS is truly useful and relevant?

INPUT DATA:
- QUESTION: {question}
- CONTEXTS:
{context_str}
- ANSWER: {answer}

CRITICAL: Return ONLY a valid JSON object matching this structure exactly. Do not include any markdown formatting, thoughts, or extra text:
{{
  "faithfulness": 0.0,
  "context_recall": 0.0,
  "answer_relevancy": 0.0,
  "context_precision": 0.0
}}"""

    attempt = 0
    max_attempts = 3  # 👉 KHỐNG CHẾ: Thử tối đa 3 lần cho BẤT KỲ lỗi nào
    text = ""

    while attempt < max_attempts:
        attempt += 1
        try:
            client = get_rotated_client()
            
            res = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], 
                model=MODEL_NAME, 
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            text = res.choices[0].message.content.strip()
            
            # Giải mã JSON
            try:
                scores = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                start_idx, end_idx = text.find('{'), text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx: 
                    scores = json.loads(text[start_idx:end_idx])
                else:
                    raise Exception("JSON_PARSE_ERROR")
            
            # Trả về kết quả thành công
            return {
                "faithfulness": float(scores.get("faithfulness", 0.0)),
                "context_recall": float(scores.get("context_recall", 0.0)),
                "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
                "context_precision": float(scores.get("context_precision", 0.0))
            }
                
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "limit" in err_str.lower():
                print(f"\n⚠️ [Lượt thử {attempt}/{max_attempts}] Rate Limit (429). Ngủ đông 65 giây để hồi TPM...")
                time.sleep(65)  # Groq khuyên nên ngủ trên 60s để reset cửa sổ TPM
            else:
                print(f"\n⚠️ Lỗi hệ thống ở lượt thử {attempt}/{max_attempts}: {err_str}.")
                time.sleep(5)
                
    # 👉 ĐÃ ĐƯA RA NGOÀI VÒNG LẶP: Nếu quá 3 lần thử (kể cả dính 429) mà vẫn tịt thì tự động fallback
    print(f"\n🚨 Không thể lấy kết quả sau {max_attempts} lần thử. Tự động Fallback về 0.0.")
    print(f"--- Nội dung thô cuối cùng từ LLM (nếu có) ---\n{text if text else 'Không có phản hồi'}\n---------------------------------")
    return {"faithfulness": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}

def start_evaluation():
    datasets_to_eval = ["KB1_Standard", "KB2_TeenCode", "KB3_Noise", "ViMedAQA"]
    
    for dataset_name in datasets_to_eval:
        raw_file = f"raw_rag_{dataset_name}.json"
        score_file = f"scores_final_{dataset_name}.json"
        
        if not os.path.exists(raw_file):
            print(f"⏭️ Bỏ qua tập {dataset_name} do không tìm thấy file raw dữ liệu thô.")
            continue
            
        with open(raw_file, "r", encoding="utf-8") as f: 
            raw_data = json.load(f)
            
        methods_list = list(raw_data.keys())
        
        if os.path.exists(score_file):
            with open(score_file, "r", encoding="utf-8") as f: score_results = json.load(f)
        else:
            score_results = {name: {"faith": [], "precision": [], "recall": [], "relevancy": [], "latency": []} for name in methods_list}
            
        total_questions = len(raw_data[methods_list[0]]["questions"])
        done_count = len(score_results[methods_list[0]]["faith"])
        
        print(f"\n{'='*80}\n📊 Đang tiến hành chấm điểm tập: {dataset_name} (Tiến độ: {done_count}/{total_questions})")
        
        for idx in range(total_questions):
            if idx < done_count: continue
            
            print(f" 🎯 Chấm điểm câu [{idx+1}/{total_questions}]", end="", flush=True)
            
            for method_name in methods_list:
                q = raw_data[method_name]["questions"][idx]
                a = raw_data[method_name]["answers"][idx]
                c = raw_data[method_name]["contexts"][idx]
                lat = raw_data[method_name]["latency"][idx]
                
                print(f" -> {method_name[:12]}", end="", flush=True)
                scores = evaluate_ragas_robust(q, a, c)
                
                score_results[method_name]["faith"].append(scores.get("faithfulness", 0.0))
                score_results[method_name]["precision"].append(scores.get("context_precision", 0.0))
                score_results[method_name]["recall"].append(scores.get("context_recall", 0.0))
                score_results[method_name]["relevancy"].append(scores.get("answer_relevancy", 0.0))
                score_results[method_name]["latency"].append(lat)
                
                # 👉 TĂNG THỜI GIAN NGHỈ CHỦ ĐỘNG: Nghỉ 8 giây giữa các phương pháp 
                # để giảm áp lực dồn dập token lên API, né lỗi 429 chủ động.
                time.sleep(8)
                
            print(" | Xong câu!")
            
            with open(score_file, "w", encoding="utf-8") as f:
                json.dump(score_results, f, ensure_ascii=False, indent=2)
                
        # (Giữ nguyên phần in bảng kết quả cuối cùng của bạn...)
        print(f"\n{'-'*100}\n📊 BẢNG KẾT QUẢ ĐÃ CHẤM XONG XỊN: {dataset_name}\n{'-'*100}")
        table_header = f"{'Method':<25} | {'Faithfulness':<12} | {'Context Prec':<12} | {'Context Recall':<12} | {'Answer Rel.':<12} | {'Latency (s)':<10}\n" + "-"*100
        print(table_header)
        log_lines = [f"🔥 CHẠY THỰC NGHIỆM TRÊN TẬP: {dataset_name}", "="*100, table_header]
        
        for name, res in score_results.items():
            avg_f = sum(res["faith"]) / len(res["faith"]) if res["faith"] else 0.0
            avg_p = sum(res["precision"]) / len(res["precision"]) if res["precision"] else 0.0
            avg_r = sum(res["recall"]) / len(res["recall"]) if res["recall"] else 0.0
            avg_rel = sum(res["relevancy"]) / len(res["relevancy"]) if res["relevancy"] else 0.0
            avg_l = sum(res["latency"]) / len(res["latency"]) if res["latency"] else 0.0
            row = f"{name:<25} | {avg_f:<12.3f} | {avg_p:<12.3f} | {avg_r:<12.3f} | {avg_rel:<12.3f} | {avg_l:<10.2f}"
            print(row)
            log_lines.append(row)
            
        with open(f"log_{dataset_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
            
if __name__ == "__main__":
    start_evaluation()