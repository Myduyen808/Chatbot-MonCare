"""
real_benchmark_4_configs.py
===========================
Đo THẬT 4 cấu hình RAG (A/B/C/D) bằng cách gọi trực tiếp vào logic thật trong
llm_chain.py / vectordb.py của MomCare. KHÔNG có bất kỳ np.random giả lập số
liệu nào — tokens & latency lấy từ lệnh gọi Groq API thật, faithfulness được
LLM-judge chấm dựa trên đáp án chuẩn thật trong file dữ liệu của bạn.

⚠️ YÊU CẦU trước khi chạy (bắt buộc, script sẽ dừng nếu thiếu):
  1. File .env có GROQ_API_KEY (hoặc GROQ_API_KEY_1/2/3) hợp lệ.
  2. VectorDB đã được build sẵn (load_vector_db() trong vectordb.py chạy được).
  3. File dữ liệu câu hỏi thật: "KB_COVID_VN.xlsx" hoặc file .csv tương ứng,
     có 2 cột: "Câu hỏi người dùng (Input)" và "Phản hồi kỳ vọng (Expected Output)".
  4. Chạy script này CÙNG THƯ MỤC với llm_chain.py và vectordb.py gốc.

Cách chạy:
    python real_benchmark_4_configs.py

Ghi chú trung thực:
  - Faithfulness ở đây là LLM-as-judge tự viết (so khớp với ground truth),
    KHÔNG phải khung RAGAS chính thức. Nếu báo cáo/khóa luận cần đúng RAGAS,
    cần cài `ragas` + `datasets` và dùng metric `faithfulness` thật của họ.
  - Token đo được là prompt_tokens mà Groq trả về cho MỌI lệnh gọi call_llm()
    xảy ra bên trong một lượt invoke (rewrite, guardrail-llm, sinh câu trả lời...),
    cộng dồn lại — đây là "chi phí token thật của cả lượt hỏi", không phải ước lượng.
"""

import os
import re
import sys
import time
import random
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import llm_chain
    from llm_chain import (
        _adaptive_hybrid_search,
        get_reranker,
        check_input_guardrails_with_llm,
        check_output_guardrails,
        context_aware_safety_check,
        generate_multi_queries,
        summarize_history_message,
        MENTAL_HEALTH_RESPONSE,
    )
    from vectordb import smart_retrieve
    from groq import Groq
except Exception as e:
    print("❌ Không import được llm_chain.py / vectordb.py thật.")
    print("   Hãy đặt file benchmark này CÙNG THƯ MỤC với 2 file đó rồi chạy lại.")
    print(f"   Lỗi chi tiết: {e}")
    import traceback
    traceback.print_exc() 
    raise SystemExit(1)    


# ==============================================================================
# ĐẾM TOKEN ĐÁNG TIN CẬY: chặn thẳng vào lệnh gọi Groq API, gắn NHÃN rõ ràng
# (question_idx, config) cho từng lệnh gọi thay vì dùng 1 biến đếm toàn cục
# rồi reset/đọc theo thời điểm (cách cũ vẫn cho ra tokens=0/latency=0 dù có
# lệnh gọi thật xảy ra — nghi là do thời điểm reset/đọc bị lệch trong một số
# trường hợp chưa xác định chắc chắn được trên máy Windows của bạn).
#
# Cách mới: mỗi lệnh gọi API được ghi vào 1 log chi tiết (_call_log) kèm nhãn
# "đang đo cho ai" tại đúng thời điểm gọi — và log này còn được xuất ra file
# 'debug_call_log.csv' để nếu vẫn còn bất thường, ta nhìn thẳng vào dữ liệu
# thô thay vì đoán mò.
# ==============================================================================
_call_log = []          # list các dict: tag, prompt_tokens, completion_tokens, prompt_chars, ts
_current_tag = {"value": None}


def _set_tag(tag):
    _current_tag["value"] = tag


def _tracked_call_llm(prompt, system_prompt="Bạn là trợ lý MomCare, chuyên chăm sóc mẹ và bé.",
                       temperature=0.3, max_retries=4):
    for attempt in range(max_retries):
        try:
            client = Groq(api_key=random.choice(llm_chain._ALL_KEYS))
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=llm_chain.MODEL_NAME,
                temperature=temperature,
            )
            tokens_used = chat_completion.usage.prompt_tokens
            completion_tokens = getattr(chat_completion.usage, "completion_tokens", 0) or 0
            tag = _current_tag["value"]
            _call_log.append({
                "tag": tag,
                "prompt_tokens": tokens_used,
                "completion_tokens": completion_tokens,
                "prompt_chars": len(prompt),
                "ts": time.time(),
            })
            print(f"✅ [ĐÃ GỌI API] tag={tag} | Độ dài prompt: {len(prompt)} ký tự -> "
                  f"Tốn: {tokens_used} tokens (+{completion_tokens} completion)")
            return chat_completion.choices[0].message.content
        except Exception as e:
            err = str(e)
            print(f"❌ [LỖI API GROQ]: {err}")
            if "429" in err:
                m = re.search(r'in (\d+)m([\d.]+)s', err)
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 10 if m else 60 * (attempt + 1)
                print(f"⏳ Rate limit - đợi {wait:.0f}s (lần {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                time.sleep(3)
    return ""


# ⭐ QUAN TRỌNG: gán đè vào chính module llm_chain, để MỌI hàm nội bộ của
# llm_chain.py (check_input_guardrails_with_llm, summarize_history_message,
# generate_multi_queries...) khi gọi call_llm(...) bên trong chúng cũng tự
# động đi qua bản có đếm token này — vì Python tra cứu tên hàm động theo
# namespace của module tại thời điểm gọi, không phải lúc import.
llm_chain.call_llm = _tracked_call_llm
call_llm = _tracked_call_llm  # để code benchmark bên dưới dùng chung 1 bản duy nhất


def judge_faithfulness_raw_response_to_float(raw: str) -> float:
    """
    Parse điểm số từ phản hồi LLM-judge. Chấp nhận CẢ dấu chấm (0.85) LẪN dấu
    phẩy kiểu Việt Nam (0,85) — bản cũ chỉ nhận dấu chấm nên khi model lỡ trả
    lời "0,85" thì regex chỉ bắt được ký tự "0" đứng đầu, khiến điểm luôn bị
    cắt cụt về 0.0 dù model thật ra chấm điểm khác hẳn 0.
    """
    raw = raw.strip()
    match = re.search(r"[01](?:[.,]\d+)?", raw)
    if not match:
        return 0.5  # không parse được -> điểm trung lập, không phải 0
    return float(match.group().replace(",", "."))


def _run_capturing_tokens(fn, tag, *args, **kwargs):
    """
    Chạy fn() với nhãn `tag` (vd (question_idx, cfg_key)) được gắn TRƯỚC khi
    gọi, để mọi lệnh gọi call_llm() xảy ra trong lúc này được ghi log đúng
    nhãn. Latency đo bằng time.perf_counter(); tokens tính bằng cách LỌC
    _call_log theo đúng tag này SAU KHI fn() đã chạy xong — không còn dùng
    kiểu "reset rồi đọc" dễ lệch nữa.
    """
    _set_tag(tag)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000
    _set_tag(None)  # để các lệnh gọi NGOÀI phạm vi (vd judge_faithfulness) không bị tính nhầm vào đây

    relevant = [r for r in _call_log if r["tag"] == tag]
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in relevant)
    return result, total_tokens, latency_ms


class ConfigurableRAGChain:
    """
    Bọc lại logic RAG THẬT (copy sát từ llm_chain.RAGChain.invoke), nhưng cho phép
    bật/tắt độc lập từng 'vũ khí tối ưu' để so sánh công bằng giữa 4 cấu hình,
    thay vì tạo số liệu giả:

      - use_history: "none" | "raw" | "summary"
            none    -> không dùng lịch sử chat (Cấu hình B)
            raw     -> nhét nguyên văn lịch sử thô vào prompt (Cấu hình A, C)
            summary -> tóm tắt từng tin nhắn bằng LLM trước khi nhét vào (Cấu hình D)
      - task_merging: True  -> 1 lệnh gọi API duy nhất vừa rewrite vừa phân loại ý định
                      False -> 2 lệnh gọi API riêng biệt (baseline chưa tối ưu)
      - k: số tài liệu truy xuất
    """

    def __init__(self, k, use_history="raw", task_merging=True, history_window=6):
        self.k = k
        self.use_history = use_history
        self.task_merging = task_merging
        self.history_window = history_window

    def _build_history_text(self, history):
        if self.use_history == "none" or not history:
            return ""
        msgs = history[-self.history_window:] if self.history_window else history
        lines = []
        for msg in msgs:
            role = "Mẹ" if msg.__class__.__name__ == "HumanMessage" else "MomCare"
            content = msg.content
            if self.use_history == "summary":
                content = summarize_history_message(content)  # gọi LLM thật để tóm tắt
            lines.append(f"{role}: {content}")
        return "LỊCH SỬ HỘI THOẠI:\n" + "\n".join(lines) + "\n\n"

    def _rewrite_and_intent(self, question, history_text):
        if self.task_merging:
            # ĐÃ GỘP: 1 lệnh gọi API duy nhất
            prompt = f"""Bạn là AI phân tích ngữ cảnh y khoa cho MomCare. Dựa vào lịch sử và câu hỏi mới:
1. Viết lại CÂU HỎI MỚI thành một câu tìm kiếm độc lập, đầy đủ ngữ cảnh.
2. Phân loại ý định: BLOCKED / SMALLTALK / RAG.

{history_text}CÂU HỎI MỚI: {question}

ĐỊNH DẠNG TRẢ LỜI (chỉ 2 dòng, không giải thích):
REWRITTEN: <câu_viết_lại_đầy_đủ>
INTENT: <RAG/SMALLTALK/BLOCKED>"""
            result = call_llm(prompt, temperature=0).strip()
            rewritten, intent = question, "RAG"
            for line in result.split("\n"):
                if line.startswith("REWRITTEN:"):
                    rewritten = line.replace("REWRITTEN:", "").strip()
                elif line.startswith("INTENT:"):
                    raw = line.replace("INTENT:", "").strip().upper()
                    if raw in ["BLOCKED", "SMALLTALK", "RAG"]:
                        intent = raw
            return rewritten, intent
        else:
            # CHƯA GỘP: 2 lệnh gọi API riêng biệt (tốn thêm 1 round-trip token)
            rewrite_prompt = (
                f"{history_text}Viết lại câu hỏi sau thành một câu tìm kiếm độc lập, "
                f"đầy đủ ngữ cảnh:\nCâu hỏi: {question}\nCâu viết lại:"
            )
            rewritten = call_llm(rewrite_prompt, temperature=0).strip() or question

            intent_prompt = (
                f'Phân loại ý định câu hỏi sau vào 1 trong 3 nhóm: BLOCKED, SMALLTALK, RAG.\n'
                f'Câu hỏi: "{question}"\nChỉ trả về đúng 1 từ, không giải thích.'
            )
            intent_raw = call_llm(intent_prompt, temperature=0).strip().upper()
            intent = intent_raw if intent_raw in ["BLOCKED", "SMALLTALK", "RAG"] else "RAG"
            return rewritten, intent

    def invoke(self, inputs):
        question = inputs["question"]
        history = inputs.get("history", [])
        history_text = self._build_history_text(history)

        # === BỎ QUA GUARDRAILS TRONG BENCHMARK ===
        # Reason: Guardrails đang block sai các câu hỏi Y khoa hợp lệ (vd: chứa "COVID-19"),
        # khiến RAG không chạy -> tokens=0, faithfulness=0. Mục tiêu benchmark là đo RAG,
        # không phải đo Guardrails.
        # blocked_msg = check_input_guardrails_with_llm(question)
        # if blocked_msg:
        #     return {"answer": blocked_msg, "docs": []}

        # context_block = context_aware_safety_check(question, history)
        # if context_block:
        #     return {"answer": context_block, "docs": []}
        # =========================================

        enriched_question, intent = self._rewrite_and_intent(question, history_text)

        if intent == "BLOCKED":
            return {"answer": MENTAL_HEALTH_RESPONSE, "docs": []}
        if intent == "SMALLTALK":
            answer = call_llm(f"Trả lời ngắn gọn, thân thiện: {enriched_question}", temperature=0.3)
            return {"answer": answer, "docs": []}

        # ... (GIỮ NGUYÊN PHẦN CODE DƯỚI ĐÂY KHÔNG THAY ĐỔI) ...
        # Truy xuất tài liệu thật (Hybrid Search thật từ vectordb + BM25 thật)
        primary_docs = _adaptive_hybrid_search(enriched_question, k=self.k)
        all_docs = list(primary_docs)
        seen = {str(d.page_content)[:200] for d in primary_docs}

        if len(question.split()) <= 5:
            extra_queries = generate_multi_queries(enriched_question, n=2)
            for q in extra_queries[1:]:
                retrieved = smart_retrieve(q, None, self.k)
                for d in retrieved:
                    key = str(d.page_content)[:200]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(d)

        if len(all_docs) > self.k:
            reranker = get_reranker()
            pairs = [(enriched_question, d.page_content) for d in all_docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
            docs = [d for _, d in ranked[: self.k]]
        else:
            docs = all_docs[: self.k]

        if not docs:
            return {"answer": "Tôi chưa tìm thấy thông tin phù hợp trong tài liệu.", "docs": []}

        context = "\n\n".join(f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs))

        prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời câu hỏi CHỈ dựa trên tài liệu sau.
Không bịa thêm thông tin ngoài tài liệu. Trình bày đầy đủ chi tiết, số liệu, không làm tròn.

TÀI LIỆU:
{context}

CÂU HỎI: {enriched_question}

TRẢ LỜI:"""
        answer = call_llm(prompt, temperature=0.3)
        answer = check_output_guardrails(answer, enriched_question)
        return {"answer": answer, "docs": docs}


# ==============================================================================
# 4 CẤU HÌNH — chỉ khác nhau ở các công tắc thật, không có số liệu hardcode nào
# ==============================================================================
CONFIGS = {
    "A": dict(name="Vanilla RAG (thô)",   k=5, use_history="raw",     task_merging=False, history_window=0),  # 0 = toàn bộ lịch sử
    "B": dict(name="Nén cực hạn",         k=1, use_history="none",    task_merging=False, history_window=0),
    "C": dict(name="Tối ưu một nửa",      k=5, use_history="raw",     task_merging=True,  history_window=0),
    "D": dict(name="MomCare Full",        k=5, use_history="summary", task_merging=True,  history_window=2),
}


def judge_faithfulness(answer: str, ground_truth: str, question: str) -> float:
    """
    LLM-as-judge thay thế cho RAGAS đầy đủ (chưa cài ragas trong môi trường này).
    Chấm 0.0 - 1.0 mức độ trung thực y khoa của câu trả lời so với đáp án chuẩn.
    """
    prompt = f"""Bạn là giám khảo y khoa nghiêm khắc. So sánh CÂU TRẢ LỜI CẦN CHẤM với ĐÁP ÁN CHUẨN.
Chấm điểm Faithfulness từ 0.0 đến 1.0:
- 1.0 = nội dung y khoa khớp hoàn toàn, không bịa, không sai đối tượng (mẹ/bé), không sai số liệu.
- 0.0 = sai lệch nghiêm trọng hoặc bịa đặt thông tin không có trong đáp án chuẩn.

CÂU HỎI: {question}
ĐÁP ÁN CHUẨN: {ground_truth}
CÂU TRẢ LỜI CẦN CHẤM: {answer}

CHỈ TRẢ VỀ 1 SỐ THẬP PHÂN DUY NHẤT, DÙNG DẤU CHẤM (ví dụ: 0.85, KHÔNG viết 0,85), KHÔNG GIẢI THÍCH GÌ THÊM."""
    raw = call_llm(prompt, temperature=0).strip()
    return judge_faithfulness_raw_response_to_float(raw)


def load_test_dataset(dataset_file=None, sample_size=20, random_state=42):
    """
    Đọc file dữ liệu THẬT. KHÔNG tự bịa dữ liệu nếu thiếu file — dừng luôn để bạn biết.

    dataset_file: chỉ định rõ file muốn dùng, ví dụ "KB1_Medical_Standard.xlsx".
                  Nếu None, sẽ dò theo thứ tự trong possible_files.
    sample_size:  số câu LẤY MẪU NGẪU NHIÊN để chạy benchmark (không chạy hết 400 câu
                  cho thí nghiệm cost-quality tradeoff, quá tốn API & thời gian).
                  Đặt None nếu muốn chạy toàn bộ file.
    random_state: cố định seed để lần chạy sau lấy đúng lại mẫu này (tái lập được).
    """
    possible_files = [
        dataset_file,
        "KB1_Medical_Standard.xlsx",
        "KB2_Mom_Style.xlsx",
        "KB3_Information_Noise.xlsx",
        "KB_COVID_VN.xlsx",
        "KB_COVID_VN.xlsx - Generalization_Dataset.csv",
    ]
    for f in possible_files:
        if f and os.path.exists(f):
            df = pd.read_csv(f) if f.endswith(".csv") else pd.read_excel(f)
            print(f"✅ Đã nạp file dữ liệu thật: {f} ({len(df)} dòng)")
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
                print(f"🎯 Đã lấy mẫu ngẫu nhiên {sample_size} câu (random_state={random_state}) để benchmark.")
            return df
    raise FileNotFoundError(
        "❌ Không tìm thấy file dữ liệu câu hỏi thật nào trong thư mục hiện tại "
        "(đã thử: KB1_Medical_Standard.xlsx, KB2_Mom_Style.xlsx, KB3_Information_Noise.xlsx, KB_COVID_VN.xlsx).\n"
        "   Script này KHÔNG tự tạo dữ liệu giả — hãy đặt file câu hỏi/đáp án thật vào đây rồi chạy lại."
    )


def build_mock_conversation_history():
    """
    ⚠️ ĐÂY LÀ LỊCH SỬ HỘI THOẠI MÔ PHỎNG (SIMULATED), KHÔNG PHẢI LOG NGƯỜI DÙNG THẬT.
    Hệ thống chưa có user thật tại thời điểm thực nghiệm, nên để đo được đúng lợi ích
    của cơ chế Summarized Memory, ta CẦN CHỦ ĐÍCH tạo ra lịch sử đủ dài.

    LƯU Ý QUAN TRỌNG: summarize_history_message() trong llm_chain.py chỉ thực sự gọi
    LLM để tóm tắt khi nội dung > 200 ký tự:
        if len(content) <= 200:
            return content   # <-- không nén, trả nguyên văn
    Nên mỗi tin nhắn dưới đây được viết CỐ Ý DÀI HƠN 200 KÝ TỰ để đảm bảo nhánh nén
    thực sự được kích hoạt và đo được — nếu không, Cấu hình D sẽ không khác gì Cấu hình A
    ở phần lịch sử, và kết luận về "tiết kiệm token nhờ tóm tắt" sẽ SAI.

    Khi viết báo cáo/khóa luận: hãy ghi rõ đây là "lịch sử hội thoại mô phỏng, được
    thiết kế có chủ đích để cô lập biến số chi phí xử lý ngữ cảnh, do hệ thống chưa
    có dữ liệu người dùng thật tại thời điểm thực nghiệm" — không trình bày như log thật.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    return [
        HumanMessage(content=(
            "Chào chatbot, tôi cần tư vấn cách phòng chống và theo dõi dịch bệnh lây nhiễm "
            "tại nhà, bé nhà tôi được 8 tháng tuổi, gần đây có tiếp xúc với người lớn trong "
            "nhà bị ho sốt, tôi lo bé bị lây nên muốn hỏi cách theo dõi và phòng ngừa cho đúng."
        )),
        AIMessage(content=(
            "Dạ chào mẹ, MomCare xin đồng hành cùng mẹ trong việc chăm sóc bé. Với trường hợp "
            "bé 8 tháng tuổi có tiếp xúc gần với người bị ho sốt trong nhà, mẹ cần theo dõi sát "
            "các dấu hiệu như sốt, ho, thở nhanh, bú kém trong 5-7 ngày tới, đồng thời hạn chế "
            "tiếp xúc gần và đảm bảo vệ sinh tay thường xuyên cho cả nhà."
        )),
        HumanMessage(content=(
            "Vậy nếu trong quá trình theo dõi mà bé bắt đầu có dấu hiệu sốt cao thì mình cần "
            "làm những bước gì trước, và khi nào thì bắt buộc phải đưa bé đi viện cấp cứu ngay "
            "chứ không thể tự theo dõi ở nhà được nữa?"
        )),
        AIMessage(content=(
            "Mẹ cần đặc biệt theo dõi sát các biểu hiện nguy hiểm như sốt cao liên tục trên 39 "
            "độ không hạ sau khi đã dùng thuốc hạ sốt đúng liều, bé lừ đừ, bỏ bú, nôn ói nhiều "
            "lần, thở nhanh hoặc rút lõm ngực, hoặc có dấu hiệu giật mình chới với bất thường — "
            "đây là các dấu hiệu cần đưa bé đi cấp cứu ngay lập tức, không nên tự xử lý tại nhà."
        )),
    ]


def execute_benchmarks(limit=None, dataset_file=None, sample_size=20, random_state=42):
    """
    limit:         nếu đặt (vd 3), chỉ chạy `limit` câu ĐẦU TIÊN trong mẫu đã lấy — dùng để test nhanh script.
    dataset_file:  tên file KB muốn dùng, vd "KB1_Medical_Standard.xlsx". None = tự dò theo thứ tự có sẵn.
    sample_size:   số câu lấy mẫu ngẫu nhiên từ file KB (400 câu) để chạy benchmark cost-quality.
    random_state:  seed cố định để tái lập đúng mẫu đã lấy ở các lần chạy sau.
    """
    print("=" * 85)
    print(" ĐO ĐẠC THẬT: 4 CẤU HÌNH RAG — GỌI TRỰC TIẾP GROQ API THẬT (KHÔNG MÔ PHỎNG)")
    print("=" * 85)
    print("⚠️ Lịch sử hội thoại dùng trong benchmark là MÔ PHỎNG (mock), không phải log")
    print("   người dùng thật — vì hệ thống chưa có user thật. Xem build_mock_conversation_history().")
    print("=" * 85)

    df = load_test_dataset(dataset_file=dataset_file, sample_size=sample_size, random_state=random_state)
    q_col = "Câu hỏi người dùng (Input)"
    gt_col = "Phản hồi kỳ vọng (Expected Output)"
    df_clean = df.dropna(subset=[q_col, gt_col])
    if limit:
        df_clean = df_clean.head(limit)
    total = len(df_clean)
    print(f"🚀 Tổng số câu hỏi đưa vào thực nghiệm: {total}\n")

    mock_history = build_mock_conversation_history()

    results = {c: {"tokens": [], "latency": [], "faithfulness": []} for c in CONFIGS}
    raw_rows = []

    for i, (idx, row) in enumerate(df_clean.iterrows()):
        q_text, gt_text = str(row[q_col]), str(row[gt_col])
        print(f"\n{'='*80}\n📝 [{i+1}/{total}] Câu hỏi: {q_text}\n{'='*80}")

        for cfg_key, cfg in CONFIGS.items():
            tag = (i, cfg_key)
            try:
                chain = ConfigurableRAGChain(
                    k=cfg["k"],
                    use_history=cfg["use_history"],
                    task_merging=cfg["task_merging"],
                    history_window=cfg["history_window"],
                )
                # Cùng một đoạn lịch sử mô phỏng cho cả 4 cấu hình trên mỗi câu hỏi,
                # để biến số duy nhất thay đổi là CÁCH XỬ LÝ lịch sử, không phải nội dung.
                result, tokens, latency_ms = _run_capturing_tokens(
                    chain.invoke, tag, {"question": q_text, "history": mock_history}
                )

                _set_tag("judge")  # tách riêng, không tính vào chi phí vận hành của config
                faithfulness = judge_faithfulness(result["answer"], gt_text, q_text)
                _set_tag(None)

                time.sleep(1.5)  # chống Rate Limit 429 khi gọi Groq API liên tục
            except Exception as e:
                print(f"⚠️ Lỗi khi chạy cấu hình {cfg_key}: {e}")
                traceback.print_exc()
                tokens, latency_ms, faithfulness = 0, 0.0, 0.0

            results[cfg_key]["tokens"].append(tokens)
            results[cfg_key]["latency"].append(latency_ms)
            results[cfg_key]["faithfulness"].append(faithfulness)
            raw_rows.append(dict(config=cfg_key, question_idx=i, question=q_text,
                                  tokens=tokens, latency_ms=latency_ms, faithfulness=faithfulness))

            print(f"  👉 [{cfg_key}] {cfg['name']:<22} | tokens={tokens:<6} | "
                  f"latency={latency_ms:.0f}ms | faithfulness={faithfulness:.2f}")

    print("\n" + "=" * 85)
    print(" KẾT QUẢ TRUNG BÌNH — SỐ LIỆU THẬT TỪ GROQ API (KHÔNG PHẢI MÔ PHỎNG)")
    print("=" * 85)
    print(f"{'Tên cấu hình':<28} | {'Tokens/Lượt':<12} | {'Độ trễ (ms)':<12} | {'Faithfulness':<12}")
    print("-" * 85)
    for cfg_key, cfg in CONFIGS.items():
        r = results[cfg_key]
        if not r["tokens"]:
            continue
        print(f"{cfg['name']:<28} | {np.mean(r['tokens']):<12.0f} | "
              f"{np.mean(r['latency']):<12.0f} | {np.mean(r['faithfulness']):<12.4f}")
    print("-" * 85)

    out_df = pd.DataFrame(raw_rows)
    out_df.to_csv("telemetry_raw_log.csv", index=False, encoding="utf-8-sig")
    print("\n✅ Đã lưu chi tiết từng câu/từng cấu hình vào 'telemetry_raw_log.csv'.")

    # Log CHI TIẾT từng lệnh gọi API (kèm nhãn) — để chẩn đoán nếu số liệu vẫn có gì bất thường
    debug_df = pd.DataFrame(_call_log)
    if not debug_df.empty:
        debug_df.to_csv("debug_call_log.csv", index=False, encoding="utf-8-sig")
        print("🔍 Đã lưu log chi tiết từng lệnh gọi API (kèm nhãn) vào 'debug_call_log.csv' để đối chiếu nếu cần.")

    print("🎯 Dùng file này để vẽ biên Pareto / quy đổi tài chính ở bước sau.")
    print("=" * 85)


if __name__ == "__main__":
    # Gợi ý chạy thử trước (đỡ tốn API quota): giới hạn 3 câu trong mẫu 20 câu đã lấy.
    #   execute_benchmarks(limit=3, dataset_file="KB1_Medical_Standard.xlsx", sample_size=20)
    #
    # Sau khi chạy thử OK, bỏ limit để chạy full mẫu 20 câu đã lấy:
    #   execute_benchmarks(dataset_file="KB1_Medical_Standard.xlsx", sample_size=20)
    #
    # Có thể đổi dataset_file sang "KB2_Mom_Style.xlsx" hoặc "KB3_Information_Noise.xlsx"
    # nếu muốn benchmark trên phong cách câu hỏi khác (thí nghiệm riêng, không gộp chung
    # với thí nghiệm cost-quality tradeoff).
    execute_benchmarks(limit=5, dataset_file="KB_COVID_VN.xlsx", sample_size=20, random_state=42)