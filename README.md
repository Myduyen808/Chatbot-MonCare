# MomCare — Hệ thống Chatbot Tư vấn Chăm sóc Mẹ Bỉm Sau Sinh

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-green?logo=langchain&logoColor=white)](https://docs.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.1--8B-orange?logo=groq&logoColor=white)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Database-purple?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-brown?labelColor=white)](https://en.wikipedia.org/wiki/MIT_License)

> **Luận văn tốt nghiệp** — Ngành Hệ thống Thông tin, Đại học Cần Thơ (K48)  
> **Sinh viên:** Trần Thị Mỹ Duyên — MSSV: B2203435  
> **Giảng viên hướng dẫn:** PGS.TS Nguyễn Thanh Hải

---

## Giới thiệu

**MomCare** là hệ thống chatbot tư vấn y tế chuyên biệt, được xây dựng trên kiến trúc **Advanced RAG (Retrieval-Augmented Generation)** nâng cao, hỗ trợ các bà mẹ sau sinh tiếp cận thông tin chăm sóc sức khỏe chính xác, an toàn và có kiểm chứng nguồn gốc.

Hệ thống giải quyết 3 vấn đề cốt lõi:
- **Thông tin sai lệch** trên mạng xã hội về chăm sóc mẹ và bé
- **Ảo giác AI (Hallucination)** của các mô hình LLM thông thường
- **Mất ngữ cảnh hội thoại** khi người dùng hỏi nhiều lượt liên tiếp

---

## Kiến trúc hệ thống

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│   Tầng 3: Intent Detection      │
│   Guardrails 3 lớp:             │
│   BLOCKED → SMALLTALK → RAG     │
└─────────────┬───────────────────┘
              │ RAG
    ▼
┌─────────────────────────────────┐
│   Tầng 4: Query Rewriting       │
│   + Multi-Query Expansion (n=3) │
└─────────────┬───────────────────┘
              │
    ▼
┌─────────────────────────────────┐
│   Tầng 5: FAISS MMR Retrieval   │
│   + CrossEncoder Re-ranking     │
│   + Map-Reduce Async (K>5)      │
└─────────────┬───────────────────┘
              │
    ▼
┌─────────────────────────────────┐
│   Tầng 6: Llama 3.1-8B (Groq)  │
│   Strict Prompting + Output     │
│   Guardrails                    │
└─────────────────────────────────┘
```

---

## Tính năng nổi bật

| Tính năng | Mô tả |
|---|---|
| **Guardrails 3 lớp** | Chặn kê đơn thuốc (Luật KCB 15/2023), nhận diện tâm lý nguy hiểm (WHO mhGAP), bypass xã giao |
| **Query Rewriting** | Tự động làm giàu câu hỏi ngắn dựa trên lịch sử hội thoại |
| **Summarized Memory** | Tóm tắt lịch sử bằng LLM thay vì cắt thô, bảo toàn thông tin y tế |
| **Task Merging** | Gộp rewrite + intent detection vào 1 lần gọi API |
| **Map-Reduce Async** | Xử lý song song tài liệu khi K>5, giảm latency ~14 lần |
| **Re-ranking** | CrossEncoder ms-marco-MiniLM-L-6-v2 tái xếp hạng kết quả |

---

## Kho tri thức (Knowledge Base)

| Định dạng | Số file | Nguồn |
|---|---|---|
| PDF | 16 | BV Từ Dũ, Bộ Y tế, WHO, UNICEF |
| DOCX | 40 | Vinmec, BV Nhi Đồng |
| CSV | 1 | 151 cặp Q&A chuẩn hóa có nguồn |
| **Tổng** | **57 file** | **2.313 chunks trong FAISS** |

---

## Kết quả thực nghiệm

### RAGAS Evaluation (300 câu — 3 kịch bản)

| Kịch bản | Faithfulness | Context Recall | Answer Relevancy | Context Precision |
|---|---|---|---|---|
| KB1 — Y khoa chuẩn | 0.743 | 0.759 | 0.636 | 0.684 |
| KB2 — Ngôn ngữ mẹ bỉm | 0.654 | 0.697 | 0.511 | 0.619 |
| KB3 — Câu có nhiễu | 0.622 | 0.687 | 0.382 | 0.542 |
| **Trung bình** | **0.673** | **0.714** | **0.510** | **0.615** |

### So sánh với nghiên cứu liên quan

| Chỉ số | Bài báo [2] CMU 2026 | MomCare |
|---|---|---|
| Expert Agreement / Faithfulness | 68.3% | **67.3%** (tương đương) |
| Context Recall | Không báo cáo | **71.4%** |
| Số kịch bản kiểm thử | 1 | **3** |
| Ngôn ngữ | Đa ngôn ngữ Ấn Độ | **Tiếng Việt** |

### Intent Classification (200 câu)

| Lớp | Accuracy |
|---|---|
| BLOCKED | 94.0% |
| SMALLTALK | 94.0% |
| RAG | 100.0% |
| **Tổng** | **97.0%** |

---

## Cài đặt và Chạy

### Yêu cầu hệ thống
- Python 3.10+
- RAM 8GB+
- Groq API Key (miễn phí tại [console.groq.com](https://console.groq.com))

### Bước 1: Clone và cài đặt

```bash
git clone https://github.com/your-username/RAG-Mom-Chatbot.git
cd RAG-Mom-Chatbot
pip install -r requirements.txt
```

### Bước 2: Cấu hình API Key

Tạo file `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
GROQ_API_KEY_1=gsk_backup_key_1
GROQ_API_KEY_2=gsk_backup_key_2
GROQ_API_KEY_3=gsk_backup_key_3
```

### Bước 3: Xây dựng Vector Database

```bash
python vectordb.py
```

### Bước 4: Chạy ứng dụng

```bash
streamlit run application.py
```

Truy cập: `http://localhost:8501`

---

## Cấu trúc dự án

```
RAG-Mom-Chatbot/
├── application.py          # Giao diện Streamlit
├── llm_chain.py            # RAGChain + Guardrails + Intent
├── vectordb.py             # FAISS + Embedding + Retrieval
├── db_config.yml           # Cấu hình đường dẫn dữ liệu
├── model_config.yml        # Cấu hình model embedding
├── data_store/
│   ├── pdf/                # 16 file PDF y khoa
│   ├── word/               # 40 file DOCX
│   └── csv/                # File Q&A chuẩn hóa
├── experiments/
│   ├── evaluate_ragas_groq.py      # RAGAS evaluation
│   ├── experiment_task_merging.py  # Thực nghiệm Task Merging
│   ├── test_intent_200.py          # Kiểm thử Intent 200 câu
│   ├── test_k_variation.py         # Biến thiên tham số K
│   └── test_stress_conversation.py # Stress test 25 lượt
└── KB/
    ├── KB1_Medical_Standard.xlsx   # 406 câu y khoa chuẩn
    ├── KB2_Mom_Style.xlsx          # 400 câu mẹ bỉm
    └── KB3_Information_Noise.xlsx  # 400 câu có nhiễu
```

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| LLM | Llama 3.1-8B-Instant (Groq LPU) |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector DB | FAISS (IndexFlatL2, Local) |
| Retrieval | MMR (fetch_k=15, lambda_mult=0.7) |
| Re-ranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Framework | LangChain + Custom RAGChain |
| UI | Streamlit |
| Evaluation | RAGAS (Faithfulness + Context Recall) |

---

## Tài liệu tham khảo

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Danescu-Niculescu-Mizil & Lee (2011). *Cornell Movie-Dialogs Corpus*. ACL 2011.
- OpenAI (2025). *Usage Policies*. https://openai.com/policies/usage-policies
- Quốc hội VN (2023). *Luật KCB số 15/2023/QH15*, Điều 7.
- WHO (2022). *mhGAP Guideline 2.0*.

---

## License

MIT License — Xem file [LICENSE](LICENSE) để biết thêm chi tiết.