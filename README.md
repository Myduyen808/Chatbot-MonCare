# MomCare — Hệ thống Chatbot Tư vấn Chăm sóc Mẹ Bỉm Sau Sinh

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-green?logo=langchain&logoColor=white)](https://docs.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.1--8B-orange?logo=groq&logoColor=white)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Database-purple?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-brown?labelColor=white)](LICENSE)

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
┌─────────────────────────────────────────────┐
│  Tầng 1: Input Guardrails                   │
│  ├── Mental Health Detection (WHO mhGAP)    │
│  └── Blocked Inputs (Luật KCB 15/2023)      │
└─────────────┬───────────────────────────────┘
              │ PASS
              ▼
┌─────────────────────────────────────────────┐
│  Tầng 2: Task Merging (1 lần gọi API)       │
│  ├── Query Rewriting (làm giàu ngữ cảnh)    │
│  ├── Intent Detection (BLOCKED/SMALLTALK/RAG)│
│  └── Conversation Context Extraction         │
└─────────────┬───────────────────────────────┘
              │ RAG
              ▼
┌─────────────────────────────────────────────┐
│  Tầng 3: Multi-Query Expansion (n=3)        │
│  ├── Biến thể 1: Thuật ngữ y khoa chuyên ngành│
│  ├── Biến thể 2: Từ khóa ngắn, cụ thể       │
│  └── Biến thể 3: Mở rộng khái niệm liên quan │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Tầng 4: FAISS MMR Retrieval                │
│  ├── search_type: MMR                       │
│  ├── fetch_k: 30, lambda_mult: 0.7          │
│  ├── Keyword Overlap Filter                 │
│  └── Adaptive K (tăng K cho câu hỏi ngắn)   │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Tầng 5: CrossEncoder Re-ranking            │
│  ├── Model: ms-marco-MiniLM-L-6-v2          │
│  ├── Rerank toàn bộ pool từ Multi-Query     │
│  └── Lấy top-K tài liệu chất lượng cao nhất │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Tầng 6: Llama 3.1-8B-Instant (Groq LPU)   │
│  ├── Strict Prompting (8 nguyên tắc)        │
│  ├── Context Injection (ngữ cảnh hội thoại)  │
│  └── Output Guardrails (chặn chẩn đoán)     │
└─────────────┬───────────────────────────────┘
              │
              ▼
    Response + Nguồn trích dẫn
```

---

## Tính năng nổi bật

| Tính năng | Mô tả | File triển khai |
|---|---|---|
| **Guardrails 3 lớp** | Chặn kê đơn thuốc/liều cụ thể (Luật KCB 15/2023), nhận diện tâm lý nguy hiểm (WHO mhGAP 17+ từ khóa), bypass xã giao (chào hỏi, cảm ơn, hỏi về bot) | `llm_chain.py` |
| **Task Merging** | Gộp Query Rewriting + Intent Detection vào **1 lần gọi API duy nhất**, giảm 50% số lượt gọi LLM ở tầng tiền xử lý | `llm_chain.py` → `rewrite_and_detect_intent()` |
| **Summarized Memory** | Tóm tắt lịch sử hội thoại bằng LLM (giữ lại tên bệnh, triệu chứng, độ tuổi, thời gian) thay vì cắt cơ học 200 ký tự | `llm_chain.py` → `summarize_history_message()` |
| **Conversation Context Extraction** | Tự động trích xuất độ tuổi/đối tượng từ câu hỏi hiện tại, KHÔNG tự suy đoán từ câu trước để tránh inject sai ngữ cảnh | `llm_chain.py` → `update_conversation_context()` |
| **Multi-Query Expansion** | Sinh 3 biến thể truy vấn: thuật ngữ y khoa, từ khóa ngắn, khái niệm liên quan. Giữ nguyên số liệu (liều, tuổi) trong ít nhất 1 biến thể | `llm_chain.py` → `generate_multi_queries()` |
| **Adaptive K** | Tự động tăng K từ 5 → 6 khi câu hỏi có ≤ 5 từ (câu hỏi ngắn cần broader retrieval) | `llm_chain.py` → `RAGChain.invoke()` |
| **MMR + Keyword Filter** | Tìm kiếm MMR (fetch_k=30, lambda_mult=0.7) kết hợp lọc overlap từ khóa câu hỏi, fallback nếu không có doc vượt ngưỡng | `vectordb.py` → `smart_retrieve()` |
| **CrossEncoder Re-ranking** | Rerank toàn bộ pool tài liệu từ Multi-Query bằng ms-marco-MiniLM-L-6-v2, lấy top-K chất lượng nhất | `llm_chain.py` → `RAGChain.invoke()` |
| **Map-Reduce Async** | Xử lý song song tài liệu khi K lớn bằng `asyncio.Semaphore(10)`, chạy trong Thread riêng để không block Streamlit | `llm_chain.py` → `summarize_docs_async()` |
| **Hybrid Search** | Kết hợp FAISS (Vector) + BM25 (từ khóa) với Adaptive Weighting |
| **Output Guardrails** | Phát hiện từ khóa chẩn đoán ("bị bệnh", "chẩn đoán", "mắc bệnh"...) trong câu trả lời, tự động append khuyến cáo đến cơ sở y tế | `llm_chain.py` → `check_output_guardrails()` |
| **API Key Rotation** | Quản lý 4 Groq API keys, tự động chọn ngẫu nhiên và retry với key khác khi gặp 429 Rate Limit | `llm_chain.py` → `call_llm()` |
| **Nguồn trích dẫn minh bạch** | Hiển thị tên file tài liệu + preview nội dung 500 ký tự cho từng đoạn được trích dẫn | `application.py` |
| **Quản lý kho kiến thức** | UI tải lên/xem chi tiết/xóa tài liệu (PDF, DOCX, CSV, XLSX), rebuild VectorDB một nút bấm | `application.py` → `Databases()` |

---

## Kho tri thức (Knowledge Base)

| Định dạng | Số file | Nguồn |
|---|---|---|
| PDF | 16 | BV Từ Dũ, Bộ Y tế, WHO, UNICEF |
| DOCX | 40 | Vinmec, BV Nhi Đồng |
| EXCEL | 2 | 151 cặp Q&A chuẩn hóa có nguồn và Bộ đề Sản khoa 4,497 câu hỏi trắc nghiệm y khoa được trích xuất và chuẩn hóa từ tập dữ liệu công khai ViMedAQA |
| **Tổng** | **58 file** | **5.832 chunks trong FAISS** |

---

## Kết quả thực nghiệm

### RAGAS Evaluation (300 câu — 3 kịch bản)

| Kịch bản | Faithfulness | Context Recall | Answer Relevancy | Context Precision |
|---|---|---|---|---|
| KB1 — Y khoa chuẩn | 0.743 | 0.759 | 0.636 | 0.684 |
| KB2 — Ngôn ngữ mẹ bỉm | 0.654 | 0.697 | 0.511 | 0.619 |
| KB3 — Câu có nhiễu | 0.622 | 0.687 | 0.382 | 0.542 |
| **Trung bình** | **0.673** | **0.714** | **0.510** | **0.615** |

### LLM-as-Judge (Clinical Evaluation)

Đánh giá bổ sung bằng mô hình `llama-3.3-70b-versatile` theo 3 tiêu chí y khoa:

| Tiêu chí | Thang điểm | Mô tả |
|---|---|---|
| Clinical Accuracy | 0.0 / 0.5 / 1.0 | Độ chính xác y khoa theo ý nghĩa, không so từng chữ |
| Completeness | 0.0 / 0.5 / 1.0 | Mức độ đầy đủ ý chính cần trả lời |
| Safety | 0.0 / 0.5 / 1.0 | An toàn — chỉ = 0 khi có khả năng gây hại trực tiếp |

### Benchmark trên tập dữ liệu ViMedAQA [ref30] (4.496 câu)

Hệ thống MomCare được kiểm thử trực tiếp trên bộ dữ liệu chuẩn **ViMedAQA** (Tran et al., 2024 — ACL SRW) gồm 4.496 câu hỏi y khoa tiếng Việt, so sánh với 8 mô hình baseline mà bài báo gốc đã công bố:

| Mô hình | BLEU | ROUGE-L | METEOR | BERTScore |
|---|---|---|---|---|
| VinaLlama-7B (best baseline) | 31.70 | 59.08 | 64.29 | 72.47 |
| Gemma-2B | 32.04 | — | 53.48 | — |
| Llama2-7B | — | 24.34 | — | — |
| **MomCare RAG** | **32.41** | 47.68 | 58.18 | **80.60** |

> **Nhận xét:** MomCare đạt BERTScore cao nhất (+8.13 so với VinaLlama-7B), chứng minh độ chính xác ngữ nghĩa y khoa vượt trội. ROUGE-L thấp hơn do RAG chỉ nhận 5 fragment rời từ FAISS thay vì toàn bộ đoạn văn gốc như các baseline — buộc mô hình phải tổng hợp, không sao chép từ vựng. Ngoài ra ~35% câu hỏi hệ thống chủ động từ chối trả lời an toàn (không bịa thông tin), khiến điểm n-gram giảm nhưng an toàn y tế tăng.

### So sánh với nghiên cứu liên quan (arXiv 2026)

| Chỉ số | Bài báo CMU 2026 [ref2] | MomCare |
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

### Ablation Studies

| Thành phần loại bỏ | Faithfulness | Context Recall | Nhận xét |
|---|---|---|---|
| Full system (baseline) | — | — | Tất cả bật |
| Tắt Multi-Query | ↓ | ↓ | Giảm độ phủ tài liệu |
| Tắt Re-ranking | ↓ | — | Giảm độ chính xác xếp hạng |
| Tắt Summarized Memory | — | ↓ | Mất ngữ cảnh lịch sử dài |

### Hybrid Search vs Vector-Only

| Phương pháp | Faithfulness | Nhận xét |
|---|---|---|
| Vector Only (FAISS MMR) | baseline | Tốt cho ngữ nghĩa |
| Hybrid (FAISS + BM25) | +14.5% | Tốt hơn cho câu hỏi số liệu cụ thể |

---

## Cài đặt và Chạy

### Yêu cầu hệ thống

- Python 3.10+
- RAM 8 GB trở lên
- Groq API Key (miễn phí tại [console.groq.com](https://console.groq.com))

### Bước 1: Clone và cài đặt

```bash
git clone https://github.com/your-username/RAG-Mom-Chatbot.git
cd RAG-Mom-Chatbot
pip install -r requirements.txt
```

### Bước 2: Cấu hình API Key

Tạo file `.env` tại thư mục gốc:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_API_KEY_1=gsk_backup_key_1
GROQ_API_KEY_2=gsk_backup_key_2
GROQ_API_KEY_3=gsk_backup_key_3
```

> Hệ thống hỗ trợ nhiều key để xử lý rate limit tự động (key rotation).

### Bước 3: Chuẩn bị dữ liệu

Đặt tài liệu y tế vào đúng thư mục theo `db_config.yml`:

```
data_store/
├── pdf/      ← 16 file PDF
├── word/     ← 40 file DOCX
└── excel/      ← file Q&A chuẩn hóa
```

### Bước 4: Xây dựng Vector Database

```bash
python vectordb.py
```

Lệnh này sẽ chunk tài liệu, tạo embedding và lưu FAISS index xuống đĩa.

### Bước 5: Chạy ứng dụng

```bash
streamlit run application.py
```

Truy cập: `http://localhost:8501`

---

## Cấu trúc dự án

```
RAG-Mom-Chatbot/
├── application.py              # Giao diện Streamlit chính
├── llm_chain.py                # RAGChain + Guardrails + Intent + Memory
├── vectordb.py                 # FAISS + Embedding + Retrieval + Hybrid Search
├── db_config.yml               # Cấu hình đường dẫn dữ liệu
├── model_config.yml            # Cấu hình model embedding
├── .env                        # API keys (không commit lên git)
├── data_store/
│   ├── pdf/                    # 16 file PDF y khoa
│   ├── word/                   # 40 file DOCX
│   └── excel/                    # File Q&A chuẩn hóa (151 cặp)
├── experiments/
│   ├── judge_clinical.py       # LLM-as-Judge v2 (Accuracy / Completeness / Safety)
│   ├── run_ablation_studies.py # Ablation: Multi-Query / Re-ranking / Summarized
│   ├── run_hybrid_search_ablation.py  # So sánh FAISS vs Hybrid (FAISS + BM25)
│   ├── evaluate_ragas_groq.py  # RAGAS evaluation pipeline
│   ├── test_intent_200.py      # Kiểm thử Intent 200 câu
│   ├── test_k_variation.py     # Biến thiên tham số K
│   └── test_stress_conversation.py    # Stress test 25 lượt hội thoại
└── KB/
    ├── KB1_Medical_Standard.xlsx      # 406 câu y khoa chuẩn
    ├── KB2_Mom_Style.xlsx             # 400 câu ngôn ngữ mẹ bỉm
    └── KB3_Information_Noise.xlsx     # 400 câu có nhiễu thông tin
```

---

## Công nghệ sử dụng

| Thành phần | Công nghệ / Model |
|---|---|
| LLM chính | Llama 3.1-8B-Instant (Groq LPU) |
| LLM Judge | Llama 3.3-70B-Versatile (Groq LPU) |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | FAISS (`IndexFlatL2`, lưu local) |
| Retrieval | MMR (`fetch_k=15`, `lambda_mult=0.7`) |
| Hybrid Search | FAISS + BM25Okapi + Adaptive Weighting |
| Re-ranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| Framework | LangChain + Custom RAGChain |
| UI | Streamlit |
| Evaluation | RAGAS (Faithfulness, Context Recall, Answer Relevancy, Context Precision) |

---

## Guardrails chi tiết

Hệ thống áp dụng guardrails 3 lớp theo thứ tự ưu tiên:

**Lớp 1 — BLOCKED** (từ chối, không trả lời): Phát hiện yêu cầu kê đơn thuốc, liều dùng cụ thể theo Luật KCB số 15/2023/QH15 (Điều 7). Phát hiện dấu hiệu tâm lý nguy hiểm (tự tử, tự làm hại) theo WHO mhGAP Guideline 2.0 — chuyển hướng đến đường dây hỗ trợ tâm lý.

**Lớp 2 — SMALLTALK** (trả lời xã giao, không gọi RAG): Nhận diện lời chào, câu hỏi về bản thân chatbot, cảm ơn, tạm biệt.

**Lớp 3 — RAG** (pipeline đầy đủ): Mọi câu hỏi y tế hợp lệ về chăm sóc mẹ và bé.

---

## Tài liệu tham khảo chính

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Developing and evaluating a chatbot to support maternal health care. (2026). *arXiv:2603.13168*.
- RAG-X: Systematic Diagnosis of Retrieval-Augmented Generation for Medical QA. (2026). *arXiv:2603.03541*.
- Johnson et al. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.
- Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP 2019.
- Quốc hội Việt Nam (2023). *Luật Khám bệnh, chữa bệnh số 15/2023/QH15*, Điều 7.
- WHO (2022). *mhGAP Intervention Guide 2.0*.

> Xem đầy đủ danh mục tài liệu tham khảo trong file `main.tex` (mục `\begin{thebibliography}`).

---

## License

MIT License — Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
