# MomCare — Trợ lý hỏi đáp tiếng Việt về chăm sóc mẹ và trẻ nhỏ

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Index-purple)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.1--8B-orange)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red?logo=streamlit&logoColor=white)](https://streamlit.io/)

> **Luận văn tốt nghiệp — Ngành Hệ thống Thông tin, Đại học Cần Thơ (K48)**  
> **Sinh viên:** Trần Thị Mỹ Duyên — **MSSV:** B2203435  
> **Giảng viên hướng dẫn:** PGS.TS Nguyễn Thanh Hải

---

## 1. Giới thiệu

**MomCare** là hệ thống hỏi đáp tiếng Việt dựa trên kiến trúc **Retrieval-Augmented Generation (RAG)**, hỗ trợ tra cứu thông tin về thai kỳ, chăm sóc mẹ sau sinh, trẻ sơ sinh, trẻ nhỏ và dinh dưỡng. Hệ thống sinh phản hồi từ kho tri thức đã được thu thập, làm sạch và lập chỉ mục, đồng thời hiển thị các đoạn tài liệu được dùng làm căn cứ.

MomCare tập trung vào ba yêu cầu chính:

- truy xuất tài liệu phù hợp và hạn chế sinh thông tin không có căn cứ;
- kiểm soát truy vấn nguy hiểm, tự hại, kê đơn hoặc liều dùng cụ thể;
- duy trì ngữ cảnh hội thoại nhiều lượt với lượng lịch sử được đưa vào mô hình nhỏ hơn.

> **Lưu ý y tế:** MomCare chỉ hỗ trợ tra cứu thông tin. Hệ thống không thay thế việc thăm khám, chẩn đoán, kê đơn hoặc tư vấn trực tiếp của nhân viên y tế.

---

## 2. Các thành phần chính

| Thành phần | Mô tả triển khai |
|---|---|
| **Input Guardrails** | Kiểm tra theo luật, phát hiện biểu đạt tự hại trực tiếp và gián tiếp, prompt injection, thuốc không rõ nguồn gốc, can thiệp nguy hiểm, yêu cầu liều dùng và một số tình huống leo thang theo lịch sử hội thoại. |
| **Adaptive Context Management** | Rolling Summary tích lũy; luôn giữ nguyên hai tin nhắn gần nhất; chỉ cập nhật phần lịch sử cũ chưa được tóm tắt khi có ít nhất hai tin nhắn và đạt ngưỡng 250 ký tự. |
| **Task Merging** | Gộp Query Rewriting và Intent Detection trong một lần gọi LLM. |
| **Intent Router** | Phân loại thành bốn nhãn: `BLOCKED`, `SMALLTALK`, `OUT_OF_SCOPE`, `RAG`. |
| **Adaptive Hybrid Search** | Tạo hai danh sách ứng viên độc lập từ FAISS và BM25, hợp nhất rồi tính điểm bằng trọng số Adaptive Alpha theo loại truy vấn. |
| **Adaptive Alpha** | Phân loại truy vấn thành `quantitative`, `exact_lexical`, `noisy_conversational`, `semantic`; đọc trọng số từ `adaptive_alpha_config.json`, có bộ tham số fallback. |
| **Table Bonus** | Cộng điểm ưu tiên cho tài liệu có `chunk_type=data_table` khi truy vấn thuộc nhóm định lượng. |
| **Multi-Query Expansion** | Chỉ kích hoạt với câu hỏi văn bản có không quá 5 từ; sinh tối đa hai biến thể bổ sung, mỗi biến thể lấy tối đa 10 tài liệu bằng MMR. |
| **Cross-Encoder Re-ranking** | Giới hạn tối đa 40 ứng viên, tái xếp hạng bằng `cross-encoder/ms-marco-MiniLM-L-6-v2`, chọn Top-5 cho prompt. |
| **Document Injection Filtering** | Loại các dòng trong tài liệu có hình thức chỉ dẫn điều khiển mô hình trước khi ghép vào prompt. |
| **Safety-first Generation** | Llama 3.1-8B-Instant chỉ được yêu cầu trả lời từ tài liệu truy xuất, giới hạn tối đa 350 token và tối đa 4 ý khi liệt kê. |
| **Output Guardrails** | Loại đoạn lặp, kiểm tra phản hồi thiếu cảnh báo trong tình huống nguy cơ cao và bổ sung lưu ý khi câu trả lời có cách diễn đạt mang tính chẩn đoán. |
| **Source Traceability** | Hiển thị tên tệp, loại tệp, số trang, `chunk_id` và phần xem trước nội dung nguồn. |

---

## 3. Kiến trúc xử lý

```text
CÂU HỎI NGƯỜI DÙNG
        │
        ▼
Lọc các lượt bị chặn khỏi lịch sử dùng cho Rewriter
        │
        ▼
Adaptive Context Management — Rolling Summary
        │
        ▼
Input Guardrails + kiểm tra leo thang theo hội thoại
        │
        ├── BLOCKED ───────────────► phản hồi an toàn, không truy xuất
        ├── SMALLTALK ─────────────► phản hồi xã giao, không truy xuất
        │
        ▼
Task Merging: Query Rewriting + Intent Detection
        │
        ├── OUT_OF_SCOPE ──────────► thông báo giới hạn phạm vi
        ├── BLOCKED/SMALLTALK ─────► xử lý theo nhãn
        └── RAG
             │
             ▼
Adaptive Hybrid Search
FAISS Top-50 thô + BM25 Top-50 thô
             │
             ▼
Hợp nhất, khử trùng lặp, Adaptive Alpha + Table Bonus
             │
             ▼
Giữ tối đa 25 ứng viên chính
             │
             ├── Câu hỏi ≤ 5 từ: thêm tối đa 2 Multi-Query
             │                    mỗi truy vấn lấy tối đa 10 tài liệu MMR
             ▼
Hợp nhất và giới hạn tối đa 40 ứng viên
             │
             ▼
Cross-Encoder Re-ranking
             │
             ▼
Top-5 tài liệu + lọc document prompt injection
             │
             ▼
Llama 3.1-8B-Instant — Safety-first Prompt
             │
             ▼
Output Guardrails
             │
             ▼
PHẢN HỒI + NGUỒN TÀI LIỆU
```

### Tham số triển khai chính

| Tham số | Giá trị |
|---|---:|
| LLM | `llama-3.1-8b-instant` |
| Temperature mặc định | 0.2 |
| Số tài liệu chính sau Adaptive Hybrid Search | 25 |
| Số ứng viên tối đa trước Cross-Encoder | 40 |
| Số tài liệu cuối đưa vào LLM | 5 |
| Số token tối đa của phản hồi RAG | 350 |
| Số ý tối đa khi liệt kê | 4 |
| Multi-Query | kích hoạt khi câu hỏi `≤ 5` từ |
| MMR cho truy vấn mở rộng | `fetch_k=30`, `lambda_mult=0.7` |
| Rolling Summary | giữ 2 tin gần nhất; ngưỡng cập nhật 2 tin và 250 ký tự |

---

## 4. Xây dựng kho tri thức

Kho tri thức được tạo từ dữ liệu tiếng Việt thuộc ba định dạng đang thống kê trong luận văn:

| Định dạng | Số tệp | Nội dung chính |
|---|---:|---|
| PDF | 19 | Hướng dẫn, phác đồ và tài liệu chuyên môn từ các cơ quan, bệnh viện và tổ chức y tế. |
| DOCX | 45 | Bài viết chuyên khoa về chăm sóc mẹ sau sinh, nuôi con bằng sữa mẹ và chăm sóc trẻ. |
| XLSX | 2 | Dữ liệu hỏi–đáp có cấu trúc, gồm các dạng FAQ, ViMedAQA và bộ đề y khoa. |
| **Tổng** | **66** | **6.176 chunk trong chỉ mục FAISS** |

### Thông tin kỹ thuật

| Thuộc tính | Giá trị |
|---|---|
| Ngôn ngữ | Tiếng Việt |
| Embedding | `keepitreal/vietnamese-sbert` |
| Kích thước vector | 768 chiều |
| Vector index | FAISS `IndexFlatL2` |
| Số ký tự trung bình/chunk | 1.017,3 |
| Số ký tự trung vị/chunk | 898 |
| Token ước lượng trung bình/chunk | 511,9 |
| Token ước lượng trung vị/chunk | 453 |

Mã nguồn vẫn hỗ trợ đọc **CSV** trong pipeline và giao diện quản lý dữ liệu. Tuy nhiên, thống kê kho tri thức cuối cùng trong luận văn gồm 19 PDF, 45 DOCX và 2 XLSX; không ghi nhận tệp CSV trong tổng số 66 tài liệu.

### Chính sách phân mảnh

- PDF được làm sạch lỗi font, ký tự điều khiển và khoảng trắng.
- DOCX được loại boilerplate và nội dung điều hướng web.
- Bản ghi `medical_exam`, `faq` và `vimedaqa` được giữ nguyên để bảo toàn quan hệ câu hỏi–đáp án–ngữ cảnh.
- Đoạn có tỷ lệ dòng chứa dữ liệu định lượng lớn được gắn `chunk_type=data_table` và không chia nhỏ.
- Văn bản thông thường được phân mảnh bằng `RecursiveCharacterTextSplitter` theo `db_config.yml`.
- Chunk thông thường phải còn ít nhất 50 ký tự; chunk bảng phải có ít nhất 80 ký tự.

---

## 5. Kết quả thực nghiệm chính

### 5.1. RAGAS trên 300 câu hỏi

Kết quả trung bình trên ba kịch bản KB1, KB2 và KB3:

| Chỉ số | Điểm trung bình |
|---|---:|
| Faithfulness | 0.741 |
| Context Precision | 0.704 |
| Context Recall | 0.780 |
| Answer Relevancy | 0.572 |

### 5.2. LLM-as-a-Judge

Trên 300 câu hỏi thuộc ba kịch bản:

| Chỉ số | Kết quả |
|---|---:|
| Tỷ lệ `HAS_ANSWER` trung bình | 87,3% |
| Tỷ lệ `NOT_FOUND` trung bình | 12,6% |
| Clinical Accuracy trên nhóm `HAS_ANSWER` | 65,4% |
| Completeness trên toàn bộ tập | 66,0% |
| Safety | 100,0% |

### 5.3. Intent Classification — 200 câu, 4 lớp cân bằng

| Lớp | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `BLOCKED` | 1.0000 | 0.9000 | 0.9474 | 50 |
| `SMALLTALK` | 0.9804 | 1.0000 | 0.9901 | 50 |
| `OUT_OF_SCOPE` | 1.0000 | 0.9000 | 0.9474 | 50 |
| `RAG` | 0.8475 | 1.0000 | 0.9174 | 50 |
| **Macro Average** | **0.9570** | **0.9500** | **0.9506** | **200** |
| **Weighted Average** | **0.9570** | **0.9500** | **0.9506** | **200** |

- **Accuracy:** 95,0% — 190/200 câu đúng.
- **Macro F1:** 0.9506.
- **Weighted F1:** 0.9506.
- Ma trận lỗi: 5 `BLOCKED → RAG`, 4 `OUT_OF_SCOPE → RAG`, 1 `OUT_OF_SCOPE → SMALLTALK`.

### 5.4. Adaptive Context Management

| Phương pháp | Khôi phục ngữ cảnh | Ký tự lịch sử trung bình | Mức giảm |
|---|---:|---:|---:|
| No Memory | 0,00% | 0 | 100,00% |
| Fixed Window | 33,33% | 1.078 | 26,84% |
| Full History | 100,00% | 1.474 | 0,00% |
| Summary Only | 100,00% | 369 | 74,96% |
| **Adaptive Context Management** | **100,00%** | **495** | **66,39%** |

Adaptive Context Management giữ tỷ lệ khôi phục ngữ cảnh tương đương Full History, trong khi giảm 66,39% kích thước lịch sử đưa vào Query Rewriting.

### 5.5. Task Merging

**Intent Detection:**

| Phương pháp | Accuracy | LLM calls | Thời gian |
|---|---:|---:|---:|
| Pipeline tách rời | 98,46% | 50 | 19,17 s |
| Task Merging | 98,46% | 33 | 25,87 s |

**Query Rewriting:**

| Phương pháp | Rewrite Success | LLM calls | Thời gian |
|---|---:|---:|---:|
| Pipeline tách rời | 75,00% | 15 | 13,29 s |
| Task Merging | 100,00% | 8 | 27,22 s |

Task Merging giảm số lần gọi LLM nhưng trong thiết lập thực nghiệm có thời gian xử lý cao hơn. README không khẳng định cơ chế này làm giảm latency hoặc token khi các chỉ số đó chưa được chứng minh trực tiếp trong bảng Task Merging.

### 5.6. Hybrid Search so với Vector Search

| Chỉ số RAGAS | Vector Search | Hybrid Search | Thay đổi |
|---|---:|---:|---:|
| Context Recall | 0.822 | 0.710 | -0.112 |
| Context Precision | 0.587 | 0.642 | +0.055 |
| Answer Relevancy | 0.619 | 0.598 | -0.021 |
| Faithfulness | 0.541 | 0.686 | +0.145 |

Thực nghiệm này sử dụng trọng số cố định để đánh giá tác động của việc bổ sung BM25. Cấu hình triển khai cuối cùng sử dụng Adaptive Alpha.

### 5.7. Adaptive Alpha — 212 câu test độc lập

| Cấu hình | Hit Rate@5 | MRR@5 | Latency |
|---|---:|---:|---:|
| Fixed α = 0.3 | 0.4717 | 0.3711 | 0.1462 s |
| Fixed α = 0.4 | 0.4575 | 0.3745 | 0.1409 s |
| Fixed α = 0.5 | 0.4623 | 0.3590 | 0.1394 s |
| Fixed α = 0.7 | 0.4575 | 0.3289 | 0.1379 s |
| **Adaptive Alpha** | **0.4764** | **0.3762** | **0.1377 s** |

Kiểm định Wilcoxon cho thấy cải thiện có ý nghĩa thống kê ở hai phép so sánh:

- Hit Rate@5: Adaptive Alpha so với α = 0.4, `p = 0.0455`.
- MRR@5: Adaptive Alpha so với α = 0.7, `p = 0.0084`.

### 5.8. ViMedAQA

MomCare xử lý thành công 4.496/4.497 mẫu đã chọn từ ViMedAQA.

| Hệ thống | ROUGE-L | BLEU | METEOR | BERTScore | Average |
|---|---:|---:|---:|---:|---:|
| **MomCare RAG** | **47.68** | **32.41** | **58.18** | **80.60** | **54.72** |

Kết quả đối chiếu với các baseline trong nghiên cứu ViMedAQA chỉ mang tính tham khảo vì MomCare tự truy xuất tối đa 5 đoạn ngữ cảnh, còn thiết lập open-book của nghiên cứu gốc được cung cấp trực tiếp paragraph tương ứng.

---

## 6. Cài đặt

### Yêu cầu

- Python 3.10 trở lên;
- RAM khuyến nghị từ 8 GB;
- Groq API key;
- các thư viện trong `requirements.txt` của dự án.

### Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu dự án chưa có `requirements.txt` hoàn chỉnh, các nhóm thư viện chính gồm Streamlit, LangChain, FAISS, sentence-transformers, rank-bm25, Groq, pandas, python-docx, pypdf, PyYAML và python-dotenv.

### Cấu hình API

Tạo tệp `.env` tại thư mục gốc:

```env
GROQ_API_KEY=gsk_your_primary_key
GROQ_API_KEY_1=gsk_optional_key_1
GROQ_API_KEY_2=gsk_optional_key_2
GROQ_API_KEY_3=gsk_optional_key_3
```

Hệ thống chọn ngẫu nhiên một key khả dụng cho mỗi lần gọi và retry tối đa 4 lần khi gặp lỗi.

### Cấu hình dữ liệu

Các đường dẫn được khai báo trong `db_config.yml`. Cấu trúc thường dùng:

```text
data_store/
├── pdf/
├── word/
├── csv/
└── excel/
```

Mô hình embedding được khai báo trong `model_config.yml`.

### Cấu hình Adaptive Alpha

Tệp `adaptive_alpha_config.json` được tạo từ bước hiệu chỉnh trên tập development. Khi tệp không tồn tại hoặc không đọc được, code dùng cấu hình fallback:

```json
{
  "quantitative": 0.30,
  "exact_lexical": 0.30,
  "noisy_conversational": 0.40,
  "semantic": 0.40,
  "table_bonus": 0.08
}
```

Các giá trị triển khai thực tế ưu tiên nội dung trong `adaptive_alpha_config.json`.

---

## 7. Chạy hệ thống

### Xây dựng lại FAISS index

```bash
python vectordb.py
```

Quy trình này đọc dữ liệu, làm sạch, phân mảnh, tạo embedding và lưu chỉ mục FAISS theo cấu hình dự án.

### Khởi động giao diện

```bash
streamlit run application.py
```

Mặc định truy cập tại:

```text
http://localhost:8501
```

Giao diện gồm hai khu vực đang được định tuyến chính thức:

- **Chatbot:** hỏi đáp, xem nguồn tài liệu và quản lý lịch sử hội thoại;
- **Quản lý Dữ liệu:** tải PDF/DOCX/CSV, xem tài liệu, xóa tài liệu và xây dựng lại VectorDB. Tệp XLSX được đọc từ thư mục cấu hình khi xây dựng chỉ mục.

---

## 8. Cấu trúc mã nguồn cốt lõi

```text
RAG-MomCare-Chatbot/
├── application.py
│   ├── Streamlit UI
│   ├── Adaptive Context Management
│   ├── lọc lượt bị chặn khỏi ngữ cảnh Rewriter
│   └── hiển thị nguồn tài liệu
│
├── llm_chain.py
│   ├── Guardrails đầu vào/đầu ra
│   ├── Task Merging
│   ├── Intent Router 4 lớp
│   ├── Rolling Summary
│   ├── Adaptive Hybrid Search + BM25 cache
│   ├── Multi-Query có điều kiện
│   ├── Cross-Encoder Re-ranking
│   ├── document injection filtering
│   └── RAGChain
│
├── vectordb.py
│   ├── loader PDF, DOCX, CSV, XLSX
│   ├── làm sạch và chunking
│   ├── Vietnamese-SBERT embeddings
│   ├── FAISS index
│   └── MMR retrieval cho nhánh Multi-Query/fallback
│
├── history_handle.py
├── db_config.yml
├── model_config.yml
├── adaptive_alpha_config.json
├── .env
└── main.tex
```

---

## 9. Công nghệ sử dụng

| Thành phần | Công nghệ / mô hình |
|---|---|
| LLM chính | Groq `llama-3.1-8b-instant` |
| LLM-as-a-Judge trong luận văn | Llama 3.3-70B |
| Embedding | `keepitreal/vietnamese-sbert` |
| Vector database | FAISS `IndexFlatL2` |
| Keyword retrieval | `BM25Okapi` |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Framework | LangChain + custom `RAGChain` |
| UI | Streamlit |
| Dữ liệu | PDF, DOCX, CSV, XLSX |
| Đánh giá | RAGAS, LLM-as-a-Judge, BLEU, ROUGE-L, METEOR, BERTScore, Hit Rate@5, MRR@5, Wilcoxon |

---

## 10. Giới hạn hiện tại

- Bộ định tuyến vẫn bỏ sót một số truy vấn nguy hiểm và truy vấn ngoài phạm vi; Recall của `BLOCKED` và `OUT_OF_SCOPE` là 90% trên benchmark 200 câu.
- Adaptive Alpha cải thiện chất lượng trung bình nhưng chỉ đạt ý nghĩa thống kê ở một số phép so sánh.
- Multi-Query có thể làm tăng nhiễu ở câu hỏi đã đủ rõ, nên chỉ được kích hoạt có điều kiện.
- Kết quả benchmark ViMedAQA không phải so sánh cùng điều kiện tuyệt đối với các baseline open-book.
- Các guardrails dựa nhiều vào luật và mẫu ngôn ngữ; cần tiếp tục mở rộng với cách diễn đạt mới.
- Hệ thống không được dùng để chẩn đoán, kê đơn hoặc xử lý tình huống khẩn cấp thay cho nhân viên y tế.

---

## 11. Tài liệu luận văn

Các mô tả thiết kế, công thức, dữ liệu thực nghiệm, bảng kết quả và danh mục tài liệu tham khảo đầy đủ được trình bày trong `main.tex`.
