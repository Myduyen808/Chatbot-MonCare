"""
Chạy lặp lại N lần cùng một bộ câu hỏi kiểm thử (KB3, 50 câu) để đo độ dao động (variance)
của các chỉ số RAGAS, phục vụ vẽ boxplot theo yêu cầu của thầy hướng dẫn.

Cách dùng:
  python run_ragas_variance.py --kb kb3 --input kb3_batch_1.csv --n_runs 5
"""
import argparse, gc, time, os, random, re, warnings, logging
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument("--kb", type=str, default="kb3")
parser.add_argument("--input", type=str, required=True, help="File CSV chứa 50 câu hỏi cố định (giữ nguyên qua các lần chạy)")
parser.add_argument("--n_runs", type=int, default=5, help="Số lần lặp lại thí nghiệm")
args = parser.parse_args()

OUTPUT_DIR = f"variance_runs_{args.kb}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(args.input, encoding="utf-8-sig")
print(f"Đã load {len(df)} câu hỏi cố định từ {args.input}")

JUDGE_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3"),
] if k]

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset
from ragas.metrics import Faithfulness, ContextRecall, AnswerRelevancy, ContextPrecision
from llm_chain import RAGChain

METRIC_NAMES = ['faithfulness', 'context_recall', 'answer_relevancy', 'context_precision']

def get_judge_llm():
    return LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=random.choice(JUDGE_KEYS)))

def get_judge_embeddings():
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

chain = RAGChain(k=5)
_embeddings = get_judge_embeddings()

# ═══════════════════════════════════════════════════════
# VÒNG LẶP N LẦN CHẠY LẠI TOÀN BỘ BATCH (RAG + RAGAS)
# ═══════════════════════════════════════════════════════
for run_id in range(1, args.n_runs + 1):
    run_file = os.path.join(OUTPUT_DIR, f"{args.kb}_run{run_id}.csv")
    if os.path.exists(run_file):
        print(f"⏭️ Đã có {run_file}, bỏ qua run {run_id}")
        continue

    print(f"\n{'='*60}\n RUN {run_id}/{args.n_runs} — {args.kb.upper()}\n{'='*60}")
    results = []

    for i, row in df.iterrows():
        q = str(row["question"])
        gt = str(row.get("ground_truth", ""))
        print(f"[Run {run_id}][{i+1}/{len(df)}] {q[:50]}...", end=" ", flush=True)

        try:
            res = chain.invoke({"question": q, "history": []})
            answer = res.get("answer", "")
            docs = res.get("docs", [])
            if not answer or len(docs) == 0:
                print("❌ skip")
                continue

            ds = Dataset.from_dict({
                "question": [q], "answer": [answer],
                "contexts": [[d.page_content[:1200] for d in docs]],
                "ground_truth": [gt],
            })
            llm = get_judge_llm()
            metrics = [Faithfulness(llm=llm), ContextRecall(llm=llm),
                       AnswerRelevancy(llm=llm, embeddings=_embeddings, strictness=1),
                       ContextPrecision(llm=llm)]
            r = evaluate(dataset=ds, metrics=metrics, raise_exceptions=False,
                         run_config=RunConfig(max_workers=1, timeout=120))
            scores = r.to_pandas().iloc[0].to_dict()
            scores["question"] = q
            scores["run_id"] = run_id
            results.append(scores)
            print("✅")
        except Exception as e:
            print(f"❌ {str(e)[:60]}")

        time.sleep(3)
        gc.collect()

    pd.DataFrame(results).to_csv(run_file, index=False, encoding="utf-8-sig")
    print(f"💾 Đã lưu {run_file}")

print("\n🎉 Hoàn tất toàn bộ các lần chạy. Dùng script vẽ boxplot để tổng hợp kết quả.")