from dotenv import load_dotenv
load_dotenv()

import json
import warnings
warnings.filterwarnings("ignore")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

CHECKPOINT_FILE = 'checkpoint_Kịch_bản_1_-_Y_khoa_thuần_túy.json'

print("Đang đọc dữ liệu từ checkpoint...")
with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
    ck = json.load(f)

questions  = ck["questions"]
answers    = ck["answers"]
references = ck["references"]
print(f"Đã load {len(questions)} mẫu.")

print("Đang tính toán điểm số...")

smoother      = SmoothingFunction().method1
rouge_scorer_ = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

bleu_scores  = []
rouge_scores = []

for i, (answer, reference) in enumerate(zip(answers, references)):
    ref_tok = reference.split()
    hyp_tok = answer.split()
    bleu  = sentence_bleu([ref_tok], hyp_tok, smoothing_function=smoother)
    rouge = rouge_scorer_.score(reference, answer)['rougeL'].fmeasure
    bleu_scores.append(bleu)
    rouge_scores.append(rouge)
    if (i + 1) % 100 == 0:
        print(f"   ... {i+1}/{len(questions)} mẫu")

avg_bleu  = sum(bleu_scores)  / len(bleu_scores)
avg_rouge = sum(rouge_scores) / len(rouge_scores)

print("\n" + "="*50)
print("KET QUA DANH GIA")
print("="*50)
print(f"  So mau   : {len(questions)}")
print(f"  BLEU     : {avg_bleu:.4f}")
print(f"  ROUGE-L  : {avg_rouge:.4f}")
print("="*50)

out_file = CHECKPOINT_FILE.replace('.json', '_eval_results.json')
import json as _j
with open(out_file, 'w', encoding='utf-8') as f:
    _j.dump({"total_samples": len(questions), "avg_bleu": round(avg_bleu,4), "avg_rouge_l": round(avg_rouge,4)}, f, ensure_ascii=False, indent=2)
print(f"Luu ket qua: {out_file}")
