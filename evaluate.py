from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import BleuScore, RougeScore
from llm_chain import RAGChain

chain = RAGChain(k=3)

test_cases = [
    "Trẻ sơ sinh bú mẹ bao nhiêu lần một ngày?",
    "Mẹ sau sinh nên ăn gì?",
    "Trẻ 3 tháng tuổi nặng bao nhiêu kg là bình thường?",
]

questions, answers, contexts, references = [], [], [], []

for q in test_cases:
    result = chain.invoke({"question": q})
    questions.append(q)
    answers.append(result["answer"])
    contexts.append([d.page_content for d in result["docs"]])
    references.append(result["answer"])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "reference": references,
})

results = evaluate(dataset, metrics=[BleuScore(), RougeScore()])
print(results)