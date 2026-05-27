import ragas
print("Ragas version:", ragas.__version__)

from ragas.metrics._bleu_score import BleuScore
from ragas.metrics._rouge_score import RougeScore

b = BleuScore()
r = RougeScore()

print("BleuScore type:", type(b))
print("RougeScore type:", type(r))
print("BleuScore MRO:", [c.__name__ for c in type(b).__mro__])

# Xem evaluate check gì
import inspect
from ragas import evaluation
src = inspect.getsource(evaluation)
# Tìm đoạn check metric
idx = src.find("initialised metric")
print("\n--- Source xung quanh lỗi ---")
print(src[max(0,idx-300):idx+300])