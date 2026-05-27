import ragas
print("Ragas version:", ragas.__version__)

# Thử tìm BleuScore và RougeScore ở đâu
import pkgutil
import importlib

for module_info in pkgutil.walk_packages(ragas.__path__, prefix="ragas."):
    try:
        mod = importlib.import_module(module_info.name)
        for attr in dir(mod):
            if "bleu" in attr.lower() or "rouge" in attr.lower():
                print(f"  Found '{attr}' in {module_info.name}")
    except Exception:
        pass