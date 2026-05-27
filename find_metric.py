import pkgutil, importlib, ragas

for mi in pkgutil.walk_packages(ragas.__path__, prefix="ragas."):
    try:
        mod = importlib.import_module(mi.name)
        if hasattr(mod, "Metric"):
            print(f"Found 'Metric' in {mi.name}")
    except Exception:
        pass