"""result_writer.py — JSON result persistence with timestamp."""
import json
import time
from pathlib import Path


def write_results(data: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {"_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **data}
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=_safe)
    print(f"  Saved: {path}")


def _safe(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")
