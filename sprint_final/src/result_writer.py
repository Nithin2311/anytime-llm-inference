"""result_writer.py — Atomic JSON writes, sprint.log logging, LaTeX fragments. Identical to v1."""
import json, logging, os, time

_results_dir = _figures_dir = _latex_dir = _logger = None

def configure(base_dir):
    global _results_dir, _figures_dir, _latex_dir, _logger
    _results_dir = os.path.join(base_dir, "results")
    _figures_dir = os.path.join(base_dir, "figures")
    _latex_dir   = os.path.join(base_dir, "latex")
    log_file     = os.path.join(_results_dir, "sprint.log")
    for d in (_results_dir, _figures_dir, _latex_dir):
        os.makedirs(d, exist_ok=True)
    _logger = logging.getLogger("sprint_v2")
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt)
        sh = logging.StreamHandler();       sh.setFormatter(fmt)
        _logger.addHandler(fh); _logger.addHandler(sh)

def results_path(f):  return os.path.join(_results_dir, f)
def figures_path(f):  return os.path.join(_figures_dir, f)
def latex_path(f):    return os.path.join(_latex_dir, f)

def write_json(filename, data):
    path = results_path(filename); tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2)
    os.replace(tmp, path); return path

def write_latex(filename, content):
    path = latex_path(filename)
    with open(path, "w") as f: f.write(content)
    return path

def already_done(filename): return os.path.exists(results_path(filename))
def load_json(filename):
    with open(results_path(filename)) as f: return json.load(f)

def log_start(eid):   _logger.info(f"[{eid}] START"); return time.time()
def log_success(eid, t0): _logger.info(f"[{eid}] SUCCESS elapsed={time.time()-t0:.1f}s")
def log_failure(eid, t0, exc): _logger.error(f"[{eid}] FAILURE elapsed={time.time()-t0:.1f}s error={exc}")


def write_results(data, output_path):
    """Atomic JSON write to an absolute path (Path or str)."""
    path = str(output_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
    return path
