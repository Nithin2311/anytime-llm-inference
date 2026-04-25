"""
result_writer.py — Shared I/O utilities for the sprint package.

Provides atomic JSON writes, sprint.log structured logging, and LaTeX
fragment generation. Each experiment calls configure() once at startup.
"""

import json
import logging
import os
import time

_results_dir = None
_figures_dir = None
_latex_dir   = None
_logger      = None


def configure(base_dir):
    global _results_dir, _figures_dir, _latex_dir, _logger
    _results_dir = os.path.join(base_dir, "results")
    _figures_dir = os.path.join(base_dir, "figures")
    _latex_dir   = os.path.join(base_dir, "latex")
    log_file     = os.path.join(_results_dir, "sprint.log")

    for d in (_results_dir, _figures_dir, _latex_dir):
        os.makedirs(d, exist_ok=True)

    _logger = logging.getLogger("sprint")
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        _logger.addHandler(fh)
        _logger.addHandler(sh)


def results_path(filename):
    return os.path.join(_results_dir, filename)


def figures_path(filename):
    return os.path.join(_figures_dir, filename)


def latex_path(filename):
    return os.path.join(_latex_dir, filename)


def write_json(filename, data):
    """Atomic write: .tmp then os.replace to avoid partial reads on crash."""
    path = results_path(filename)
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path


def write_latex(filename, content):
    path = latex_path(filename)
    with open(path, "w") as f:
        f.write(content)
    return path


def already_done(filename):
    """Return True if results/filename exists — allows idempotent re-runs."""
    return os.path.exists(results_path(filename))


def load_json(filename):
    with open(results_path(filename)) as f:
        return json.load(f)


def log_start(experiment_id):
    _logger.info(f"[{experiment_id}] START")
    return time.time()


def log_success(experiment_id, start_time):
    elapsed = time.time() - start_time
    _logger.info(f"[{experiment_id}] SUCCESS elapsed={elapsed:.1f}s")


def log_failure(experiment_id, start_time, exc):
    elapsed = time.time() - start_time
    _logger.error(f"[{experiment_id}] FAILURE elapsed={elapsed:.1f}s error={exc}")
