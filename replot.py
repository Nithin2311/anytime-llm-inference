"""
replot.py — Regenerate fixed figures from existing JSON result files.
No GPU / model required.

Usage:
    python replot.py
"""

import json
import sys

# ── scheduler_comparison.png ──────────────────────────────────────────────────
print("Regenerating scheduler_comparison.png ...")
try:
    with open("scheduler_comparison.json") as f:
        data = json.load(f)
    # Import the fixed plot function from the updated module
    from compare_schedulers import plot_comparison
    plot_comparison(data["stateless_metrics"], data["kvcached_metrics"])
    print("  OK → scheduler_comparison.png")
except Exception as e:
    print(f"  FAILED: {e}", file=sys.stderr)

# ── evt_wcet_analysis.png ─────────────────────────────────────────────────────
print("Regenerating evt_wcet_analysis.png ...")
try:
    with open("evt_wcet_results.json") as f:
        evt_data = json.load(f)

    from evt_wcet_analysis import plot_evt
    # plot_evt accepts qq_model=None and falls back gracefully
    plot_evt(evt_data["results"], qq_model=None)
    print("  OK → evt_wcet_analysis.png")
except Exception as e:
    print(f"  FAILED: {e}", file=sys.stderr)

print("Done.")
