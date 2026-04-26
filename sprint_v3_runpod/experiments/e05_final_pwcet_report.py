"""
e05_final_pwcet_report.py — Final publishable pWCET table.

Decision logic (read automatically from E01–E04 results):
  CASE A: E01 IID PASS + E03 Gumbel valid (xi < 0.15)
    → Use Gumbel on spaced data. Gumbel pWCET is valid.
  CASE B: E01 IID PASS + E03 Gumbel invalid (xi >= 0.15)
    → Use block maxima GEV. Report: block size, xi of maxima, GEV pWCET.
  CASE C: E01 IID FAIL + E03 Gumbel invalid
    → Block maxima GEV is mandatory. IID on raw data violated but block
      maxima are approximately independent.
  CASE D: E01 IID FAIL + E03 Gumbel valid (unusual)
    → Spaced data is correlated but Gumbel fits well; use block maxima
      Gumbel as conservative option, note limitation.

Outputs a final LaTeX table + JSON for the report.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from evt_utils_v3 import (
    pot_tail, fit_gev, gumbel_pwcet, gev_pwcet,
    anderson_darling_gumbel, ljungbox_iid,
    block_maxima, analyze_block_maxima, bootstrap_gev_pwcet_ci
)
from result_writer import write_results
from fig_style import apply_style, DOUBLE

apply_style()

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LENS        = [32, 64, 128, 256, 512, 1024]
EXCEEDANCE_PROB = 1e-6
POT_FRACTION    = 0.20
N_BOOTSTRAP     = 1000


def pick_best_method(e01_data: dict, e03_data: dict, e04_data: dict, cell_key: str) -> str:
    """Return 'gumbel_spaced', 'gev_spaced', or 'block_maxima_gev'."""
    iid_pass = e01_data.get("cells", {}).get(cell_key, {}).get("iid_pass", False)
    gumbel_valid = e03_data.get("cells", {}).get(cell_key, {}).get("gumbel_valid", False)

    if iid_pass and gumbel_valid:
        return "gumbel_spaced"
    elif iid_pass and not gumbel_valid:
        return "gev_spaced"
    else:
        return "block_maxima_gev"


def main():
    print("=" * 60)
    print("E05  Final pWCET Report — Decision-Driven Methodology")
    print("=" * 60)

    # Load prerequisite results
    paths = {
        "e00": RESULTS_DIR / "e00_spaced_profiling.json",
        "e01": RESULTS_DIR / "e01_iid_validation.json",
        "e03": RESULTS_DIR / "e03_gev_xi_refit.json",
        "e04": RESULTS_DIR / "e04_block_maxima.json",
    }
    data = {}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{k} results not found — run {k} first.")
        with open(p) as f:
            data[k] = json.load(f)

    raw = data["e00"]["raw"]
    rows = []

    print(f"\n  Method decision per cell:")
    print(f"  {'Cell':<22} {'Method':<20} {'xi':>7} {'pWCET(1e-6)':>14} "
          f"{'CI_lo':>8} {'CI_hi':>8} {'IID':>5} {'Gumbel':>7}")
    print("  " + "-" * 95)

    for seq_len in SEQ_LENS:
        for layer in ["l16", "full"]:
            cell_key = f"seq{seq_len}_{layer}"
            if cell_key not in raw:
                continue

            arr  = np.array(raw[cell_key])
            method = pick_best_method(data["e01"], data["e03"], data["e04"], cell_key)

            if method == "gumbel_spaced":
                tail   = pot_tail(arr, POT_FRACTION)
                pw     = gumbel_pwcet(tail, EXCEEDANCE_PROB)
                ci_r   = bootstrap_gev_pwcet_ci(tail, EXCEEDANCE_PROB, N_BOOTSTRAP)
                xi     = 0.0
                ci_lo  = ci_r.get("ci_lower", np.nan)
                ci_hi  = ci_r.get("ci_upper", np.nan)

            elif method == "gev_spaced":
                tail   = pot_tail(arr, POT_FRACTION)
                gev    = fit_gev(tail)
                pw_val = gev_pwcet(gev, EXCEEDANCE_PROB)
                pw     = pw_val if np.isfinite(pw_val) and pw_val < 1e9 else np.nan
                ci_r   = bootstrap_gev_pwcet_ci(tail, EXCEEDANCE_PROB, N_BOOTSTRAP)
                xi     = gev["xi"]
                ci_lo  = ci_r.get("ci_lower", np.nan)
                ci_hi  = ci_r.get("ci_upper", np.nan)

            else:  # block_maxima_gev
                # Find best block size from E04
                bm_e04 = data["e04"].get("cells", {}).get(cell_key, {})
                best_b = data["e04"].get("min_iid_valid_block", {}).get(cell_key, 10) or 10
                bm_key = f"b{best_b}"
                if bm_key in bm_e04:
                    bm = bm_e04[bm_key]
                    pw  = bm.get("pwcet_gev") or bm.get("pwcet_gumbel")
                    xi  = bm["gev"]["xi"]
                    ci_r = bm.get("ci", {})
                    ci_lo = ci_r.get("ci_lower", np.nan)
                    ci_hi = ci_r.get("ci_upper", np.nan)
                else:
                    # Fallback: compute directly
                    bm_data = analyze_block_maxima(arr, best_b, EXCEEDANCE_PROB, N_BOOTSTRAP)
                    pw  = bm_data.get("pwcet_gev") or bm_data.get("pwcet_gumbel")
                    xi  = bm_data["gev"]["xi"]
                    ci_lo = bm_data["ci"].get("ci_lower", np.nan)
                    ci_hi = bm_data["ci"].get("ci_upper", np.nan)

            iid_pass = data["e01"].get("cells", {}).get(cell_key, {}).get("iid_pass", False)
            gum_val  = data["e03"].get("cells", {}).get(cell_key, {}).get("gumbel_valid", False)

            pw_str = f"{pw:.2f}ms" if pw is not None and np.isfinite(pw) else "unbounded"
            ci_str = (f"[{ci_lo:.2f},{ci_hi:.2f}]"
                      if np.isfinite(ci_lo) and np.isfinite(ci_hi) else "n/a")
            print(f"  {cell_key:<22} {method:<20} {xi:>7.4f} "
                  f"{pw_str:>14} {ci_lo:>8.2f} {ci_hi:>8.2f} "
                  f"{'Y' if iid_pass else 'N':>5} {'Y' if gum_val else 'N':>7}")

            rows.append({
                "seq_len": seq_len,
                "layer": layer,
                "cell": cell_key,
                "method": method,
                "xi": round(xi, 4),
                "pwcet_1e6_ms": round(float(pw), 3) if (
                    pw is not None and np.isfinite(pw)
                ) else None,
                "ci_lower_ms": round(float(ci_lo), 3) if np.isfinite(ci_lo) else None,
                "ci_upper_ms": round(float(ci_hi), 3) if np.isfinite(ci_hi) else None,
                "iid_pass": iid_pass,
                "gumbel_valid": gum_val,
                "mean_ms": round(float(arr.mean()), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
            })

    # ── LaTeX table (full pass rows only, for report) ─────────────────────────
    tex_path = RESULTS_DIR / "table_final_pwcet.tex"
    with open(tex_path, "w") as f:
        f.write("% E05: Final pWCET table — spaced profiling + validated EVT method\n")
        f.write("\\begin{tabular}{@{}rcccccc@{}}\n\\toprule\n")
        f.write("Seq & Method & $\\xi$ & Emp$\\times$1.10 & pWCET($10^{-6}$) & 95\\% CI & $P(\\text{miss})$ \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            if row["layer"] != "full":
                continue
            emp = round(row["p99_ms"] * 1.10, 2) if row["p99_ms"] else "---"
            pw  = f"{row['pwcet_1e6_ms']:.2f}" if row["pwcet_1e6_ms"] else "unbounded"
            ci  = (f"[{row['ci_lower_ms']:.2f}, {row['ci_upper_ms']:.2f}]"
                   if row["ci_lower_ms"] else "n/a")
            method_short = row["method"].replace("gumbel_spaced", "Gumbel").replace(
                "gev_spaced", "GEV (POT)").replace("block_maxima_gev", "BM-GEV")
            f.write(f"{row['seq_len']} & {method_short} & {row['xi']:.3f} & "
                    f"{emp}~ms & {pw}~ms & {ci} & $<10^{{-6}}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\n  LaTeX table: {tex_path}")

    # ── Figure: method map + pWCET comparison ─────────────────────────────────
    full_rows = [r for r in rows if r["layer"] == "full"]
    seqs = [r["seq_len"] for r in full_rows]
    pws  = [r["pwcet_1e6_ms"] or 0 for r in full_rows]
    methods = [r["method"] for r in full_rows]
    method_colors = {
        "gumbel_spaced": "tab:green",
        "gev_spaced": "tab:orange",
        "block_maxima_gev": "tab:blue",
    }
    bar_colors = [method_colors.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    bars = ax.bar(range(len(full_rows)), pws, color=bar_colors, alpha=0.85,
                  edgecolor="black", lw=0.5)
    ax.axhline(45, ls="--", color="red", lw=1, label="D=45ms SLO")
    ax.set_xticks(range(len(full_rows)))
    ax.set_xticklabels([f"seq={s}" for s in seqs], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("pWCET($10^{-6}$) ms")
    ax.set_title("Final pWCET — Full Pass (validated EVT method)")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="tab:green", label="Gumbel (spaced)"),
        Patch(color="tab:orange", label="GEV (spaced, POT)"),
        Patch(color="tab:blue", label="Block-maxima GEV"),
    ]
    ax.legend(handles=legend_elements, fontsize=6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e05_final_pwcet.png", dpi=150)
    plt.close()

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "e05_final_pwcet_report",
        "exceedance_prob": EXCEEDANCE_PROB,
        "rows": rows,
        "methodology_summary": {
            "gumbel_spaced_cells":   sum(1 for r in rows if r["method"] == "gumbel_spaced"),
            "gev_spaced_cells":      sum(1 for r in rows if r["method"] == "gev_spaced"),
            "block_maxima_cells":    sum(1 for r in rows if r["method"] == "block_maxima_gev"),
        },
    }
    write_results(output, RESULTS_DIR / "e05_final_pwcet.json")
    print("\nPASS: E05 complete\n")


if __name__ == "__main__":
    main()
