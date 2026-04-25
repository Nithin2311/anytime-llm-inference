"""
admission_control.py — Admission control for real-time LLM inference.

A real-time system must REJECT requests it cannot serve within deadline before
they start executing. This script formalises the admission test:

  ADMIT iff WCET_safe(seq_len) ≤ deadline_ms
  where WCET_safe = empirical_max × 1.10

Produces:
  1. Admissible-region analysis across (seq_len × deadline) pairs.
  2. The exact admissibility boundary: max seq_len admissible at each deadline.
  3. A simulation of 200 random requests to show admit/reject rates in practice.

Requires: wcet_results.json  (from profile_wcet.py)

Outputs:
  admission_control_results.json
  admission_control.png
"""

import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import fig_style as fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE   = "admission_control_results.json"
FIGURE_FILE    = "admission_control.png"
SAFETY_FACTOR  = 1.10
DEADLINE_MS    = 45.0
SIM_N_REQUESTS = 200
random.seed(42)
np.random.seed(42)


# ── Core admission logic ───────────────────────────────────────────────────────

def load_wcet_table(wcet_file="wcet_results.json"):
    """Load (seq_len → WCET_safe) mapping for the Full(22) pass."""
    with open(wcet_file) as f:
        data = json.load(f)
    table = {}
    for seq_len_str, cells in data["results"].items():
        if "None" in cells:
            table[int(seq_len_str)] = round(cells["None"]["wcet_ms"] * SAFETY_FACTOR, 3)
    return dict(sorted(table.items()))


def wcet_safe(seq_len: int, table: dict) -> float:
    """
    Ceiling lookup: return WCET_safe for the smallest profiled bin ≥ seq_len.
    If seq_len exceeds all profiled lengths, use the maximum.
    """
    for profiled_len, wcet in table.items():
        if seq_len <= profiled_len:
            return wcet
    return max(table.values())


def admit(seq_len: int, deadline: float, table: dict) -> bool:
    return wcet_safe(seq_len, table) <= deadline


# ── Admissible region ──────────────────────────────────────────────────────────

def compute_admissible_region(table, deadlines, seq_lengths):
    """
    Returns a 2-D boolean matrix: region[i][j] = admit(seq_lengths[j], deadlines[i])
    """
    return np.array(
        [[admit(sl, d, table) for sl in seq_lengths] for d in deadlines],
        dtype=bool,
    )


def max_admissible_seq_len(table, deadlines):
    """For each deadline, the largest seq_len that is admissible."""
    profiled = sorted(table.keys())
    results  = []
    for d in deadlines:
        max_sl = 0
        for sl in profiled:
            if table[sl] <= d:
                max_sl = sl
        results.append(max_sl)
    return results


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_requests(table, n=SIM_N_REQUESTS):
    """
    Simulate n arriving requests.  Each request has a randomly drawn:
      - seq_len   : Uniform(50, 800) tokens (realistic prompt + history range)
      - deadline  : Uniform(25, 65) ms

    Returns list of (seq_len, deadline, admitted) tuples.
    """
    events = []
    for _ in range(n):
        sl = random.randint(50, 800)
        d  = random.uniform(25.0, 65.0)
        events.append((sl, round(d, 1), admit(sl, d, table)))
    return events


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_admission(table, sim_events):
    fs.apply()

    fig, axes = plt.subplots(1, 3, figsize=fs.TRIPLE)

    deadlines_fine   = np.linspace(18, 70, 300)
    seq_lengths_fine = np.arange(20, 1050, 10)

    region = compute_admissible_region(table, deadlines_fine, seq_lengths_fine)

    # ── Panel 1: Admissible region heatmap ────────────────────────────────────
    ax1 = axes[0]
    # imshow expects [rows=y, cols=x]; rows = seq_len (y-axis), cols = deadline (x-axis)
    im = ax1.imshow(
        region.T,                          # transpose: rows=seq_len, cols=deadline
        origin="lower",
        extent=[deadlines_fine[0], deadlines_fine[-1],
                seq_lengths_fine[0], seq_lengths_fine[-1]],
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=1,
    )
    ax1.axvline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.5,
                label=f"D = {DEADLINE_MS:.0f} ms")
    ax1.set_xlabel("Deadline (ms)")
    ax1.set_ylabel("Input Sequence Length (tokens)")
    ax1.set_title("Admissible Region\n(green = ADMIT, red = REJECT)")
    ax1.legend(loc="upper left", fontsize=8)
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Admitted")

    # ── Panel 2: Boundary curve — max admissible seq_len vs deadline ──────────
    ax2 = axes[1]
    boundary = max_admissible_seq_len(table, deadlines_fine)
    ax2.plot(deadlines_fine, boundary, linewidth=2.5, color="#2ca02c", label="Max admissible seq len")
    ax2.fill_between(deadlines_fine, 0, boundary, alpha=0.15, color="#2ca02c")
    ax2.axvline(DEADLINE_MS, color="black", linestyle="--", linewidth=1.2,
                label=f"D = {DEADLINE_MS:.0f} ms")

    # Annotate profiled WCET points
    for sl, w in table.items():
        ax2.scatter([w], [sl], marker="o", s=55, color="#d62728", zorder=5)
        ax2.annotate(f"seq={sl}", xy=(w, sl), xytext=(w + 0.8, sl + 20),
                     fontsize=7.5, color="#d62728")

    ax2.set_xlabel("Deadline (ms)")
    ax2.set_ylabel("Max Admissible Sequence Length (tokens)")
    ax2.set_title("Admissibility Boundary\n(WCET_safe × 1.10 = deadline)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 1100)

    # ── Panel 3: Simulation scatter — admit vs reject ─────────────────────────
    ax3 = axes[2]
    admit_pts  = [(d, sl) for sl, d, a in sim_events if a]
    reject_pts = [(d, sl) for sl, d, a in sim_events if not a]
    n_admit  = len(admit_pts)
    n_reject = len(reject_pts)

    if admit_pts:
        ax3.scatter(*zip(*admit_pts),  s=18, alpha=0.65, color="#2ca02c",
                    label=f"ADMIT ({n_admit})", zorder=3)
    if reject_pts:
        ax3.scatter(*zip(*reject_pts), s=18, alpha=0.65, color="#d62728",
                    label=f"REJECT ({n_reject})", zorder=3)

    # Boundary line overlay
    ax3.plot(deadlines_fine, boundary, linewidth=1.8, color="black",
             linestyle="--", label="Admissibility boundary", zorder=4)

    ax3.set_xlabel("Deadline (ms)")
    ax3.set_ylabel("Sequence Length (tokens)")
    ax3.set_title(f"Admission Control Simulation\n(n={SIM_N_REQUESTS} random requests)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_ylim(0, 850)

    fig.suptitle("Admission Control Analysis — TinyLlama-1.1B (RTX 4000 Ada, WCET_safe = max × 1.10)",
                 fontsize=7.5, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{FIGURE_FILE}'")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ADMISSION CONTROL ANALYSIS — TinyLlama-1.1B")
    print(f"Safety factor: ×{SAFETY_FACTOR}  |  Reference deadline: {DEADLINE_MS} ms")
    print("=" * 60 + "\n")

    table = load_wcet_table()

    print("WCET_safe table (Full pass, ×1.10):")
    print(f"  {'Seq Len':>8}  {'WCET (ms)':>10}  {'Admissible @ 45ms':>18}")
    print("  " + "-" * 42)
    for sl, w in table.items():
        status = "YES" if w <= DEADLINE_MS else "NO"
        print(f"  {sl:>8}  {w:>10.3f}  {status:>18}")

    print()

    # Admissibility at reference deadline
    print(f"Admissibility at D = {DEADLINE_MS} ms:")
    for sl in sorted(table):
        ok = admit(sl, DEADLINE_MS, table)
        print(f"  seq={sl:>4} → WCET_safe={table[sl]:.2f} ms → {'ADMIT' if ok else 'REJECT'}")

    print()

    # Boundary: max admissible seq_len at key deadlines
    print("Max admissible seq_len at key deadlines:")
    key_deadlines = [25, 30, 35, 40, 45, 50, 55, 60]
    print(f"  {'Deadline':>10}  {'Max seq_len':>12}  {'WCET_safe':>10}")
    print("  " + "-" * 36)
    for d in key_deadlines:
        max_sl = 0
        best_w = 0.0
        for sl, w in sorted(table.items()):
            if w <= d:
                max_sl = sl
                best_w = w
        print(f"  {d:>10.0f}  {max_sl:>12}  {best_w:>10.3f}")

    print()

    # Simulation
    sim_events = simulate_requests(table)
    n_admit  = sum(1 for _, _, a in sim_events if a)
    n_reject = sum(1 for _, _, a in sim_events if not a)
    print(f"Simulation ({SIM_N_REQUESTS} random requests):")
    print(f"  ADMIT : {n_admit}  ({100*n_admit/SIM_N_REQUESTS:.1f}%)")
    print(f"  REJECT: {n_reject}  ({100*n_reject/SIM_N_REQUESTS:.1f}%)")
    print()

    # Save
    output = {
        "safety_factor":      SAFETY_FACTOR,
        "reference_deadline": DEADLINE_MS,
        "wcet_safe_table":    {str(k): v for k, v in table.items()},
        "simulation": {
            "n_requests":    SIM_N_REQUESTS,
            "n_admit":       n_admit,
            "n_reject":      n_reject,
            "admit_rate_pct": round(100 * n_admit / SIM_N_REQUESTS, 2),
            "events": [
                {"seq_len": sl, "deadline_ms": d, "admitted": a}
                for sl, d, a in sim_events
            ],
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to '{RESULTS_FILE}'")

    plot_admission(table, sim_events)
