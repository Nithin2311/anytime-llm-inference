# sprint_runpod — Anytime LLM Inference: Academic Review Sprint

Self-contained package for running 8 experiments (E0–E7) on a RunPod A100 SXM4
instance to address all peer-review recommendations before resubmission.

The entire pipeline runs unattended inside a tmux session and survives SSH
disconnects. Total wall-clock time: ~10–12 hours.

---

## Hardware Recommendation

| Component | Recommended | Notes |
|-----------|-------------|-------|
| GPU | A100 SXM4 80GB | Required for unified hardware baseline |
| CPU | 16 vCPU (standard bundle) | Any RunPod default works |
| RAM | 60 GB | Model + dataset fit comfortably |
| Disk | 200 GB | HuggingFace model cache + results |

> RunPod pod type: **A100 SXM4** — select "Secure Cloud" for SXM4 availability.
> Standard "Community Cloud" pods typically offer PCIe A100; SXM4 has higher
> memory bandwidth and is preferred for consistent WCET measurements.

---

## Deploy Steps

### 1. Upload this folder to the pod

From your local machine:
```bash
scp -r sprint_runpod/ root@<pod-ip>:/workspace/
```

### 2. Validate environment (run once after pod startup)
```bash
cd /workspace/sprint_runpod
bash setup.sh
```
This will:
- Confirm CUDA and GPU name
- `pip install -r requirements.txt`
- Pre-download TinyLlama weights from HuggingFace
- Dry-run every experiment to check imports

### 3. Launch the sprint inside tmux
```bash
bash tmux_launch.sh
```
Creates session `sprint` with 3 windows:
- `[0] orchestrator` — E0 through E7 running sequentially
- `[1] gpu-watch` — live `nvidia-smi` every 2 seconds
- `[2] log-tail` — live tail of `results/sprint.log`

Detach at any time with `Ctrl-B D`. The sprint keeps running.

### 4. Re-attach after WiFi disconnect
```bash
bash monitor.sh
```

### 5. Check progress without attaching
```bash
tail -50 results/sprint.log
cat results/SPRINT_SUMMARY.md
```

### 6. Resume after a failure
```bash
bash run_sprint.sh --resume   # skips experiments with existing result files
```

### 7. Retrieve results (run on local machine)
```bash
rsync -avz --progress \
  root@<pod-ip>:/workspace/sprint_runpod/results/ ./sprint_runpod/results/
rsync -avz --progress \
  root@<pod-ip>:/workspace/sprint_runpod/figures/ ./sprint_runpod/figures/
rsync -avz --progress \
  root@<pod-ip>:/workspace/sprint_runpod/latex/   ./sprint_runpod/latex/
```

---

## Directory Layout

```
sprint_runpod/
├── README.md               # this file
├── SPRINT_PLAN.md          # day-by-day schedule and acceptance gates
├── requirements.txt        # Python dependencies
├── setup.sh                # one-shot environment validation
├── tmux_launch.sh          # create tmux session and start sprint
├── run_sprint.sh           # master orchestrator (E0->E7)
├── monitor.sh              # re-attach to running session
├── src/
│   ├── early_exit_model.py     # EarlyExitTinyLlama
│   ├── dynamic_scheduler.py    # stateless / KV-cached / async routers
│   ├── benchmark_utils.py      # PubMedQA helpers, bootstrap CI
│   ├── result_writer.py        # atomic JSON writes, logging, LaTeX output
│   └── fig_style.py            # IEEE figure size constants
├── experiments/
│   ├── e0_wcet_profile.py          # WCET re-profile, 50 runs/cell
│   ├── e1_evt_reprofiling.py       # EVT WCET, 500 runs/cell, Gumbel_r
│   ├── e2_threshold_ablation.py    # tau sweep {0.5..0.9}
│   ├── e3_forced_exit_quality.py   # ROUGE-L + accuracy at D=20/25ms
│   ├── e4_pot_sensitivity.py       # POT fraction sensitivity {10..25%}
│   ├── e5_deadline_sweep_ext.py    # deadline sweep with KV-cached router
│   ├── e6_accuracy_ci.py           # bootstrap 95% CI on accuracy
│   └── e7_wcet_ci.py               # parametric bootstrap CI on pWCET(1e-6)
├── results/                # JSON outputs + sprint.log + SPRINT_SUMMARY.md
├── figures/                # PNG figures (300 DPI, IEEE two-column)
└── latex/                  # .tex table snippets for report_v2.tex
```

---

## Expected Outputs

After a successful run, `results/SPRINT_SUMMARY.md` shows all 8 experiments as
**PASS**. The `latex/` directory contains 8 `.tex` snippets ready to paste into
`report_v2.tex`:

| File | Replaces / Extends |
|------|--------------------|
| `table_ii_wcet.tex` | Table II body rows |
| `table_xi_evt.tex` | Table XI body rows |
| `table_threshold_ablation.tex` | New table (Section IV-C) |
| `table_forced_exit_quality.tex` | Extends Table VI |
| `table_pot_sensitivity.tex` | New table (Appendix) |
| `table_deadline_sweep_ext.tex` | Companion to Table VI |
| `table_accuracy_ci.tex` | New table (Section IV-B) |
| `table_wcet_ci.tex` | New table (Section III-D) |
