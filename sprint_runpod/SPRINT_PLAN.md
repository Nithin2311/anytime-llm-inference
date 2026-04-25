# 5-Day Academic Sprint Plan
**Hardware:** RunPod A100 SXM4 80GB  
**Goal:** Address all academic reviewer recommendations before resubmission  
**Branch:** main → results committed after sprint

---

## Experiment Map

| ID | Name | GPU Required | Est. Runtime | Output Files |
|----|------|-------------|--------------|--------------|
| E0 | WCET Re-profile (50 runs/cell) | Yes | 15–20 min | `wcet_results.json`, `wcet_profile.png`, `table_ii_wcet.tex` |
| E1 | EVT Re-profile (500 runs/cell, A100) | Yes | 2–3 h | `evt_wcet_results.json`, `evt_wcet_analysis.png`, `table_xi_evt.tex` |
| E2 | Threshold Ablation (tau sweep) | Yes | 45 min | `threshold_ablation_results.json`, `threshold_ablation.png`, `table_threshold_ablation.tex` |
| E3 | Forced-Exit Quality (ROUGE-L + Acc) | Yes | 40 min | `forced_exit_quality_results.json`, `forced_exit_quality.png`, `table_forced_exit_quality.tex` |
| E4 | POT Sensitivity (10–25%) | Yes | 2–3 h | `pot_sensitivity_results.json`, `pot_sensitivity.png`, `table_pot_sensitivity.tex` |
| E5 | Deadline Sweep Ext (KV router) | Yes | 60 min | `deadline_sweep_ext_results.json`, `deadline_sweep_ext.png`, `table_deadline_sweep_ext.tex` |
| E6 | Accuracy Bootstrap CI | Yes | 30 min | `accuracy_ci_results.json`, `accuracy_ci.png`, `table_accuracy_ci.tex` |
| E7 | WCET Parametric Bootstrap CI | Yes | 3–4 h | `wcet_ci_results.json`, `wcet_ci.png`, `table_wcet_ci.tex` |

**Total estimated GPU time:** ~10–12 hours  
**Total wall-clock (sequential):** ~10–12 hours on A100 SXM4

---

## Day-by-Day Schedule

### Day 1 — Environment + Baselines (E0, E1)
- [ ] Pod provisioned: A100 SXM4 80GB, 16 vCPU, 60 GB RAM, 200 GB disk
- [ ] `bash setup.sh` — validate CUDA, install deps, dry-run all experiments
- [ ] `bash tmux_launch.sh` — start sprint session
- [ ] E0 completes (~20 min) — new A100 WCET table written
- [ ] E1 completes (~3 h) — unified-hardware EVT results

**Acceptance gates:**
- `results/wcet_results.json` exists, `"hardware": "A100 SXM4"`
- `results/evt_wcet_results.json` exists; pWCET(1e-6) < 45 ms for seq <= 256

### Day 2 — Router Quality (E2, E3)
- [ ] E2 completes — threshold ablation shows tau=0.7 optimal
- [ ] E3 completes — ROUGE-L >= 0.35 vs full model at D=20ms

**Acceptance gates:**
- `results/threshold_ablation_results.json` has all 5 tau entries
- `results/forced_exit_quality_results.json` has both ROUGE-L fields populated

### Day 3 — EVT Robustness + Extended Sweep (E4, E5)
- [ ] E4 completes — POT sensitivity shows <2 ms spread across 10–25%
- [ ] E5 completes — KV router deadline sweep shows 0% forced exit at D >= 35ms

**Acceptance gates:**
- `results/pot_sensitivity_results.json` has entries for all 4 POT fractions
- `results/deadline_sweep_ext_results.json` has all 10 deadline entries

### Day 4 — Confidence Intervals (E6, E7)
- [ ] E6 completes — bootstrap CI width <30pp (sufficiently tight for n=30)
- [ ] E7 completes — pWCET CI upper bound still <45 ms for seq <= 256

**Acceptance gates:**
- `results/accuracy_ci_results.json` has `ci_lower` and `ci_upper` for both routers
- `results/wcet_ci_results.json` has CI fields for all 24 cells

### Day 5 — Results Harvest + LaTeX Integration
- [ ] All 8 `latex/*.tex` snippets generated
- [ ] Rsync results back to local machine
- [ ] Integrate LaTeX snippets into `report_v2.tex`
- [ ] Verify figures render at 300 DPI in IEEE two-column format
- [ ] Commit and push results branch

**Rsync command (run on local machine):**
```bash
rsync -avz --progress \
  root@<pod-ip>:/workspace/sprint_runpod/results/ ./sprint_runpod/results/
rsync -avz --progress \
  root@<pod-ip>:/workspace/sprint_runpod/figures/ ./sprint_runpod/figures/
rsync -avz --progress \
  root@<pod-ip>:/workspace/sprint_runpod/latex/   ./sprint_runpod/latex/
```

---

## Reviewer → Experiment Cross-Reference

| Reviewer Comment | Addressed By | LaTeX File |
|-----------------|-------------|-----------|
| Hardware mismatch (EVT used RTX 4000 Ada, not A100) | E1 | `table_xi_evt.tex` |
| No justification for 20% POT fraction choice | E4 | `table_pot_sensitivity.tex` |
| No confidence intervals on accuracy | E6 | `table_accuracy_ci.tex` |
| No CI on pWCET bounds | E7 | `table_wcet_ci.tex` |
| No quality metric for forced-exit outputs | E3 | `table_forced_exit_quality.tex` |
| Threshold tau=0.7 not ablated | E2 | `table_threshold_ablation.tex` |
| Deadline sweep only covers stateless router | E5 | `table_deadline_sweep_ext.tex` |
| WCET table not on submission hardware | E0 | `table_ii_wcet.tex` |

---

## Recovery Procedures

### SSH Disconnect
```bash
bash monitor.sh        # re-attach; sprint keeps running in tmux
```

### Experiment Failed Mid-Run
The orchestrator logs FAIL but continues with remaining experiments. To re-run after fixing:
```bash
bash run_sprint.sh --resume   # skips experiments with existing result files
```

### OOM / CUDA Error
1. Check `results/sprint.log` for the offending experiment
2. Re-run the single failing experiment: `python experiments/eX_name.py`

### Partial Results
Each experiment is idempotent — if `results/eX_*.json` exists it is skipped.
Delete the file to force a re-run:
```bash
rm results/wcet_results.json   # example: re-run E0
```
