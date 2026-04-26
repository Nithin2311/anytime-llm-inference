# Sprint V3 — IID + GEV Resolution Summary

**Completed:** Sun Apr 26 21:08:50 UTC 2026
**Total time:** ~42 min wall clock (E00 dominated at 1348s = 22.5 min)
**GPU:** NVIDIA A100-SXM4-80GB, 81920 MiB
**Results:** PASS=6  FAIL=0  (E02 initially failed due to filename mismatch in
run_all.sh — fixed in this commit and re-ran successfully)

## Experiment Results

| #   | Experiment                | Status | Resolves                                |
|-----|---------------------------|--------|-----------------------------------------|
| E00 | e00_spaced_profiling      | PASS   | Protocol: 200ms sleep, 50 warm-up       |
| E01 | e01_iid_validation        | PASS   | IID: Ljung-Box on spaced data           |
| E02 | e02_outlier_warmup_study  | PASS   | GEV: warm-up artifact diagnosis         |
| E03 | e03_gev_xi_refit          | PASS   | GEV: xi refit after spacing             |
| E04 | e04_block_maxima_pwcet    | PASS   | GEV: block maxima fallback              |
| E05 | e05_final_pwcet_report    | PASS   | Final: decision-driven pWCET table      |

## Timing

- e00_spaced_profiling:    1348s
- e01_iid_validation:         2s
- e02_outlier_warmup_study:  32s   (after filename fix)
- e03_gev_xi_refit:          65s
- e04_block_maxima_pwcet:   760s
- e05_final_pwcet_report:     2s

## Headline Findings

1. **IID test still FAILS even with 200ms inter-run spacing.**
   12/12 cells reject the white-noise null at lag >= 5.
   Reference without spacing: 12/12 FAIL. 200ms spacing is **not** enough.

2. **Heavy tails are intrinsic, not warm-up artifacts.**
   E02: GEV xi on warm-only runs = 1.71 (cold-included = 1.89,
   extended-warmup = 1.60). Reference without spacing was xi = 1.36.
   Anderson-Darling rejects Gumbel in every variant.

3. **POT-GEV xi on spaced data is large and worse than the unspaced reference.**
   12/12 cells reject Gumbel; xi ranges 0.21 to 1.18.

4. **Block-maxima EVT is the only methodology that yields any valid Gumbel fits.**
   At b=20 or b=25, several cells (notably seq128_l16, seq512_l16,
   seq1024_full) produce well-behaved fits with xi ~ 0; many do not.

5. **E05 selected `block_maxima_gev` for ALL 12 cells**, but no cell
   satisfies *both* IID-on-block-maxima AND Gumbel acceptance — so the
   reported pWCETs use the best block size per cell and many CIs are
   extremely wide (e.g. seq256_full: 47100 ms with CI 2212-640819 ms).

## Implications for the Paper

The paper's Gumbel pWCET claim (P(TPOT > 45 ms) < 1e-6 = 18 ms) is
**unsupported by these data**. The block-maxima fallback gives wildly
varying numbers per cell (28 ms to 187201 ms), and the underlying
samples remain non-IID even with spacing. Recommended next steps:

- Larger inter-run spacing (1 s, 5 s) to break thermal/cache coupling.
- Larger sample sizes per cell (e.g. 5000 runs) so block-maxima can use
  larger blocks with usable n.
- Explicitly randomized cell order across the whole campaign rather than
  within a single python process.
- Consider non-EVT bounds (e.g. measured P99 + safety factor) as a more
  defensible WCET surrogate given the data quality.

## Output Files

```
results/e00_spaced_profiling.json
results/e00_spaced_timeseries.png
results/e01_acf_spaced.png
results/e01_iid_validation.json
results/e02_outlier_warmup.json
results/e02_warmup_study.png
results/e03_gev_xi_refit.json
results/e03_gev_xi_refit.png
results/e04_block_maxima.json
results/e04_block_maxima.png
results/e05_final_pwcet.json
results/e05_final_pwcet.png
results/table_final_pwcet.tex
```
