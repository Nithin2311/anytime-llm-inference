"""
evt_utils_v3.py — EVT toolkit: GEV fitting, block maxima, AD test, Ljung-Box.

GEV convention: scipy.stats.genextreme uses shape c = -xi.
  c < 0  →  xi > 0  →  Frechet (heavy tail)
  c = 0  →  xi = 0  →  Gumbel  (light tail)
  c > 0  →  xi < 0  →  Weibull (bounded)
"""

import numpy as np
from scipy.stats import genextreme, gumbel_r, anderson
from statsmodels.stats.diagnostic import acorr_ljungbox


# ── POT helpers ──────────────────────────────────────────────────────────────

def pot_tail(data: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    """Return the top `fraction` of data as the POT tail sample."""
    n_tail = max(10, int(len(data) * fraction))
    return np.sort(data)[-n_tail:]


# ── GEV fitting ──────────────────────────────────────────────────────────────

def fit_gev(tail: np.ndarray) -> dict:
    """
    Fit GEV to tail sample via MLE.

    Returns dict with:
      xi      : shape parameter (positive = Frechet / heavy tail)
      mu      : location
      sigma   : scale
      c_scipy : scipy shape (c = -xi)
    """
    c, loc, scale = genextreme.fit(tail, method="MLE")
    xi = -c  # convert scipy sign convention
    return {"xi": float(xi), "mu": float(loc), "sigma": float(scale), "c_scipy": float(c)}


def gev_pwcet(gev: dict, exceedance_prob: float) -> float:
    """Compute pWCET quantile from GEV parameters at given exceedance probability."""
    c = gev["c_scipy"]
    loc = gev["mu"]
    scale = gev["sigma"]
    return float(genextreme.ppf(1.0 - exceedance_prob, c, loc=loc, scale=scale))


def gumbel_pwcet(tail: np.ndarray, exceedance_prob: float) -> float:
    """Fit Gumbel to tail and return pWCET quantile."""
    loc, scale = gumbel_r.fit(tail, method="MLE")
    return float(gumbel_r.ppf(1.0 - exceedance_prob, loc=loc, scale=scale))


def bootstrap_gev_pwcet_ci(
    tail: np.ndarray,
    exceedance_prob: float = 1e-6,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Bootstrap 95% CI on GEV pWCET(exceedance_prob)."""
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(n_bootstrap):
        resample = rng.choice(tail, size=len(tail), replace=True)
        try:
            gev = fit_gev(resample)
            est = gev_pwcet(gev, exceedance_prob)
            if np.isfinite(est) and est < 1e6:
                estimates.append(est)
        except Exception:
            pass

    if len(estimates) < 50:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "n_valid": len(estimates)}

    alpha = (1 - ci) / 2
    lo = float(np.percentile(estimates, 100 * alpha))
    hi = float(np.percentile(estimates, 100 * (1 - alpha)))
    return {"ci_lower": lo, "ci_upper": hi, "n_valid": len(estimates)}


# ── Anderson-Darling for Gumbel ───────────────────────────────────────────────

def anderson_darling_gumbel(data: np.ndarray) -> dict:
    """
    Anderson-Darling goodness-of-fit test for Gumbel (extreme value) distribution.

    Returns stat, critical_value_5pct, and gumbel_rejected (bool).
    """
    result = anderson(data, dist="gumbel_r")
    # critical values at [15%, 10%, 5%, 2.5%, 1%]
    # index 2 → 5% significance level
    crit_5pct = float(result.critical_values[2])
    stat = float(result.statistic)
    return {
        "statistic": stat,
        "critical_value_5pct": crit_5pct,
        "gumbel_rejected": bool(stat > crit_5pct),
        "significance_levels": list(result.significance_level),
        "critical_values": list(result.critical_values),
    }


# ── Ljung-Box IID test ────────────────────────────────────────────────────────

def ljungbox_iid(data: np.ndarray, lags: list = None) -> dict:
    """
    Ljung-Box test for serial autocorrelation (IID assumption check).

    H0: data is uncorrelated (IID).
    Reject H0 (IID violated) if p < 0.05.

    Returns dict with per-lag results and overall iid_pass (bool).
    """
    if lags is None:
        lags = [1, 5, 10, 20]

    result = acorr_ljungbox(data, lags=lags, return_df=True)
    pvalues = result["lb_pvalue"].tolist()
    stats = result["lb_stat"].tolist()

    lag_results = {}
    for lag, stat, pval in zip(lags, stats, pvalues):
        lag_results[f"lag_{lag}"] = {
            "statistic": float(stat),
            "pvalue": float(pval),
            "iid_pass": bool(pval > 0.05),
        }

    iid_pass_all = all(v["iid_pass"] for v in lag_results.values())
    min_pvalue = float(min(pvalues))

    return {
        "lags": lag_results,
        "iid_pass": iid_pass_all,
        "min_pvalue": min_pvalue,
    }


# ── Block maxima ──────────────────────────────────────────────────────────────

def block_maxima(data: np.ndarray, block_size: int) -> np.ndarray:
    """
    Extract block maxima from data with given block size.
    Truncates trailing samples that don't form a complete block.
    """
    n_blocks = len(data) // block_size
    blocks = np.array(data[: n_blocks * block_size]).reshape(n_blocks, block_size)
    return blocks.max(axis=1)


def analyze_block_maxima(
    data: np.ndarray,
    block_size: int,
    exceedance_prob: float = 1e-6,
    n_bootstrap: int = 500,
) -> dict:
    """
    Full block maxima EVT analysis:
    1. Extract block maxima
    2. Ljung-Box IID check on maxima
    3. GEV fit + xi
    4. AD test for Gumbel on maxima
    5. pWCET with bootstrap CI
    """
    maxima = block_maxima(data, block_size)
    n_blocks = len(maxima)

    iid = ljungbox_iid(maxima, lags=[1, 5, min(10, n_blocks // 4)])
    gev = fit_gev(maxima)
    ad = anderson_darling_gumbel(maxima)
    pwcet = gev_pwcet(gev, exceedance_prob)
    ci = bootstrap_gev_pwcet_ci(maxima, exceedance_prob, n_bootstrap)

    gumbel_pwcet_val = gumbel_pwcet(maxima, exceedance_prob)

    return {
        "block_size": block_size,
        "n_blocks": n_blocks,
        "maxima_mean": float(maxima.mean()),
        "maxima_p99": float(np.percentile(maxima, 99)),
        "maxima_max": float(maxima.max()),
        "iid": iid,
        "gev": gev,
        "ad": ad,
        "pwcet_gev": float(pwcet) if np.isfinite(pwcet) and pwcet < 1e6 else None,
        "pwcet_gumbel": float(gumbel_pwcet_val),
        "ci": ci,
    }


# ── Full cell analysis ────────────────────────────────────────────────────────

def analyze_cell(
    data: np.ndarray,
    pot_fraction: float = 0.20,
    exceedance_prob: float = 1e-6,
    n_bootstrap: int = 1000,
    block_sizes: list = None,
) -> dict:
    """
    Complete EVT analysis for one (seq_len, layer) cell:
    - POT tail extraction
    - GEV fit + xi
    - Gumbel AD test
    - Ljung-Box on raw data
    - Block maxima analysis at multiple block sizes
    - pWCET estimates (Gumbel and GEV)
    """
    if block_sizes is None:
        block_sizes = [5, 10, 20]

    tail = pot_tail(data, pot_fraction)
    gev = fit_gev(tail)
    ad = anderson_darling_gumbel(tail)
    iid = ljungbox_iid(data)
    pwcet_gumbel = gumbel_pwcet(tail, exceedance_prob)
    pwcet_gev = gev_pwcet(gev, exceedance_prob)
    ci_gev = bootstrap_gev_pwcet_ci(tail, exceedance_prob, n_bootstrap)

    bm_results = {}
    for b in block_sizes:
        if len(data) // b >= 20:
            bm_results[f"b{b}"] = analyze_block_maxima(
                data, b, exceedance_prob, n_bootstrap // 2
            )

    return {
        "n_samples": len(data),
        "mean_ms": float(data.mean()),
        "p99_ms": float(np.percentile(data, 99)),
        "max_ms": float(data.max()),
        "empirical_wcet_1_10": float(data.max() * 1.10),
        "pot_fraction": pot_fraction,
        "n_tail": len(tail),
        "gev": gev,
        "ad_gumbel": ad,
        "iid_ljungbox": iid,
        "pwcet_gumbel_ms": float(pwcet_gumbel),
        "pwcet_gev_ms": float(pwcet_gev) if np.isfinite(pwcet_gev) and pwcet_gev < 1e9 else None,
        "pwcet_gev_ci": ci_gev,
        "block_maxima": bm_results,
        "gumbel_valid": not ad["gumbel_rejected"],
        "iid_pass": iid["iid_pass"],
    }
