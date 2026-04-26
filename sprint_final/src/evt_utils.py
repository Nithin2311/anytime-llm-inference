"""
evt_utils.py — EVT fitting utilities for probabilistic WCET analysis.

Provides GEV and Gumbel fits via POT, Anderson-Darling goodness-of-fit,
parametric bootstrap CI, Ljung-Box independence test.
Used by E01, E04, E07, E08.
"""

import numpy as np
from scipy.stats import gumbel_r, genextreme, anderson


def pot_tail(samples, fraction):
    """Return the top-fraction tail of samples (minimum 20 points)."""
    n_tail = max(int(fraction * len(samples)), 20)
    return np.sort(samples)[-n_tail:]


def fit_gumbel(samples, fraction=0.20):
    """Fit Gumbel_r (Type I EVT) to POT tail. Returns loc, scale, pWCET."""
    tail = pot_tail(samples, fraction)
    loc, scale = gumbel_r.fit(tail)
    return {
        "loc":      float(loc),
        "scale":    float(scale),
        "wcet_1e4": float(gumbel_r.ppf(1.0 - 1e-4, loc=loc, scale=scale)),
        "wcet_1e6": float(gumbel_r.ppf(1.0 - 1e-6, loc=loc, scale=scale)),
        "n_tail":   len(tail),
    }


def fit_gev(samples, fraction=0.20):
    """
    Fit Generalized Extreme Value distribution to POT tail.

    GEV shape ξ interpretation:
      ξ > +0.15  →  Fréchet  (heavy tail; Gumbel underestimates pWCET — anti-conservative)
      |ξ| ≤ 0.15 →  Gumbel   (Type I is justified)
      ξ < -0.15  →  Weibull  (bounded tail; Gumbel overestimates — conservative)
    """
    tail = pot_tail(samples, fraction)
    # scipy genextreme uses sign convention: c = -ξ; negate to get standard ξ
    c_fit, loc, scale = genextreme.fit(tail)
    xi = -c_fit  # convert to standard EVT notation
    return {
        "xi":           float(xi),
        "loc":          float(loc),
        "scale":        float(scale),
        "wcet_1e4":     float(genextreme.ppf(1.0 - 1e-4, c=c_fit, loc=loc, scale=scale)),
        "wcet_1e6":     float(genextreme.ppf(1.0 - 1e-6, c=c_fit, loc=loc, scale=scale)),
        "n_tail":       len(tail),
        "gumbel_valid": bool(abs(xi) < 0.15),
    }


def anderson_darling_gumbel(samples, fraction=0.20):
    """
    Anderson-Darling goodness-of-fit test for Gumbel distribution on POT tail.
    Returns AD statistic and critical value at 5% significance.
    fit_not_rejected_at_5pct=True means Gumbel cannot be rejected at α=0.05.
    """
    tail = pot_tail(samples, fraction)
    result = anderson(tail, dist="gumbel_r")
    # scipy returns critical_values at [15, 10, 5, 2.5, 1]% — index 2 is 5%
    crit_5pct = float(result.critical_values[2]) if len(result.critical_values) >= 3 else None
    return {
        "ad_stat":                  float(result.statistic),
        "ad_crit_5pct":             crit_5pct,
        "fit_not_rejected_at_5pct": (
            bool(result.statistic < crit_5pct) if crit_5pct is not None else None
        ),
        "n_tail": len(tail),
    }


def bootstrap_wcet_ci(samples, fraction=0.20, n_bootstrap=1000, rng_seed=42):
    """
    Parametric bootstrap 95% CI on pWCET(1e-6) via Gumbel POT fit.

    Each bootstrap replicate:
      1. Resample POT tail with replacement
      2. Fit Gumbel_r to resampled tail
      3. Compute pWCET(1e-6)
    Returns 95% CI (2.5th, 97.5th percentiles of bootstrap distribution).
    """
    rng = np.random.default_rng(rng_seed)
    tail = pot_tail(samples, fraction)
    boot_wcets = []
    for _ in range(n_bootstrap):
        resamp = rng.choice(tail, size=len(tail), replace=True)
        try:
            loc, scale = gumbel_r.fit(resamp)
            w = float(gumbel_r.ppf(1.0 - 1e-6, loc=loc, scale=scale))
            if np.isfinite(w):
                boot_wcets.append(w)
        except Exception:
            pass
    boot_wcets = np.array(boot_wcets)
    point = fit_gumbel(samples, fraction)["wcet_1e6"]
    return {
        "wcet_1e6_point": round(float(point), 4),
        "ci_lower":       round(float(np.percentile(boot_wcets, 2.5)), 4),
        "ci_upper":       round(float(np.percentile(boot_wcets, 97.5)), 4),
        "ci_width":       round(float(np.percentile(boot_wcets, 97.5)
                                      - np.percentile(boot_wcets, 2.5)), 4),
        "n_valid":        len(boot_wcets),
        "n_tail":         len(tail),
    }


def ljung_box_test(samples, max_lag=20):
    """
    Ljung-Box test for serial autocorrelation.
    H0: no autocorrelation in lags 1..max_lag.
    p > 0.05 → fail to reject H0 → samples are approximately i.i.d.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        df = acorr_ljungbox(samples, lags=max_lag, return_df=True)
        lb_stat   = float(df["lb_stat"].iloc[-1])
        lb_pvalue = float(df["lb_pvalue"].iloc[-1])
        return {
            "lb_stat":              round(lb_stat, 4),
            "lb_pvalue":            round(lb_pvalue, 6),
            "max_lag":              max_lag,
            "independent_at_5pct":  bool(lb_pvalue > 0.05),
        }
    except ImportError:
        return {
            "lb_stat": None, "lb_pvalue": None,
            "max_lag": max_lag, "independent_at_5pct": None,
            "note": "statsmodels not installed",
        }


def sample_acf(samples, n_lags=20):
    """Compute sample ACF at lags 1..n_lags. Returns list of (lag, acf_value) pairs."""
    n    = len(samples)
    mean = np.mean(samples)
    var  = np.var(samples)
    if var == 0:
        return [(i, 0.0) for i in range(1, n_lags + 1)]
    result = []
    for lag in range(1, n_lags + 1):
        cov = np.mean((samples[:n - lag] - mean) * (samples[lag:] - mean))
        result.append((lag, round(float(cov / var), 6)))
    return result


def bootstrap_accuracy_ci(correct_flags, n_bootstrap=2000, rng_seed=42):
    """
    Non-parametric bootstrap 95% CI on classification accuracy.
    correct_flags: list/array of booleans (True = correct prediction on scored query).
    """
    rng   = np.random.default_rng(rng_seed)
    flags = np.array(correct_flags, dtype=float)
    n     = len(flags)
    boot_accs = [
        float(np.mean(rng.choice(flags, size=n, replace=True))) * 100.0
        for _ in range(n_bootstrap)
    ]
    return {
        "accuracy":    round(float(np.mean(flags)) * 100.0, 1),
        "ci_lower":    round(float(np.percentile(boot_accs, 2.5)),  2),
        "ci_upper":    round(float(np.percentile(boot_accs, 97.5)), 2),
        "ci_width":    round(float(np.percentile(boot_accs, 97.5)
                                   - np.percentile(boot_accs, 2.5)), 2),
        "n":           n,
        "n_bootstrap": n_bootstrap,
    }
