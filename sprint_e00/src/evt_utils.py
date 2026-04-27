"""
evt_utils.py — EVT + empirical WCET utilities for sprint_camera_ready.

Extended from sprint_final with:
  - fit_gev_block_maxima()   GEV on block maxima (camera-ready EVT fallback)
  - nonevt_bounds()          Empirical P99/P99.9/P99.99 + safety-factor bounds
  - bootstrap_percentile_ci() Bootstrap CI on arbitrary percentile
"""

import numpy as np
from scipy.stats import gumbel_r, genextreme, anderson


# ── Original helpers (unchanged) ───────────────────────────────────────────

def pot_tail(samples, fraction):
    n_tail = max(int(fraction * len(samples)), 20)
    return np.sort(samples)[-n_tail:]


def fit_gumbel(samples, fraction=0.20):
    tail = pot_tail(samples, fraction)
    loc, scale = gumbel_r.fit(tail)
    return {
        "loc": float(loc), "scale": float(scale),
        "wcet_1e4": float(gumbel_r.ppf(1 - 1e-4, loc=loc, scale=scale)),
        "wcet_1e6": float(gumbel_r.ppf(1 - 1e-6, loc=loc, scale=scale)),
        "n_tail": len(tail),
    }


def fit_gev(samples, fraction=0.20):
    tail = pot_tail(samples, fraction)
    c_fit, loc, scale = genextreme.fit(tail)
    xi = -c_fit
    return {
        "xi": float(xi), "loc": float(loc), "scale": float(scale),
        "wcet_1e4": float(genextreme.ppf(1 - 1e-4, c=c_fit, loc=loc, scale=scale)),
        "wcet_1e6": float(genextreme.ppf(1 - 1e-6, c=c_fit, loc=loc, scale=scale)),
        "n_tail": len(tail), "gumbel_valid": bool(abs(xi) < 0.15),
    }


def anderson_darling_gumbel(samples, fraction=0.20):
    tail = pot_tail(samples, fraction)
    result = anderson(tail, dist="gumbel_r")
    crit5  = float(result.critical_values[2]) if len(result.critical_values) >= 3 else None
    return {
        "ad_stat": float(result.statistic), "ad_crit_5pct": crit5,
        "fit_not_rejected_at_5pct": bool(result.statistic < crit5) if crit5 else None,
        "n_tail": len(tail),
    }


def bootstrap_wcet_ci(samples, fraction=0.20, n_bootstrap=1000, rng_seed=42):
    rng  = np.random.default_rng(rng_seed)
    tail = pot_tail(samples, fraction)
    boot = []
    for _ in range(n_bootstrap):
        rs = rng.choice(tail, size=len(tail), replace=True)
        try:
            loc, scale = gumbel_r.fit(rs)
            w = float(gumbel_r.ppf(1 - 1e-6, loc=loc, scale=scale))
            if np.isfinite(w):
                boot.append(w)
        except Exception:
            pass
    boot = np.array(boot)
    pt   = fit_gumbel(samples, fraction)["wcet_1e6"]
    return {
        "wcet_1e6_point": round(float(pt), 4),
        "ci_lower":  round(float(np.percentile(boot, 2.5)), 4),
        "ci_upper":  round(float(np.percentile(boot, 97.5)), 4),
        "ci_width":  round(float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)), 4),
        "n_valid":   len(boot), "n_tail": len(tail),
    }


def ljung_box_test(samples, max_lag=20):
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        df = acorr_ljungbox(samples, lags=max_lag, return_df=True)
        lb_stat   = float(df["lb_stat"].iloc[-1])
        lb_pvalue = float(df["lb_pvalue"].iloc[-1])
        return {
            "lb_stat": round(lb_stat, 4), "lb_pvalue": round(lb_pvalue, 6),
            "max_lag": max_lag, "independent_at_5pct": bool(lb_pvalue > 0.05),
        }
    except ImportError:
        return {"lb_stat": None, "lb_pvalue": None, "max_lag": max_lag,
                "independent_at_5pct": None, "note": "statsmodels not installed"}


def sample_acf(samples, n_lags=20):
    n, mean, var = len(samples), np.mean(samples), np.var(samples)
    if var == 0:
        return [(i, 0.0) for i in range(1, n_lags + 1)]
    return [(lag, round(float(np.mean((samples[:n-lag]-mean)*(samples[lag:]-mean))/var), 6))
            for lag in range(1, n_lags + 1)]


def bootstrap_accuracy_ci(correct_flags, n_bootstrap=2000, rng_seed=42):
    rng   = np.random.default_rng(rng_seed)
    flags = np.array(correct_flags, dtype=float)
    n     = len(flags)
    boot  = [float(np.mean(rng.choice(flags, size=n, replace=True))) * 100.0
             for _ in range(n_bootstrap)]
    return {
        "accuracy":  round(float(np.mean(flags)) * 100.0, 1),
        "ci_lower":  round(float(np.percentile(boot, 2.5)), 2),
        "ci_upper":  round(float(np.percentile(boot, 97.5)), 2),
        "ci_width":  round(float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)), 2),
        "n": n, "n_bootstrap": n_bootstrap,
    }


# ── New: Block-maxima GEV ───────────────────────────────────────────────────

def fit_gev_block_maxima(samples, block_size):
    """
    Fit GEV to block maxima of `samples` with the given block size.

    Returns GEV parameters, pWCET estimates, Anderson-Darling result,
    and the number of usable blocks.  Requires at least 10 block maxima;
    returns None if block_size is too large.
    """
    samples = np.asarray(samples)
    n_blocks = len(samples) // block_size
    if n_blocks < 10:
        return None

    maxima = np.array([
        float(np.max(samples[i * block_size:(i + 1) * block_size]))
        for i in range(n_blocks)
    ])

    try:
        c_fit, loc, scale = genextreme.fit(maxima)
        xi = -c_fit
        wcet_1e6 = float(genextreme.ppf(1 - 1e-6, c=c_fit, loc=loc, scale=scale))
        wcet_1e4 = float(genextreme.ppf(1 - 1e-4, c=c_fit, loc=loc, scale=scale))

        ad_result = anderson(maxima, dist="gumbel_r")
        crit5 = float(ad_result.critical_values[2]) if len(ad_result.critical_values) >= 3 else None
        ad_pass = bool(ad_result.statistic < crit5) if crit5 else None
    except Exception as e:
        return {"error": str(e), "block_size": block_size, "n_blocks": n_blocks}

    # Bootstrap CI on pWCET(1e-6)
    rng = np.random.default_rng(42)
    boot_wcets = []
    for _ in range(500):
        rs = rng.choice(maxima, size=len(maxima), replace=True)
        try:
            cf, lc, sc = genextreme.fit(rs)
            w = float(genextreme.ppf(1 - 1e-6, c=cf, loc=lc, scale=sc))
            if np.isfinite(w):
                boot_wcets.append(w)
        except Exception:
            pass

    ci_lo = float(np.percentile(boot_wcets, 2.5)) if boot_wcets else None
    ci_hi = float(np.percentile(boot_wcets, 97.5)) if boot_wcets else None
    ci_w  = (ci_hi - ci_lo) if (ci_lo is not None and ci_hi is not None) else None

    return {
        "block_size":       block_size,
        "n_blocks":         n_blocks,
        "xi":               round(float(xi), 4),
        "loc":              round(float(loc), 4),
        "scale":            round(float(scale), 4),
        "wcet_1e6":         round(wcet_1e6, 4) if np.isfinite(wcet_1e6) else None,
        "wcet_1e4":         round(wcet_1e4, 4) if np.isfinite(wcet_1e4) else None,
        "ad_stat":          round(float(ad_result.statistic), 4),
        "ad_crit_5pct":     crit5,
        "gumbel_accepted":  ad_pass,
        "gumbel_valid":     bool(abs(xi) < 0.15),
        "ci_lower_1e6":     round(ci_lo, 4) if ci_lo is not None else None,
        "ci_upper_1e6":     round(ci_hi, 4) if ci_hi is not None else None,
        "ci_width_1e6":     round(ci_w, 4) if ci_w is not None else None,
    }


# ── New: Non-EVT empirical bounds ───────────────────────────────────────────

def bootstrap_percentile_ci(samples, percentile, n_bootstrap=2000, rng_seed=42):
    """Bootstrap 95% CI on a given percentile (e.g. 99.9)."""
    rng    = np.random.default_rng(rng_seed)
    n      = len(samples)
    point  = float(np.percentile(samples, percentile))
    boot   = [float(np.percentile(rng.choice(samples, size=n, replace=True), percentile))
              for _ in range(n_bootstrap)]
    return {
        "percentile":  percentile,
        "point":       round(point, 4),
        "ci_lower":    round(float(np.percentile(boot, 2.5)), 4),
        "ci_upper":    round(float(np.percentile(boot, 97.5)), 4),
        "ci_width":    round(float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)), 4),
        "n":           n,
        "n_bootstrap": n_bootstrap,
    }


def nonevt_bounds(samples, n_bootstrap=2000):
    """
    Non-EVT WCET bounds — defensible without EVT distributional assumptions.

    Method A: Empirical percentile bounds (P99, P99.9, P99.99) with
              bootstrap 95% CIs.
    Method B: Safety-factor bound = P99 + 3 * sigma  (Gaussian tail approx.)
    Method C: Hoeffding bound = max(samples) + epsilon for chosen epsilon.
              Distribution-free; uses range = max - min.
    """
    samples = np.asarray(samples, dtype=float)
    n       = len(samples)
    mu      = float(np.mean(samples))
    sigma   = float(np.std(samples))
    smin    = float(np.min(samples))
    smax    = float(np.max(samples))
    rng_val = smax - smin

    # Method A
    percentiles_A = {}
    for pct in [99.0, 99.9, 99.99]:
        percentiles_A[f"p{pct}"] = bootstrap_percentile_ci(samples, pct,
                                                             n_bootstrap=n_bootstrap)

    # Method B: P99 + 3*sigma
    p99     = float(np.percentile(samples, 99))
    bound_B = p99 + 3.0 * sigma

    # Method C: Hoeffding — for exceedance probability delta,
    #   P(sample > max_obs + epsilon) <= exp(-2 n epsilon^2 / range^2)
    #   Solving for epsilon at delta=1e-6: epsilon = range * sqrt(log(1/delta) / (2n))
    delta   = 1e-6
    if rng_val > 0 and n > 0:
        import math
        epsilon = rng_val * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
        bound_C = smax + epsilon
    else:
        bound_C = smax

    return {
        "n":          n,
        "mean_ms":    round(mu, 4),
        "std_ms":     round(sigma, 4),
        "min_ms":     round(smin, 4),
        "max_ms":     round(smax, 4),
        "method_A":   percentiles_A,
        "method_B": {
            "name":        "P99 + 3*sigma",
            "p99_ms":      round(p99, 4),
            "sigma_ms":    round(sigma, 4),
            "bound_ms":    round(float(bound_B), 4),
            "description": "Gaussian tail approximation; conservative if tail is heavy",
        },
        "method_C": {
            "name":        "Hoeffding (distribution-free)",
            "max_obs_ms":  round(smax, 4),
            "range_ms":    round(rng_val, 4),
            "epsilon_ms":  round(float(bound_C - smax), 4) if bound_C else None,
            "bound_ms":    round(float(bound_C), 4),
            "delta":       delta,
            "description": "P(new sample > bound) < 1e-6 with no distributional assumption",
        },
    }
