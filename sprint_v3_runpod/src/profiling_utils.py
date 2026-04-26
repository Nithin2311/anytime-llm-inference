"""
profiling_utils.py — GPU latency profiling with optional inter-run sleep.

Spaced profiling (sleep_ms > 0) allows GPU thermal state, L2 cache, and
CUDA stream state to return to baseline between measurements, which is
the standard fix for autocorrelation in timing samples (Ljung-Box IID test).
"""

import time
import torch
import numpy as np


def profile_layer_latency(
    model,
    seq_len: int,
    n_warmup: int = 50,
    n_runs: int = 500,
    sleep_ms: float = 200.0,
    exit_layer: str = "full",
    device: str = "cuda",
) -> np.ndarray:
    """
    Profile per-token latency for one (seq_len, exit_layer) cell.

    Parameters
    ----------
    model        : EarlyExitModel instance
    seq_len      : input sequence length in tokens
    n_warmup     : warm-up passes before timing begins (discarded)
    n_runs       : number of timed runs to collect
    sleep_ms     : milliseconds to sleep between timed runs (0 = no sleep)
    exit_layer   : "l16" for L16 exit, "full" for full 22-layer pass
    device       : cuda device string

    Returns
    -------
    np.ndarray of shape (n_runs,) with per-run latencies in milliseconds
    """
    input_ids = torch.randint(
        100, 30000, (1, seq_len), dtype=torch.long, device=device
    )

    # Warm-up: run without timing to stabilize GPU state
    with torch.no_grad():
        for _ in range(n_warmup):
            if exit_layer == "l16":
                out, hidden = model.forward(input_ids)
                _ = model.get_l16_confidence(hidden)
            else:
                out, _ = model.forward(input_ids)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        with torch.no_grad():
            if exit_layer == "l16":
                out, hidden = model.forward(input_ids)
                _ = model.get_l16_confidence(hidden)
            else:
                out, _ = model.forward(input_ids)
        end_evt.record()
        torch.cuda.synchronize()

        latencies.append(start_evt.elapsed_time(end_evt))

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    return np.array(latencies, dtype=np.float64)


def profile_warmup_artifact(
    model,
    seq_len: int = 128,
    n_runs: int = 600,
    device: str = "cuda",
) -> np.ndarray:
    """
    Profile WITHOUT any warm-up discard to expose warm-up artifacts.
    First n_runs measurements include any JIT/cache cold-start outliers.
    """
    input_ids = torch.randint(
        100, 30000, (1, seq_len), dtype=torch.long, device=device
    )
    torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        with torch.no_grad():
            out, _ = model.forward(input_ids)
        end_evt.record()
        torch.cuda.synchronize()
        latencies.append(start_evt.elapsed_time(end_evt))

    return np.array(latencies, dtype=np.float64)
