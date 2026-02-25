import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def load_config(path: str | None) -> dict:
    cfg_path = Path(path or "config.json")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Defaults (no T: duration is determined by completing `cycles` cycles)
    cfg.setdefault("cycles", 16)
    cfg.setdefault("f_start", 1.0)
    cfg.setdefault("f_end", 10.0)
    cfg.setdefault("amp_max", 1.0)
    cfg.setdefault("chirp_type", 1)  # 1=linear, 2=log, 3=hyperbolic, 4=Hann-tapered log
    cfg.setdefault("hann_alpha", 0.25)

    return cfg


def phase_linear(t: np.ndarray, f_start: float, f_end: float, T: float) -> np.ndarray:
    df = f_end - f_start
    return 2 * np.pi * (f_start * t + 0.5 * df * t**2 / T)


def phase_log(t: np.ndarray, f_start: float, f_end: float, T: float) -> np.ndarray:
    # Logarithmic sweep: f(t) = f_start * r^(t/T)
    r = f_end / f_start
    if r <= 0:
        raise ValueError("f_start and f_end must be positive for log chirp.")
    if abs(r - 1.0) < 1e-12:
        return 2 * np.pi * f_start * t

    ln_r = np.log(r)
    # N(t) = f_start * T / ln(r) * (r^(t/T) - 1)
    N_t = f_start * T / ln_r * (np.power(r, t / T) - 1.0)
    return 2 * np.pi * N_t


def phase_hyperbolic(t: np.ndarray, f_start: float, f_end: float, T: float) -> np.ndarray:
    # Hyperbolic sweep: 1/f(t) varies linearly between 1/f_start and 1/f_end
    inv_f_start = 1.0 / f_start
    inv_f_end = 1.0 / f_end
    a = inv_f_start
    b = (inv_f_end - inv_f_start) / T

    if abs(b) < 1e-12:
        return 2 * np.pi * f_start * t

    # N(t) = (1/b) * ln(1 + b t / a)
    N_t = (1.0 / b) * np.log(1.0 + b * t / a)
    return 2 * np.pi * N_t


def tukey_window(t: np.ndarray, T: float, alpha: float) -> np.ndarray:
    """
    Hann-tapered Tukey-style window.
    alpha in (0, 1]: fraction of T used for tapering at both ends.
    """
    if alpha <= 0:
        return np.ones_like(t)
    if alpha >= 1:
        # Full Hann window over [0, T]
        u = t / T
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * u))

    u = t / T
    w = np.ones_like(u)

    # Rising edge
    idx_rise = u < alpha / 2.0
    w[idx_rise] = 0.5 * (1.0 - np.cos(2.0 * np.pi * u[idx_rise] / alpha))

    # Falling edge
    idx_fall = u > 1.0 - alpha / 2.0
    w[idx_fall] = 0.5 * (
        1.0 - np.cos(2.0 * np.pi * (1.0 - u[idx_fall]) / alpha)
    )

    return w


def cycles_linear(t: float | np.ndarray, f_start: float, f_end: float, T: float) -> np.ndarray:
    """Number of cycles up to time t for linear chirp."""
    return phase_linear(np.asarray(t, dtype=float), f_start, f_end, T) / (2.0 * np.pi)


def cycles_log(t: float | np.ndarray, f_start: float, f_end: float, T: float) -> np.ndarray:
    """Number of cycles up to time t for log chirp."""
    return phase_log(np.asarray(t, dtype=float), f_start, f_end, T) / (2.0 * np.pi)


def cycles_hyperbolic(t: float | np.ndarray, f_start: float, f_end: float, T: float) -> np.ndarray:
    """Number of cycles up to time t for hyperbolic chirp."""
    return phase_hyperbolic(np.asarray(t, dtype=float), f_start, f_end, T) / (2.0 * np.pi)


def invert_cycles_bisection(
    N_target: float,
    N_func,
    T_ub: float,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> float:
    """
    Invert N(t) = N_target on [0, T_ub] by bisection, assuming
    N(t) is monotone increasing and N(0)=0, N(T_ub) >= N_target.
    """
    lo, hi = 0.0, T_ub
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if N_func(mid) < N_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def duration_for_cycles(
    cycles: int, f_start: float, f_end: float, chirp_type: int
) -> float:
    """
    Return the duration T_end (seconds) such that the chirp completes
    exactly `cycles` full cycles.
    """
    if chirp_type == 1:
        # Linear: N(T) = (f_start + f_end)/2 * T  =>  T = 2*cycles/(f_start+f_end)
        return 2.0 * cycles / (f_start + f_end)
    if chirp_type in (2, 4):
        # Log: N(T) = f_start*T/ln(r)*(r-1), r = f_end/f_start  =>  T = cycles*ln(r)/(f_start*(r-1))
        r = f_end / f_start
        if r <= 0:
            raise ValueError("f_start and f_end must be positive for log chirp.")
        if abs(r - 1.0) < 1e-12:
            return cycles / f_start
        return cycles * np.log(r) / (f_start * (r - 1.0))
    if chirp_type == 3:
        # Hyperbolic: N(T) = T * ln(f_start/f_end) / (1/f_end - 1/f_start)  =>  T = cycles*(1/f_end-1/f_start)/ln(f_start/f_end)
        inv_start = 1.0 / f_start
        inv_end = 1.0 / f_end
        return cycles * (inv_end - inv_start) / np.log(f_start / f_end)
    raise ValueError(f"Unknown chirp_type: {chirp_type}")


def generate_chirp_points(cfg: dict) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Generate chirp points for exactly `cycles` complete cycles.
    Returns (t_points, y_points, effective_cycles, T_end).
    """
    cycles = cfg["cycles"]
    f_start = cfg["f_start"]
    f_end = cfg["f_end"]
    amp_max = cfg["amp_max"]
    chirp_type = cfg["chirp_type"]
    hann_alpha = cfg["hann_alpha"]

    # Duration to complete exactly `cycles` cycles (no fixed T)
    T = duration_for_cycles(cycles, f_start, f_end, chirp_type)

    if chirp_type in (1, 2, 4):
        if chirp_type == 1:
            N_func = lambda t: cycles_linear(t, f_start, f_end, T)
        else:
            N_func = lambda t: cycles_log(t, f_start, f_end, T)
    elif chirp_type == 3:
        N_func = lambda t: cycles_hyperbolic(t, f_start, f_end, T)
    else:
        raise ValueError(f"Unknown chirp_type: {chirp_type}")

    N_T = float(N_func(T))
    effective_cycles = int(np.round(N_T))  # should equal cycles

    k_values = np.arange(0, 4 * effective_cycles + 1)

    t_points = np.empty_like(k_values, dtype=float)
    for i, k in enumerate(k_values):
        N_target = k / 4.0
        if N_target > N_T:
            t_points[i] = T
        else:
            t_points[i] = invert_cycles_bisection(N_target, N_func, T)

    if chirp_type == 4:
        sign_pattern = np.array([0.0, 1.0, 0.0, -1.0], dtype=float)
        signs = sign_pattern[k_values % 4]
        window = tukey_window(t_points, T, hann_alpha)
        y_points = amp_max * window * signs
    else:
        pattern = np.array([0.0, amp_max, 0.0, -amp_max], dtype=float)
        y_points = pattern[k_values % 4]

    return t_points, y_points, effective_cycles, T


POINTS_PER_SEGMENT = 17


def segment_chirp_points(
    t_points: np.ndarray,
    y_points: np.ndarray,
    chirp_name: str,
) -> None:
    """
    Split the full set of points into segments of up to 17 points each
    (boundary points repeated between adjacent segments). Print each segment's
    x,y points plus start time, end time, and duration. All times are quantized
    to 1 decimal place.
    """
    n = len(t_points)
    if n < 2:
        print(f"  [Segments skipped: need at least 2 points, got {n}]")
        return

    # Build segment index ranges with max length POINTS_PER_SEGMENT
    segments: list[tuple[int, int]] = []
    start_idx = 0
    step = POINTS_PER_SEGMENT - 1  # overlap of 1 point
    while start_idx < n:
        end_idx = min(start_idx + POINTS_PER_SEGMENT, n)
        segments.append((start_idx, end_idx))
        if end_idx >= n:
            break
        start_idx += step

    num_segments = len(segments)

    for seg_idx, (start_idx, end_idx) in enumerate(segments):
        t_seg = t_points[start_idx:end_idx]
        y_seg = y_points[start_idx:end_idx]
        t_start = round(float(t_seg[0]), 1)
        t_end = round(float(t_seg[-1]), 1)
        duration = round(t_end - t_start, 1)

        print(f"  --- Segment {seg_idx + 1}/{num_segments} ({chirp_name}) ---")
        print(
            f"  Start time: {t_start:.1f} s, End time: {t_end:.1f} s, "
            f"Duration: {duration:.1f} s"
        )
        # Scaled x: segment time mapped to [-100, 100]
        if duration > 0:
            scaled_x = -100.0 + 200.0 * (t_seg - t_start) / duration
        else:
            scaled_x = np.zeros_like(t_seg)
        quantized_scaled_x = np.round(scaled_x)

        print("  quantized_scaled_x, scaled_x, time (s), amplitude (int):")
        for i in range(len(t_seg)):
            sx_q = int(quantized_scaled_x[i])
            sx = float(scaled_x[i])
            t_q = round(float(t_seg[i]), 1)
            amp_int = int(round(float(y_seg[i])))
            print(f"    {sx_q}, {sx:.1f}, {t_q:.1f}, {amp_int}")
        print()


def export_chirp_segments_csv(cfg: dict, chirp_type: int, filepath: str) -> None:
    """
    Write segment data for the given chirp type to a CSV.
    Columns: quantized_scaled_x, time, amplitude.
    Amplitude is scaled to -100..100 regardless of amp_max.
    """
    import csv
    cfg_type = dict(cfg)
    cfg_type["chirp_type"] = chirp_type
    t_points, y_points, _, _ = generate_chirp_points(cfg_type)
    amp_max = max(float(cfg["amp_max"]), 1e-10)

    n = len(t_points)
    if n < 2:
        raise ValueError(f"Not enough points to export ({n})")

    segments = []
    start_idx = 0
    step = POINTS_PER_SEGMENT - 1
    while start_idx < n:
        end_idx = min(start_idx + POINTS_PER_SEGMENT, n)
        segments.append((start_idx, end_idx))
        if end_idx >= n:
            break
        start_idx += step

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["quantized_scaled_x", "time", "amplitude"])
        for seg_idx, (start_idx, end_idx) in enumerate(segments):
            t_seg = t_points[start_idx:end_idx]
            y_seg = y_points[start_idx:end_idx]
            t_start = float(t_seg[0])
            t_end = float(t_seg[-1])
            duration = t_end - t_start
            if duration > 0:
                scaled_x = -100.0 + 200.0 * (t_seg - t_start) / duration
            else:
                scaled_x = np.zeros_like(t_seg)
            quantized_scaled_x = np.round(scaled_x).astype(int)
            # Amplitude scaled to -100..100
            amp_scaled = np.clip(np.round((y_seg / amp_max) * 100.0), -100, 100).astype(int)
            time_q = np.round(t_seg, 1)
            for i in range(len(t_seg)):
                writer.writerow([int(quantized_scaled_x[i]), float(time_q[i]), int(amp_scaled[i])])
            # Blank line after each segment except possibly the last
            writer.writerow([])


def add_chirp_segments_to_model_yaml(
    cfg: dict,
    chirp_type: int,
    model_path: str,
    max_segments: int | None = None,
) -> None:
    """
    Append curves and points for all segments of the selected chirp
    into an EdgeTX model YAML file.
    Each segment becomes one curve; its points go into the global `points` map.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML editing. Please `pip install pyyaml`.")

    if not Path(model_path).is_file():
        raise FileNotFoundError(model_path)

    with open(model_path, "r", encoding="utf-8") as f:
        model = yaml.safe_load(f) or {}

    # Ensure curves/points are dicts (YAML may contain null here)
    curves_raw = model.get("curves")
    if not isinstance(curves_raw, dict):
        curves_raw = {}
    model["curves"] = curves_raw

    points_raw = model.get("points")
    if not isinstance(points_raw, dict):
        points_raw = {}
    model["points"] = points_raw

    curves = curves_raw
    points_map = points_raw

    # Normalize keys to ints
    def _int_keys(d: dict) -> list[int]:
        return [int(k) for k in d.keys()] if d else []

    next_curve_idx = (max(_int_keys(curves)) + 1) if curves else 0
    next_point_idx = (max(_int_keys(points_map)) + 1) if points_map else 0

    cfg_type = dict(cfg)
    cfg_type["chirp_type"] = chirp_type
    t_points, y_points, _, _ = generate_chirp_points(cfg_type)
    amp_max = max(float(cfg["amp_max"]), 1e-10)

    n = len(t_points)
    if n < 2:
        raise ValueError(f"Not enough points to add to model ({n})")

    # Same segmenting logic as elsewhere
    segments: list[tuple[int, int]] = []
    start_idx = 0
    step = POINTS_PER_SEGMENT - 1
    while start_idx < n:
        end_idx = min(start_idx + POINTS_PER_SEGMENT, n)
        segments.append((start_idx, end_idx))
        if end_idx >= n:
            break
        start_idx += step

    for seg_idx, (start_idx, end_idx) in enumerate(segments):
        if max_segments is not None and seg_idx >= max_segments:
            break
        t_seg = t_points[start_idx:end_idx]
        y_seg = y_points[start_idx:end_idx]
        n_pt = len(t_seg)

        # Amplitude (y) scaled to -100..100
        amp_scaled = np.clip(np.round((y_seg / amp_max) * 100.0), -100, 100).astype(int)

        # Quantized scaled x: segment time -> [-100, 100]
        t_start = float(t_seg[0])
        t_end = float(t_seg[-1])
        duration = t_end - t_start
        if duration > 0:
            scaled_x = -100.0 + 200.0 * (t_seg - t_start) / duration
        else:
            scaled_x = np.zeros_like(t_seg)
        quantized_scaled_x = np.round(scaled_x).astype(int)

        # Add curve entry
        curve_name = f"chirp{chirp_type}_seg{seg_idx + 1}"
        curves[next_curve_idx] = {
            "type": 1,
            "smooth": 1,
            "points": n_pt,
            "name": curve_name,
        }

        # 1) Points for all y values (amplitudes)
        for val in amp_scaled:
            points_map[next_point_idx] = {"val": int(val)}
            next_point_idx += 1

        # 2) Points for all x values (quantized_scaled_x) with -100 and 100 omitted
        x_inner = quantized_scaled_x[1:-1]  # drop first and last
        for val in x_inner:
            points_map[next_point_idx] = {"val": int(val)}
            next_point_idx += 1

        next_curve_idx += 1

    # Write back YAML (preserving a simple style)
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(model, f, sort_keys=False)


CHIRP_NAMES = {
    1: "Linear sine chirp",
    2: "Log sine chirp",
    3: "Hyperbolic sine chirp",
    4: "Hann-tapered log-sine chirp",
}
CHIRP_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
PSD_YLIM = (1e-6, 10000.0)  # fixed y-range for PSD (no dynamic autoscale)


def build_plots(cfg: dict, fig: plt.Figure, *, skip_segment_print: bool = False) -> None:
    """Draw chirp and PSD subplots on fig. Set skip_segment_print=True in GUI to avoid console spam."""
    fig.clear()
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 2], hspace=0.4, wspace=0.3)
    ax_chirps = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    ax_psd_all = fig.add_subplot(gs[:, 1])

    f_start = cfg["f_start"]
    f_end = cfg["f_end"]
    amp_max = cfg["amp_max"]
    hann_alpha = cfg["hann_alpha"]

    for idx, chirp_type in enumerate((1, 2, 3, 4)):
        cfg_type = dict(cfg)
        cfg_type["chirp_type"] = chirp_type
        t_points, y_points, effective_cycles, T_end = generate_chirp_points(cfg_type)
        T_end_q = round(T_end, 1)  # for display only; curve uses exact T_end so it passes through points

        if not skip_segment_print:
            segment_chirp_points(t_points, y_points, CHIRP_NAMES[chirp_type])

        # Use exact T_end for curve/phase so the curve passes through the discrete points
        t_curve = np.linspace(0.0, T_end, 2000)
        if chirp_type == 1:
            y_curve_base = np.sin(phase_linear(t_curve, f_start, f_end, T_end))
        elif chirp_type == 2:
            y_curve_base = np.sin(phase_log(t_curve, f_start, f_end, T_end))
        elif chirp_type == 3:
            y_curve_base = np.sin(phase_hyperbolic(t_curve, f_start, f_end, T_end))
        elif chirp_type == 4:
            y_curve_base = np.sin(phase_log(t_curve, f_start, f_end, T_end))
        else:
            raise ValueError(f"Unknown chirp_type: {chirp_type}")

        if chirp_type == 4:
            window_curve = tukey_window(t_curve, T_end, hann_alpha)
            y_curve = amp_max * window_curve * y_curve_base
        else:
            y_curve = amp_max * y_curve_base

        # PSD of the chirp (uniform sampling)
        fs = (len(t_curve) - 1) / T_end  # sample rate in Hz

        # Full-precision signal
        f_psd, P_psd_full = scipy_signal.welch(
            y_curve, fs=fs, nperseg=min(256, len(y_curve) // 4)
        )

        # Quantized signal (amplitude rounded to integer steps)
        y_curve_q = np.round(y_curve)
        _, P_psd_quant = scipy_signal.welch(
            y_curve_q, fs=fs, nperseg=min(256, len(y_curve_q) // 4)
        )

        # Restrict to 0.1–100 Hz
        mask = (f_psd >= 0.01) & (f_psd <= 200.0)
        f_psd = f_psd[mask]
        P_psd_full = P_psd_full[mask]
        P_psd_quant = P_psd_quant[mask]

        ax = ax_chirps[idx]
        color = CHIRP_COLORS[chirp_type]
        ax.plot(t_curve, y_curve, label=CHIRP_NAMES[chirp_type], color=color)

        if chirp_type == 4:
            # Peaks/troughs may not be exactly ±amp_max due to window
            is_peak = y_points > 0
            is_trough = y_points < 0
        else:
            is_peak = y_points == amp_max
            is_trough = y_points == -amp_max

        ax.scatter(
            t_points[is_peak],
            y_points[is_peak],
            color="red",
            label="Peaks",
            zorder=3,
        )
        ax.scatter(
            t_points[is_trough],
            y_points[is_trough],
            color="blue",
            label="Troughs",
            zorder=3,
        )
        ax.scatter(
            t_points[y_points == 0],
            y_points[y_points == 0],
            color="black",
            label="Zero crossings (0)",
            zorder=3,
        )

        # Draw vertical lines showing all segments (max length 17 points, with overlaps)
        n_pts = len(t_points)
        if n_pts >= 2:
            start_idx = 0
            step = POINTS_PER_SEGMENT - 1
            boundary_indices = []
            while start_idx < n_pts:
                boundary_indices.append(start_idx)
                end_idx = min(start_idx + POINTS_PER_SEGMENT, n_pts)
                if end_idx >= n_pts:
                    break
                start_idx += step
            # Also add final point as a boundary
            if boundary_indices[-1] != n_pts - 1:
                boundary_indices.append(n_pts - 1)

            for idx in boundary_indices:
                x_boundary = t_points[idx]
                ax.axvline(
                    x_boundary,
                    color="magenta",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.8,
                    zorder=2,
                )

        # Compute segment count for this chirp (same logic as in segment_chirp_points)
        n_pts = len(t_points)
        seg_count = 0
        if n_pts >= 2:
            start_idx = 0
            step = POINTS_PER_SEGMENT - 1
            while start_idx < n_pts:
                end_idx = min(start_idx + POINTS_PER_SEGMENT, n_pts)
                seg_count += 1
                if end_idx >= n_pts:
                    break
                start_idx += step

        title = (
            f"{CHIRP_NAMES[chirp_type]} "
            f"({effective_cycles} cycles, {seg_count} segments, "
            f"f_start={f_start} Hz, f_end={f_end} Hz) — final time = {T_end_q:.1f} s"
        )
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        # Only bottom subplot gets the x-label to reduce clutter
        if idx == 3:
            ax.set_xlabel("Time (s)")
        else:
            ax.tick_params(labelbottom=False)

        # Right column: combined PSD plot (all chirps)
        # Full-precision PSD
        ax_psd_all.semilogy(
            f_psd,
            P_psd_full,
            color=color,
            linewidth=0.9,
            label=f"{CHIRP_NAMES[chirp_type]} full",
        )
        ax_psd_all.semilogy(
            f_psd,
            P_psd_quant,
            color=color,
            linewidth=0.9,
            linestyle="--",
            label=f"{CHIRP_NAMES[chirp_type]} quantized",
        )

    ax_psd_all.set_xlim(1, 100.0)
    ax_psd_all.set_xscale("log")
    ax_psd_all.set_ylim(*PSD_YLIM)
    ax_psd_all.set_title("Power Spectral Density (all chirps: full vs quantized)")
    ax_psd_all.set_xlabel("Frequency (Hz)")
    ax_psd_all.set_ylabel("Power / frequency")
    ax_psd_all.grid(True, which="both")
    ax_psd_all.legend()
    fig.tight_layout()


def main(path: str | None = None) -> None:
    cfg = load_config(path)
    fig = plt.figure(figsize=(12, 8))
    build_plots(cfg, fig, skip_segment_print=False)
    plt.show()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_path)