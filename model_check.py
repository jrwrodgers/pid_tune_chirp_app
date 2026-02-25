"""
Plot EdgeTX model curves from a YAML model file.

Usage:
    python model_check.py path/to/model.yml

For each curve in the model's `curves` section, this script:
- Reconstructs its Y values (amplitudes) from the global `points` map.
- Reconstructs X positions from the second block of points (quantized_scaled_x)
  if present, otherwise uses an even grid from -100 to 100.
- Plots the curve as a subplot titled with the curve's `name`.

NOTE: This follows the convention used by this project when adding chirp
segments to a model:
  - For each curve with N points:
      * First N entries in `points` for that curve are amplitudes (Y).
      * Next (N-2) entries are inner X values (quantized_scaled_x), with
        the endpoints -100 and 100 implied.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml  # type: ignore[import]
except Exception as e:  # pragma: no cover
    yaml = None


def load_model(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Please `pip install pyyaml`.")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def extract_curve_series(model: dict):
    """
    Returns list of (name, x, y) for each curve in the model.
    """
    curves = model.get("curves") or {}
    points_map = model.get("points") or {}
    if not isinstance(curves, dict) or not isinstance(points_map, dict):
        return []

    # Flatten point values in ascending key order
    def _sorted_vals(d: dict):
        items = sorted(d.items(), key=lambda kv: int(kv[0]))
        return [v.get("val", 0) for k, v in items]

    all_vals = _sorted_vals(points_map)

    # Sort curves by index
    curve_items = sorted(curves.items(), key=lambda kv: int(kv[0]))

    series = []
    cursor = 0

    for _, cdef in curve_items:
        if not isinstance(cdef, dict):
            continue
        n_pts = int(cdef.get("points", 0) or 0)
        name = cdef.get("name") or ""
        if n_pts <= 0:
            continue

        if cursor + n_pts > len(all_vals):
            # Not enough data, stop
            break

        # Y values (amplitudes)
        y_vals = np.array(all_vals[cursor : cursor + n_pts], dtype=float)
        cursor += n_pts

        # X inner values, if present (N-2)
        x_inner = None
        if cursor + max(n_pts - 2, 0) <= len(all_vals) and n_pts >= 3:
            x_inner = np.array(all_vals[cursor : cursor + (n_pts - 2)], dtype=float)
            cursor += n_pts - 2

        if x_inner is not None and len(x_inner) == n_pts - 2:
            # Reconstruct full x with -100 and 100 endpoints implied
            x_vals = np.concatenate(
                [
                    np.array([-100.0]),
                    x_inner.astype(float),
                    np.array([100.0]),
                ]
            )
        else:
            # Fallback: evenly spaced from -100..100
            x_vals = np.linspace(-100.0, 100.0, n_pts)

        series.append((name, x_vals, y_vals))

    return series


def plot_model_curves(path: str) -> None:
    model = load_model(path)
    series = extract_curve_series(model)

    if not series:
        print("No curves found to plot.")
        return

    n = len(series)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, x, y) in zip(axes, series):
        ax.plot(x, y, marker="o")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        title = name if name else "Unnamed curve"
        ax.set_title(title)

    axes[-1].set_xlabel("X (scaled -100..100)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_check.py path/to/model.yml")
        raise SystemExit(1)
    plot_model_curves(sys.argv[1])

