"""
GUI for EdgeTX Tuning Chirp: parameters on the left, plots on the right.
Plots update when parameters change (on button click or Enter in entry).
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from chirp_gen import (
    load_config,
    build_plots,
    export_chirp_segments_csv,
    add_chirp_segments_to_model_yaml,
    CHIRP_NAMES,
)


def run_gui(config_path: str | None = None) -> None:
    root = tk.Tk()
    root.title("EdgeTX Tuning Chirp")
    root.geometry("1400x800")
    root.minsize(1000, 600)

    # Load initial config
    cfg = load_config(config_path)

    # --- Left panel: parameters ---
    left = ttk.Frame(root, padding=10, width=220)
    left.pack(side=tk.LEFT, fill=tk.Y)
    left.pack_propagate(False)

    ttk.Label(left, text="Parameters", font=("", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))

    vars_ = {}
    row = 0

    def add_param(label: str, key: str, default, width=12):
        nonlocal row
        ttk.Label(left, text=label).pack(anchor=tk.W)
        if isinstance(default, int):
            v = tk.IntVar(root, value=default)
            w = ttk.Spinbox(left, from_=1, to=10000, width=width, textvariable=v)
        else:
            v = tk.DoubleVar(root, value=default)
            w = ttk.Entry(left, width=width, textvariable=v)
        w.pack(anchor=tk.W, pady=(0, 8))
        vars_[key] = (v, isinstance(default, int))
        row += 1

    add_param("Cycles", "cycles", cfg.get("cycles", 33))
    add_param("f_start (Hz)", "f_start", cfg.get("f_start", 1.0))
    add_param("f_end (Hz)", "f_end", cfg.get("f_end", 10.0))
    add_param("Amplitude (amp_max)", "amp_max", cfg.get("amp_max", 1.0))
    add_param("Hann alpha", "hann_alpha", cfg.get("hann_alpha", 0.25))

    def get_cfg() -> dict:
        out = {}
        for key, (var, is_int) in vars_.items():
            try:
                if is_int:
                    out[key] = int(var.get())
                else:
                    out[key] = float(var.get())
            except (ValueError, tk.TclError):
                return None
        out["chirp_type"] = 1
        return out

    def update_plots() -> None:
        cfg_new = get_cfg()
        if cfg_new is None:
            messagebox.showerror("Invalid parameters", "Please enter valid numbers.")
            return
        try:
            build_plots(cfg_new, fig, skip_segment_print=True)
            canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Plot error", str(e))

    ttk.Button(left, text="Update plots", command=update_plots).pack(anchor=tk.W, pady=(12, 0))

    # Function selection + export / add-curve actions
    ttk.Label(left, text="Function", font=("", 10, "bold")).pack(anchor=tk.W, pady=(16, 4))
    chirp_options = [(ct, CHIRP_NAMES.get(ct, f"Type {ct}")) for ct in (1, 2, 3, 4)]
    sel_var = tk.IntVar(root, value=1)
    combo = ttk.Combobox(
        left,
        values=[f"{ct} – {name}" for ct, name in chirp_options],
        state="readonly",
        width=24,
    )
    combo.current(0)
    combo.pack(anchor=tk.W, pady=(0, 6))

    def selected_type() -> int:
        idx = combo.current()
        return chirp_options[idx][0]

    def do_export_selected() -> None:
        cfg_new = get_cfg()
        if cfg_new is None:
            messagebox.showerror("Invalid parameters", "Please enter valid numbers.")
            return
        ct = selected_type()
        name = CHIRP_NAMES.get(ct, "chirp")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"chirp_{ct}_{name.replace(' ', '_').lower()}.csv",
        )
        if not path:
            return
        try:
            export_chirp_segments_csv(cfg_new, ct, path)
            messagebox.showinfo("Export", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def do_add_curve_to_model() -> None:
        cfg_new = get_cfg()
        if cfg_new is None:
            messagebox.showerror("Invalid parameters", "Please enter valid numbers.")
            return
        ct = selected_type()
        model_path = filedialog.askopenfilename(
            title="Select EdgeTX model YAML",
            filetypes=[("YAML", "*.yml *.yaml"), ("All files", "*.*")],
        )
        if not model_path:
            return
        try:
            # For this test, only add the first segment as a curve
            add_chirp_segments_to_model_yaml(cfg_new, ct, model_path, max_segments=1)
            messagebox.showinfo("Model updated", f"First segment added to {model_path}")
        except Exception as e:
            messagebox.showerror("Add curve error", str(e))

    ttk.Button(left, text="Export segment CSV", command=do_export_selected).pack(anchor=tk.W, pady=(4, 2))
    ttk.Button(left, text="Add curves to model file", command=do_add_curve_to_model).pack(anchor=tk.W, pady=(2, 8))

    def on_return(_event):
        update_plots()

    # --- Right panel: matplotlib figure ---
    right = ttk.Frame(root, padding=5)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    fig = Figure(figsize=(12, 8), dpi=100)
    build_plots(cfg, fig, skip_segment_print=True)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    for c in left.winfo_children():
        if isinstance(c, (ttk.Entry, ttk.Spinbox)):
            c.bind("<Return>", on_return)
        if isinstance(c, ttk.Spinbox):
            c.bind("<ButtonRelease-1>", lambda e: update_plots())
            c.bind("<KeyRelease>", lambda e: update_plots())

    root.mainloop()


if __name__ == "__main__":
    import sys
    run_gui(sys.argv[1] if len(sys.argv) > 1 else None)
