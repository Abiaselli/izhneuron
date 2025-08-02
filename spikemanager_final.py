from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

@dataclass
class TransistorParams:
    mu_n: float
    cox_n: float
    mu_p: float
    cox_p: float
    cu: float
    cu2: float
    alpha: float
    beta: float
    gamma: float

def get_k(mos_type: str, params: TransistorParams):
    if mos_type == "NMOS":
        return params.mu_n * params.cox_n
    elif mos_type == "PMOS":
        return params.mu_p * params.cox_p
    else:
        raise ValueError("MOS type must be NMOS or PMOS")

def compute_wl_all(
    wl_known1, wl_known2,
    target: str,
    m1_type: str, m2_type: str, m7_type: str, m6_type: str,
    params: TransistorParams
):
    k_m1 = get_k(m1_type, params)
    k_m2 = get_k(m2_type, params)
    k_m7 = get_k(m7_type, params)
    k_m6 = get_k(m6_type, params)
    
    wl_result = {}
    if target == "M1":
        wl_result["M1"] = 2 * params.cu * 0.04 / (params.alpha * k_m1 * wl_known1 * wl_known2)
        wl_result["M2"] = wl_known1
        wl_result["M7"] = wl_known2
    elif target == "M2":
        wl_result["M2"] = 2 * params.cu * 0.04 / (params.alpha * k_m2 * wl_known1 * wl_known2)
        wl_result["M1"] = wl_known1
        wl_result["M7"] = wl_known2
    elif target == "M7":
        wl_result["M7"] = 2 * params.cu * 0.04 / (params.alpha * k_m7 * wl_known1 * wl_known2)
        wl_result["M1"] = wl_known1
        wl_result["M2"] = wl_known2

    # Always compute M6
    wl_result["M6"] = 2 * params.cu2 * 0.02 / (params.gamma * k_m6 * 1 * 1)  # simplified denominator

    return wl_result

# Tooltip helper class
class CreateToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, _):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "9", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, _):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def launch_full_gui():
    root = tk.Tk()
    root.title("Izhikevich MOSFET W/L Solver")

    # Entries for mobility and Cox
    def labeled_entry(row, label):
        tk.Label(root, text=label).grid(row=row, column=0, sticky='e')
        entry = tk.Entry(root)
        entry.grid(row=row, column=1)
        return entry

    mu_n_entry = labeled_entry(0, "μ_n (NMOS)")
    cox_n_entry = labeled_entry(1, "Cox_n (NMOS)")
    mu_p_entry = labeled_entry(2, "μ_p (PMOS)")
    cox_p_entry = labeled_entry(3, "Cox_p (PMOS)")
    cu_entry = labeled_entry(4, "Cu")
    cu2_entry = labeled_entry(5, "Cu2")
    alpha_entry = labeled_entry(6, "α (alpha)")
    gamma_entry = labeled_entry(7, "γ (gamma)")

    # Dropdown for which MOS to solve
    tk.Label(root, text="Solve for").grid(row=8, column=0, sticky="e")
    solve_target = tk.StringVar()
    solve_menu = ttk.Combobox(root, textvariable=solve_target, values=["M1", "M2", "M7"], state="readonly")
    solve_menu.grid(row=8, column=1)
    solve_menu.current(0)
    CreateToolTip(solve_menu, "Choose which transistor's W/L to solve for based on the others")

    # Type selectors
    def typedrop(row, label):
        tk.Label(root, text=f"{label} type").grid(row=row, column=2, sticky='e')
        var = tk.StringVar()
        menu = ttk.Combobox(root, textvariable=var, values=["NMOS", "PMOS"], state="readonly", width=6)
        menu.grid(row=row, column=3)
        menu.current(0)
        CreateToolTip(menu, f"Select device type for {label}")
        return var

    m1_type = typedrop(0, "M1")
    m2_type = typedrop(1, "M2")
    m6_type = typedrop(2, "M6")
    m7_type = typedrop(3, "M7")

    # W/L input labels and fields
    val1_label = tk.Label(root, text="M2 (W/L)")
    val1_label.grid(row=9, column=0, sticky="e")
    val1_entry = tk.Entry(root)
    val1_entry.grid(row=9, column=1)

    val2_label = tk.Label(root, text="M7 (W/L)")
    val2_label.grid(row=10, column=0, sticky="e")
    val2_entry = tk.Entry(root)
    val2_entry.grid(row=10, column=1)

    # Output box
    output_box = tk.Text(root, width=50, height=6)
    output_box.grid(row=12, column=0, columnspan=4, padx=5, pady=5)

    def update_labels(*_):
        t = solve_target.get()
        if t == "M1":
            val1_label.config(text="M2 (W/L)")
            val2_label.config(text="M7 (W/L)")
        elif t == "M2":
            val1_label.config(text="M1 (W/L)")
            val2_label.config(text="M7 (W/L)")
        elif t == "M7":
            val1_label.config(text="M1 (W/L)")
            val2_label.config(text="M2 (W/L)")

    solve_target.trace("w", update_labels)

    def calculate():
        try:
            params = TransistorParams(
                mu_n=float(mu_n_entry.get()), cox_n=float(cox_n_entry.get()),
                mu_p=float(mu_p_entry.get()), cox_p=float(cox_p_entry.get()),
                cu=float(cu_entry.get()), cu2=float(cu2_entry.get()),
                alpha=float(alpha_entry.get()), beta=1.0, gamma=float(gamma_entry.get())
            )
            t = solve_target.get()
            v1 = float(val1_entry.get())
            v2 = float(val2_entry.get())
            mos_types = {
                "M1": m1_type.get(), "M2": m2_type.get(),
                "M6": m6_type.get(), "M7": m7_type.get()
            }
            result = compute_wl_all(
                wl_known1=v1, wl_known2=v2,
                target=t,
                m1_type=mos_types["M1"], m2_type=mos_types["M2"],
                m6_type=mos_types["M6"], m7_type=mos_types["M7"],
                params=params
            )
            output_box.delete(1.0, tk.END)
            for k, v in result.items():
                output_box.insert(tk.END, f"{k} (W/L) = {v:.6e}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_output():
        content = output_box.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("No Data", "No output to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filepath:
            with open(filepath, 'w') as f:
                f.write(content)
            messagebox.showinfo("Saved", f"Output saved to {filepath}")

    tk.Button(root, text="Compute", command=calculate).grid(row=11, column=0, columnspan=2)
    tk.Button(root, text="Save Output", command=save_output).grid(row=11, column=2, columnspan=2)

    update_labels()
    root.mainloop()
# Additions to integrate the spiking models into the GUI
izhikevich_params = [
    [0.02,   0.2,   -65,   6,   14], [0.02,   0.25,  -65,   6,   0.5],
    [0.02,   0.2,   -50,   2,   15], [0.02,   0.25,  -55,   0.05, 0.6],
    [0.02,   0.2,   -55,   4,   10], [0.01,   0.2,   -65,   8,   30],
    [0.02,  -0.1,   -55,   6,   0],  [0.2,    0.26,  -65,   0,   0],
    [0.02,   0.2,   -65,   6,   7],  [0.05,   0.26,  -60,   0,   0],
    [0.1,    0.26,  -60,  -1,   0],  [0.02,  -0.1,   -55,   6,   0],
    [0.03,   0.25,  -60,   4,   0],  [0.03,   0.25,  -52,   0,   0],
    [0.03,   0.25,  -60,   4,   0],  [1.0,    1.5,   -60,   0, -65],
    [1.0,    0.2,   -60, -21,  0],   [0.02,   1.0,   -55,   4,   0],
    [-0.02, -1.0,   -60,   8,  80],  [-0.026,-1.0,   -45,   0,  80]
]
izh_labels = [
    "tonic spiking", "phasic spiking", "tonic bursting", "phasic bursting",
    "mixed mode", "spike freq adaptation", "Class 1", "Class 2", "spike latency",
    "subthreshold oscillations", "resonator", "integrator", "rebound spike",
    "rebound burst", "threshold variability", "bistability", "DAP", "accommodation",
    "inhibition-induced spiking", "inhibition-induced bursting"
]

def add_izh_selector(root, a_entry, b_entry, d_entry, i_entry):
    def apply_params(event):
        idx = izhikevich_menu.current()
        a_val, b_val, _, d_val, i_val = izhikevich_params[idx]
        a_entry.delete(0, tk.END)
        a_entry.insert(0, str(a_val))
        b_entry.delete(0, tk.END)
        b_entry.insert(0, str(b_val))
        d_entry.delete(0, tk.END)
        d_entry.insert(0, str(d_val))
        i_entry.delete(0, tk.END)
        i_entry.insert(0, str(i_val))

    tk.Label(root, text="Spiking Model").grid(row=13, column=0, sticky='e')
    izhikevich_var = tk.StringVar()
    izhikevich_menu = ttk.Combobox(root, textvariable=izhikevich_var, values=izh_labels, state="readonly", width=30)
    izhikevich_menu.grid(row=13, column=1, columnspan=3)
    izhikevich_menu.bind("<<ComboboxSelected>>", apply_params)

    return izhikevich_menu


launch_full_gui()
