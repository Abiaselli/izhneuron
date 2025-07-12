import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

MAX_COSH_ARG = np.log(np.finfo(float).max)  # ≈709

def _safe_cosh(x):
        # clip x to [–709, +709] so np.cosh never overflows
        return np.cosh(np.clip(x, -MAX_COSH_ARG, MAX_COSH_ARG))

# Default BSIM4 parameters (example values; adjust as needed)
default_params = {
    'VTH0':   0.7,       # Long-channel threshold (V)
    'K1':     0.53,      # Body effect coefficient 1
    'K2':     0.0,       # Body effect coefficient 2
    'K3':     80.0,      # Narrow-width effect coefficient
    'K3B':    0.0,       # Narrow-width body-bias coefficient
    'W0':     2.5e-6,    # Narrow-width characteristic length (m)
    'DVT0':   2.2,       # Short-channel effect coefficient
    'DVT1':   0.53,      # Short-channel effect coefficient
    'DVT2':   0.0,       # Short-channel body-bias effect
    'DVT0W':  0.0,       # Narrow-short interaction coeff
    'DVT1W':  0.0,
    'DVT2W':  0.0,
    'DSUB':   0.0,       # DIBL coefficient
    'ETA0':   0.0,       # DIBL base
    'ETAB':   0.0,       # DIBL body-bias coefficient
    'DVTP0':  0.0,       # DITS coefficient
    'DVTP1':  0.0,       # DITS coefficient
    'TOXE':   3e-9,      # Oxide thickness (m)
    'eps_ox': 3.9 * 8.85e-12,  # Oxide permittivity (F/m)
    'eps_si': 11.7 * 8.85e-12, # Silicon permittivity (F/m)
    'q':      1.602e-19, # Elementary charge (C)
    'ni':     1.45e16,   # Intrinsic carrier concentration (1/m^3)
    'NDEP':   1e24,      # Doping concentration (1/m^3)
    'LINT':   0.0,       # Length bias (m)
    'WINT':   0.0,       # Width bias (m)
    'CGSO':   0.0,       # Gate-Source overlap cap per width (F/m)
    'CGDO':   0.0,       # Gate-Drain overlap cap per width (F/m)
    'CF':     0.0,       # Fringe cap per width (F/m)
    'RDSW':   200.0,     # Source/drain resistance per width (Ω·µm)
    'PCLM':   1.3,       # Channel-length modulation exponent
    'T':      300.0      # Temperature (K)
}

# Helper functions
def effective_dimensions(W, L, p):
    """Compute effective channel width and length."""
    Leff = L - 2*p['LINT']
    Weff = W - 2*p['WINT']
    return Weff, Leff

def calc_phi_s(p):
    """Surface potential approx: 2*Phi_F."""
    k = 1.38064852e-23  # Boltzmann constant
    T = p['T']
    Vt = k * T / p['q']
    Phi_F = Vt * np.log(p['NDEP'] / p['ni'])
    return 2 * Phi_F

def calc_Vbi(p):
    """Built-in potential for source/drain junction."""
    k = 1.38064852e-23
    T = p['T']
    Vt = k * T / p['q']
    return Vt * np.log((p['NDEP']**2) / (p['ni']**2))

def char_length(p, Vbs=0.0, is_width_dep=False):
    """Characteristic length for SCE or narrow-short interaction."""
    # Using oxide & silicon permittivities and surface potential
    phi_s = calc_phi_s(p)
    if not is_width_dep:
        factor = 1 + p['DVT2'] * Vbs
        eps_si = p['eps_si']
        return (eps_si * p['TOXE'] * np.sqrt(2*eps_si*phi_s/(p['q']*p['NDEP']))) / (p['eps_ox'] * factor)
    else:
        factor = 1 + p['DVT2W'] * Vbs
        eps_si = p['eps_si']
        return (eps_si * p['TOXE'] * np.sqrt(2*eps_si*phi_s/(p['q']*p['NDEP']))) / (p['eps_ox'] * factor)

def calc_Vth(W, L, Vgs=0.0, Vbs=0.0, Vds=0.0, params=None):
    """Calculate threshold voltage Vth based on BSIM4 simplified model."""
    # Merge default and user parameters
    p = default_params.copy()
    if params:
        p.update(params)

    # Effective dimensions
    Weff, Leff = effective_dimensions(W, L, p)
    # Thermal voltage
    k = 1.38064852e-23
    T = p['T']
    Vt = k * T / p['q']
    phi_s = calc_phi_s(p)
    Vbi = calc_Vbi(p)

    # Body effect (long-channel)
    dVth_body = p['K1']*(np.sqrt(phi_s+Vbs) - np.sqrt(phi_s)) - p['K2']*Vbs

    # Short-channel effect
    lt = char_length(p, Vbs=Vbs)

    arg = p['DVT1'] * Leff / lt
    denom = _safe_cosh(arg) - 1 + 1e-12
    theta_sce = 0.5 * p['DVT0'] / denom

    dVth_sce = -theta_sce * (Vbi - phi_s)

    # Narrow-width (primary)
    dVth_nw1 = (p['K3'] + p['K3B']*Vbs) * (p['TOXE']*phi_s) / (Weff + p['W0'] + 1e-12)

    # Narrow-width secondary (if enabled)
    dVth_nw2 = 0.0
    if p['DVT0W'] != 0.0:
        ltW = char_length(p, Vbs=Vbs, is_width_dep=True)
        theta_nw = 0.5*p['DVT0W'] / (np.cosh(p['DVT1W']*Leff/ltW) - 1 + 1e-12)
        dVth_nw2 = -theta_nw * (Vbi - phi_s)

    # DIBL (if Vds > 0)
    dVth_dibl = 0.0
    if Vds != 0.0:
        lt0 = char_length(p, Vbs=0.0)
        theta_dibl = 0.5 / (np.cosh(p['DSUB']*Leff/lt0) - 1 + 1e-12)
        dVth_dibl = -theta_dibl * (p['ETA0'] + p['ETAB']*Vbs) * Vds

    # DITS (if enabled)
    dVth_dits = 0.0
    if p['DVTP0'] != 0.0 and Vds != 0.0:
        dVth_dits = -Vt * np.log(Leff / (Leff + p['DVTP0']*(1 + np.exp(-p['DVTP1']*Vds))) + 1e-12)

    # Sum contributions
    Vth = (
        p['VTH0'] + dVth_body + dVth_sce +
        dVth_nw1 + dVth_nw2 + dVth_dibl + dVth_dits
    )
    return Vth

def calc_capacitances(W, L, params=None):
    """Approximate gate capacitances: Cox, overlap, fringe."""
    p = default_params.copy()
    if params:
        p.update(params)
    Weff, Leff = effective_dimensions(W, L, p)

    # Oxide capacitance per area
    Cox = p['eps_ox'] / p['TOXE']
    C_ox = Cox * Weff * Leff

    # Overlap capacitances
    C_gs = p['CGSO'] * Weff
    C_gd = p['CGDO'] * Weff

    # Fringe capacitance (both sides)
    C_fringe = p['CF'] * Weff * 2

    # Total gate capacitance
    C_total = C_ox + C_gs + C_gd + C_fringe
    return {
        'C_ox': C_ox,
        'C_gs': C_gs,
        'C_gd': C_gd,
        'C_fringe': C_fringe,
        'C_total': C_total
    }

def calc_resistance(W, L, Vgs, params=None):
    """Estimate on-resistance R_on and series R_s/d."""
    p = default_params.copy()
    if params:
        p.update(params)
    Weff, Leff = effective_dimensions(W, L, p)

    # Channel mobility approx (neglect velocity saturation)
    mu_eff = 200e-4  # m^2/V·s, example value (200 cm^2/V·s)
    Cox = p['eps_ox'] / p['TOXE']
    Id_lin = mu_eff * Cox * (Weff/Leff) * (Vgs - calc_Vth(W,L,Vgs,0,0,p)) * 1e-3  # A at Vds=1mV
    R_on_channel = 1e-3 / Id_lin if Id_lin > 0 else np.inf

    # Series source/drain resistance
    Rs = (p['RDSW'] / Weff)  # Ω
    Rd = Rs

    return {
        'R_on_channel': R_on_channel,
        'R_source': Rs,
        'R_drain': Rd,
        'R_on_total': R_on_channel + Rs + Rd
    }



def mosfet_ids(Vgs, Vth, mu, Cox, W, L):
    """MOSFET saturation region current"""
    K = 0.5 * mu * Cox * (W / L)
    V_ov = Vgs - Vth
    return K * V_ov**2 if V_ov > 0 else 0

def neuron_mos_dynamics(y, t, params):
    u, v = y
    Vgs = u  # assuming Vgs is mapped from membrane potential
    Vth, mu, Cox, W, L = params['Vth'], params['mu'], params['Cox'], params['W'], params['L']
    
    Ids = mosfet_ids(Vgs, Vth, mu, Cox, W, L)
    
    du_dt = -u + Ids
    dv_dt = -v + u**2  # simple f(u), can be replaced
    
    return [du_dt, dv_dt]


def dvdu_with_current(y, t, params):
    u, v = y
    Vgs, Vds, mu, Cox, W, L, I_inj = params
    Vth = calc_Vth(W, L, Vgs, Vbs=0.0, Vds=Vds)
    Ids = mosfet_ids(Vgs, Vth, mu, Cox, W, L)
    du_dt = -u + Ids + I_inj
    dv_dt = -v + u**2
    return [du_dt, dv_dt]

class BsimMosNeuronApp:
    def __init__(self, root):
        self.root = root
        root.title("BSIM-MOS Izhikevich Neuron Simulator")

        # Parameter frames
        param_frame = ttk.LabelFrame(root, text="MOS & BSIM Parameters")
        param_frame.pack(padx=10, pady=5, fill="x")

        self.entries = {}
        for label, default in [("mu (m^2/Vs)", 200e-4), ("Cox (F/m^2)", 3.45e-3)]:
            ttk.Label(param_frame, text=label).pack(anchor="w")
            entry = ttk.Entry(param_frame)
            entry.insert(0, str(default))
            entry.pack(fill="x")
            self.entries[label] = entry

        # Search ranges
        range_frame = ttk.LabelFrame(root, text="Search Ranges")
        range_frame.pack(padx=10, pady=5, fill="x")

        for label, default in [("W_min (µm)", 0.5), ("W_max (µm)", 5.0),
                               ("L_min (µm)", 0.5), ("L_max (µm)", 5.0),
                               ("I_min (µA)", 0.1), ("I_max (µA)", 10.0)]:
            ttk.Label(range_frame, text=label).pack(anchor="w")
            entry = ttk.Entry(range_frame)
            entry.insert(0, str(default))
            entry.pack(fill="x")
            self.entries[label] = entry

        bias_frame = ttk.LabelFrame(root, text="Bias Voltages")
        bias_frame.pack(padx=10, pady=5, fill="x")
        for label, default in [("Vgs (V)", 0.0), ("Vds (V)", 1.0)]:
            ttk.Label(bias_frame, text=label).pack(anchor="w")
            entry = ttk.Entry(bias_frame)
            entry.insert(0, str(default))
            entry.pack(fill="x")
            self.entries[label] = entry
        # Spike detection threshold
        ttk.Label(root, text="Spike Threshold (u) :").pack(anchor="w", padx=10)
        self.threshold_entry = ttk.Entry(root)
        self.threshold_entry.insert(0, "1.0")
        self.threshold_entry.pack(fill="x", padx=10)

        # Simulation settings
        ttk.Label(root, text="Sim time (ms) and dt:").pack(anchor="w", padx=10)
        sim_frame = ttk.Frame(root)
        sim_frame.pack(fill="x", padx=10)
        self.tmax_entry = ttk.Entry(sim_frame, width=10)
        self.tmax_entry.insert(0, "100")
        self.tmax_entry.pack(side="left")
        self.dt_entry = ttk.Entry(sim_frame, width=10)
        self.dt_entry.insert(0, "0.1")
        self.dt_entry.pack(side="left")

        # Action buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Find Min Params", command=self.find_min_params).pack()

        # Results display
        self.result_text = tk.Text(root, height=5, padx=10)
        self.result_text.pack(fill="x", padx=10, pady=5)

    def find_min_params(self):
        try:
            mu = float(self.entries["mu (m^2/Vs)"].get())
            Cox = float(self.entries["Cox (F/m^2)"].get())
            W_min = float(self.entries["W_min (µm)"].get()) * 1e-6
            W_max = float(self.entries["W_max (µm)"].get()) * 1e-6
            L_min = float(self.entries["L_min (µm)"].get()) * 1e-6
            L_max = float(self.entries["L_max (µm)"].get()) * 1e-6
            I_min = float(self.entries["I_min (µA)"].get()) * 1e-6
            I_max = float(self.entries["I_max (µA)"].get()) * 1e-6
            threshold = float(self.threshold_entry.get())
            tmax = float(self.tmax_entry.get())
            dt = float(self.dt_entry.get())
            Vgs = float(self.entries["Vgs (V)"].get())
            Vds = float(self.entries["Vds (V)"].get())
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric values.")
            return

        Ws = np.linspace(W_min, W_max, 10)
        Ls = np.linspace(L_min, L_max, 10)
        Is = np.linspace(I_min, I_max, 10)
        t = np.arange(0, tmax, dt)
        y0 = [0.0, 0.0]


        found = False
        for W in Ws:
            for L in Ls:
                for I in Is:
                    try:
                        params = (Vgs, Vds, mu, Cox, W, L, I)
                        # quietly ignore any overflow / divide warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            Vth = calc_Vth(W, L, Vgs, Vbs=0.0, Vds=Vds)
                            Ids = mosfet_ids(Vgs, Vth, mu, Cox, W, L)
                        # integrate…
                        sol = odeint(dvdu_with_current, y0, t, args=(params,))
                        u = sol[:,0]
                    except Exception as e:
                        # print to console so you still see what's going on
                        print(f"Skipped W={W:.2e}, L={L:.2e}: {e}")
                        continue

                    if np.max(u) > threshold:
                        # …report success and break out
                        self.result_text.delete(1.0, tk.END)
                        self.result_text.insert(tk.END,
                                               f"Spike detected!\nW = {W*1e6:.2f} µm, "
                                               f"L = {L*1e6:.2f} µm, I = {I*1e6:.2f} µA")
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "No spiking parameters found in the specified ranges.")

if __name__ == "__main__":
    root = tk.Tk()
    app = BsimMosNeuronApp(root)
    root.mainloop()
