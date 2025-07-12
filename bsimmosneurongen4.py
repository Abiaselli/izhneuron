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

# 32 nm PTM NMOS defaults (for M1, M4 & M6) 
default_params_nmos = {
    'VTH0':   0.63,        # vth0             :contentReference[oaicite:0]{index=0}
    'K1':     0.4,         # k1               :contentReference[oaicite:1]{index=1}
    'K2':     0.0,         # k2               :contentReference[oaicite:2]{index=2}
    'K3':     0.0,         # k3               :contentReference[oaicite:3]{index=3}
    'K3B':    0.0,         # k3b              :contentReference[oaicite:4]{index=4}
    'W0':     2.5e-6,      # w0               :contentReference[oaicite:5]{index=5}
    'DVT0':   1.0,         # dvt0             :contentReference[oaicite:6]{index=6}
    'DVT1':   2.0,         # dvt1             :contentReference[oaicite:7]{index=7}
    'DVT2':   0.0,         # dvt2             :contentReference[oaicite:8]{index=8}
    'DVT0W':  0.0,         # dvt0w            :contentReference[oaicite:9]{index=9}
    'DVT1W':  0.0,         # dvt1w            :contentReference[oaicite:10]{index=10}
    'DVT2W':  0.0,         # dvt2w            :contentReference[oaicite:11]{index=11}
    'DSUB':   0.1,         # dsub             :contentReference[oaicite:12]{index=12}
    'ETA0':   0.0115,      # eta0             :contentReference[oaicite:13]{index=13}
    'ETAB':   0.0,         # etab             :contentReference[oaicite:14]{index=14}
    'DVTP0':  1e-11,       # dvtp0            :contentReference[oaicite:15]{index=15}
    'DVTP1':  0.1,         # dvtp1            :contentReference[oaicite:16]{index=16}
    'TOXE':   1.6e-9,      # toxe             :contentReference[oaicite:17]{index=17}
    'eps_ox': 3.9*8.85e-12,
    'eps_si': 11.7*8.85e-12,
    'q':      1.602e-19,
    'ni':     1.45e16,
    'NDEP':   4.12e18,     # ndep             :contentReference[oaicite:18]{index=18}
    'LINT':   0.0,         # lint             :contentReference[oaicite:19]{index=19}
    'WINT':   5e-9,        # wint             :contentReference[oaicite:20]{index=20}
    'CGSO':   8.5e-11,     # cgso             :contentReference[oaicite:21]{index=21}
    'CGDO':   8.5e-11,     # cgdo             :contentReference[oaicite:22]{index=22}
    'CF':     0.0,
    'RDSW':   190.0,       # rdsw             :contentReference[oaicite:23]{index=23}
    'PCLM':   0.02,        # pclm             :contentReference[oaicite:24]{index=24}
    'T':      300.0
}

# 32 nm PTM PMOS defaults (for M2 & M7)
default_params_pmos = {
    'VTH0':   -0.5808,     # vth0             :contentReference[oaicite:25]{index=25}
    'K1':     0.4,         # k1               :contentReference[oaicite:26]{index=26}
    'K2':     -0.01,       # k2               :contentReference[oaicite:27]{index=27}
    'K3':     0.0,
    'K3B':    0.0,
    'W0':     2.5e-6,      # w0               :contentReference[oaicite:28]{index=28}
    'DVT0':   1.0,         # dvt0             :contentReference[oaicite:29]{index=29}
    'DVT1':   2.0,         # dvt1             :contentReference[oaicite:30]{index=30}
    'DVT2':   -0.032,      # dvt2             :contentReference[oaicite:31]{index=31}
    'DVT0W':  0.0,
    'DVT1W':  0.0,
    'DVT2W':  0.0,
    'DSUB':   0.1,         # dsub             :contentReference[oaicite:32]{index=32}
    'ETA0':   0.0115,      # eta0             :contentReference[oaicite:33]{index=33}
    'ETAB':   0.0,
    'DVTP0':  1e-11,       # dvtp0            :contentReference[oaicite:34]{index=34}
    'DVTP1':  0.05,        # dvtp1            :contentReference[oaicite:35]{index=35}
    'TOXE':   1.62e-9,     # toxe             :contentReference[oaicite:36]{index=36}
    'eps_ox': 3.9*8.85e-12,
    'eps_si': 11.7*8.85e-12,
    'q':      1.602e-19,
    'ni':     1.45e16,
    'NDEP':   3.07e18,     # ndep             :contentReference[oaicite:37]{index=37}
    'LINT':   0.0,         # lint             :contentReference[oaicite:38]{index=38}
    'WINT':   5e-9,        # wint             :contentReference[oaicite:39]{index=39}
    'CGSO':   8.5e-11,     # cgso             :contentReference[oaicite:40]{index=40}
    'CGDO':   8.5e-11,     # cgdo             :contentReference[oaicite:41]{index=41}
    'CF':     0.0,
    'RDSW':   240.0,       # rdsw             :contentReference[oaicite:42]{index=42}
    'PCLM':   0.12,        # pclm             :contentReference[oaicite:43]{index=43}
    'T':      300.0
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

def dvdu_wijekoon(y, t, sim):
    V, U = y
    # unpack simulation settings
    Vds = sim["Vds"]
    I    = sim["I_inj"]
    α, β, γ = sim["alpha"], sim["beta"], sim["gamma"]
    Cv, Cu = sim["Cv"], sim["Cu"]
    # the five W/L’s
    WL1, WL2 = sim["WL_M1"], sim["WL_M2"]
    WL4, WL6 = sim["WL_M4"], sim["WL_M6"]
    WL7     = sim["WL_M7"]

    # compute currents through each MOS:
    Vth1 = calc_Vth(WL1[0], WL1[1], V, 0.0, Vds,
                    params=default_params_nmos)
    I1    = mosfet_ids(V    , Vth1, sim["mu"], sim["Cox"], *WL1)

    Vth4 = calc_Vth(WL4[0], WL4[1], U, 0.0, Vds,
                    params=default_params_nmos)
    I4   = mosfet_ids(U    , Vth4, sim["mu"], sim["Cox"], *WL4)

    Vth2 = calc_Vth(WL2[0], WL2[1], V, 0.0, Vds,
                    params=default_params_pmos)
    # in a PMOS you may invert voltages & swap signs as needed…
    I2   = mosfet_ids(V    , Vth2, sim["mu"], sim["Cox"], *WL2)

    Vth6 = calc_Vth(WL6[0], WL6[1], U, 0.0, Vds,
                    params=default_params_nmos)
    I6   = mosfet_ids(U    , Vth6, sim["mu"], sim["Cox"], *WL6)

    Vth7 = calc_Vth(WL7[0], WL7[1], V, 0.0, Vds,
                    params=default_params_pmos)
    I7   = mosfet_ids(V    , Vth7, sim["mu"], sim["Cox"], *WL7)

    k = sim["mu"] * sim["Cox"]

    # piecewise dV/dt:
    if V >= U - sim["Vt"]:
        dVdt = (k/Cv) * ( α*(0.5*WL1[0]/WL1[1]*(V-sim["Vt"])**2)
                        -β*(0.5*WL4[0]/WL4[1]*(U-sim["Vt"])**2)
                        + I/k )
    else:
        dVdt = (k/Cv) * ( α*(0.5*WL1[0]/WL1[1]*(V-sim["Vt"])**2)
                        -β*(0.5*WL4[0]/WL4[1]*(U-sim["Vt"])**2)
                        + I/k )

    # slow variable:
    dUdt = (k/Cu) * ( α*(0.5 * WL1[0]/WL1[1]
                         * WL2[1]/WL2[0]   # note (L/W) for M2
                         * WL7[0]/WL7[1]
                         * (V-sim["Vt"])**2)
                     -γ*(0.5 * WL6[0]/WL6[1] * (U-sim["Vt"])**2) )

    return [dVdt, dUdt]

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
        # in BsimMosNeuronApp.__init__():
        tx_frame = ttk.LabelFrame(root, text="Transistor W/L Ranges (µm)")
        tx_frame.pack(padx=10, pady=5, fill="x")

        # define each device with a label and default (W,L) min/max
        self.tx_names = ["M1", "M2", "M4", "M6", "M7"]
        defaults = {
            "M1": (2.3, 1.0),
            "M2": (2.3, 1.0),
            "M4": (1.3, 2.0),
            "M6": (1.3, 18.0),
            "M7": (1.3, 14.0),
        }

        self.tx_entries = {}
        for name in self.tx_names:
            lf = ttk.LabelFrame(tx_frame, text=name)
            lf.pack(fill="x", padx=5, pady=2)
            wmin, lmin = defaults[name]
            for lab, val in [("W_min", wmin), ("W_max", wmin*1.2),
                            ("L_min", lmin), ("L_max", lmin*1.2)]:
                ttk.Label(lf, text=f"{lab} (µm)").pack(side="left")
                e = ttk.Entry(lf, width=6); e.insert(0, str(val)); e.pack(side="left", padx=2)
                self.tx_entries[(name, lab)] = e

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
        # build per‐transistor W/L lists (in meters!)
        tx_WL = {}
        n_steps = 5  # e.g. sample 5 points each
        for name in self.tx_names:
            Wmin = float(self.tx_entries[(name,"W_min")].get()) * 1e-6
            Wmax = float(self.tx_entries[(name,"W_max")].get()) * 1e-6
            Lmin = float(self.tx_entries[(name,"L_min")].get()) * 1e-6
            Lmax = float(self.tx_entries[(name,"L_max")].get()) * 1e-6
            tx_WL[name] = (
            np.linspace(Wmin, Wmax, n_steps),
            np.linspace(Lmin, Lmax, n_steps)
            )

        # a list of dicts, one per combination:
        combo_list = []
        for W1, L1 in zip(*tx_WL["M1"]):
            for W2, L2 in zip(*tx_WL["M2"]):
                for W4, L4 in zip(*tx_WL["M4"]):
                    for W6, L6 in zip(*tx_WL["M6"]):
                        for W7, L7 in zip(*tx_WL["M7"]):
                            combo_list.append({
                                "WL_M1":(W1,L1),
                                "WL_M2":(W2,L2),
                                "WL_M4":(W4,L4),
                                "WL_M6":(W6,L6),
                                "WL_M7":(W7,L7),
                            })

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

import torch
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def char_length_torch(p, Vbs=0.0, is_width_dep=False):
    phi_s = calc_phi_s(p)
    eps_si, eps_ox = p['eps_si'], p['eps_ox']
    factor = 1 + (p['DVT2W'] if is_width_dep else p['DVT2']) * Vbs
    return (eps_si * p['TOXE'] * math.sqrt(2*eps_si*phi_s/(p['q']*p['NDEP']))) / (eps_ox * factor)

_MAX_COSH_ARG = 80.0  # safe for torch.float32

def _safe_cosh_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.cosh(torch.clamp(x, -_MAX_COSH_ARG, _MAX_COSH_ARG))

def calc_Vth_torch(W, L, Vgs, Vbs, Vds, p):
    # Merge BSIM contributions into threshold voltage Vth
    # W, L, Vgs, Vbs, Vds may be torch tensors [B]
    # p is a Python dict of floats
    Weff = W - 2*p['WINT']
    Leff = L - 2*p['LINT']
    k_b = 1.38064852e-23
    Vt_th = k_b * p['T'] / p['q']
    phi_s = calc_phi_s(p)
    Vbi   = calc_Vbi(p)

    # Body effect
    dVth_body = p['K1']*(torch.sqrt(phi_s+Vbs)-math.sqrt(phi_s)) - p['K2']*Vbs

    # Short-channel effect
    lt0 = char_length_torch(p, Vbs, is_width_dep=False)
    arg = p['DVT1']*Leff/lt0
    theta_sce = 0.5*p['DVT0']/( _safe_cosh_torch(arg) - 1 + 1e-12 )
    dVth_sce = -theta_sce * (Vbi - phi_s)

    # Narrow-width
    dVth_nw1 = (p['K3'] + p['K3B']*Vbs) * (p['TOXE']*phi_s)/(Weff + p['W0'] + 1e-12)
    dVth_nw2 = torch.zeros_like(Weff)
    if p['DVT0W'] != 0.0:
        ltW = char_length_torch(p, Vbs, is_width_dep=True)
        argW = p['DVT1W']*Leff/ltW
        theta_nw = 0.5*p['DVT0W']/( _safe_cosh_torch(argW) - 1 + 1e-12 )
        dVth_nw2 = -theta_nw * (Vbi - phi_s)

    # DIBL
    dVth_dibl = torch.zeros_like(Weff)
    if torch.any(Vds != 0.0):
        theta_dibl = 0.5 / (_safe_cosh_torch(p['DSUB']*Leff/lt0) - 1 + 1e-12)
        dVth_dibl = -theta_dibl * (p['ETA0'] + p['ETAB']*Vbs) * Vds

    # DITS
    dVth_dits = torch.zeros_like(Weff)
    if p['DVTP0'] != 0.0 and torch.any(Vds != 0.0):
        dVth_dits = -Vt_th * torch.log(
            Leff/(Leff + p['DVTP0']*(1 + torch.exp(-p['DVTP1']*Vds))) + 1e-12
        )

    Vth = p['VTH0'] + dVth_body + dVth_sce + dVth_nw1 + dVth_nw2 + dVth_dibl + dVth_dits
    return Vth
def batch_euler_step(VU, sim, dt):
    # VU: [batch,2] tensor of [V, U] per combination
    V, U = VU[:,0], VU[:,1]
    # unpack sim constants as scalars or tensors on device
    k    = sim["mu"] * sim["Cox"]
    Cv, Cu = sim["Cv"], sim["Cu"]
    Vt   = sim["Vt"]
    alpha, beta, gamma = sim["alpha"], sim["beta"], sim["gamma"]
    Iinj = sim["I_inj"]
    Vds  = sim["Vds"]

    n_steps = 5
    tx_WL = {}
    for name in self.tx_names:
            Wmin = float(self.tx_entries[(name,"W_min")].get())*1e-6
            Wmax = float(self.tx_entries[(name,"W_max")].get())*1e-6
            Lmin = float(self.tx_entries[(name,"L_min")].get())*1e-6
            Lmax = float(self.tx_entries[(name,"L_max")].get())*1e-6
            tx_WL[name] = (np.linspace(Wmin,Wmax,n_steps), np.linspace(Lmin,Lmax,n_steps))
    # Pre‐stack the WL arrays:
    combo_list = []
    for W1,L1 in zip(*tx_WL["M1"]):
          for W2,L2 in zip(*tx_WL["M2"]):
            for W4,L4 in zip(*tx_WL["M4"]):
              for W6,L6 in zip(*tx_WL["M6"]):
                for W7,L7 in zip(*tx_WL["M7"]):
                  for Iinj in Is:
                    combo_list.append({
                      "WL_M1":(W1,L1),"WL_M2":(W2,L2),
                      "WL_M4":(W4,L4),"WL_M6":(W6,L6),
                      "WL_M7":(W7,L7),"I_inj":Iinj
                    })


    # A helper to compute Id across a batch:
    def batch_Id(Vgate, WL, default_params):
        W = WL[:,0]; L = WL[:,1]
        Vth = calc_Vth_torch(W, L, Vgate, Vds, default_params)  # see note below
        K   = 0.5 * sim["mu"] * sim["Cox"] * (W/L)
        Vov = Vgate - Vth
        return torch.clamp(Vov, min=0)**2 * K

    I1 = batch_Id(V, sim["WL_M1_tensor"], default_params_nmos)
    I4 = batch_Id(U, sim["WL_M4_tensor"], default_params_nmos)
    I2 = batch_Id(V, sim["WL_M2_tensor"], default_params_pmos)
    I6 = batch_Id(U, sim["WL_M6_tensor"], default_params_nmos)
    I7 = batch_Id(V, sim["WL_M7_tensor"], default_params_pmos)

    # piecewise dV/dt:
    cond = V >= (U - Vt)
    dVdt = torch.where(
      cond,
      (k/Cv)*( alpha*(0.5*WL1[:,0]/WL1[:,1])*(V - Vt)**2
             - beta*(0.5*sim["WL_M4_tensor"][:,0]/sim["WL_M4_tensor"][:,1])*(U - Vt)**2
             + Iinj/k),
      # else path (you may customize)
      (k/Cv)*( alpha*(0.5*WL1[:,0]/WL1[:,1])*(V - Vt)**2
             - beta*(0.5*sim["WL_M4_tensor"][:,0]/sim["WL_M4_tensor"][:,1])*(U - Vt)**2
             + Iinj/k)
    )
    dUdt = (k/Cu)*( alpha*(0.5*WL1[:,0]/WL1[:,1])
                       *(sim["WL_M2_tensor"][:,1]/sim["WL_M2_tensor"][:,0])
                       *(sim["WL_M7_tensor"][:,0]/sim["WL_M7_tensor"][:,1])
                       *(V - Vt)**2)- gamma*(0.5*sim["WL_M6_tensor"][:,0]/sim["WL_M6_tensor"][:,1])*(U - Vt)**2

    V_next = V + dt*dVdt
    U_next = U + dt*dUdt
    return torch.stack([V_next, U_next], dim=1)

def run_batch_sim(combo_list, sim, tmax, dt):
    B = len(combo_list)
    # prepare sim‐constant tensors on device
    for dev in ["M1","M2","M4","M6","M7"]:
      WL = torch.tensor([c[f"WL_{dev}"] for c in combo_list], device=device)
      sim[f"WL_{dev}_tensor"] = WL

    VU = torch.zeros(B,2, device=device)  # initial [V=0, U=0]
    steps = int(tmax/dt)
    spike_mask = torch.zeros(B, device=device, dtype=torch.bool)

    for _ in range(steps):
        VU = batch_euler_step(VU, sim, dt)
        spike_mask |= (VU[:,1] > sim["spike_threshold"])  # U-axis spike test

    return spike_mask.cpu().numpy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BsimMosNeuronApp(root)
    root.mainloop()
