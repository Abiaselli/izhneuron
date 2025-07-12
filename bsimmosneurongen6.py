import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import math
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# 1) BSIM4 parameters for 32 nm PTM
# ──────────────────────────────────────────────────────────────────────────────
default_params_nmos = {
    'VTH0':   0.63, 'K1': 0.4,   'K2': 0.0,   'K3': 0.0,   'K3B': 0.0,
    'W0':     2.5e-6, 'DVT0': 1.0, 'DVT1': 2.0, 'DVT2': 0.0,
    'DVT0W':  0.0,   'DVT1W': 0.0, 'DVT2W': 0.0,
    'DSUB':   0.1,   'ETA0': 0.0115,'ETAB':0.0,
    'DVTP0':  1e-11, 'DVTP1':0.1,
    'TOXE':   1.6e-9, 'eps_ox':3.9*8.85e-12, 'eps_si':11.7*8.85e-12,
    'q':      1.602e-19, 'ni':1.45e16, 'NDEP':4.12e18,
    'LINT':   0.0,   'WINT':5e-9,
    'CGSO':   8.5e-11,'CGDO':8.5e-11,'CF':0.0,
    'RDSW':   190.0, 'PCLM':0.02, 'T':300.0
}

default_params_pmos = {
    'VTH0':   -0.5808,'K1':0.4,   'K2':-0.01, 'K3':0.0,   'K3B':0.0,
    'W0':     2.5e-6, 'DVT0':1.0, 'DVT1':2.0, 'DVT2':-0.032,
    'DVT0W':  0.0,   'DVT1W':0.0, 'DVT2W':0.0,
    'DSUB':   0.1,   'ETA0':0.0115,'ETAB':0.0,
    'DVTP0':  1e-11, 'DVTP1':0.05,
    'TOXE':   1.62e-9,'eps_ox':3.9*8.85e-12,'eps_si':11.7*8.85e-12,
    'q':      1.602e-19,'ni':1.45e16,'NDEP':3.07e18,
    'LINT':   0.0,   'WINT':5e-9,
    'CGSO':   8.5e-11,'CGDO':8.5e-11,'CF':0.0,
    'RDSW':   240.0, 'PCLM':0.12, 'T':300.0
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Safe cosh to avoid overflow
# ──────────────────────────────────────────────────────────────────────────────
_MAX_COSH_ARG = 80.0  # safe for torch.float32

def _safe_cosh_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.cosh(torch.clamp(x, -_MAX_COSH_ARG, _MAX_COSH_ARG))

# ──────────────────────────────────────────────────────────────────────────────
# 3) BSIM helper functions (Torch versions)
# ──────────────────────────────────────────────────────────────────────────────
def calc_phi_s(p):
    # Surface potential approx (2·ΦF)
    k_b = 1.38064852e-23
    Vt_th = k_b * p['T'] / p['q']
    Phi_F = Vt_th * math.log(p['NDEP']/p['ni'])
    return 2*Phi_F

def calc_Vbi(p):
    # Built-in potential
    k_b = 1.38064852e-23
    Vt_th = k_b * p['T'] / p['q']
    return Vt_th * math.log((p['NDEP']**2)/(p['ni']**2))

def char_length_torch(p, Vbs=0.0, is_width_dep=False):
    phi_s = calc_phi_s(p)
    eps_si, eps_ox = p['eps_si'], p['eps_ox']
    factor = 1 + (p['DVT2W'] if is_width_dep else p['DVT2']) * Vbs
    return (eps_si * p['TOXE'] * math.sqrt(2*eps_si*phi_s/(p['q']*p['NDEP']))) / (eps_ox * factor)

def calc_Vth_torch(W, L, Vgs, Vbs, Vds, p):
    """
    W, L:  1D torch Tensors of shape [B]
    Vgs, Vbs, Vds: Python floats
    p: parameter dict of Python floats (BSIM4)
    Returns: Vth as a torch Tensor [B]
    """
    # Effective geometry
    Weff = W - 2*p['WINT']
    Leff = L - 2*p['LINT']

    # Scalar surface & built-in potentials
    phi_s = calc_phi_s(p)      # Python float
    Vbi    = calc_Vbi(p)       # Python float

    # Broadcast them to tensors of shape [B]
    phi_s_t = W.new_full(W.shape, phi_s)
    Vbi_t   = W.new_full(W.shape, Vbi)

    # 1) Body-effect
    dVth_body = (
        p['K1']*(torch.sqrt(phi_s_t + Vbs) - torch.sqrt(phi_s_t))
        - p['K2'] * Vbs
    )

    # 2) Short-channel effect (SCE)
    lt0 = char_length_torch(p, Vbs, is_width_dep=False)
    arg = p['DVT1'] * Leff / lt0
    theta_sce = 0.5 * p['DVT0'] / (_safe_cosh_torch(arg) - 1 + 1e-12)
    dVth_sce = -theta_sce * (Vbi_t - phi_s_t)

    # 3) Narrow-width effects
    #   a) first term
    dVth_nw1 = ((p['K3'] + p['K3B']*Vbs) * (p['TOXE']*phi_s_t)) \
               / (Weff + p['W0'] + 1e-12)
    #   b) optional second term
    if p['DVT0W'] != 0.0:
        ltW = char_length_torch(p, Vbs, is_width_dep=True)
        argW = p['DVT1W'] * Leff / ltW
        theta_nw = 0.5 * p['DVT0W'] / (_safe_cosh_torch(argW) - 1 + 1e-12)
        dVth_nw2 = -theta_nw * (Vbi_t - phi_s_t)
    else:
        dVth_nw2 = W.new_zeros(W.shape)

    # 4) DIBL
    if p['DSUB'] != 0.0 and Vds != 0.0:
        theta_dibl = 0.5 / (_safe_cosh_torch(p['DSUB']*Leff/lt0) - 1 + 1e-12)
        dVth_dibl = -theta_dibl * (p['ETA0'] + p['ETAB']*Vbs) * Vds
    else:
        dVth_dibl = W.new_zeros(W.shape)

    # 5) DITS
    k_b   = 1.38064852e-23
    Vt_th = k_b * p['T'] / p['q']
    if p['DVTP0'] != 0.0:
        # use math.exp on the scalar Vds
        exp_term = math.exp(-p['DVTP1'] * Vds)
        # Leff is a tensor, so adding a float broadcasts correctly
        arg2     = Leff / (Leff + p['DVTP0'] * (1 + exp_term)) + 1e-12
        dVth_dits = -Vt_th * torch.log(arg2)
    else:
        dVth_dits = W.new_zeros(W.shape)


    # Final threshold
    return (
        p['VTH0']
        + dVth_body
        + dVth_sce
        + dVth_nw1
        + dVth_nw2
        + dVth_dibl
        + dVth_dits
    )

# ──────────────────────────────────────────────────────────────────────────────
# 4) Batch integrate du/dv on GPU/CPU (Euler method)
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_Id_torch(Vgate, WL, p, sim):
    W, L = WL[:,0], WL[:,1]
    Vth = calc_Vth_torch(W, L, Vgate, 0.0, sim["Vds"], p)
    K   = 0.5 * sim["mu"] * sim["Cox"] * (W/L)
    Vov = torch.clamp(Vgate - Vth, min=0.0)
    return K * Vov**2

def batch_euler_step(VU, sim, dt):
    V, U = VU[:,0], VU[:,1]
    k   = sim["mu"] * sim["Cox"]
    Cv, Cu = sim["Cv"], sim["Cu"]
    Vt  = sim["Vt"]
    α, β, γ = sim["alpha"], sim["beta"], sim["gamma"]
    Iinj = sim["I_inj_tensor"]
    Vds  = sim["Vds"]

    # currents through each MOS
    I1 = batch_Id_torch(V, sim["WL_M1_tensor"], default_params_nmos, sim)
    I4 = batch_Id_torch(U, sim["WL_M4_tensor"], default_params_nmos, sim)
    I2 = batch_Id_torch(V, sim["WL_M2_tensor"], default_params_pmos, sim)
    I6 = batch_Id_torch(U, sim["WL_M6_tensor"], default_params_nmos, sim)
    I7 = batch_Id_torch(V, sim["WL_M7_tensor"], default_params_pmos, sim)

    # dV/dt piecewise
    cond = V >= (U - Vt)
    dVdt = torch.where(
        cond,
        (k/Cv)*(α*(0.5*sim["WL_M1_tensor"][:,0]/sim["WL_M1_tensor"][:,1])*(V - Vt)**2
               - β*(0.5*sim["WL_M4_tensor"][:,0]/sim["WL_M4_tensor"][:,1])*(U - Vt)**2
               + Iinj/k),
        (k/Cv)*(α*(0.5*sim["WL_M1_tensor"][:,0]/sim["WL_M1_tensor"][:,1])*(V - Vt)**2
               - β*(0.5*sim["WL_M4_tensor"][:,0]/sim["WL_M4_tensor"][:,1])*(U - Vt)**2
               + Iinj/k)
    )

    # dU/dt
    dUdt = (k/Cu)*(
        α*(0.5 * sim["WL_M1_tensor"][:,0]/sim["WL_M1_tensor"][:,1]
           * sim["WL_M2_tensor"][:,1]/sim["WL_M2_tensor"][:,0]
           * sim["WL_M7_tensor"][:,0]/sim["WL_M7_tensor"][:,1]
           * (V - Vt)**2)
        - γ*(0.5*sim["WL_M6_tensor"][:,0]/sim["WL_M6_tensor"][:,1]*(U - Vt)**2)
    )

    V_next = V + dt*dVdt
    U_next = U + dt*dUdt
    return torch.stack([V_next, U_next], dim=1)

def run_batch_sim(sim, tmax, dt):
    B = sim["I_inj_tensor"].shape[0]
    steps = int(tmax / dt)
    VU = torch.zeros(B,2, device=device)
    spike_mask = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(steps):
        VU = batch_euler_step(VU, sim, dt)
        spike_mask |= (VU[:,1] > sim["spike_threshold"])
    return spike_mask.cpu().numpy()

# ──────────────────────────────────────────────────────────────────────────────
# 5) Tkinter GUI with per‐transistor W/L sweeps and GPU acceleration
# ──────────────────────────────────────────────────────────────────────────────
class BsimMosNeuronApp:
    def __init__(self, root):
        self.root = root
        root.title("BSIM-MOS Izhikevich Neuron GPU Simulator")

        # MOS/BSIM params
        frame1 = ttk.LabelFrame(root, text="Technology Params")
        frame1.pack(padx=10, pady=5, fill="x")
        self.entries = {}
        for label, default in [("mu (m²/Vs)", 200e-4), ("Cox (F/m²)", 3.45e-3)]:
            ttk.Label(frame1, text=label).pack(anchor="w")
            e = ttk.Entry(frame1); e.insert(0, str(default)); e.pack(fill="x")
            self.entries[label] = e

        # Bias voltages
        frame2 = ttk.LabelFrame(root, text="Bias Voltages")
        frame2.pack(padx=10, pady=5, fill="x")
        for label, default in [("Vgs (V)", 0.0), ("Vds (V)", 1.0)]:
            ttk.Label(frame2, text=label).pack(anchor="w")
            e = ttk.Entry(frame2); e.insert(0, str(default)); e.pack(fill="x")
            self.entries[label] = e

        # Current sweep
        frame3 = ttk.LabelFrame(root, text="I_inj Sweep (µA)")
        frame3.pack(padx=10, pady=5, fill="x")
        for label, default in [("I_min", 0.1), ("I_max", 10.0)]:
            ttk.Label(frame3, text=label).pack(anchor="w")
            e = ttk.Entry(frame3); e.insert(0, str(default)); e.pack(fill="x")
            self.entries[label] = e

        # Transistor W/L ranges
        self.tx_names = ["M1","M2","M4","M6","M7"]
        defaults = {"M1":(2.3,1.0),"M2":(2.3,1.0),
                    "M4":(1.3,2.0),"M6":(1.3,18.0),"M7":(1.3,14.0)}
        frame4 = ttk.LabelFrame(root, text="Transistor W/L Ranges (µm)")
        frame4.pack(padx=10, pady=5, fill="x")
        self.tx_entries = {}
        for name in self.tx_names:
            lf = ttk.LabelFrame(frame4, text=name); lf.pack(fill="x", padx=5, pady=2)
            w0,l0 = defaults[name]
            for lab, val in [("W_min",w0),("W_max",w0*1.2),
                             ("L_min",l0),("L_max",l0*1.2)]:
                ttk.Label(lf, text=f"{lab}").pack(side="left")
                e = ttk.Entry(lf, width=6); e.insert(0,str(val)); e.pack(side="left", padx=2)
                self.tx_entries[(name,lab)] = e

        # Simulation controls
        frame5 = ttk.Frame(root); frame5.pack(padx=10, pady=5, fill="x")
        ttk.Label(frame5, text="Spike Thresh (U)").pack(side="left")
        self.threshold_e = ttk.Entry(frame5); self.threshold_e.insert(0,"1.0"); self.threshold_e.pack(side="left")
        ttk.Label(frame5, text="Sim t (ms)").pack(side="left", padx=(10,0))
        self.tmax_e = ttk.Entry(frame5, width=6); self.tmax_e.insert(0,"100"); self.tmax_e.pack(side="left")
        ttk.Label(frame5, text="dt (ms)").pack(side="left")
        self.dt_e = ttk.Entry(frame5, width=6); self.dt_e.insert(0,"0.1"); self.dt_e.pack(side="left")

        # Run button & result
        btn = ttk.Button(root, text="Find Min Params", command=self.find_min_params)
        btn.pack(pady=5)
        self.result_text = tk.Text(root, height=4); self.result_text.pack(fill="x", padx=10)

    def find_min_params(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            mu   = float(self.entries["mu (m²/Vs)"].get())
            Cox  = float(self.entries["Cox (F/m²)"].get())
            Vgs  = float(self.entries["Vgs (V)"].get())
            Vds  = float(self.entries["Vds (V)"].get())
            I_min= float(self.entries["I_min"].get())*1e-6
            I_max= float(self.entries["I_max"].get())*1e-6
            thresh = float(self.threshold_e.get())
            tmax   = float(self.tmax_e.get())/1e3   # s
            dt     = float(self.dt_e.get())/1e3     # s
        except ValueError:
            return messagebox.showerror("Error","Check numeric inputs!")

        # build W/L arrays
        n_steps = 5
        tx_WL = {}
        for name in self.tx_names:
            Wmin = float(self.tx_entries[(name,"W_min")].get())*1e-6
            Wmax = float(self.tx_entries[(name,"W_max")].get())*1e-6
            Lmin = float(self.tx_entries[(name,"L_min")].get())*1e-6
            Lmax = float(self.tx_entries[(name,"L_max")].get())*1e-6
            tx_WL[name] = (np.linspace(Wmin,Wmax,n_steps), np.linspace(Lmin,Lmax,n_steps))

        Is = np.linspace(I_min,I_max,n_steps)
        # Cartesian product of W/L for each transistor and I_inj
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

        # assemble sim dict
        sim = {
          "mu":mu, "Cox":Cox,
          "alpha":1.0,"beta":1.0,"gamma":1.0,
          "Cv":0.1e-6,"Cu":1e-6,
          "Vt":0.5,"Vds":Vds,
          "spike_threshold":thresh
        }
        B = len(combo_list)
        # pre-stack tensors
        for dev in self.tx_names:
            sim[f"WL_{dev}_tensor"] = torch.tensor(
                [c[f"WL_{dev}"] for c in combo_list],
                dtype=torch.float32, device=device
            )
        sim["I_inj_tensor"] = torch.tensor(
            [c["I_inj"] for c in combo_list],
            dtype=torch.float32, device=device
        )

        # run batch sim
        mask = run_batch_sim(sim, tmax, dt)
        idxs = np.nonzero(mask)[0]
        self.result_text.delete(1.0,"end")
        if len(idxs)>0:
            i0 = idxs[0]
            c = combo_list[i0]
            self.result_text.insert("end",
                f"Spike @ I={c['I_inj']*1e6:.2f}µA  " +
                "  ".join(f"{dev} W/L={c[f'WL_{dev}'][0]*1e6:.2f}/"
                          f"{c[f'WL_{dev}'][1]*1e6:.2f}µm"
                          for dev in self.tx_names)
            )
        else:
            self.result_text.insert("end","No spiking combo found.")

if __name__ == "__main__":
    root = tk.Tk()
    app  = BsimMosNeuronApp(root)
    root.mainloop()
