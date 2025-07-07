import numpy as np

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
    theta_sce = 0.5*p['DVT0'] / (np.cosh(p['DVT1']*Leff/lt) - 1 + 1e-12)
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

# Example usage
if __name__ == "__main__":
    W = 1e-6  # 1 µm
    L = 1e-6  # 1 µm
    Vgs = 1.2  # Gate-source voltage
    Vbs = 0.0  # Bulk-source voltage
    Vds = 0.05 # Drain-source voltage

    params = {
        # Override or add model parameters here
        'VTH0': 0.65,
        'CGSO': 2e-10,  # Example gate-source overlap
        'CGDO': 2e-10,
        'CF':   5e-15
    }

    Vth = calc_Vth(W, L, Vgs, Vbs, Vds, params)
    caps = calc_capacitances(W, L, params)
    res = calc_resistance(W, L, Vgs, params)

    print(f"Calculated Vth: {Vth:.3f} V")
    print("Capacitances (F):", caps)
    print("Resistances (Ω):", res)
