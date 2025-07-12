import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


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

def simulate(params, tmax=100, dt=0.1):
    t = np.arange(0, tmax, dt)
    y0 = [0.0, 0.0]  # Initial values for u and v
    sol = odeint(neuron_mos_dynamics, y0, t, args=(params,))
    return t, sol


def dvdu_system(y, t, params):
    u, v = y
    Vgs, Vds, Vt, mu, Cox, W, L, a, b = params
    
    Ids = mosfet_ids(Vgs, Vds, Vt, mu, Cox, W, L)
    
    du_dt = a * (b * v - u) + Ids
    dv_dt = -v + u**2  # example dynamics

    return [du_dt, dv_dt]

def run_sim(params, tmax=100, dt=0.1):
    t = np.arange(0, tmax, dt)
    y0 = [0, 0]  # initial u, v
    sol = odeint(dvdu_system, y0, t, args=(params,))
    return t, sol

class MOSSimApp:
    def __init__(self, root):
        self.root = root
        root.title("MOSFET du/dv Simulator")

        # Input fields
        self.entries = {}
        for label in ["Vth", "mu", "Cox", "W", "L"]:
            tk.Label(root, text=label).pack()
            entry = tk.Entry(root)
            entry.insert(0, "1.0")
            entry.pack()
            self.entries[label] = entry

        self.run_button = tk.Button(root, text="Run Simulation", command=self.run)
        self.run_button.pack()

        # Canvas for plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    def run(self):
        try:
            vals = {key: float(self.entries[key].get()) for key in self.entries}
            t, sol = simulate(vals)
            self.ax.clear()
            self.ax.plot(t, sol[:, 0], label='u(t)')
            self.ax.plot(t, sol[:, 1], label='v(t)')
            self.ax.legend()
            self.ax.set_title("MOSFET du/dv Simulation")
            self.canvas.draw()
        except Exception as e:
            print("Simulation error:", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = MOSSimApp(root)
    root.mainloop()
