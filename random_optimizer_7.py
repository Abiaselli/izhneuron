
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import subprocess


MODEL_DIR = r"C:/Users/abias/.cursor-tutor/vccdoe/forml"
NGSPICE_PATH = r"C:\msys64\Spice64\bin\ngspice.exe"
SPICE_FILE = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/izhikevich_ngspice_unique_models.sp"
RAW_FILE = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/neuron_output.txt"
NUM_DEVICES = 9

PARAM_KEYS = [
    "VTH0", "VFB", "K1", "K2", "K3", "U0", "UA", "UB", "UC", "ETA0",
    "DVT0", "DVT1", "DVT2", "DVT1W", "VSAT", "A0", "AGS", "B0", "B1",
    "KETA", "VOFF", "NFACTOR", "DSUB", "CDSC", "PCLM", "PDIBLC1",
    "PDIBLC2", "DROUT", "PSCBE1", "PSCBE2", "DELTA", "RDSW", "RSH",
    "RSHG", "CGSO", "CGDO", "CGBO", "AIGC", "BIGC", "CIGC", "AIGS",
    "BIGS", "CIGS", "RBPB", "RBPD", "RBPS", "RBDB", "RBSB", "KF",
    "AF", "EF", "NOIA", "NOIB", "NOIC"
]
PARAMS_PER_DEVICE = len(PARAM_KEYS)
TOTAL_PARAMS = 2 * NUM_DEVICES * PARAMS_PER_DEVICE

BSIM4_CONSTANTS = [
    "LEVEL = 14", "VERSION = 4.81", "BINUNIT = 2", "PARAMCHK = 1",
    "MOBMOD = 1", "MTRLMOD = 0", "MTRLCOMPATMOD = 0", "RDSMOD = 0",
    "IGCMOD = 0", "IGBMOD = 0", "CVCHARGEMOD = 0", "CAPMOD = 2",
    "RGATEMOD = 0", "RBODYMOD = 0", "FNOIMOD = 1", "TNOIMOD = 0",
    "DIOMOD = 1", "TEMPMOD = 0", "PERMOD = 1", "GEOMOD = 0"
]

def format_model_block(name, mtype, param_values, geometry):
    lines = [f".model {name} {name.lower()}_{mtype}"]
    lines += [f"+ {c}" for c in BSIM4_CONSTANTS]
    for i, key in enumerate(PARAM_KEYS):
        val = param_values[i]
        lines.append(f"+ {key} = {val:.6g}")
    W, L = geometry
    lines.append(f"+ W = {W:.6g}")
    lines.append(f"+ L = {L:.6g}")
    return "\n".join(lines)

def write_modified_netlist(base_path, target_path):
    with open(base_path, 'r') as f:
        lines = f.readlines()

    def random_pulse(start):
        high = 5
        delay = f"{start}n"
        rise = f"{random.randint(1, 20)}n"
        fall = f"{random.randint(1, 20)}n"
        width = f"{random.randint(1, 500)}n"
        period = f"{random.randint(100, 500)}u"
        return f"PULSE(0 {high} {delay} {rise} {fall} {width} {period})"

    updated_lines = []
    for line in lines:
        if line.strip().startswith("VREQ"):
            updated_lines.append(f"VREQ N016 0 {random_pulse(0)}\n")
        elif line.strip().startswith("VACK "):
            updated_lines.append(f"VACK N002 0 {random_pulse(1)}\n")
        elif "V§ACK1" in line or "V" in line and "ACK1" in line:
            updated_lines.append(f"VACK1 N026 0 {random_pulse(1)}\n")
        elif line.strip().startswith("Vc "):
            updated_lines.append(f"Vc N030 0 {random.uniform(0.5, 1.5):.2f}\n")
        elif line.strip().startswith("Vd "):
            updated_lines.append(f"Vd N006 0 {random.uniform(0.5, 1.5):.2f}\n")
        elif line.strip().startswith("Vdd1 "):
            updated_lines.append(f"Vdd1 N003 0 {random.uniform(0.9, 1.5):.2f}\n")
        elif line.strip().startswith("Vdd2 "):
            updated_lines.append(f"Vdd2 N001 0 {random.uniform(0.9, 1.5):.2f}\n")
        else:
            updated_lines.append(line)

    with open(target_path, 'w') as f:
        f.writelines(updated_lines)

def write_model_files(params, geometry):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i in range(NUM_DEVICES):
        n_name, p_name = f"NM{i+1}", f"PM{i+1}"
        n_params = params[i * PARAMS_PER_DEVICE:(i + 1) * PARAMS_PER_DEVICE]
        p_params = params[(NUM_DEVICES + i) * PARAMS_PER_DEVICE:(NUM_DEVICES + i + 1) * PARAMS_PER_DEVICE]
        n_geom = geometry[i]
        p_geom = geometry[NUM_DEVICES + i]

        with open(os.path.join(MODEL_DIR, f"model_{n_name}.nmos"), "w") as f:
            f.write(format_model_block(n_name, "nmos", n_params, n_geom))
        with open(os.path.join(MODEL_DIR, f"model_{p_name}.pmos"), "w") as f:
            f.write(format_model_block(p_name, "pmos", p_params, p_geom))


def run_ngspice():
    result = subprocess.run([NGSPICE_PATH, "-b", SPICE_FILE], capture_output=True, text=True)
    if result.returncode != 0:
        print("SPICE error:", result.stderr)
        return False
    return True
def parse_raw_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    data = np.array([[float(v) for v in line.strip().split()] for line in lines if line.strip()])
    return data[:, 0], data[:, 1], data[:, 2]
from scipy.interpolate import interp1d
import pandas as pd

# Load Izhikevich target waveform
izh_df = pd.read_csv(r"C:\Users\abias\.cursor-tutor\vccdoe\forml\izhikevich_waveform_data.csv")
eval_time = np.linspace(0, 6, 6000)
target_v = interp1d(izh_df["Time (ms)"], izh_df["V (mV)"], fill_value="extrapolate")(eval_time)
target_u = interp1d(izh_df["Time (ms)"], izh_df["U (mV)"], fill_value="extrapolate")(eval_time)

from scipy.fft import fft

def score_waveform(times, u_values, v_values):
    try:
        # Resample simulation waveforms to match target
        sim_v = interp1d(times, v_values, fill_value="extrapolate")(eval_time)
        sim_u = interp1d(times, u_values, fill_value="extrapolate")(eval_time)

        # Time-domain errors
        v_error_td = np.mean((sim_v * 1000 - target_v) ** 2)
        u_error_td = np.mean((sim_u * 1000 - target_u) ** 2)

        # Frequency-domain (Fourier) errors
        sim_v_fft = np.abs(fft(sim_v))
        tgt_v_fft = np.abs(fft(target_v))
        v_error_fd = np.mean((sim_v_fft - tgt_v_fft) ** 2)

        sim_u_fft = np.abs(fft(sim_u))
        tgt_u_fft = np.abs(fft(target_u))
        u_error_fd = np.mean((sim_u_fft - tgt_u_fft) ** 2)

        # Weighted combination
        score = 0.5 * (v_error_td + u_error_td) + 0.5 * (v_error_fd + u_error_fd)
        return score
    except Exception as e:
        print("Scoring failed:", e)
        return 1e6


def run_random_search(trials=25):
    best_score = float("inf")
    best_params = None
    best_geometry = None

    for t in range(trials):
        params = [random.uniform(0.01, 1.0) for _ in range(TOTAL_PARAMS)]
        geometry = [(random.uniform(100e-9, 1e-6), random.uniform(100e-9, 1e-6)) for _ in range(2 * NUM_DEVICES)]

        write_model_files(params, geometry)
        backup_netlist = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/backup/izhikevich_ngspice_unique_models.sp"
        active_netlist = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/izhikevich_ngspice_unique_models.sp"
        write_modified_netlist(backup_netlist, active_netlist)

        subprocess.run([NGSPICE_PATH, "-b", SPICE_FILE], stdout=subprocess.PIPE)

        try:
            with open(RAW_FILE, "r") as f:
                data = f.readlines()
                score = sum(abs(float(line.split()[1]) - float(line.split()[3])) for line in data)
        except:
            score = float("inf")

        if score < best_score:
            best_score = score
            best_params = params
            best_geometry = geometry

            # Save best netlist
            with open(active_netlist, "r") as src, open(os.path.join(MODEL_DIR, "best_netlist.sp"), "w") as dst:
                dst.writelines(src.readlines())

            print("New best found — saving netlist and model")

        print(f"Trial {t+1}/{trials} - Score: {score:.4f}")

    print("Best score:", best_score)
    if best_params:
        write_model_files(best_params, best_geometry)
        # Replace active netlist with the best one
        import shutil
        shutil.copyfile(os.path.join(MODEL_DIR, "best_netlist.sp"), active_netlist)


run_random_search(50)
