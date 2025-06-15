
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import cma
import subprocess

MODEL_DIR = r"C:/Users/abias/.cursor-tutor/vccdoe/forml"
NGSPICE_PATH = r"C:\msys64\Spice64\bin\ngspice.exe"
SPICE_FILE = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/izhikevich_ngspice_unique_models.sp"
RAW_FILE = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/neuron_output.txt"
NUM_DEVICES = 9
PARAM_KEYS = ['VTO', 'KP', 'LAMBDA', 'PHI', 'GAMMA', 'RD', 'RS', 'IS',
              'CGBO', 'CGDO', 'CGSO', 'CBD', 'CBS', 'PB', 'RSH', 'CJ',
              'MJ', 'CJSW', 'MJSW', 'JS', 'TOX', 'NSS', 'TPG', 'LD',
              'U0', 'KF', 'AF', 'FC']
PARAMS_PER_DEVICE = len(PARAM_KEYS)
TOTAL_PARAMS = 2 * NUM_DEVICES * PARAMS_PER_DEVICE

# Fixed geometry map
GEOM = {
    'NM1': (1e-7, 2e-7), 'NM2': (1.25e-7, 1.3e-6), 'NM3': (9e-8, 6.5e-7),
    'NM4': (1.25e-7, 3e-7), 'NM5': (1.25e-7, 3e-7), 'NM6': (1e-7, 1e-6),
    'NM7': (1e-7, 1e-7), 'NM8': (1e-7, 2e-7), 'NM9': (1e-7, 1e-7),
    'PM1': (3.5e-7, 1.75e-7), 'PM2': (1.5e-7, 7e-7), 'PM3': (6e-7, 4e-7),
    'PM4': (1.25e-7, 3e-7), 'PM5': (1e-7, 1e-7), 'PM6': (1e-7, 1e-6),
    'PM7': (1e-7, 1e-7), 'PM8': (1e-7, 1e-7), 'PM9': (2.5e-7, 1e-7)
}

def parse_ascii_wrdata(file_path):
    try:
        data = np.loadtxt(file_path)
        time = data[:, 0]
        v017 = data[:, 1]
        v012 = data[:, 3]
        return {"time": time, "v(x1.n017)": v017, "v(x1.n012)": v012}
    except Exception as e:
        print(f"Failed to parse .wrdata: {e}")
        return None

def run_ngspice():
    try:
        result = subprocess.run([NGSPICE_PATH, "-b", "-o", "ngspice.log", SPICE_FILE], cwd=MODEL_DIR, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"SPICE execution failed: {e}")
        return False

def render_model(name, type_, param_dict):
    lines = [f".model {name} {type_}"]
    for k, v in param_dict.items():
        lines.append(f"+   {k} = {v:.5e}")
    w, l = GEOM[name]

    # Ensure Pd and Ps are ≥ W
    pd = max(param_dict.get('RD', 0.0), w)
    ps = max(param_dict.get('RS', 0.0), w)

    lines.append(f"+   W = {w:.5e}")
    lines.append(f"+   L = {l:.5e}")
    lines.append(f"+   Pd = {pd:.5e}")
    lines.append(f"+   Ps = {ps:.5e}")
    lines.append("+   NSUB = 1e16")
    lines.append("+   TNOM = 27")
    return '\n'.join(lines)


def write_model_files(params):
    os.makedirs(MODEL_DIR, exist_ok=True)
    idx = 0
    for i in range(1, NUM_DEVICES + 1):
        name = f"NM{i}"
        p = dict(zip(PARAM_KEYS, params[idx:idx+PARAMS_PER_DEVICE]))
        idx += PARAMS_PER_DEVICE
        with open(os.path.join(MODEL_DIR, f"model_{name}.nmos"), "w") as f:
            f.write(render_model(name, "nmos", p))
    for i in range(1, NUM_DEVICES + 1):
        name = f"PM{i}"
        p = dict(zip(PARAM_KEYS, params[idx:idx+PARAMS_PER_DEVICE]))
        idx += PARAMS_PER_DEVICE
        with open(os.path.join(MODEL_DIR, f"model_{name}.pmos"), "w") as f:
            f.write(render_model(name, "pmos", p))

def fuzzy_score_from_data(time, v_017, v_012, debug=False):
    def compute_metrics(time, voltage, target_rate_hz, min_peak_height):
        duration_ms = (time[-1] - time[0]) * 1e3
        peaks, props = find_peaks(voltage, height=min_peak_height)
        rate = len(peaks) / (duration_ms / 1000.0)
        peak_heights = props['peak_heights'] if 'peak_heights' in props else np.zeros(len(peaks))
        height_error = np.mean(np.abs(peak_heights - min_peak_height)) if len(peak_heights) else 1.0
        rate_error = np.abs(rate - target_rate_hz)
        return rate_error, height_error, peaks

    try:
        dt = time[1] - time[0]
        freqs_017, fft_017 = rfftfreq(len(v_017), dt), np.abs(rfft(v_017))
        freqs_012, fft_012 = rfftfreq(len(v_012), dt), np.abs(rfft(v_012))
        dom_freq_017 = freqs_017[np.argmax(fft_017)]
        dom_freq_012 = freqs_012[np.argmax(fft_012)]
    except:
        return 1e6

    rate_err_017, height_err_017, peaks_017 = compute_metrics(time, v_017, 4/0.0012, 0.180)
    rate_err_012, height_err_012, peaks_012 = compute_metrics(time, v_012, 3/0.001, 0.110)
    freq_error = (np.abs(dom_freq_017 - 3333) + np.abs(dom_freq_012 - 3000)) / 1000.0
    score = rate_err_017 + rate_err_012 + 5 * (height_err_017 + height_err_012) + freq_error

    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time * 1e3, v_017 * 1e3)
        plt.plot(time[peaks_017] * 1e3, v_017[peaks_017] * 1e3, "x")
        plt.title("v(x1.n017)")
        plt.subplot(2, 1, 2)
        plt.plot(time * 1e3, v_012 * 1e3, color="orange")
        plt.plot(time[peaks_012] * 1e3, v_012[peaks_012] * 1e3, "x", color="red")
        plt.title("v(x1.n012)")
        plt.tight_layout()
        plt.show()

    return score

def fitness_function(params, debug=False):
    write_model_files(params)
    if not run_ngspice():
        return 1e6
    data = parse_ascii_wrdata(RAW_FILE)
    if data is None:
        return 1e6
    return fuzzy_score_from_data(data["time"], data["v(x1.n017)"], data["v(x1.n012)"], debug=debug)

def run_cma_es():
    
    x0 = []
    lower = []
    upper = []
    for _ in range(2 * NUM_DEVICES):
        x0 += [0.5, 1e-6, 600, 0.001, 5e-10, 1e-10, 1e-10, 2e-10, 0, 2e-10,
               5e-10, 4e-10, 0, 0.8, 0, 0, 0.5, 0, 0.5, 0, 2e-6, 0, 1, 0,
               600, 0, 1, 0.5]
        lower += [0.2, 1e-8, 200, 1e-5] + [0]*24
        upper += [1.0, 1e-3, 1000, 0.1] + [1]*24
    lower = np.array(lower)
    upper = np.array(upper)

    # Initial solution clipped to bounds
    x0 = np.clip(x0, lower, upper)

    opts = {'bounds': [lower, upper], 'scaling_of_variables': np.array(upper) - np.array(lower)}
    es = cma.CMAEvolutionStrategy(x0, 0.01, opts)
    while not es.stop():
        solutions = es.ask()
        scores = [fitness_function(x) for x in solutions]
        es.tell(solutions, scores)
        es.disp()
    print("Best score:", es.best.f)
    print("Best params:", es.best.x)
    fitness_function(es.best.x, debug=True)

if __name__ == "__main__":
    run_cma_es()
