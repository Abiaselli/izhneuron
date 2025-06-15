
import os
import subprocess
import numpy as np
import cma

MODEL_DIR = r"C:/Users/abias/.cursor-tutor/vccdoe/forml"
NGSPICE_PATH = r"C:\msys64\Spice64\bin\ngspice.exe"
SPICE_FILE = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/izhikevich_ngspice_unique_models.sp"
RAW_FILE = r"C:/Users/abias/.cursor-tutor/vccdoe/forml/neuron_output.txt"

NUM_DEVICES = 9
PARAM_KEYS = ['VTO', 'KP', 'U0', 'RD', 'CGSO', 'CGDO']
PARAMS_PER_DEVICE = len(PARAM_KEYS)
TOTAL_PARAMS = 2 * NUM_DEVICES * PARAMS_PER_DEVICE

def run_ngspice():
    result = subprocess.run([NGSPICE_PATH, "-b", SPICE_FILE], capture_output=True, text=True)
    if result.returncode != 0:
        print("SPICE error:", result.stderr)
        return False
    return True

def parse_ascii_wrdata(filepath):
    try:
        data = np.loadtxt(filepath)
        t = data[:, 0]
        v_n017 = data[:, 1]
        v_n012 = data[:, 3]
        return {
            "time": t,
            "v(x1.n017)": v_n017,
            "v(x1.n012)": v_n012
        }
    except Exception as e:
        print("Failed to parse ASCII wrdata:", e)
        return None

def render_model(name, type_, param_dict):
    lines = [f".model {name} {type_}"]
    for k, v in param_dict.items():
        lines.append(f"+   {k} = {v:.5e}")
    lines.append("+   NSUB = 1e16\n+   TNOM = 27")
    return '\n'.join(lines)

def write_model_files(params):
    os.makedirs(MODEL_DIR, exist_ok=True)
    idx = 0
    for i in range(1, NUM_DEVICES + 1):
        p = dict(zip(PARAM_KEYS, params[idx:idx+PARAMS_PER_DEVICE]))
        idx += PARAMS_PER_DEVICE
        with open(os.path.join(MODEL_DIR, f"model_NM{i}.nmos"), "w") as f:
            f.write(render_model(f"NM{i}", "nmos", p))
    for i in range(1, NUM_DEVICES + 1):
        p = dict(zip(PARAM_KEYS, params[idx:idx+PARAMS_PER_DEVICE]))
        idx += PARAMS_PER_DEVICE
        with open(os.path.join(MODEL_DIR, f"model_PM{i}.pmos"), "w") as f:
            f.write(render_model(f"PM{i}", "pmos", p))

def score_output(data):
    if data is None:
        return 1e6
    u_vals = data.get("v(x1.n012)", [])
    v_vals = data.get("v(x1.n017)", [])
    if len(u_vals) == 0 or len(v_vals) == 0:
        return 1e6
    u_vals = np.array(u_vals)
    v_vals = np.array(v_vals)
    u_spikes = (u_vals > 0.1).astype(int)
    spike_count_u = np.sum(np.diff(u_spikes) == 1)
    score = abs(np.max(v_vals) - 0.18) + abs(np.min(v_vals) - 0.0)
    score += abs(spike_count_u - 3) * 100
    return score

def fitness_function(params):
    write_model_files(params)
    if not run_ngspice():
        return 1e6
    data = parse_ascii_wrdata(RAW_FILE)
    return score_output(data)

def run_cma_es():
    x0 = []
    lower = []
    upper = []
    for _ in range(2 * NUM_DEVICES):
        x0 += [0.5, 1e-6, 600, 0.001, 5e-10, 1e-10]
        lower += [0.2, 1e-7, 200, 0.0001, 1e-10, 1e-11]
        upper += [0.8, 1e-4, 1000, 0.01, 1e-9, 1e-9]

    sigma = 0.1
    es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': [lower, upper]})
    while not es.stop():
        solutions = es.ask()
        scores = [fitness_function(x) for x in solutions]
        es.tell(solutions, scores)
        es.logger.add()
        es.disp()
    res = es.result
    print("Best score:", res.fbest)
    print("Best params:", res.xbest)

if __name__ == "__main__":
    run_cma_es()
