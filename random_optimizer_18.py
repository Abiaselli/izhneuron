import os
import re
import random
import shutil
import subprocess

# Paths & constants
NMOS_MODEL_SRC = r"C:\Users\abias\.cursor-tutor\vccdoe\forml\65nm_bulk_nmos.model"
PMOS_MODEL_SRC = r"C:\Users\abias\.cursor-tutor\vccdoe\forml\65nm_bulk_pmos.model"
BASE_NETLIST   = r"C:\Users\abias\.cursor-tutor\vccdoe\forml\backup\newizhtest3.sp"
MODEL_DIR      = r"C:\Users\abias\.cursor-tutor\vccdoe\forml\random_search_models"
NGSPICE_PATH   = r"C:\msys64\Spice64\bin\ngspice.exe"
RAW_OUTPUT     = r"C:\Users\abias\.cursor-tutor\vccdoe\forml\neuron_output.txt"

# Define separate BSIM parameter ranges
param_ranges_nmos = {
    'vth0':  (0.1, 0.9),
    'k1':    (0.2, 1.0),
    'k2':    (0.0, 0.1),
    'dvt0':  (0.5, 3.0),
    'dvt1':  (0.1, 2.0),
    'dvt2':  (-0.05, 0.05),
    'k3':    (0.0, 200.0),
    'w0':    (1e-7, 5e-6),
    'dsub':  (0.0, 0.2),
    'eta0':  (0.0, 0.01),
    'pclm':  (0.01, 0.1),
}

param_ranges_pmos = {
    'vth0':  (-0.9, -0.1),  # flipped sign for PMOS
    'k1':    (0.2, 1.0),
    'k2':    (0.0, 0.1),
    'dvt0':  (0.5, 3.0),
    'dvt1':  (0.1, 2.0),
    'dvt2':  (-0.05, 0.05),
    'k3':    (0.0, 200.0),
    'w0':    (1e-7, 5e-6),
    'dsub':  (0.0, 0.2),
    'eta0':  (0.0, 0.01),
    'pclm':  (0.01, 0.1),
}

param_line_re = re.compile(r'^\s*\+(?P<key>\w+)\s*=\s*[-+.\deE]+', re.IGNORECASE)

def randomize_bsim_params(src_path, dst_path, ranges):
    """Read BSIM model and randomize parameter values."""
    randomized = {}
    with open(src_path, 'r') as f:
        lines = f.readlines()

    out = []
    for line in lines:
        m = param_line_re.match(line)
        if m:
            key = m.group('key').lower()
            if key in ranges:
                low, high = ranges[key]
                new_val = random.uniform(low, high)
                randomized[key] = new_val
                line = re.sub(
                    r'(\+' + key + r'\s*=\s*)[-+.\deE]+',
                    lambda m: m.group(1) + f"{new_val:.6g}",
                    line,
                    flags=re.IGNORECASE
                )
        out.append(line)

    with open(dst_path, 'w') as f:
        f.writelines(out)
    return randomized

def run_random_search(trials=50):
    os.makedirs(MODEL_DIR, exist_ok=True)

    for t in range(1, trials + 1):
        print(f"\n--- Trial {t}/{trials} ---")
        trial_dir = os.path.join(MODEL_DIR, f"trial_{t:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        # Output file paths
        trial_nmos = os.path.join(trial_dir, f"nmos_trial.model")
        trial_pmos = os.path.join(trial_dir, f"pmos_trial.model")
        trial_netlist = os.path.join(trial_dir, "izhikevich_trial.sp")
        trial_outfile = os.path.join(trial_dir, os.path.basename(RAW_OUTPUT))
        trial_log = os.path.join(trial_dir, "param_log.txt")

        # Randomize NMOS and PMOS models
        randomized_nmos = randomize_bsim_params(NMOS_MODEL_SRC, trial_nmos, param_ranges_nmos)
        randomized_pmos = randomize_bsim_params(PMOS_MODEL_SRC, trial_pmos, param_ranges_pmos)

        # Create trial-specific netlist
        with open(BASE_NETLIST, 'r') as f:
            net = f.readlines()
        with open(trial_netlist, 'w') as f:
            for line in net:
                if line.lstrip().lower().startswith('.include'):
                    f.write(f'.include "{trial_nmos}"\n')
                    f.write(f'.include "{trial_pmos}"\n')
                else:
                    f.write(line)

        # Save parameter values to log
        with open(trial_log, 'w') as f:
            f.write("[NMOS Parameters]\n")
            for k, v in randomized_nmos.items():
                f.write(f"{k} = {v:.6g}\n")
            f.write("\n[PMOS Parameters]\n")
            for k, v in randomized_pmos.items():
                f.write(f"{k} = {v:.6g}\n")

        # Run NGSpice (interactive so .plot/.control blocks work)
        result = subprocess.run([NGSPICE_PATH, trial_netlist])
        if result.returncode != 0:
            print("NGSpice error, skipping this trial.")
            continue

        # Save the output raw file if generated
        if os.path.exists(RAW_OUTPUT) and os.path.abspath(RAW_OUTPUT) != os.path.abspath(trial_outfile):
            shutil.copy(RAW_OUTPUT, trial_outfile)

        print(f"Saved trial to {trial_dir}")
        input("Press Enter to continue (Ctrl+C to stop)...")

    print("\nRandom search complete.")

if __name__ == "__main__":
    run_random_search(trials=50)
