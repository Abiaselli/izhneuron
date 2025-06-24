from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

circuit = Circuit('Izhikevich Neuron with AER Handshake')

# Include NMOS and PMOS models
circuit.include('modelcard.nmos')
circuit.include('modelcard.pmos')

circuit = Circuit('Izhikevich Neuron with Full Modelcards')

# Include NMOS and PMOS model files (model_NM1.nmos ... model_PM9.pmos)
for i in range(1, 10):
    circuit.include(f'model_NM{i}.nmos')
    circuit.include(f'model_PM{i}.pmos')

# Power and tuning supplies
circuit.V('dd1', 'VDD1', circuit.gnd, 0.71@u_V)
circuit.V('dd2', 'VDD2', circuit.gnd, 0.98@u_V)
circuit.V('c', 'VC', circuit.gnd, 0.77@u_V)
circuit.V('d', 'VD', circuit.gnd, 0.84@u_V)

# Input current
circuit.I('stim', 'N033', 'N017', 2e-12@u_A)

# Capacitors
circuit.C('u', 'N012', 'N033', 300e-15@u_F)
circuit.C('v', 'N017', circuit.gnd, 20e-15@u_F)
circuit.C('spike', 'N000', circuit.gnd, 25e-15@u_F)

# Selected transistors
circuit.M(7, 'N017', 'N026', 'VC', 'VC', model='PM1')
circuit.M(6, 'N012', 'N012', 'N033', 'N033', model='PM2')
circuit.M(5, 'N017', 'N012', 'N033', 'N033', model='PM3')
circuit.M(3, 'N017', 'N007', 'VDD1', 'VDD1', model='NM4')
circuit.M(1, 'N007', 'N007', 'VDD1', 'VDD1', model='NM5')
circuit.M(10, 'N021', 'N017', 'VDD2', 'VDD2', model='NM6')
circuit.M(12, 'N013', 'N021', 'N019', 'N019', model='NM7')
circuit.M(18, 'N000', 'N013', 'VDD2', 'VDD2', model='NM8')  # REQ driver
circuit.M(15, 'N000', 'N013', circuit.gnd, circuit.gnd, model='PM9')

# ACK pulses
circuit.PulseVoltageSource('ACK1', 'N002', circuit.gnd,
    initial_value=0@u_V, pulsed_value=0.12@u_V,
    delay_time=1@u_ns, rise_time=9@u_ns,
    fall_time=11@u_ns, pulse_width=152@u_ns, period=282@u_us)

circuit.PulseVoltageSource('ACK2', 'N026', circuit.gnd,
    initial_value=0@u_V, pulsed_value=0.14@u_V,
    delay_time=1@u_ns, rise_time=3@u_ns,
    fall_time=9@u_ns, pulse_width=319@u_ns, period=462@u_us)

# Probe
circuit.R('probe', 'out', 'N017', 1@u_kΩ)

# Simulate
simulator = circuit.simulator()
analysis = simulator.transient(step_time=10@u_ns, end_time=6@u_ms)

# Plot
time = np.array(analysis.time)
v_mem = np.array(analysis['N017'])
u_var = np.array(analysis['N012'])

plt.figure(figsize=(12, 6))
plt.plot(time * 1e3, v_mem, label='Membrane potential v (N017)', linewidth=2)
plt.plot(time * 1e3, u_var, label='Recovery variable u (N012)', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.title('Izhikevich Neuron Simulation with AER ACK Pulses')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
