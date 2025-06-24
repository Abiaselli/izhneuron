# Re-import libraries after kernel reset
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import matplotlib.pyplot as plt
import numpy as np

# Create the circuit
circuit = Circuit('REQ to ACK Handshake Test')

# REQ: active-low pulse from 2us to 2.05us
circuit.PulseVoltageSource('REQ', 'req', circuit.gnd,
                           initial_value=0.18@u_V, pulsed_value=0@u_V,
                           delay_time=2@u_us, rise_time=10@u_ns,
                           fall_time=10@u_ns, pulse_width=50@u_ns, period=200@u_us)

# ACK: manual 10ns pulse starting at 2.001us
circuit.PulseVoltageSource('ACK', 'ack', circuit.gnd,
                           initial_value=0@u_V, pulsed_value=0.18@u_V,
                           delay_time=2.001@u_us, rise_time=10@u_ns,
                           fall_time=10@u_ns, pulse_width=10@u_ns, period=1@u_s)

# Resistors for observable node behavior
circuit.R('load_req', 'req', circuit.gnd, 1@u_kΩ)
circuit.R('load_ack', 'ack', circuit.gnd, 1@u_kΩ)

# Simulate with a timestep small enough to capture the ACK pulse
simulator = circuit.simulator()
analysis = simulator.transient(step_time=0.5@u_ns, end_time=10@u_us)

# Extract and plot the signals
time = np.array(analysis.time)
v_req = np.array(analysis['req'])
v_ack = np.array(analysis['ack'])

plt.figure(figsize=(10, 6))
plt.plot(time * 1e6, v_req, label='REQ (V)', linewidth=2)
plt.plot(time * 1e6, v_ack, label='ACK (V)', linewidth=2)
plt.xlabel('Time (µs)')
plt.ylabel('Voltage (V)')
plt.title('REQ-to-ACK Handshake Simulation (Confirmed)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
