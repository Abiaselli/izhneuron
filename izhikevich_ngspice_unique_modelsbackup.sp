
* Begin neuron subcircuit
* Izhikevich Neuron with NMOS/PMOS BSIM3 models
.include "modelcard_corrected.nmos"
.include "modelcard_corrected.pmos"

* Begin neuron subcircuit
.include "model_NM1.nmos"
.include "model_NM2.nmos"
.include "model_NM3.nmos"
.include "model_NM4.nmos"
.include "model_NM5.nmos"
.include "model_NM6.nmos"
.include "model_NM7.nmos"
.include "model_NM8.nmos"
.include "model_NM9.nmos"
.include "model_PM1.pmos"
.include "model_PM2.pmos"
.include "model_PM3.pmos"
.include "model_PM4.pmos"
.include "model_PM5.pmos"
.include "model_PM6.pmos"
.include "model_PM7.pmos"
.include "model_PM8.pmos"
.include "model_PM9.pmos"
.subckt izh_neuron in out vdd vss

  Vc N030 0 140e-3
  I1 N033 N017 2e-12
  Vdd1 N003 0 180e-3
  Vdd2 N001 0 180e-3
  Vd N006 0 140e-3
  Cu N012 N033 300e-15
  Cv N017 0 20e-15
  C1 N016 0 25e-15
M7 N017 N002 N030 N030 PM1 l=175e-9 w=350e-9
M6 N012 N012 N033 N033 PM2 l=700e-9 w=150e-9
M5 N017 N012 N033 N033 PM3 l=400e-9 w=600e-9
M2 N007 N017 N033 N033 PM4 l=300e-9 w=125e-9
M13 N019 N002 N001 N001 NM1 l=200e-9 w=100e-9
M8 N012 N005 N006 N006 NM2 l=1.3e-6 w=125e-9
M4 N012 N007 N003 N003 NM3 l=650e-9 w=90e-9
M3 N017 N007 N003 N003 NM4 l=300e-9 w=125e-9
M1 N007 N007 N003 N003 NM5 l=300e-9 w=125e-9
M10 N021 N017 N001 N001 NM6 l=1e-6 w=100e-9
M12 N013 N021 N019 N019 NM7 l=100e-9 w=100e-9
M18 N016 N013 N001 N001 NM8 l=200e-9 w=100e-9
M9 N021 N017 0 0 PM5 l=100e-9 w=100e-9
M11 N013 N021 0 0 PM6 l=1e-6 w=100e-9
M17 N005 N002 N001 N001 NM9 l=100e-9 w=100e-9
M16 N005 N002 N028 N028 PM7 l=100e-9 w=100e-9
M14 N028 N013 0 0 PM8 l=100e-9 w=100e-9
M15 N016 N013 0 0 PM9 l=100e-9 w=250e-9
VREQ N016 0 PULSE(0 0.18 0s 1n 1n 50n 100n)
VACK N002 0 PULSE(0 0.18 0.001ms 1n 1n 50n 100n)
Vprobe out N017 0
  .tran 10ns 1ms 10ns 10ns
  .end
.ends izh_neuron

* Simulation control
X1 in out vdd 0 izh_neuron
.tran 100n 100u
.control
run
set filetype=binary
wrdata C:/Users/abias/.cursor-tutor/vccdoe/forml/neuron_output.raw v(x1.n017) v(x1.n012)
plot v(x1.n017) v(x1.n012)
.endc
.end