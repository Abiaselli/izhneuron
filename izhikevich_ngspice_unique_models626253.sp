
* Begin neuron subcircuit
* Izhikevich Neuron with NMOS/PMOS BSIM3 models

* Begin neuron subcircuit
.include "model_M1.pmos"
.include "model_M2.nmos"
.include "model_M3.pmos"
.include "model_M4.pmos"
.include "model_M5.nmos"
.include "model_M6.nmos"
.include "model_M7.nmos"
.include "model_M8.pmos"
.include "model_M9.nmos"
.include "model_M10.pmos"
.include "model_M11.nmos"
.include "model_M12.pmos"
.include "model_M13.pmos"
.include "model_M14.nmos"
.include "model_M15.nmos"
.include "model_M16.nmos"
.include "model_M17.pmos"
.include "model_M18.pmos"
.subckt izh_neuron in out vdd vss


.model M1 pmos
.model M3 pmos
.model M4 pmos
.model M8 pmos
.model M10 pmos
.model M12 pmos
.model M13 pmos
.model M17 pmos
.model M18 pmos
.model M2 nmos
.model M5 nmos
.model M6 nmos
.model M7 nmos
.model M9 nmos
.model M14 nmos
.model M15 nmos
.model M16 nmos
.model M11 nmos
Vc N030 0 0.14
  I1 N033 N017 2n
Vdd1 N003 0 0.18
Vdd2 N001 0 0.18
Vd N006 0 0.14
  Cu N012 N033 300e-15
  Cv N017 0 20e-15
  C1 N016 0 25e-15
M7 N017 N026 N030 N030 M7 
M6 N012 N012 N033 N033 M6
M5 N017 N012 N033 N033 M5
M2 N007 N017 N033 N033 M2
M13 N019 N002 N001 N001 M13
M8 N012 N005 N006 N006 M8
M4 N012 N007 N003 N003 M4
M3 N017 N007 N003 N003 M3
M1 N007 N007 N003 N003 M1
M10 N021 N017 N001 N001 M10
M12 N013 N021 N019 N019 M12
M18 N016 N013 N001 N001 M18
M9 N021 N017 0 0 M9
M11 N013 N021 0 0 M11
M17 N005 N002 N001 N001 M17
M16 N005 N002 N028 N028 M16
M14 N028 N013 0 0 M14
M15 N016 N013 0 0 M15
*dgsb

* REQ is received on N016 (input)
* VACK signal to be sent on N002 and N026 after 1ns delay
VACK1 N002 0 PULSE(0 0.18 1n 10n 10n 100n 2000n)
VACK2 N026 0 PULSE(0 0.18 1n 10n 10n 100n 2000n)
  .end

.ends izh_neuron

.probe V(X1.N016) V(X1.N017) V(X1.N012)

* Simulation control
X1 in out vdd 0 izh_neuron
* Fake REQ generator (active-low)
VREQ REQ 0 PULSE(0.18 0 2u 10n 10n 50n 200u)



* Transient sim
.tran 1n 10u

.control
run


reset
run

plot v(X1.N016) v(X1.N012)
.endc



.endc
.end