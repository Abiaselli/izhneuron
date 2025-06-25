
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


.model M1 nmos
.model M3 nmos
.model M4 nmos
.model M8 nmos
.model M10 nmos
.model M12 nmos
.model M13 nmos
.model M17 nmos
.model M18 nmos
.model M2 pmos
.model M5 pmos
.model M6 pmos
.model M7 pmos
.model M9 pmos
.model M14 pmos
.model M15 pmos
.model M16 pmos
.model M11 pmos
Vc N028 0 0.14
  I1 N031 N016 200n
Vdd1 N003 0 0.18
Vdd2 N001 0 0.18
Vd N006 0 0.14
  Cu N012 N033 500e-15
  Cv N016 0 100e-15
  C1 N020 0 250e-15
*dgsb
M7 N016 N002 N028 N028 M7 l=175n w=350n
M6 N012 N012 N031 N031 M6 l=700n w=150n
M5 N016 N012 N031 N031 M5 l=400n w=600n
M2 N007 N016 N031 N031 M2 l=300n w=125n
M13 N018 N002 N001 N001 M13 l=200n w=100n
M8 N012 N005 N006 N006 M8 l=1.3u w=125n
M4 N012 N007 N003 N003 M4 l=650n w=90n
M3 N016 N007 N003 N003 M3 l=300n w=125n
M1 N007 N007 N003 N003 M1 l=300n w=125n
M10 N021 N016 N001 N001 M10 l=1u w=100n
M12 N013 N021 N018 N018 M12 l=100n w=100n
M18 N020 N013 N001 N001 M18 l=200n w=100n
M9 N021 N016 0 0 M9 l=100n w=100n
M11 N013 N021 0 0 M11 l=1u w=100n
M17 N005 N002 N001 N001 M17 l=100n w=100n
M16 N005 N002 N026 N026 M16 l=100n w=100n
M14 N026 N013 0 0 M14 l=100n w=100n
M15 N020 N013 0 0 M15 l=100n w=250n

* VACK signal to be sent on N002 and N026 after 1ns delay
VACK1 N002 0 PULSE(0 0.18 5n 10n 10n 100n 2000n)
VACK2 N026 0 PULSE(0 0.18 5n 10n 10n 100n 2000n)
  .end

.ends izh_neuron

.probe  V(X1.N016) V(X1.N012)

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