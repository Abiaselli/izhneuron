* NMOS Id vs VGS (BSIM4.81) test circuit

.include "bsim4_model.lib"     * Include your BSIM4.81 model here

.options nomod post
.temp 27

* Supply voltage for VDS (kept constant)
VDD Drain 0 1.2

* Gate voltage sweep source (VGS)
VGS Gate 0 0

* NMOS instance
* D     G     S  B      model
M1 Drain Gate 0 0 BSIM4_Extract L=0.18u W=1u NF=1

* Sweep gate voltage from 0V to 1.2V
.dc VGS -5 8 0.01

* Control block for plotting Id vs VGS
.control
  run
  setplot dc1
  plot -i(VDD) vs V(Gate)
  write id_vgs.raw
.endc

.end
