# Integrating diagenetic equations using Python

This repo was created as an attempt to reproduce the plots shown at the kickoff of the AstroTOM ("Turing or Milankovitch") project by Niklas Hohmann, from his Matlab scripts. 

AstroTOM is an OpenSSI 2021b project from the Netherlands eScience Center and Utrecht University (UU).

Dr. Emilia Jarochowska (UU) is the lead applicant of this project.

After replacing central differencing for the gradients in the five diagenetic equations from l'Heureux (2018) by forward and backward differencing depending on the sign of U and W as a first step and a Fiadeiro-Veronis spatial difference scheme as a second step, it turns out that these equations can be integrated for more than 13.190 years (the full Tstar) with an explicit solver (in time), but not with a simple Eulerian scheme. A Runge-Kutta solver, with an adaptive timestep will, however, suffice.
