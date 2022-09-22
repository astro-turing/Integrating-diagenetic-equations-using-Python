# Last modified by Niklas Hohmann (n.hohmann@uw.edu.pl) Oct 2021
## Parameters for Scenario A
#Taken from table 1 (p. 7)
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from LMAHeureuxPorosityDiffV2 import LMAHeureuxPorosityDiff
from pde import CartesianGrid, ScalarField
from scipy.integrate import solve_ivp
import time

Scenario = 'A'
CA0 = 0.6
CAIni = CA0
CC0 = 0.3
CCIni = CC0
cCa0 = 0.326
cCaIni = cCa0
cCO30 = 0.326
cCO3Ini = cCO30
Phi0 = 0.6
PhiIni = 0.5

ShallowLimit = 50

DeepLimit = 150

sedimentationrate = 0.1
m1 = 2.48
m2 = m1
n1 = 2.8
n2 = n1
KA = 10 ** (- 6.19)

KC = 10 ** (- 6.37)
rhos0 = 2.95 * CA0 + 2.71 * CC0 + 2.8 * (1 - (CA0 + CC0))

rhos = rhos0

rhow = 1.023
beta = 0.1
D0Ca = 131.9
k1 = 1

k2 = k1
k3 = 0.1
k4 = k3
muA = 100.09
DCa = 131.9
DCO3 = 272.6
b = 5

PhiNR = Phi0

PhiInfty = 0.01

Xstar = D0Ca / sedimentationrate
Tstar = Xstar / sedimentationrate 

number_of_depths = 500

max_depth = 500

Depths = CartesianGrid([[0, max_depth * (1 + 0.5/number_of_depths)/Xstar]],\
                        [number_of_depths], periodic=False)
AragoniteSurface = ScalarField(Depths, CAIni)
CalciteSurface = ScalarField(Depths, CCIni)
CaSurface = ScalarField(Depths, cCaIni)
CO3Surface = ScalarField(Depths, cCO3Ini)
PorSurface = ScalarField(Depths, PhiIni) 

# I need those two fields for computing coA, which is rather involved.
# There may be a simpler way of selecting these depths, but I haven't
# found one yet. For now these two Heaviside step functions.
not_too_shallow = ScalarField.from_expression(Depths, 
                  "heaviside(x-{}, 0)".format(ShallowLimit/Xstar))
not_too_deep = ScalarField.from_expression(Depths, 
               "heaviside({}-x, 0)".format(DeepLimit/Xstar)) 

slices_for_all_fields = [slice(i * number_of_depths, (i+1) * number_of_depths) \
                         for i in range(5)]            

eq = LMAHeureuxPorosityDiff(Depths, slices_for_all_fields, CA0, CC0, cCa0, cCO30, Phi0, 
                            sedimentationrate, Xstar, Tstar, k1, k2, k3, k4, 
                            m1, m2, n1, n2, b, beta, rhos, rhow, rhos0, KA, KC, 
                            muA, D0Ca, PhiNR, PhiInfty, DCa, DCO3, 
                            not_too_shallow, not_too_deep)     

depths = ScalarField.from_expression(Depths, "x").data * Xstar                              

# Let us try to reach 710 years, like Niklas.
end_time = 1/Tstar
number_of_steps = 1e4
time_step = end_time/number_of_steps
# t_eval = np.linspace(0,end_time, num = int(number_of_steps))

state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                     CO3Surface, PorSurface)

y0 = state.data.ravel()               

start_computing = time.time()
sol = solve_ivp(eq.fun_numba, (0, end_time), y0, atol = 1e-7, rtol = 1e-7, \
                method="BDF", vectorized = False,\
                first_step = time_step, jac = eq.jac)
end_computing = time.time()

print("Time taken for solve_ivp is {0:.2f}s.".format(end_computing - start_computing))

""" print("sol.status = {0}, sol.success =  {1}".format(sol.status, sol.success))
print()
print("sol.t = {0}, sol.y =  {1}".format(sol.t, sol.y)) """

fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize = (5, 25))
ax0.plot(depths, (sol.y)[slices_for_all_fields[0], -1], label = "CA")
ax0.legend(loc='upper right')
ax1.plot(depths, (sol.y)[slices_for_all_fields[1], -1], label = "CC")
ax1.legend(loc='upper right')
ax2.plot(depths, (sol.y)[slices_for_all_fields[2], -1], label = "cCa")
ax2.legend(loc='upper right')
ax3.plot(depths, (sol.y)[slices_for_all_fields[3], -1], label = "cCO3")
ax3.legend(loc='upper right')
ax4.plot(depths, (sol.y)[slices_for_all_fields[4], -1], label = "Phi")
ax4.legend(loc='upper right')

fig.tight_layout()
fig.savefig("../Results/Final_compositions_and_concentrations_" + datetime.now().\
                      strftime("%d_%m_%Y_%H_%M_%S") + ".png")
# plt.plot(depths,sol(timeslice,:,5))
## Componentwise Plots
# timeslice = 5
# tiledlayout(5,1)
# nexttile
# plt.plot(depths,sol(timeslice,:,1))
# plt.xlabel('Depth (cm)')
# plt.title('Aragonite')
# plt.ylim(np.array([0,1]))
# plt.xlim(np.array([0,np.amax(depths)]))
# nexttile
# plt.plot(depths,sol(timeslice,:,2))
# plt.xlabel('Depth (cm)')
# plt.title('Calcite')
# plt.ylim(np.array([0,1]))
# plt.xlim(np.array([0,np.amax(depths)]))
# nexttile
# plt.plot(depths,sol(timeslice,:,3))
# plt.xlabel('Depth (cm)')
# plt.title('Ca')
# plt.xlim(np.array([0,np.amax(depths)]))
# nexttile
# plt.plot(depths,sol(timeslice,:,4))
# plt.xlabel('Depth (cm)')
# plt.title('CO3')
# plt.xlim(np.array([0,np.amax(depths)]))
# nexttile
# plt.plot(depths,sol(timeslice,:,5))
# plt.xlabel('Depth (cm)')
# plt.title('Porosity')
# plt.ylim(np.array([0,1]))
# plt.xlim(np.array([0,np.amax(depths)]))
# sgtitle(join(np.array([num2str(times(timeslice)),' Years'])))
