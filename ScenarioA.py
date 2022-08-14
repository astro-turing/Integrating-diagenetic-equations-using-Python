# Last modified by Niklas Hohmann (n.hohmann@uw.edu.pl) Oct 2021
## Parameters for Scenario A
#Taken from table 1 (p. 7)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from LMAHeureuxPorosityDiffV2 import LMAHeureuxPorosityDiff
from pde import CartesianGrid, ScalarField, FileStorage
from pde import Controller, PlotTracker, PrintTracker, RealtimeIntervals
from scipy.integrate import solve_ivp

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

Depths = CartesianGrid([[0, 502/Xstar]], [400], periodic=False)
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

eq = LMAHeureuxPorosityDiff(Depths, CA0, CC0, cCa0, cCO30, Phi0, 
                            sedimentationrate, Xstar, Tstar, k1, k2, k3, k4, 
                            m1, m2, n1, n2, b, beta, rhos, rhow, rhos0, KA, KC, 
                            muA, D0Ca, PhiNR, PhiInfty, DCa, DCO3, 
                            not_too_shallow, not_too_deep)     

depths = ScalarField.from_expression(Depths, "x").data                                    

# Let us try to years 710 years, like Niklas.
end_time = 100/Tstar
number_of_steps = 1e4
time_step = end_time/number_of_steps
t_eval = np.linspace(0,end_time, num = int(number_of_steps))

state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                     CO3Surface, PorSurface)

y0 = state.data.flat                

sol = solve_ivp(eq.fun, (0, end_time), y0, t_eval = t_eval, vectorized = True,
                first_step = time_step)

""" print("sol.status = {0}, sol.success =  {1}".format(sol.status, sol.success))
print()
print("sol.t = {0}, sol.y =  {1}".format(sol.t, sol.y)) """

fig, (ax0, ax1, ax2, ax3, ax4) =plt.subplots(1, 5)
ax0.plot(depths, (sol.y)[0: 400, -1])
ax1.plot(depths, (sol.y)[400:800, -1])
ax2.plot(depths, (sol.y)[800:1200, -1])
ax3.plot(depths, (sol.y)[1200:1600, -1])
ax4.plot(depths, (sol.y)[1600:2000, -1])

fig.show()
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
