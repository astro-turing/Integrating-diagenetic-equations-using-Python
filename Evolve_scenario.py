#!/usr/bin/env python

from datetime import datetime
import os
from pde import CartesianGrid, ScalarField, FileStorage
from pde.grids.operators.cartesian import _make_derivative
import numpy as np
import matplotlib.pyplot as plt
from LHeureux_model import LMAHeureuxPorosityDiff
from marlpde.marlpde import Scenario, Solver

Scenario_parameters = Scenario()
Solver_parameters = Solver()

KA = Scenario_parameters.Ka.magnitude
KC = Scenario_parameters.Kc.magnitude
CA0 = Scenario_parameters.cara0.magnitude
CAIni = Scenario_parameters.cara00.magnitude
CC0 = Scenario_parameters.ccal0.magnitude
CCIni = Scenario_parameters.ccal00.magnitude
cCa0 = Scenario_parameters.ca0.magnitude/np.sqrt(KC)
cCaIni = Scenario_parameters.ca00.magnitude/np.sqrt(KC)
cCO30 = Scenario_parameters.co30.magnitude/np.sqrt(KC)
cCO3Ini = Scenario_parameters.co300.magnitude/np.sqrt(KC)
Phi0 = Scenario_parameters.phi0.magnitude
PhiIni = Scenario_parameters.phi00.magnitude

ShallowLimit = Scenario_parameters.xdis.magnitude

DeepLimit = ShallowLimit + Scenario_parameters.Th.magnitude

sedimentationrate = Scenario_parameters.S.magnitude
m1 = Scenario_parameters.m.magnitude
m2 = m1
n1 = Scenario_parameters.nn.magnitude
n2 = n1
rhoa = Scenario_parameters.rhoa.magnitude
rhoc = Scenario_parameters.rhoc.magnitude
rhot = Scenario_parameters.rhot.magnitude
rhos0 = rhoa * CA0 + rhoc * CC0 + rhot * (1 - (CA0 + CC0))

rhos = rhos0

rhow = Scenario_parameters.rhow.magnitude
beta = Scenario_parameters.beta.magnitude
D0Ca = Scenario_parameters.D0ca.magnitude
k1 = Scenario_parameters.k1.magnitude
k2 = Scenario_parameters.k2.magnitude
k3 = Scenario_parameters.k3.magnitude
k4 = Scenario_parameters.k4.magnitude
muA = Scenario_parameters.mua.magnitude
DCa = Scenario_parameters.D0ca.magnitude
DCO3 = Scenario_parameters.D0co3.magnitude
b = Scenario_parameters.b.magnitude/1e4
PhiNR = Scenario_parameters.phi00.magnitude
PhiInfty = Scenario_parameters.phiinf.magnitude

Xstar = D0Ca / sedimentationrate
Tstar = Xstar / sedimentationrate 

max_depth = Scenario_parameters.length.magnitude

NUMBER_OF_DEPTHS = Solver_parameters.N

depths = CartesianGrid([[0, max_depth/Xstar]], [NUMBER_OF_DEPTHS], periodic=False)
# We will be needing forward and backward differencing for
# Fiadeiro-Veronis differentiation.
depths.register_operator("grad_back", \
    lambda grid: _make_derivative(grid, method="backward"))
depths.register_operator("grad_forw", \
    lambda grid: _make_derivative(grid, method="forward"))

AragoniteSurface = ScalarField(depths, CAIni)
CalciteSurface = ScalarField(depths, CCIni)
CaSurface = ScalarField(depths, cCaIni)
CO3Surface = ScalarField(depths, cCO3Ini)
PorSurface = ScalarField(depths, PhiIni)

# I need those two fields for computing coA, which is rather involved.
# There may be a simpler way of selecting these depths, but I haven't
# found one yet. For now these two Heaviside step functions.
not_too_shallow = ScalarField.from_expression(depths,
                  f"heaviside(x-{ShallowLimit/Xstar}, 0)")
not_too_deep = ScalarField.from_expression(depths,
               f"heaviside({DeepLimit/Xstar}-x, 0)")    

eq = LMAHeureuxPorosityDiff(AragoniteSurface, CalciteSurface, CaSurface, 
                            CO3Surface, PorSurface, CA0, CC0, cCa0, cCO30, 
                            Phi0, sedimentationrate, Xstar, Tstar, k1, k2, 
                            k3, k4, m1, m2, n1, n2, b, beta, rhos, rhow, rhos0, 
                            KA, KC, muA, D0Ca, PhiNR, PhiInfty, PhiIni, DCa, DCO3, 
                            not_too_shallow, not_too_deep)             

end_time = Solver_parameters.tmax/Tstar
time_step = Solver_parameters.dt
number_of_steps = end_time/time_step

state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                     CO3Surface, PorSurface)

# Store your results somewhere in a subdirectory of a parent directory.
store_folder = "../Results/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S" + "/")
os.makedirs(store_folder)
stored_results = store_folder + "LMAHeureuxPorosityDiff.hdf5"
storage = FileStorage(stored_results)

sol, info = eq.solve(state, t_range=end_time, dt=time_step, method="explicit", \
               scheme = "rk", tracker=["progress", storage.tracker(0.01)], \
               backend = "numba", ret_info = True, adaptive = True)
print()
print(f"Meta-information about the solution : {info}")        

covered_time = Tstar * end_time
plt.title(f"Situation after {covered_time:.2f} years")
# Marker size
ms = 3
plotting_depths = ScalarField.from_expression(depths, "x").data * Xstar
plt.plot(plotting_depths, sol.data[0], "v", ms = ms, label = "CA")
plt.plot(plotting_depths, sol.data[1], "^", ms = ms, label = "CC")
plt.plot(plotting_depths, sol.data[2], ">", ms = ms, label = "cCa")
plt.plot(plotting_depths, sol.data[3], "<", ms = ms, label = "cCO3")
plt.plot(plotting_depths, sol.data[4], "o", ms = ms, label = "Phi")
plt.xlabel("Depth (cm)")
plt.ylabel("Compositions and concentrations (dimensionless)")
plt.legend(loc='upper right')
plt.plot()
plt.show()