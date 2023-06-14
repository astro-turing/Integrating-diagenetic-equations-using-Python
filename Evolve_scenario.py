from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pde import CartesianGrid, ScalarField, FileStorage, plot_kymographs
from pde import Controller, PlotTracker
from pde import ScipySolver, ExplicitSolver
from pde.grids.operators.cartesian import _make_derivative
import time
import os
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

number_of_depths = Solver_parameters.N

depths = CartesianGrid([[0, max_depth/Xstar]], [number_of_depths], periodic=False)
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
# tspan = np.arange(0,end_time+time_step, time_step)

state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                     CO3Surface, PorSurface)

# simulate the pde
tracker = PlotTracker(interval=10, plot_args={"vmin": 0, "vmax": 1.6})
# Store your results somewhere in a subdirectory of a parent directory.
store_folder = "../Results/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S" + "/")
os.makedirs(store_folder)
stored_results = store_folder + "LMAHeureuxPorosityDiff.hdf5"
storage = FileStorage(stored_results)

sol, info = eq.solve(state, t_range=end_time, dt=time_step, method="explicit", \
               scheme = "rk", tracker=["progress", storage.tracker(0.01)], \
               backend = "numba", ret_info = True, adaptive = True)
print()
print("Meta-information about the solution : {}".format(info))        

sol.plot()

""" trackers = [
    "progress",  # show progress bar during simulation
    "steady_state",  # abort when steady state is reached
    storage.tracker(interval=1),  # store data every simulation time unit
    PlotTracker(show=True),  # show images during simulation
    # print some output every 5 real seconds:
    PrintTracker(interval=RealtimeIntervals(duration=5)),
]


solver = ScipySolver(eq, method = "Radau", vectorized = False, backend="numba",\
                     first_step = time_step)
solver = ExplicitSolver(eq, scheme="rk", adaptive=True, \
    backend="numba", tolerance=1e-2)   
controller1 = Controller(solver, t_range = (0, end_time), tracker=trackers)

start_computing = time.time()

sol = controller1.run(state, dt = time_step)

end_computing = time.time()

print("Time taken for running the controller is {0:.2f}s.".\
    format(end_computing - start_computing))
print()

sol.label = "Explicit solver"
print("Diagnostic information:")
print(controller1.diagnostics)

sol.plot() """
