# Last modified by Niklas Hohmann (n.hohmann@uw.edu.pl) Oct 2021
## Parameters for Scenario A
#Taken from table 1 (p. 7)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from LMAHeureuxPorosityDiffV2 import LMAHeureuxPorosityDiff
from pde import CartesianGrid, ScalarField, FileStorage, plot_kymographs
from pde import Controller, PlotTracker
from pde import ScipySolver, ExplicitSolver
from pde.grids.operators.cartesian import _make_derivative
import time
import os

Scenario = 'A'

KA = 10 ** (- 6.19)
KC = 10 ** (- 6.37)
CA0 = 0.6
CAIni = CA0
CC0 = 0.3
CCIni = CC0
cCa0 = 0.326e-3/np.sqrt(KC)
cCaIni = cCa0
cCO30 = 0.326e-3/np.sqrt(KC)
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
b = 5e-4

PhiNR = Phi0

PhiInfty = 0.01

Xstar = D0Ca / sedimentationrate
Tstar = Xstar / sedimentationrate 

max_depth = 500
number_of_depths = 200
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
                  "heaviside(x-{}, 0)".format(ShallowLimit/Xstar))
not_too_deep = ScalarField.from_expression(depths, 
               "heaviside({}-x, 0)".format(DeepLimit/Xstar))    

eq = LMAHeureuxPorosityDiff(AragoniteSurface, CalciteSurface, CaSurface, 
                            CO3Surface, PorSurface, CA0, CC0, cCa0, cCO30, 
                            Phi0, sedimentationrate, Xstar, Tstar, k1, k2, 
                            k3, k4, m1, m2, n1, n2, b, beta, rhos, rhow, rhos0, 
                            KA, KC, muA, D0Ca, PhiNR, PhiInfty, DCa, DCO3, 
                            not_too_shallow, not_too_deep)             

end_time = Tstar/Tstar
number_of_steps = 1e6
time_step = end_time/number_of_steps
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
