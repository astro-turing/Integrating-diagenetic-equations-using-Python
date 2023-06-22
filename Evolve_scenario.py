#!/usr/bin/env python

from datetime import datetime
import os
from pde import CartesianGrid, ScalarField, FileStorage
from pde.grids.operators.cartesian import _make_derivative
import matplotlib.pyplot as plt
from LHeureux_model import LMAHeureuxPorosityDiff
from marlpde.marlpde import Map_Scenario, Solver

def retrieve_solver_parameters():
    '''
    Function to retrieve the solver parameters defined in marlpde.py.
    '''
    Solver_parameters = Solver()
    NUMBER_OF_DEPTHS = Solver_parameters.N
    end_time = Solver_parameters.tmax/Tstar
    time_step = Solver_parameters.dt
    number_of_steps = end_time/time_step

    return all_solver_parameters

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

if __name__ == '__main__':
    scenario_parameters = read_Scenario_parameters()
    solver_parameters = read_solver_parameters()
    solution = solve_stuff()
    plot = plot_stuff()