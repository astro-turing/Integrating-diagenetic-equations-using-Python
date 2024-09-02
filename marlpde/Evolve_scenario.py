#!/usr/bin/env python

from datetime import datetime
import time
import os
from dataclasses import asdict
import inspect
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from pde import CartesianGrid, ScalarField, FileStorage
from pde.grids.operators.cartesian import _make_derivative
from parameters import Map_Scenario, Solver, Tracker
from LHeureux_model import LMAHeureuxPorosityDiff

def integrate_equations(solver_parms, tracker_parms, pde_parms):
    '''
    This function retrieves the parameters of the Scenario to be simulated and 
    the solution parameters for the integration. It then integrates the five
    partial differential equations from L'Heureux, stores and returns the 
    solution, to be used for plotting.
    '''

    Xstar = pde_parms["Xstar"]
    Tstar = pde_parms["Tstar"]
    max_depth = pde_parms["max_depth"]
    ShallowLimit = pde_parms["ShallowLimit"]
    DeepLimit = pde_parms["DeepLimit"]
    CAIni = pde_parms["CAIni"]
    CCIni = pde_parms["CCIni"]
    cCaIni = pde_parms["cCaIni"]
    cCO3Ini = pde_parms["cCO3Ini"]
    PhiIni = pde_parms["PhiIni"]

    Number_of_depths = pde_parms["N"]

    depths = CartesianGrid([[0, max_depth/Xstar]], [Number_of_depths], periodic=False)
    # We will be needing forward and backward differencing for
    # Fiadeiro-Veronis differentiation.
    depths.register_operator("grad_back", \
        lambda grid: _make_derivative(grid, method="backward"))
    depths.register_operator("grad_forw", \
        lambda grid: _make_derivative(grid, method="forward"))
    
    # I need those two fields for computing coA, which is rather involved.
    # There may be a simpler way of selecting these depths, but I haven't
    # found one yet. For now these two Heaviside step functions.
    not_too_shallow = ScalarField.from_expression(depths,
                      f"heaviside(x-{ShallowLimit/Xstar}, 0)")
    not_too_deep = ScalarField.from_expression(depths,
                   f"heaviside({DeepLimit/Xstar}-x, 0)")    
    
    # Not all keys from pde_parms are LMAHeureuxPorosityDiff arguments.
    # Taken from https://stackoverflow.com/questions/334655/passing-a-\
    # dictionary-to-a-function-as-keyword-parameters
    filtered_pde_parms = {k: v for k, v in pde_parms.items() if k in [p.name for
                          p in 
                          inspect.signature(LMAHeureuxPorosityDiff).parameters.\
                          values()]}

    slices_for_all_fields = [slice(i * Number_of_depths, (i+1) * Number_of_depths) \
                             for i in range(5)]            
    
    eq = LMAHeureuxPorosityDiff(depths, slices_for_all_fields, not_too_shallow, 
                                not_too_deep, **filtered_pde_parms)     
    
    # Setting initial values for all five fields in three steps.
    # This is perhaps more involved than necessary, since the
    # end result is a 1D Numpy array with a length of five times
    # the number of depths. The number five comes from the five
    # fields we are tracking, by integrating five pdes.
    # First step.
    AragoniteSurface = ScalarField(depths, CAIni)
    CalciteSurface = ScalarField(depths, CCIni)
    CaSurface = ScalarField(depths, cCaIni)
    CO3Surface = ScalarField(depths, cCO3Ini)
    PorSurface = ScalarField(depths, PhiIni)
    
    # Second step.
    state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                         CO3Surface, PorSurface)
    # Third step.
    y0 = state.data.ravel()   
    
    start_computing = time.time()
    with tqdm(total=number_of_progress_updates) as pbar:
        t0 = solver_parms["t_span"][0]
        end_time = solver_parms["t_span"][1]
        no_progress_updates = tracker_parms["no_progress_updates"]
        progress_bar_args = [pbar, (end_time - t0) / no_progress_updates, t0]
  
        sol = solve_ivp(fun=eq.fun_numba, y0=y0, **solver_parms, 
                    t_eval= tracker_parms["t_eval"], 
                    events = [eq.zeros, eq.zeros_CA, eq.zeros_CC, \
                    eq.ones_CA_plus_CC, eq.ones_Phi, eq.zeros_U, eq.zeros_W],  
                    args=progress_bar_args)
    end_computing = time.time()
    
    print()
    print("Number of rhs evaluations = {0} \n".format(sol.nfev))
    print("Number of Jacobian evaluations = {0} \n".format(sol.njev))
    print("Number of LU decompositions = {0} \n".format(sol.nlu))
    print("Status = {0} \n".format(sol.status))
    print("Success = {0} \n".format(sol.success))
    f = sol.t_events[0]
    print(("Times, in years, at which any field at any depth crossed zero: "\
      +', '.join(['%.2f']*len(f))+"") % tuple([Tstar * time for time in f]))
    print()
    g = sol.t_events[1]
    print(("Times, in years, at which CA at any depth crossed zero: "\
      +', '.join(['%.2f']*len(g))+"") % tuple([Tstar * time for time in g]))
    print()
    h = sol.t_events[2]
    print(("Times, in years, at which CC at any depth crossed zero: "\
      +', '.join(['%.2f']*len(h))+"") % tuple([Tstar * time for time in h]))
    print()
    k = sol.t_events[3]
    print(("Times, in years, at which CA + CC at any depth crossed one: "\
      +', '.join(['%.2f']*len(k))+"") % tuple([Tstar * time for time in k]))
    print()
    l = sol.t_events[4]
    print(("Times, in years, at which the porosity at any depth crossed one: "\
      +', '.join(['%.2f']*len(l))+"") % tuple([Tstar * time for time in l]))
    print()
    m = sol.t_events[5]
    print(("Times, in years, at which U at any depth crossed zero: "\
      +', '.join(['%.2f']*len(m))+"") % tuple([Tstar * time for time in m]))
    print()
    n = sol.t_events[6]
    print(("Times, in years, at which W at any depth crossed zero: "\
      +', '.join(['%.2f']*len(n))+"") % tuple([Tstar * time for time in n]))
    print()
    
    print("Message from solve_ivp = {0} \n".format(sol.message))
    print(("Time taken for solve_ivp is "
          "{0:.2f}s. \n".format(end_computing - start_computing)))
    
    if sol.status == 0:
        covered_time = Tstar * end_time
    else:
       covered_time = pbar.n * Tstar * end_time /number_of_progress_updates 
    # Store your results somewhere in a subdirectory of a parent directory.
    store_folder = "../Results/" + \
                   datetime.now().strftime("%d_%m_%Y_%H_%M_%S" + "/")
    os.makedirs(store_folder)
    stored_results = store_folder + "LMAHeureuxPorosityDiff.hdf5"
    # Keep a record of all parameters.
    storage_parms = solver_parms | tracker_parms | pde_parms
    storage = FileStorage(stored_results, info=storage_parms)

    return sol, covered_time_span, depths, Xstar, store_folder

def Plot_results(sol, covered_time, depths, Xstar, store_folder):
    '''
    Plot the five fields at the end of the integration interval as a function
    of depth.
    '''
    fig, ax = plt.subplots()
    fig.suptitle(f"Distributions after {covered_time:.2f} years")
    # Marker size
    ms = 5
    plotting_depths = ScalarField.from_expression(depths, "x").data * Xstar
    ax.plot(plotting_depths, sol.data[0], "v", ms = ms, label = "CA")
    ax.plot(plotting_depths, sol.data[1], "^", ms = ms, label = "CC")
    ax.plot(plotting_depths, sol.data[2], ">", ms = ms, label = "cCa")
    ax.plot(plotting_depths, sol.data[3], "<", ms = ms, label = "cCO3")
    ax.plot(plotting_depths, sol.data[4], "o", ms = ms, label = "Phi")
    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Compositions and concentrations (dimensionless)")
    ax.legend(loc='upper right')
    fig.savefig(store_folder + 'Final_distributions.pdf', bbox_inches="tight")

if __name__ == '__main__':
    pde_parms = asdict(Map_Scenario()) 
    solver_parms = asdict(Solver()) 
    tracker_parms = asdict(Tracker())
    solution, covered_time, depths, Xstar, store_folder = \
        integrate_equations(solver_parms, tracker_parms, pde_parms)
    Plot_results(solution, covered_time, depths, Xstar, store_folder)

