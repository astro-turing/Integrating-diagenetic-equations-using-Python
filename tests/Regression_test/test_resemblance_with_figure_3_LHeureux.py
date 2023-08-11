from dataclasses import asdict, replace
import numpy as np
import h5py
from numpy.testing import assert_allclose
from pde import CartesianGrid, ScalarField
from marlpde.parameters import Map_Scenario, Solver
from marlpde.Evolve_scenario import integrate_equations

def load_hdf5_data(path_to_output, data="data"):
    '''
    Load ground truth data which are stored as hdf5 files.     
    Return the five fields at all depths integrated up to T*.
    It turns out conversion to a Numpy array does not seem necessary for unit
    or regression testing.
    '''
    hf = h5py.File(path_to_output, 'r')
    hf_data = hf.get(data)
    return hf_data


def test_integration_Scenario_A():
    '''
    We have reproduced figure 3e from L'Heureux, so we consider this correct.
    The output data for Scenario A with Phi0 = 0.6 and PhiIni = 0.5 are stored
    in tests/Regression_test_data.
    Results from intermediate times have also been stored in there, i.e. they
    are included in the hdf5 files, but they are currently not used.    
    '''
    rtol=0.1
    atol=0.01

    path_to_ground_truth_Scenario_A = \
    'tests/Regression_test/data/LMAHeureuxPorosityDiff_Phi0_0.6_PhiIni_0.5.hdf5'
    
    Scenario_A_data = load_hdf5_data(path_to_ground_truth_Scenario_A)

    # Concatenate the dict containing the Scenario parameters with the
    # dict containing the solver parameters (such as required tolerance).
    all_kwargs = asdict(Map_Scenario()) | asdict(Solver()) | {"PhiNR": 0.6}
    # integrate_equations returns four variables, we only need the first one.
    solution, _, _, _ = \
        integrate_equations(**all_kwargs)
     
    # Test the final distribution of all five fields over depths
    assert_allclose(solution.data, Scenario_A_data[-1, :, :],
                    rtol=rtol, atol=atol)

def test_high_porosity_integration():
    '''
    Test Scenario A evolution after T*, but this time with high initial and
    boundary values of the porosity.   
    '''
    rtol=0.1
    atol=0.01

    path_to_ground_truth_high_porosities = \
    'tests/Regression_test/data/LMAHeureuxPorosityDiff_Phi0_PhiIni_0.8.hdf5'
    
    high_porosity_data = load_hdf5_data(path_to_ground_truth_high_porosities)

    # replace from dataclasses cannot be applied here, since b would be
    # divided by 1e4 twice. So we will modify porosity values at the dict level.
    Scenario_parameters = asdict(Map_Scenario()) |\
                                {"Phi0": 0.8, "PhiIni": 0.8, "PhiNR": 0.8}

    # Smaller initial time step needed for these high porosities.
    # If not, we get a ZeroDivisionError from "wrapped_stepper".
    Solver_parms = replace(Solver(), dt = 5e-7)
    # Concatenate the dict containing the Scenario parameters with the
    # dict containing the solver parameters (such as required tolerance).
    all_kwargs = Scenario_parameters  | asdict(Solver_parms)
    # integrate_equations returns four variables, we only need the first one.
    solution, _, _, _ = integrate_equations(**all_kwargs)
     
    # Test the final distribution of all five fields over depths
    assert_allclose(solution.data, high_porosity_data[-1, :, :],
                    rtol=rtol, atol=atol)

def test_cross_check_with_Matlab_output():
    '''
    This test was written to cross check with the output from a completely
    independent codebase.
    Using the Matlab code (https://github.com/MindTheGap-ERC/LMA-Matlab) 
    with k3=k4=0.01 one can achieve stable integrations over T*. 
    With the extra options 'InitialStep',1e-6,'MaxStep',1e-5 for pdepe
    one can get good resemblance between Python and Matlab output, at least
    for Phi0=PhiIni=0.5. 
    Matlab uses a slightly different depth grid, it was set to use
    201 depths, including the surface itself (depth=0) and the bottom 
    (depth=500cm), so an interpolation is needed.
    '''
    atol=0.05

    path_to_Matlab_output =  ("tests/Regression_test/data/"
                              "Matlab_output_Scenario_A_Phi0_PhiIni_"
                              "0.5_k3_k4_0.01.h5")

    Matlab_output = load_hdf5_data(path_to_Matlab_output, \
                                         data="Solutions after_T*")
    
    Matlab_depths = np.linspace(0, 500, Matlab_output.shape[1])

    Scenario_parameters = asdict(Map_Scenario()) |\
                                   {"Phi0": 0.5, "PhiIni": 0.5, "PhiNR": 0.5, \
                                    "k3": 0.01, "k4": 0.01}
    Xstar =  Scenario_parameters["Xstar"]

    max_depth = Scenario_parameters["max_depth"]  

    all_kwargs = Scenario_parameters  | asdict(Solver())
    
    solution, _, _, _ = integrate_equations(**all_kwargs)

    Number_of_depths = all_kwargs["N"]

    # It may seem a bit cumbersome to arrive at Python_plotting_depths in this manner,
    # but we are trying to mimick the way the grid has been setup in Evolve_scenario.py.
    Python_depth_grid = CartesianGrid([[0, max_depth/Xstar]], [Number_of_depths], \
                           periodic=False)
    
    Python_plotting_depths = ScalarField.from_expression(Python_depth_grid, "x").data * Xstar

    # Now we need to interpolate the Matlab field values CA, CC, cCa, cCO3 and Phi to
    # the Python grid values.
    Matlab_output_interpolated = np.empty((Matlab_output.shape[0], Number_of_depths))
    for field in range(Matlab_output.shape[0]):
        Matlab_output_interpolated[field, :] = np.interp(Python_plotting_depths, Matlab_depths, \
                                                         Matlab_output[field, :, 0])
        
    # Compare the Python and Matlab output over all depths, except for close to the surface,
    # since we are dealing with a boundary layer near the surface - see chapter 7, page 56 of
    # Willem Hundsdorfer: "Numerical Solution of Advection-Diffusion-Reaction Equations".
    # Consequently, both the Matlab and Python solutions for the concentrations and the 
    # porosity jump a bit up and down near the surface, due to the high Peclet numbers.
    assert_allclose(solution.data[:, 2:], Matlab_output_interpolated[:, 2:], atol = atol)
    
    


