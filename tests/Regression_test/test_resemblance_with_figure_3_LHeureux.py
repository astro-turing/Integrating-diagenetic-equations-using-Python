from dataclasses import asdict, replace
import h5py
from numpy.testing import assert_allclose
from marlpde.parameters import Map_Scenario
from marlpde.Evolve_scenario import integrate_equations

def load_hdf5_data(path_to_ouput):
    '''
    Load ground truth data which are stored as hdf5 files.     
    Return the five fields at all depths integrated up to T*.
    It turns out conversion to a Numpy array does not seem necessary for unit
    or regression testing.
    '''
    hf = h5py.File(path_to_ouput, 'r')
    hf_data = hf.get("data")
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

    # integrate_equations returns four variables, we only need the first one.
    solution, _, _, _ = \
        integrate_equations(**asdict(Map_Scenario()))
     
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

    high_porosity_parameters = \
        asdict(replace(Map_Scenario(), Phi0 = 0.8, PhiIni = 0.8))

    # integrate_equations returns four variables, we only need the first one.
    solution, _, _, _ = \
        integrate_equations(**high_porosity_parameters)
     
    # Test the final distribution of all five fields over depths
    assert_allclose(solution.data, high_porosity_data[-1, :, :],
                    rtol=rtol, atol=atol)