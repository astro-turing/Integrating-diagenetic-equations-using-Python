# ~\~ language=Python filename=marlpde/marlpde.py
# ~\~ begin <<lit/python-interface.md|marlpde/marlpde.py>>[init]
# import numpy
import configparser
from pathlib import Path
from dataclasses import (dataclass, asdict, make_dataclass, fields)
from subprocess import (run)
import h5py as h5
from pint import UnitRegistry
import numpy as np

u = UnitRegistry()
quantity = u.Quantity

@dataclass
class Scenario:
    '''
    Sets all the Scenario parameter values from the FORTRAN code from 
    L'Heureux (2018). Strictly, the initial and boundary porosities are not
    part of the Scenario parameters, but they are included here.
    '''
    mua: quantity    = 100.09 * u.g/u.mol
    rhoa: quantity   = 2.95 * u.g/u.cm**3
    rhoc: quantity   = 2.71 * u.g/u.cm**3
    rhot: quantity   = 2.8 * u.g/u.cm**3
    rhow: quantity   = 1.023 * u.g/u.cm**3
    D0ca: quantity   = 131.9 * u.cm**2/u.a
    D0co3: quantity  = 272.6 * u.cm**2/u.a
    Ka: quantity     = 10**(-6.19) * u.M**2
    Kc: quantity     = 10**(-6.37) * u.M**2
    beta: quantity   = 0.1 * u.cm / u.a
    b: quantity      = 5.0 / u.kPa
    k1: quantity     = 1.0 / u.a
    k2: quantity     = 1.0 / u.a
    k3: quantity     = 0.1 / u.a
    k4: quantity     = 0.1 / u.a
    nn: quantity     = 2.8 * u.dimensionless
    m: quantity      = 2.48 * u.dimensionless
    S: quantity      = 0.1 * u.cm / u.a
    # cAthy: quantity  = 0.1 * u.dimensionless
    phiinf: quantity = 0.01 * u.dimensionless
    phi0: quantity   = 0.6 * u.dimensionless
    ca0: quantity    = 0.326e-3 * u.M
    co30: quantity   = 0.326e-3 * u.M
    ccal0: quantity  = 0.3 * u.dimensionless
    cara0: quantity  = 0.6 * u.dimensionless
    xdis: quantity   = 50.0 * u.cm       # x_d   (start of dissolution zone)
    xcem: quantity   = -100.0 * u.cm
    xcemf: quantity  = 1000.0 * u.cm
    length: quantity = 500.0 * u.cm
    Th: quantity     = 100.0 * u.cm      # h_d   (height of dissolution zone)
    phi00: quantity  = 0.5 * u.dimensionless
    ca00: quantity   = 0.326e-3 * u.M    # sqrt(Kc) / 2
    co300: quantity  = 0.326e-3 * u.M    # sqrt(Kc) / 2
    ccal00: quantity = 0.3 * u.dimensionless
    cara00: quantity = 0.6 * u.dimensionless

def map_Scenario():
    '''
    Maps the Fortran 'Scenario' parameters to the equivalent parameters in the
    Matlab and Python codes. These codes use slightly different parameter 
    names. Besides the mapping, also a few numerical conversions are applied.
    '''
    mapping = {"Ka":"KA",
               "Kc":"KC", 
               "cara0": "CA0", 
               "cara00": "CAIni",
               "ccal0": "CC0",
               "ccal00": "CCIni",
               "phi0": "Phi0", 
               "phi00":"PhiIni", 
               "xdis": "ShallowLimit",
               "S": "sedimentationrate",
               "m": "m1",
               "m1": "m2",
               "nn": "n1",
               "n1": "n2",
               "rhoa": "rhoa",
               "rhoc": "rhoc",
               "rhot": "rhot",
               "rhos0": "rhos",
               "rhow": "rhow",
               "beta": "beta",
               "D0ca": "D0Ca",
               "k1": "k1",
               "k2": "k2",
               "k3": "k3",
               "k4": "k4",
               "mua": "muA",
               "D0Ca": "DCa", 
               "D0co3": "DCO3",  
               "Phi0": "PhiNR",
               "phiinf": "PhiInfty",
               "length": "max_depth"
    }

    all_fields = fields(Scenario)

    allowed_fields = [field for field in all_fields
                      if field.name in mapping.keys()]
    
    derived_fields = [(mapping[field.name], field.type, field.default.magnitude) 
                      for field in allowed_fields]
                    
    # Here we append the fields that cannot be initialised directly from the
    # Scenario dataclass.
    derived_fields.append(("cCa0", float, None),
                          ("cCaIni", float, None),
                          ("cCO30"), float, None),
                          ("cCO3Ini", float, None),
                          ("DeepLimit", float, None),
                          ("rhos0", float, None),
                          ("Xstar", float, None),
                          ("Tstar", float, None),
                          ("b", float, None))   

    def post_init(self):
        # The Python parameters that need additional conversion
        # are initialised here.   
        self.cCa0 = self.ca0/np.sqrt(self.KC)
        self.cCaIni = self.ca00/np.sqrt(self.KC)
        self.cCO30 =  self.co30/np.sqrt(self.KC)
        self.cCO3Ini = self.co300/np.sqrt(self.KC)
        self.DeepLimit = self.ShallowLimit + self.Th
        self.rhos0 = self.rhoa * self.CA0 + self.rhoc * self.CC0 + \
                     self. rhot * (1 - (self.CA0 + self.CC0))
        self.Xstar = self.D0Ca / self.sedimentationrate
        self.Tstar = self.Xstar / self.sedimentationrate
        self.b = self.b/1e4
               
    derived_dataclass = make_dataclass("Mapped parameters", derived_fields,
                                       namespace={"__post_init__": post_init})

    return asdict(derived_dataclass())

@dataclass
class Solver:
    '''
    Initialises all the parameters for the solver.
    So parameters like time interval, time step and tolerance.
    '''
    dt: float     = 1.e-6
    eps: float    = 1.e-2
    tmax: int     = Scenario().D0ca.magnitude/Scenario().S.magnitude**2
    outt: int     =   1_000      # timesteps inbetween writing
    outx: int     =  25_000
    N: int        = 200

def cAthy(s: Scenario):
    return ((1 - s.phi0) * s.b * 9.81 * u['m/s²'] * s.rhow).to('cm⁻¹')

def write_input_cfg(path: Path, solver: Solver, scenario: Scenario):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["Solver"] = asdict(solver)

    units = { k: v.units for (k, v) in asdict(Scenario()).items() }
    magnitudes = { k: v.to(units[k]).magnitude
                   for (k, v) in asdict(scenario).items() }
    magnitudes["cAthy"] = cAthy(scenario).magnitude
    cfg["Scenario"] = magnitudes
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "input.cfg", "w") as f_cfg:
        cfg.write(f_cfg)

def run_marl_pde(path: Path, exe_dir: Path = Path(".")):
    run(exe_dir / "marl-pde", cwd=path, check=True)

def output_data(path: Path):
    return h5.File(path / "output.h5", mode="r")
# ~\~ end
