# Last modified by Niklas Hohmann (n.hohmann@uw.edu.pl) Oct 2021
## Parameters for Scenario A
#Taken from table 1 (p. 7)
import numpy as np
import matplotlib.pyplot as plt
from LMAHeureuxPorosityDiffV2 import LMAHeureuxPorosityDiffV2

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

depths = np.arange(0,500+2,2)
## Define Initial Conditions
#Initial conditions: homogeneous sediment at all depths (eqs 36)
AragoniteInitial = lambda depth = None: CAIni
CalciteInitial = lambda depth = None: CCIni
CaInitial = lambda depth = None: cCaIni
CO3Initial = lambda depth = None: cCO3Ini
PorInitial = lambda depth = None: PhiIni
## Define Boundary Conditions
# Boundary conditions: Constant support of input at the sediment-water interface (eqs. 35)
# Lack of diffusive flux at the bottom is hardcoded into the function LMAHeureux
AragoniteSurface = lambda time = None: CA0
CalciteSurface = lambda time = None: CC0
CaSurface = lambda time = None: cCa0
CO3Surface = lambda time = None: cCO30
PorSurface = lambda time = None: Phi0
## options
options = odeset('MaxStep',0.01,'RelTol',1e-10,'AbsTol',1e-10)
times = np.array([np.arange(0,1000+10,10)])
## run solver
sol = LMAHeureuxPorosityDiffV2(AragoniteInitial,CalciteInitial,CaInitial,
                               CO3Initial,PorInitial,AragoniteSurface,
                               CalciteSurface,CaSurface,CO3Surface,
                               PorSurface,times,depths,sedimentationrate,k1,k2,k3,k4,m1,m2,n1,n2,b,beta,rhos,rhow,rhos0,KA,KC,muA,D0Ca,PhiNR,PhiInfty,options,Phi0,DCa,DCO3,DeepLimit,ShallowLimit)
## plot results
#through time
timeslice = 10
print("type(sol) = {}".format(type(sol)))
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
