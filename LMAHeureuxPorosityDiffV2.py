from curses import KEY_C1
import numpy as np
from pde import FieldCollection, PDEBase, ScalarField
from sympy import evaluate
np.seterr(all="raise")
    
""" def LMAHeureuxPorosityDiffV2(AragoniteInitial = None,CalciteInitial = None,CaInitial = None,
    CO3Initial = None,PorInitial = None, AragoniteSurface = None, CalciteSurface = None,CaSurface = None,
    CO3Surface = None,PorSurface = None,times = None,depths = None,sedimentationrate = None, k1 = None, 
    k2 = None,k3 = None,k4 = None,m1 = None,m2 = None,n1 = None,n2 = None,b = None, beta = None,
    rhos = None,rhow = None,rhos0 = None,KA = None,KC = None,muA = None,D0Ca = None,PhiNR = None,
    PhiInfty = None,options = None,Phi0 = None,DCa = None,DCO3 = None,DeepLimit = None, 
    ShallowLimit = None): 
    ## Define Local constants
    Xstar = D0Ca / sedimentationrate
    
    Tstar = Xstar / sedimentationrate
    
    xmesh = depths / Xstar
    
    tspan = times / Tstar
    
    Da = k2 * Tstar
    
    lambda_ = k3 / k2
    
    nu1 = k1 / k2
    
    nu2 = k4 / k3
    
    dCa = DCa / D0Ca
    
    dCO3 = DCO3 / D0Ca
    
    delta = rhos / (muA * np.sqrt(KC))
    
    KRat = KC / KA
    
    g = 100 * 9.81
    
    auxcon = beta / (D0Ca * b * g * rhow * (PhiNR - PhiInfty))
    
    rhorat0 = (rhos0 / rhow - 1) * beta / sedimentationrate
    
    rhorat = (rhos / rhow - 1) * beta / sedimentationrate
    
    presum = 1 - rhorat0 * Phi0 ** 3 * (1 - np.exp(10 - 10 / Phi0)) / (1 - Phi0)
    
    ## Define Initial conditions
    InitialConditions = lambda depth = None: np.array([[AragoniteInitial(depth)],[CalciteInitial(depth)],
                                                       [CaInitial(depth)],[CO3Initial(depth)],[PorInitial(depth)]])
    ## Define Boundary conditions
    
    def BoundaryConditions(AragoniteSurface, CalciteSurface, CaSurface, CO3Surface, 
                           PorSurface, ul, t): 
        #eq. 35 top
        ql = np.array([[0],[0],[0],[0],[0]])
        pl = np.array([[ul(1) - AragoniteSurface(t)],[ul(2) - CalciteSurface(t)],[ul(3) - CaSurface(t)],
            [ul(4) - CO3Surface(t)],[ul(5) - PorSurface(t)]])
        pr = np.array([[0],[0],[0],[0],[0]])
        # eq 35 bottom
        qr = np.array([[1],[1],[1],[1],[1]])
        return pl,ql,pr,qr
        
    ## Define System of PDEs
    def PDEDef(x = None,__ = None,u = None,dudx = None): 
        ##System of PDEs of LHeureux, described in eqs. 40 to 43
        #abbreciations for readability
        CA = u(1)
        CC = u(2)
        cCa = u(3)
        cCO3 = u(4)
        Phi = u(5)
        #formulas for compact representation
    #dPhi=(auxcon*((Phi^3)/(1-Phi))*(1-exp(10-10/Phi))); # eq. 25 + 17 in comb with eq. 44
        dPhislash = (auxcon * (Phi / ((1 - Phi) ** 2)) * (np.exp(10 - 10 / Phi) * 
                    (2 * Phi ** 2 + 7 * Phi - 10) + Phi * (3 - 2 * Phi)))
        #OmegaPA=max(0,cCa*cCO3*KRat-1)^m1; #eq. 45
    #OmegaDA=(max(0,1-cCa*cCO3*KRat)^m2)*(x*Xstar <= DeepLimit && x*Xstar >= ShallowLimit); #eq. 45
    #OmegaPC=max(0,cCa*cCO3-1)^n1; #eq. 45
    #OmegaDC=max(0,1-cCa*cCO3)^n2; #eq. 45
        coA = CA * (((np.amax(0,1 - cCa * cCO3 * KRat) ** m2) * (x * Xstar <= DeepLimit 
              and x * Xstar >= ShallowLimit)) - nu1 * (np.amax(0,cCa * cCO3 * KRat - 1) ** m1))
        coC = CC * ((np.amax(0,cCa * cCO3 - 1) ** n1) - nu2 * (np.amax(0,1 - cCa * cCO3) ** n2))
        U = (presum + rhorat * Phi ** 3 * (1 - np.exp(10 - 10 / Phi)) / (1 - Phi))
        
        W = (presum - rhorat * Phi ** 2 * (1 - np.exp(10 - 10 / Phi)))
        
        Wslash = - rhorat * 2 * (Phi - (Phi + 5) * np.exp(10 - 10 / Phi))
        #Describe eqs. 40 to 43
        c = np.array([[1],[1],[Phi],[Phi],[1]])
        
        f = np.array([[0],[0],[Phi * dCa * dudx(3)],[Phi * dCO3 * dudx(4)],
                     [(auxcon * ((Phi ** 3) / (1 - Phi)) * (1 - np.exp(10 - 10 / Phi))) 
                     * dudx(5)]])
        
        s = np.array([[(- U * dudx(1) - Da * ((1 - CA) * coA + lambda_ * CA * coC))],
                     [(- U * dudx(2) + Da * (lambda_ * (1 - CC) * coC + CC * coA))],
                     [(- Phi * W * dudx(3) + Da * (1 - Phi) * (delta - cCa) * (coA - lambda_ * coC))],
                     [(- Phi * W * dudx(4) + Da * (1 - Phi) * (delta - cCO3) * (coA - lambda_ * coC))],
                     [(Da * (1 - Phi) * (coA - lambda_ * coC) - dudx(5) * (W + Wslash * Phi + dudx(5) * dPhislash))]])
        return c,f,s
    
    ## Solve PDE
    sol = pdepe(0,PDEDef,InitialConditions,BoundaryConditions,xmesh,tspan,options)
    # return c,f,s
    
    return sol """


class LMAHeureuxPorosityDiff(PDEBase):
    """SIR-model with diffusive mobility"""

    def __init__(self, AragoniteSurface, CalciteSurface, CaSurface, 
                CO3Surface, PorSurface, CA0, CC0, cCa0, cCO30, Phi0, 
                sedimentationrate, Xstar, Tstar, k1, k2, k3, k4, m1, m2, n1, 
                n2, b, beta, rhos, rhow, rhos0, KA, KC, muA, D0Ca, PhiNR, 
                PhiInfty, DCa, DCO3, not_too_shallow, not_too_deep):

        self.AragoniteSurface = AragoniteSurface
        self.CalciteSurface = CalciteSurface
        self.CaSurface = CaSurface
        self.CO3Surface = CO3Surface
        self.PorSurface = PorSurface
        self.bc_CA = [{"value": CA0}, {"curvature" : 0}]
        self.bc_CC = [{"value": CC0}, {"curvature": 0}]
        self.bc_cCa = [{"value": cCa0}, {"derivative": 0}]
        self.bc_cCO3 = [{"value": cCO30}, {"derivative": 0}]
        self.bc_Phi = [{"value": Phi0}, {"derivative": 0}]
        self.sedimentationrate = sedimentationrate
        self.Xstar = Xstar
        self.Tstar = Tstar
        self.k1 = k1
        self.k2 = k2
        self.nu1 = k1/k2
        self.k3 = k3
        self.k4 = k4
        self.nu2 = k4/k3
        self.m1 = m1
        self.m2 = m2
        self.n1 = n1
        self.n2 = n2
        self.b = b
        self.beta = beta
        self.rhos = rhos
        self.rhow = rhow
        self.rhos0 = rhos0
        self.KA = KA
        self.KC = KC
        self.KRat = self.KC/self.KA
        self.muA = muA
        self.D0Ca = D0Ca
        self.PhiNR = PhiNR
        self.PhiInfty = PhiInfty 
        self.Phi0 = Phi0
        self.DCa = DCa
        self.DCO3 = DCO3
        self.not_too_shallow = not_too_shallow
        self.not_too_deep = not_too_deep

    def get_state(self, AragoniteSurface, CalciteSurface, CaSurface, CO3Surface, 
                  PorSurface):
        """generate a suitable initial state"""
        AragoniteSurface.label = "ARA"
        CalciteSurface.label = "CAL"
        CaSurface.label = "Ca"
        CO3Surface.label = "CO3"
        PorSurface.label = "Po"

        return FieldCollection([AragoniteSurface, CalciteSurface, CaSurface, 
                                CO3Surface, PorSurface])

    def evolution_rate(self, state, t=0):
        CA, CC, cCa, cCO3, Phi = state

        g = 100 * 9.81
        dCa = self.DCa / self.D0Ca
        dCO3 = self.DCO3 / self.D0Ca
        delta = self.rhos / (self.muA * np.sqrt(self.KC))
        Xstar = self.Xstar
        Tstar = self.Tstar
        Da = self.k2 * Tstar
        lambda_ = self.k3 / self.k2
        auxcon = self.beta / (self.D0Ca * self.b * g * self.rhow * \
                 (self.PhiNR - self.PhiInfty))
        rhorat0 = (self.rhos0 / self.rhow - 1) * self.beta / \
                  self.sedimentationrate
        rhorat = (self.rhos / self.rhow - 1) * self.beta / \
                 self.sedimentationrate
        presum = 1 - rhorat0 * self.Phi0 ** 3 * \
                 (1 - np.exp(10 - 10 / self.Phi0)) / (1 - self.Phi0)   

        dPhislash = (auxcon * (Phi / ((1 - Phi) ** 2)) * (np.exp(10 - 10 / Phi) * 
                    (2 * Phi ** 2 + 7 * Phi - 10) + Phi * (3 - 2 * Phi)))      

        two_factors = cCa * cCO3
        two_factors_upp_lim = two_factors.to_scalar(lambda f: np.fmin(f,1))
        two_factors_low_lim = two_factors.to_scalar(lambda f: np.fmax(f,1))

        three_factors = two_factors * self.KRat
        three_factors_upp_lim = three_factors.to_scalar(lambda f: np.fmin(f,1))
        three_factors_low_lim = three_factors.to_scalar(lambda f: np.fmax(f,1))

        coA = CA * (((1 - three_factors_upp_lim) ** self.m2) * \
                    (self.not_too_deep * self.not_too_shallow) - self.nu1 * \
                    (three_factors_low_lim - 1) ** self.m1)
 
        # coA = CA * (((np.amax(0,1 - cCa * cCO3 * self.KRat) ** self.m2) * \
        #      (Depths.data * Xstar <= self.DeepLimit and \
        #      Depths.data * Xstar >= self.ShallowLimit)) - self.nu1 * \
        #      (np.amax(0,cCa * cCO3 * self.KRat - 1) ** self.m1))

        coC = CC * (((two_factors_low_lim - 1) ** self.n1) - self.nu2 * \
                    (1 - two_factors_upp_lim) ** self.n2)

        # coC = CC * ((np.amax(0,cCa * cCO3 - 1) ** self.n1) - self.nu2 * \
        #      (np.amax(0,1 - cCa * cCO3) ** self.n2))

        U = (presum + rhorat * Phi ** 3 * (1 - np.exp(10 - 10 / Phi)) / (1 - Phi))
        
        W = (presum - rhorat * Phi ** 2 * (1 - np.exp(10 - 10 / Phi)))
        
        Wslash = - rhorat * 2 * (Phi - (Phi + 5) * np.exp(10 - 10 / Phi))

        dCA_dt = - U * CA.gradient(self.bc_CA) - Da * ((1 - CA) * coA + lambda_ * CA * coC)

        dCC_dt = - U * CC.gradient(self.bc_CC) + Da * (lambda_ * (1 - CC) * coC + CC * coA)

        dcCa_dx = cCa.gradient(self.bc_cCa)[0]

        dcCa_dt = ((Phi * dCa * dcCa_dx).gradient(self.bc_cCa))/Phi -W * dcCa_dx \
                  + Da * (1 - Phi) * (delta - cCa) * (coA - lambda_ * coC)/Phi

        dcCO3_dx = cCO3.gradient(self.bc_cCO3)[0]

        dcCO3_dt = (Phi * dCO3 * dcCO3_dx).gradient(self.bc_cCO3)/Phi \
                   -W * dcCO3_dx + Da * (1 - Phi) * (delta - cCO3) * \
                   (coA - lambda_ * coC)/Phi

        dPhi_dx = Phi.gradient(self.bc_Phi)[0]

        dPhi_dt = ((auxcon * ((Phi ** 3) / (1 - Phi)) * (1 - np.exp(10 - 10 / Phi))) * dPhi_dx).gradient(self.bc_Phi) \
                  + Da * (1 - Phi) * (coA - lambda_ * coC) - dPhi_dx * (W + Wslash * Phi + dPhi_dx * dPhislash)

        return FieldCollection([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt])