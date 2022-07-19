import numpy as np
    
def LMAHeureuxPorosityDiffV2(AragoniteInitial = None,CalciteInitial = None,CaInitial = None,
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
    
    def BoundaryConditions(AragoniteSurface, CalciteSurface, CaSurface, CO3Surface, PorSurface, ul, t): 
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
                     [(auxcon * ((Phi ** 3) / (1 - Phi)) * (1 - np.exp(10 - 10 / Phi))) * dudx(5)]])
        
        s = np.array([[(- U * dudx(1) - Da * ((1 - CA) * coA + lambda_ * CA * coC))],
                     [(- U * dudx(2) + Da * (lambda_ * (1 - CC) * coC + CC * coA))],
                     [(- Phi * W * dudx(3) + Da * (1 - Phi) * (delta - cCa) * (coA - lambda_ * coC))],
                     [(- Phi * W * dudx(4) + Da * (1 - Phi) * (delta - cCO3) * (coA - lambda_ * coC))],
                     [(Da * (1 - Phi) * (coA - lambda_ * coC) - dudx(5) * (W + Wslash * Phi + dudx(5) * dPhislash))]])
        return c,f,s
    
    ## Solve PDE
    sol = pdepe(0,PDEDef,InitialConditions,BoundaryConditions,xmesh,tspan,options)
    # return c,f,s
    
    return sol