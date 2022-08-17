import numpy as np
from pde import FieldCollection, PDEBase, ScalarField, FieldBase
from numba import jit
np.seterr(divide="raise", over="raise", under="warn", invalid="raise")
    
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

    def __init__(self, Depths, slices_for_all_fields, CA0, CC0, cCa0, cCO30, Phi0, 
                sedimentationrate, Xstar, Tstar, k1, k2, k3, k4, m1, m2, n1, n2, 
                b, beta, rhos, rhow, rhos0, KA, KC, muA, D0Ca, PhiNR, PhiInfty, 
                DCa, DCO3, not_too_shallow, not_too_deep):  
        self.Depths = Depths    
        self.slices_for_all_fields = slices_for_all_fields
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
        self.g = 100 * 9.81
        self.dCa = self.DCa / self.D0Ca
        self.dCO3 = self.DCO3 / self.D0Ca
        self.delta = self.rhos / (self.muA * np.sqrt(self.KC))
        self.Da = self.k2 * self.Tstar
        self.lambda_ = self.k3 / self.k2
        self.auxcon = self.beta / (self.D0Ca * self.b * self.g * self.rhow * \
                 (self.PhiNR - self.PhiInfty))
        self.rhorat0 = (self.rhos0 / self.rhow - 1) * self.beta / \
                  self.sedimentationrate
        self.rhorat = (self.rhos / self.rhow - 1) * self.beta / \
                 self.sedimentationrate
        self.presum = 1 - self.rhorat0 * self.Phi0 ** 3 * \
                 (1 - np.exp(10 - 10 / self.Phi0)) / (1 - self.Phi0)  
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

    def evolution_rate(self, state: FieldBase, t: float = 0) -> FieldBase:
        return super().evolution_rate(state, t)                            

    def fun(self, t, y):
        CA = ScalarField(self.Depths, y[self.slices_for_all_fields[0]])
        CC = ScalarField(self.Depths, y[self.slices_for_all_fields[1]])
        cCa = ScalarField(self.Depths, y[self.slices_for_all_fields[2]])
        cCO3 = ScalarField(self.Depths, y[self.slices_for_all_fields[3]])
        Phi = ScalarField(self.Depths, y[self.slices_for_all_fields[4]])

        # dPhislash = (self.auxcon * (Phi / ((1 - Phi) ** 2)) * (np.exp(10 - 10 / Phi) * 
        #            (2 * Phi ** 2 + 7 * Phi - 10) + Phi * (3 - 2 * Phi)))    

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

        F = 1 - np.exp(10 - 10 / Phi)

        U = self.presum + self.rhorat * Phi ** 3 * F / (1 - Phi)
        
        W = self.presum - self.rhorat * Phi ** 2 * F
        
        # Wslash = - self.rhorat * 2 * (Phi - (Phi + 5) * np.exp(10 - 10 / Phi))

        dCA_dt = - U * CA.gradient(self.bc_CA) - self.Da * ((1 - CA) * coA + self.lambda_ * CA * coC)

        dCC_dt = - U * CC.gradient(self.bc_CC) + self.Da * (self.lambda_ * (1 - CC) * coC + CC * coA)

        dcCa_dx = cCa.gradient(self.bc_cCa)[0]

        dcCa_dt = ((Phi * self.dCa * dcCa_dx).gradient(self.bc_cCa))/Phi -W * dcCa_dx \
                  + self.Da * (1 - Phi) * (self.delta - cCa) * (coA - self.lambda_ * coC)/Phi

        dcCO3_dx = cCO3.gradient(self.bc_cCO3)[0]

        dcCO3_dt = (Phi * self.dCO3 * dcCO3_dx).gradient(self.bc_cCO3)/Phi \
                   -W * dcCO3_dx + self.Da * (1 - Phi) * (self.delta - cCO3) * \
                   (coA - self.lambda_ * coC)/Phi

        dPhi = self.auxcon * F * (Phi ** 3) / (1 - Phi)

        # dPhi_dx = Phi.gradient(self.bc_Phi)[0]

        # dPhi_dt = ((self.auxcon * ((Phi ** 3) / (1 - Phi)) * \
        #           (1 - np.exp(10 - 10 / Phi))) * dPhi_dx).gradient(self.bc_Phi) \
        #          + self.Da * (1 - Phi) * (coA - self.lambda_ * coC) - dPhi_dx * \
        #            (W + Wslash * Phi + dPhi_dx * dPhislash)


        # This is closer to the original form of (43) from l' Heureux than
        # the Matlab implementation.
        dPhi_dt = - (W * Phi).gradient(self.bc_Phi) \
                  + dPhi * Phi.laplace(self.bc_Phi) \
                  + self.Da * (1 - Phi) * (coA - self.lambda_ * coC)

        return FieldCollection([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt]).data.ravel()

    def fun_numba(self, t, y):
        """ the numba-accelerated evolution equation """      

        # make attributes locally available
        KRat = self.KRat
        m1 = self.m1
        m2  = self.m2
        n1 = self.n1
        n2 = self.n2
        nu1 = self.nu1
        nu2 = self.nu2
        not_too_deep = self.not_too_deep.data
        not_too_shallow = self.not_too_shallow.data
        presum = self.presum
        rhorat= self.rhorat
        lambda_ = self.lambda_
        Da = self.Da
        dCa = self.dCa
        dCO3 = self.dCO3
        delta = self.delta
        auxcon = self.auxcon
        CA_sl= self.slices_for_all_fields[0]
        CC_sl = self.slices_for_all_fields[1]
        cCa_sl = self.slices_for_all_fields[2]
        cCO3_sl = self.slices_for_all_fields[3]
        Phi_sl = self.slices_for_all_fields[4]

        CA = ScalarField(self.Depths, y[CA_sl])
        CC = ScalarField(self.Depths, y[CC_sl])
        cCa = ScalarField(self.Depths, y[cCa_sl])
        cCO3 = ScalarField(self.Depths, y[cCO3_sl])
        Phi = ScalarField(self.Depths, y[Phi_sl])          

        gradient_CA = CA.grid.make_operator("gradient", bc=self.bc_CA)
        gradient_CC = CC.grid.make_operator("gradient", bc=self.bc_CC)
        gradient_cCa = cCa.grid.make_operator("gradient", bc=self.bc_cCa)
        gradient_cCO3 = cCO3.grid.make_operator("gradient", bc=self.bc_cCO3)
        gradient_Phi = Phi.grid.make_operator("gradient", bc=self.bc_Phi)
        laplace_Phi = Phi.grid.make_operator("laplace", bc=self.bc_Phi)

        @jit(nopython = True, nogil= True, cache = True, parallel = True)
        def pde_rhs(y):
            """ compiled helper function evaluating right hand side """
            CA = y[CA_sl]
            CA_grad = gradient_CA(CA)[0]
            CC = y[CC_sl]
            CC_grad = gradient_CC(CC)[0]
            cCa = y[cCa_sl]
            cCa_grad = gradient_cCa(cCa)[0]
            cCO3 = y[cCO3_sl]
            cCO3_grad = gradient_cCO3(cCO3)[0]
            Phi = y[Phi_sl]
            Phi_laplace = laplace_Phi(Phi)

            helper_cCa_grad = gradient_cCa(Phi * dCa * cCa_grad)[0]
            helper_cCO3_grad = gradient_cCO3(Phi * dCO3 * cCO3_grad)[0]

            rate = np.empty_like(y)

            # state_data.size should be the same as len(CA) or len(CC), check this.
            # So the number of depths, really.
            no_depths = CA.size

            two_factors = np.empty(no_depths)
            two_factors_upp_lim = np.empty(no_depths)
            two_factors_low_lim = np.empty(no_depths)
            three_factors = np.empty(no_depths)
            three_factors_upp_lim = np.empty(no_depths)
            three_factors_low_lim = np.empty(no_depths)
            coA = np.empty(no_depths)
            coC = np.empty(no_depths)
            U = np.empty(no_depths)
            W = np.empty(no_depths)
            F = np.empty(no_depths)

            for i in range(no_depths):
                F[i] = 1 - np.exp(10 - 10 / Phi[i])

                U[i] = presum + rhorat * Phi[i] ** 3 * F[i]/ (1 - Phi[i])
        
                W[i] = presum - rhorat * Phi[i] ** 2 * F[i]

            helper_Phi_grad = gradient_Phi(W * Phi)[0]                        

            dPhi = np.empty(no_depths)

            for i in range(no_depths):
                two_factors[i] = cCa[i] * cCO3[i]
                two_factors_upp_lim[i] = min(two_factors[i],1)
                two_factors_low_lim[i] = max(two_factors[i],1)
                three_factors[i] = two_factors[i] * KRat
                three_factors_upp_lim[i] = min(three_factors[i],1)
                three_factors_low_lim[i] = max(three_factors[i],1)

                coA[i] = CA[i] * (((1 - three_factors_upp_lim[i]) ** m2) * \
                    (not_too_deep[i] * not_too_shallow[i]) - nu1 * \
                    (three_factors_low_lim[i] - 1) ** m1)

                coC[i] = CC[i] * (((two_factors_low_lim[i] - 1) ** n1) - nu2 * \
                    (1 - two_factors_upp_lim[i]) ** n2)
                
                # This is dCA_dt
                rate[CA_sl.start + i] = - U[i] * CA_grad[i] - Da * ((1 - CA[i]) \
                                        * coA[i] + lambda_ * CA[i] * coC[i])

                # This is dCC_dt
                rate[CC_sl.start + i] = - U[i] * CC_grad[i] + Da * (lambda_ * \
                                        (1 - CC[i]) * coC[i] + CC[i] * coA[i])

                # This is dcCa_dt
                rate[cCa_sl.start + i] =  helper_cCa_grad[i]/Phi[i] -W[i] * \
                                          cCa_grad[i] + Da * (1 - Phi[i]) * \
                                          (delta - cCa[i]) * (coA[i] - lambda_ \
                                          * coC[i])/Phi[i]

                # This is dcCO3_dt
                rate[cCO3_sl.start + i] =  helper_cCO3_grad[i]/Phi[i] -W[i] * \
                                           cCO3_grad[i] + Da * (1 - Phi[i]) * \
                                           (delta - cCO3[i]) * (coA[i] - \
                                           lambda_ * coC[i])/Phi[i]

                dPhi[i] = auxcon * F[i] * (Phi[i] ** 3) / (1 - Phi[i])        

                # This is dPhi_dt
                rate[Phi_sl.start + i] = - helper_Phi_grad[i] + dPhi[i] * \
                                         Phi_laplace[i] + Da * (1 - Phi[i]) \
                                         * (coA[i] - lambda_ * coC[i])
            return rate

        return pde_rhs(y)