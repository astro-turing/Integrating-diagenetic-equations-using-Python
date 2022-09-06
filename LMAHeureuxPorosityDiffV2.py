import numpy as np
from pde import FieldCollection, PDEBase, ScalarField, FieldBase
from numba import jit, prange
np.seterr(divide="raise", over="raise", under="warn", invalid="raise")
from scipy.sparse import csr_matrix, find   
from Compute_jacobian import Jacobian

class LMAHeureuxPorosityDiff(PDEBase):
    """SIR-model with diffusive mobility"""

    def __init__(self, Depths, slices_for_all_fields, CA0, CC0, cCa0, cCO30, Phi0, 
                sedimentationrate, Xstar, Tstar, k1, k2, k3, k4, m1, m2, n1, n2, 
                b, beta, rhos, rhow, rhos0, KA, KC, muA, D0Ca, PhiNR, PhiInfty, 
                DCa, DCO3, not_too_shallow, not_too_deep):  
        self.no_fields = 5
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
        self.not_too_shallow = not_too_shallow.data
        self.not_too_deep = not_too_deep.data      
        self.CA_sl= self.slices_for_all_fields[0]
        self.CC_sl = self.slices_for_all_fields[1]
        self.cCa_sl = self.slices_for_all_fields[2]
        self.cCO3_sl = self.slices_for_all_fields[3]
        self.Phi_sl = self.slices_for_all_fields[4]
        self.gradient_CA = self.Depths.make_operator("gradient", bc=self.bc_CA)
        self.gradient_CC = self.Depths.make_operator("gradient", bc=self.bc_CC)
        self.gradient_cCa = self.Depths.make_operator("gradient", \
            bc=self.bc_cCa)
        self.gradient_cCO3 = self.Depths.make_operator("gradient", \
            bc=self.bc_cCO3)
        self.gradient_Phi = self.Depths.make_operator("gradient", \
            bc=self.bc_Phi)
        self.laplace_Phi = self.Depths.make_operator("laplace", bc=self.bc_Phi)        

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

        # Implementing equation 6 from l'Heureux.
        denominator = 1 - 2 * ScalarField(self.Depths, np.log(Phi.data))
        # Implementing equation 6 from l'Heureux.
        dcCa_dt = ((Phi * self.dCa * dcCa_dx/denominator).gradient(self.bc_cCa))/Phi -W * dcCa_dx \
                  + self.Da * (1 - Phi) * (self.delta - cCa) * (coA - self.lambda_ * coC)/Phi

        dcCO3_dx = cCO3.gradient(self.bc_cCO3)[0]

        # Implementing equation 6 from l'Heureux.
        dcCO3_dt = (Phi * self.dCO3 * dcCO3_dx/denominator).gradient(self.bc_cCO3)/Phi \
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
        CA = y[self.CA_sl]
        CC = y[self.CC_sl]
        cCa = y[self.cCa_sl]
        cCO3 = y[self.cCO3_sl]
        Phi = y[self.Phi_sl]   

        rhs = LMAHeureuxPorosityDiff.pde_rhs(CA, CC, cCa, cCO3, Phi, self.KRat, \
            self.m1, self.m2, self.n1, self.n2, self.nu1, self.nu2, \
            self.not_too_deep, self.not_too_shallow, self.presum, self.rhorat, \
            self.lambda_, self.Da, self.dCa, self.dCO3, self.delta, self.auxcon, \
            self.gradient_CA, self.gradient_CC, self.gradient_cCa, \
            self.gradient_cCO3,self.gradient_Phi, self.laplace_Phi, \
            no_depths = self.Depths.shape[0])

        # print("Right-hand side evaluated")

        return rhs

    def jac(self, t, y):

        CA = y[self.CA_sl]
        CC = y[self.CC_sl]
        cCa = y[self.cCa_sl]
        cCO3 = y[self.cCO3_sl]
        Phi = y[self.Phi_sl]  

        jacob_csr = csr_matrix(Jacobian(CA, CC, cCa, cCO3, Phi, \
            self.KRat, self.m1, self.m2, \
            self.n1, self.n2, self.nu1, self.nu2, self.not_too_deep, \
            self.not_too_shallow, self.lambda_, \
            self.Da, self.delta, \
            no_depths = self.Depths.shape[0], no_fields = self.no_fields))

        # Check that we should have 5**2 sets of Jacobian values for 5 fields,
        # with each set comprised of no_depth numbers.
        """     no_sets_of_jacobian_values = len(all_jac_values_rows_and_cols)
        assert no_sets_of_jacobian_values == self.no_fields ** 2
       
        no_depths = self.Depths.shape[0]
        n = self.no_fields * no_depths
        jacob_csr = csr_matrix((n, n))
 
        for item in all_jac_values_rows_and_cols: """
            # item[0] are the self.Depths.shape[0] Jacobian values.
            # item[1] is a tuple of self.Depths.shape[0] row indices and
            # self.Depths.shape[0] column indices.
            # This loop fills the sparse matrix.
        """     data, row_and_col = item
            jacob_csr += csr_matrix((data, row_and_col), shape = (n, n)) """

        # Check that the non-zero values in the Jacobian occur ath the same 
        # indices as the Jacobian sparsity matrix.
        # Perhaps we do not need this check for every computation of the
        # Jacobian.
        """ sparsity_matrix = self.jacobian_sparsity()
        nonzero_sparsity = find(sparsity_matrix)
        nonzero_jacobian = find(jacob_csr) """
        # print("Jacobian matrix evaluated")
        # assert np.allclose(nonzero_sparsity[0], nonzero_jacobian[0]) 
        # assert np.allclose(nonzero_sparsity[1], nonzero_jacobian[1])       

        return jacob_csr

    @jit(nopython = True, nogil= True, parallel = True)
    def pde_rhs(CA, CC, cCa, cCO3, Phi, KRat, m1, m2, n1, n2, nu1, nu2, \
            not_too_deep, not_too_shallow, presum, rhorat, lambda_, Da, dCa, \
            dCO3, delta, auxcon, gradient_CA, gradient_CC, gradient_cCa, \
            gradient_cCO3,gradient_Phi, laplace_Phi, no_depths):
        """ compiled helper function evaluating right hand side """
        CA_grad = gradient_CA(CA)[0]
        CC_grad = gradient_CC(CC)[0]
        cCa_grad = gradient_cCa(cCa)[0]
        cCO3_grad = gradient_cCO3(cCO3)[0]
        Phi_laplace = laplace_Phi(Phi)

        # Implementing equation 6 from l'Heureux.
        denominator = 1 - 2 * np.log(Phi)
        helper_cCa_grad = gradient_cCa(Phi * dCa * cCa_grad/denominator)[0]
        helper_cCO3_grad = gradient_cCO3(Phi * dCO3 * cCO3_grad/denominator)[0]

        rate = np.empty(5 * no_depths)

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

        for i in prange(no_depths):
            F[i] = 1 - np.exp(10 - 10 / Phi[i])

            U[i] = presum + rhorat * Phi[i] ** 3 * F[i]/ (1 - Phi[i])
    
            W[i] = presum - rhorat * Phi[i] ** 2 * F[i]

        helper_Phi_grad = gradient_Phi(W * Phi)[0]                        

        dPhi = np.empty(no_depths)

        for i in prange(no_depths):
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
            rate[i] = - U[i] * CA_grad[i] - Da * ((1 - CA[i]) \
                                    * coA[i] + lambda_ * CA[i] * coC[i])

            # This is dCC_dt
            rate[no_depths + i] = - U[i] * CC_grad[i] + Da * (lambda_ * \
                                    (1 - CC[i]) * coC[i] + CC[i] * coA[i])

            # This is dcCa_dt
            rate[2 * no_depths + i] =  helper_cCa_grad[i]/Phi[i] -W[i] * \
                                        cCa_grad[i] + Da * (1 - Phi[i]) * \
                                        (delta - cCa[i]) * (coA[i] - lambda_ \
                                        * coC[i])/Phi[i]

            # This is dcCO3_dt
            rate[3 * no_depths + i] =  helper_cCO3_grad[i]/Phi[i] -W[i] * \
                                        cCO3_grad[i] + Da * (1 - Phi[i]) * \
                                        (delta - cCO3[i]) * (coA[i] - \
                                        lambda_ * coC[i])/Phi[i]

            dPhi[i] = auxcon * F[i] * (Phi[i] ** 3) / (1 - Phi[i])        

            # This is dPhi_dt
            rate[4 * no_depths + i] = - helper_Phi_grad[i] + dPhi[i] * \
                                        Phi_laplace[i] + Da * (1 - Phi[i]) \
                                        * (coA[i] - lambda_ * coC[i])
        return rate

    def jacobian_sparsity(self):
        no_depths = self.Depths.shape[0]
        n = self.no_fields * no_depths
        jacob_csr = csr_matrix((n, n))
        data = np.ones(no_depths)
        row = np.arange(no_depths)
        col = np.arange(no_depths)
        for i in range(self.no_fields):
            for j in range(self.no_fields):
                if i < self.no_fields - 1 or j > 1:
                    jacob_csr += csr_matrix((data, (i * no_depths + row, \
                        j * no_depths + col)), shape = (n, n))

        return jacob_csr


