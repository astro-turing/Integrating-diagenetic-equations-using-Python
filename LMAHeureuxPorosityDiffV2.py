import numpy as np
from pde import FieldCollection, PDEBase, ScalarField
from numba import njit, prange
np.seterr(divide="raise", over="raise", under="warn", invalid="raise")
    
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

        # Fiadeiro-Veronis integration involves a coth and a reciprocal, which can
        # easily lead to FloatingPointError: overflow encountered in double_scalars.
        # To avoid this, better revert to either backwards or central differencing 
        # the Peclet number is very large or very small.
        self.Peclet_min = 1e-2
        self.Peclet_max = 1/self.Peclet_min
        self.check_implementation = False

    def get_state(self, AragoniteSurface, CalciteSurface, CaSurface, CO3Surface, 
                  PorSurface):
        # Return initial state and register forward and backward difference
        # operators. 
        AragoniteSurface.label = "ARA"
        CalciteSurface.label = "CAL"
        CaSurface.label = "Ca"
        CO3Surface.label = "CO3"
        PorSurface.label = "Po"

        return FieldCollection([AragoniteSurface, CalciteSurface, CaSurface, 
                                CO3Surface, PorSurface])

    def evolution_rate(self, state, t=0):
        CA, CC, cCa, cCO3, Phi = state   

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
 
        coC = CC * (((two_factors_low_lim - 1) ** self.n1) - self.nu2 * \
                    (1 - two_factors_upp_lim) ** self.n2)

        F = 1 - np.exp(10 - 10 / Phi)

        U = self.presum + self.rhorat * Phi ** 3 * F / (1 - Phi)
        
        W = self.presum - self.rhorat * Phi ** 2 * F
        
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

        # This is closer to the original form of (43) from l' Heureux than
        # the Matlab implementation.
        dPhi_dt = - (W * Phi).gradient(self.bc_Phi) \
                  + dPhi * Phi.laplace(self.bc_Phi) \
                  + self.Da * (1 - Phi) * (coA - self.lambda_ * coC)

        return FieldCollection([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt])

    def _make_pde_rhs_numba(self, state):
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
        # The following three numbers are also needed for Fiadeiro-Veronis.
        Peclet_min = self.Peclet_min
        Peclet_max = self.Peclet_max
        delta_x = state.grid._axes_coords[0][1] - state.grid._axes_coords[0][0]
        grad_back_CA = state.grid.make_operator("grad_back", bc = self.bc_CA)
        grad_forw_CA = state.grid.make_operator("grad_forw", bc = self.bc_CA)
        grad_back_CC = state.grid.make_operator("grad_back", bc = self.bc_CC)
        grad_forw_CC = state.grid.make_operator("grad_forw", bc = self.bc_CC)
        grad_back_cCa = state.grid.make_operator("grad_back", bc = self.bc_cCa)
        grad_forw_cCa = state.grid.make_operator("grad_forw", bc = self.bc_cCa)
        laplace_cCa = state.grid.make_operator("laplace", bc = self.bc_cCa)
        grad_back_cCO3 = state.grid.make_operator("grad_back", bc = self.bc_cCO3)
        grad_forw_cCO3 = state.grid.make_operator("grad_forw", bc = self.bc_cCO3)
        laplace_cCO3 = state.grid.make_operator("laplace", bc = self.bc_cCO3)
        grad_back_Phi = state.grid.make_operator("grad_back", bc = self.bc_Phi)
        grad_forw_Phi = state.grid.make_operator("grad_forw", bc = self.bc_Phi)
        laplace_Phi = state.grid.make_operator("laplace", bc = self.bc_Phi)

        @njit(nogil = True, parallel = True)
        def pde_rhs(state_data, t=0):
            """ compiled helper function evaluating right hand side """
            # Instead of the default central differenced gradient from py-pde
            # construct forward and backward differenced gradients and apply
            # either one of them, based on the sign of U.
            CA = state_data[0]
            CA_grad_back = grad_back_CA(CA)
            CA_grad_forw = grad_forw_CA(CA)

            CC = state_data[1]
            CC_grad_back = grad_back_CC(CC)
            CC_grad_forw = grad_forw_CC(CC)

            cCa = state_data[2]
            cCa_grad_back = grad_back_cCa(cCa)
            cCa_grad_forw = grad_forw_cCa(cCa)
            cCa_laplace = laplace_cCa(cCa)

            cCO3 = state_data[3]
            cCO3_grad_back = grad_back_cCO3(cCO3)
            cCO3_grad_forw = grad_forw_cCO3(cCO3)
            cCO3_laplace = laplace_cCO3(cCO3)

            Phi = state_data[4]
            Phi_grad_back = grad_back_Phi(Phi)
            Phi_grad_forw = grad_forw_Phi(Phi)
            Phi_laplace = laplace_Phi(Phi)

            rate = np.empty_like(state_data)

            # state_data.size should be the same as len(CA) or len(CC), check this.
            # So the number of depths, really.
            no_depths = state_data[0].size

            denominator = np.empty(no_depths)
            common_helper1 = np.empty(no_depths)
            common_helper2 = np.empty(no_depths)
            helper_cCa_grad = np.empty(no_depths)
            helper_cCO3_grad = np.empty(no_depths)
            F = np.empty(no_depths)
            U = np.empty(no_depths)
            W = np.empty(no_depths)
            two_factors = np.empty(no_depths)
            two_factors_upp_lim = np.empty(no_depths)
            two_factors_low_lim = np.empty(no_depths)
            three_factors = np.empty(no_depths)
            three_factors_upp_lim = np.empty(no_depths)
            three_factors_low_lim = np.empty(no_depths)
            coA = np.empty(no_depths)
            coC = np.empty(no_depths)
            common_helper3 = np.empty(no_depths)
            dPhi = np.empty(no_depths)
            dW_dx = np.empty(no_depths)
            one_minus_Phi = np.empty(no_depths)
            CA_grad = np.empty(no_depths)
            CC_grad = np.empty(no_depths)
            cCa_grad = np.empty(no_depths)
            cCO3_grad = np.empty(no_depths)
            Phi_grad = np.empty(no_depths)            

            for i in prange(no_depths):
                F[i] = 1 - np.exp(10 - 10 / Phi[i])

                U[i] = presum + rhorat * Phi[i] ** 3 * F[i]/ (1 - Phi[i])

                if U[i] > 0:
                    CA_grad[i] = CA_grad_back[i]
                    CC_grad[i] = CC_grad_back[i]
                else:
                    CA_grad[i] = CA_grad_forw[i]
                    CC_grad[i] = CC_grad_forw[i]

                W[i] = presum - rhorat * Phi[i] ** 2 * F[i]

                # Implementing equation 6 from l'Heureux.
                denominator[i] = 1 - 2 * np.log(Phi[i])

                # Fiadeiro-Veronis scheme for equations 42 and 43
                # from l'Heureux. 
                Peclet_cCa = W[i] * delta_x * denominator[i]/ (2. * dCa )
                if np.abs(Peclet_cCa) < Peclet_min:
                    sigma_cCa = 0
                elif np.abs(Peclet_cCa) > Peclet_max:
                    sigma_cCa = np.sign(W[i])
                else:
                     sigma_cCa = np.cosh(Peclet_cCa)/np.sinh(Peclet_cCa) - \
                        1/Peclet_cCa

                Peclet_cCO3 = W[i] * delta_x * denominator[i]/ (2. * dCO3)
                if np.abs(Peclet_cCO3) < Peclet_min:
                    sigma_cCO3 = 0
                elif np.abs(Peclet_cCO3) > Peclet_max:
                    sigma_cCO3 = np.sign(W[i])
                else:
                    sigma_cCO3 = np.cosh(Peclet_cCO3)/np.sinh(Peclet_cCO3) - \
                        1/Peclet_cCO3

                one_minus_Phi[i] = 1 - Phi[i]                 
                dPhi[i] = auxcon * F[i] * (Phi[i] ** 3) / one_minus_Phi[i]
                Peclet_Phi = W[i] * delta_x * denominator[i] / (2. * dPhi[i])
                if np.abs(Peclet_Phi) < Peclet_min:
                    sigma_Phi = 0
                elif np.abs(Peclet_Phi) > Peclet_max:
                    sigma_Phi = np.sign(W[i])
                else:
                    sigma_Phi = np.cosh(Peclet_Phi)/np.sinh(Peclet_Phi) - \
                        1/Peclet_Phi

                cCa_grad[i] = 0.5 * ((1-sigma_cCa) * cCa_grad_forw[i] + \
                              (1+sigma_cCa) * cCa_grad_back[i])
                cCO3_grad[i] = 0.5 * ((1-sigma_cCO3) * cCO3_grad_forw[i] + \
                              (1+sigma_cCO3) * cCO3_grad_back[i])
                Phi_grad[i] = 0.5 * ((1-sigma_Phi) * Phi_grad_forw [i]+ \
                              (1+sigma_Phi) * Phi_grad_back[i])

                common_helper1[i] = Phi[i]/denominator[i]
                common_helper2[i] = Phi_grad[i] * (2 + denominator[i]) \
                                    / denominator[i] ** 2
                helper_cCa_grad[i] = dCa * (common_helper2[i] * cCa_grad[i] \
                                     + common_helper1[i] * cCa_laplace[i])
                helper_cCO3_grad[i] = dCO3 * (common_helper2[i] * cCO3_grad[i] \
                                     + common_helper1[i] * cCO3_laplace[i])            

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

                common_helper3[i] = coA[i] - lambda_* coC[i]
                   
                dW_dx[i] = -rhorat * Phi_grad[i] * (2 * Phi[i] * F[i] + 10 * (F[i] - 1))    
       
                # This is dCA_dt
                rate[0][i] = - U[i] * CA_grad[i] - Da * ((1 - CA[i]) \
                             * coA[i] + lambda_ * CA[i] * coC[i])

                # This is dCC_dt
                rate[1][i] = - U[i] * CC_grad[i] + Da * (lambda_ * \
                             (1 - CC[i]) * coC[i] + CC[i] * coA[i])              

                # This is dcCa_dt
                rate[2][i] =  helper_cCa_grad[i]/Phi[i] - W[i] * \
                              cCa_grad[i] + Da * one_minus_Phi[i] * \
                              (delta - cCa[i]) * common_helper3[i] \
                              /Phi[i]                                 

                # This is dcCO3_dt
                rate[3][i] =  helper_cCO3_grad[i]/Phi[i] - W[i] * \
                              cCO3_grad[i] + Da * one_minus_Phi[i] * \
                              (delta - cCO3[i]) * common_helper3[i] \
                              /Phi[i]                       

                # This is dPhi_dt
                rate[4][i] = - (dW_dx[i] * Phi[i] + W[i] * Phi_grad[i]) \
                             + dPhi[i] * Phi_laplace[i] + Da * one_minus_Phi[i] \
                             * common_helper3[i] 

            return rate

        return pde_rhs   
