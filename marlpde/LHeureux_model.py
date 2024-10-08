from pde.grids.operators.cartesian import _make_derivative
from pde import FieldCollection, ScalarField
import numpy as np
from numba import njit

np.seterr(divide="raise", over="raise", under="warn", invalid="raise")

class LMAHeureuxPorosityDiff():
    '''
    This class defines all the parameters for the diagenetic model.
    '''
    def __init__(self, Depths, slices_all_fields, not_too_shallow, not_too_deep,
                 CA0, CC0, cCa0, cCO30, Phi0, sedimentationrate, Xstar, Tstar, 
                 k1, k2, k3, k4, m1, m2, n1, n2, b, beta, rhos, rhow, rhos0, 
                 KA, KC, muA, D0Ca, PhiNR, PhiInfty, PhiIni, DCa, DCO3, 
                 FV_switch):  
        self.no_fields = 5
        self.Depths = Depths    
        self.Depths.register_operator("grad_back", \
            lambda grid: _make_derivative(grid, method="backward"))
        self.Depths.register_operator("grad_forw", \
            lambda grid: _make_derivative(grid, method="forward"))
        self.delta_x = self.Depths._axes_coords[0][1] - \
                       self.Depths._axes_coords[0][0]
        self.slices_all_fields = slices_all_fields
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
        self.FV_switch = FV_switch

        self.CA_sl= self.slices_all_fields[0]
        self.CC_sl = self.slices_all_fields[1]
        self.cCa_sl = self.slices_all_fields[2]
        self.cCO3_sl = self.slices_all_fields[3]
        self.Phi_sl = self.slices_all_fields[4]
        
        # Fiadeiro-Veronis integration involves a coth and a reciprocal, which can
        # easily lead to FloatingPointError: overflow encountered in double_scalars.
        # To avoid this, better revert to either backwards or central differencing 
        # the Peclet number is very large or very small.
        self.Peclet_min = 1e-2
        self.Peclet_max = 1/self.Peclet_min

        self.last_t = 0.  # Helper variable for progress bar.

        # Make operators for use in Numba compiled (i.e. jit compiled) functions.
        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and apply
        # either one of them, based on the sign of U.
        self.CA_grad_back_op = self.Depths.make_operator("grad_back", self.bc_CA)
        self.CA_grad_forw_op = self.Depths.make_operator("grad_forw", self.bc_CA)
        self.CC_grad_back_op = self.Depths.make_operator("grad_back", self.bc_CC)
        self.CC_grad_forw_op = self.Depths.make_operator("grad_forw", self.bc_CC)
        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and give them
        # appropriate weights according to a Fiadeiro-Veronis scheme.
        self.cCa_grad_back_op = self.Depths.make_operator("grad_back", self.bc_cCa)
        self.cCa_grad_forw_op = self.Depths.make_operator("grad_forw", self.bc_cCa)      
        self.cCa_laplace_op = self.Depths.make_operator("laplace", self.bc_cCa)
        self.cCO3_grad_back_op = self.Depths.make_operator("grad_back", self.bc_cCO3)
        self.cCO3_grad_forw_op = self.Depths.make_operator("grad_forw", self.bc_cCO3)      
        self.cCO3_laplace_op = self.Depths.make_operator("laplace", self.bc_cCO3)
        self.Phi_grad_back_op = self.Depths.make_operator("grad_back", self.bc_Phi)
        self.Phi_grad_forw_op = self.Depths.make_operator("grad_forw", self.bc_Phi)
        self.Phi_laplace_op = self.Depths.make_operator("laplace", self.bc_Phi)

        # Make sure integration stops when we field values become less than zero
        # or more than one, in some cases. Or just register that this is happening
        # and continue integration, which corresponds to "False".
        setattr(self.zeros.__func__, "terminal", False)
        setattr(self.zeros_CA.__func__, "terminal", False)
        setattr(self.zeros_CC.__func__, "terminal", False)
        setattr(self.ones_CA_plus_CC.__func__, "terminal", False)
        setattr(self.ones_Phi.__func__, "terminal", False)
        setattr(self.zeros_U.__func__, "terminal", False)
        setattr(self.zeros_W.__func__, "terminal", False)

        # Accommodate for fixed porosity diffusion coefficient.
        # Beware that the functional Jacobian has not been aligned with this,
        # i.e. it was derived using a porosity diffusion coefficient depending
        # on a time-varying porosity. This will all be fine if you do not set
        # 'jac=eq.jac' in solve_ivp, but leave it empty such that a numerically
        # approximated Jacobian will be used.
        self.PhiIni = PhiIni
        self.F_fixed = 1 - np.exp(10 - 10 / self.PhiIni)
        self.dPhi_fixed = self.auxcon * self.F_fixed *\
                          self.PhiIni ** 3 / (1 - self.PhiIni)         

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

    @staticmethod
    @njit
    def calculate_sigma(Peclet, W_data, Peclet_min, Peclet_max):
        # Assuming the arrays are 1D.
        sigma = np.empty(Peclet.size)
        for i in range(sigma.size):
            if np.abs(Peclet[i]) < Peclet_min:
                sigma[i] = 0
            elif np.abs(Peclet[i]) > Peclet_max:
                sigma[i] = np.sign(W_data[i])
            else:
                sigma[i] = np.cosh(Peclet[i])/np.sinh(Peclet[i]) - \
                    1/Peclet[i]
        return sigma                       

    def fun(self, t, y, progress_proxy, progress_dt, t0):
        """ 
        For tqdm to monitor progress. 
        From 
        https://stackoverflow.com/questions/59047892/
        how-to-monitor-the-process-of-scipy-odeint """
        if self.last_t == 0.:
            self.last_t = t0        
        n = int((t - self.last_t) / progress_dt)
        progress_proxy.update(n)
        self.last_t += n * progress_dt        
        
        CA = ScalarField(self.Depths, y[self.slices_all_fields[0]])
        CC = ScalarField(self.Depths, y[self.slices_all_fields[1]])
        cCa = ScalarField(self.Depths, y[self.slices_all_fields[2]])
        cCO3 = ScalarField(self.Depths, y[self.slices_all_fields[3]])
        Phi = ScalarField(self.Depths, y[self.slices_all_fields[4]])

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
        
        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and apply
        # either one of them, based on the sign of U.
        CA_grad_back = CA.apply_operator("grad_back", self.bc_CA)
        CA_grad_forw = CA.apply_operator("grad_forw", self.bc_CA)
        CA_grad = ScalarField(self.Depths, np.where(U.data>0, CA_grad_back.data, \
            CA_grad_forw.data))

        CC_grad_back = CC.apply_operator("grad_back", self.bc_CC)
        CC_grad_forw = CC.apply_operator("grad_forw", self.bc_CC)    
        CC_grad = ScalarField(self.Depths, np.where(U.data>0, CC_grad_back.data, \
            CC_grad_forw.data))

        W = self.presum - self.rhorat * Phi ** 2 * F
        
        dCA_dt = - U * CA_grad - self.Da * ((1 - CA) \
                 * coA + self.lambda_ * CA * coC)

        dCC_dt = - U * CC_grad + self.Da * (self.lambda_ \
                 * (1 - CC) * coC + CC * coA)

        # Implementing equation 6 from l'Heureux.
        denominator = 1 - 2 * np.log(Phi)
        # dPhi = self.auxcon * F * (Phi ** 3) / (1 - Phi)
        dPhi = self.dPhi_fixed

        if self.FV_switch:
            # Fiadeiro-Veronis scheme for equations 42 and 43
            # from l'Heureux. 
            common_Peclet  = W * self.delta_x / 2. 
            Peclet_cCa =  common_Peclet * denominator/ self.dCa       
            sigma_cCa_data = LMAHeureuxPorosityDiff.calculate_sigma(\
                Peclet_cCa.data, W.data, self.Peclet_min, self.Peclet_max)
            sigma_cCa = ScalarField(self.Depths, sigma_cCa_data)

            Peclet_cCO3 = common_Peclet * denominator/ self.dCO3      
            sigma_cCO3_data = LMAHeureuxPorosityDiff.calculate_sigma(\
                Peclet_cCO3.data, W.data, self.Peclet_min, self.Peclet_max)
            sigma_cCO3 = ScalarField(self.Depths, sigma_cCO3_data)

            Peclet_Phi = common_Peclet / dPhi
            sigma_Phi_data = LMAHeureuxPorosityDiff.calculate_sigma(\
                Peclet_Phi.data, W.data, self.Peclet_min, self.Peclet_max)
            sigma_Phi = ScalarField(self.Depths, sigma_Phi_data)
        else:
            sigma_cCa = 0
            sigma_cCO3 = 0
            sigma_Phi = 0

        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and give them
        # appropriate weights according to a Fiadeiro-Veronis scheme.
        cCa_grad_back = cCa.apply_operator("grad_back", self.bc_cCa)
        cCa_grad_forw = cCa.apply_operator("grad_forw", self.bc_cCa)      
        cCa_grad = 0.5 * ((1-sigma_cCa) * cCa_grad_forw +\
             (1+sigma_cCa) * cCa_grad_back)
        cCa_laplace = cCa.laplace(self.bc_cCa)

        cCO3_grad_back = cCO3.apply_operator("grad_back", self.bc_cCO3)
        cCO3_grad_forw = cCO3.apply_operator("grad_forw", self.bc_cCO3)
        cCO3_grad = 0.5 * ((1-sigma_cCO3) * cCO3_grad_forw +\
             (1+sigma_cCO3) * cCO3_grad_back)
        cCO3_laplace = cCO3.laplace(self.bc_cCO3)

        Phi_grad_back = Phi.apply_operator("grad_back", self.bc_Phi)
        Phi_grad_forw = Phi.apply_operator("grad_forw", self.bc_Phi)
        Phi_grad = 0.5 * ((1-sigma_Phi) * Phi_grad_forw +\
             (1+sigma_Phi) * Phi_grad_back)
        Phi_laplace = Phi.laplace(self.bc_Phi)

        Phi_denom = Phi/denominator
        grad_Phi_denom = Phi_grad * (denominator + 2) / denominator ** 2

        common_helper = coA - self.lambda_ * coC
       
        dcCa_dt = (cCa_grad * grad_Phi_denom + Phi_denom * cCa_laplace) \
                  * self.dCa /Phi - W * cCa_grad \
                  + self.Da * (1 - Phi) * (self.delta - cCa) * common_helper / Phi

        dcCO3_dt = (cCO3_grad * grad_Phi_denom + Phi_denom * cCO3_laplace) \
                   * self.dCO3/Phi - W * cCO3_grad \
                   + self.Da * (1 - Phi) * (self.delta - cCO3) * common_helper / Phi

        dW_dx = -self.rhorat * Phi_grad * (2 * Phi * F + 10 * (F - 1))

        dPhi_dt = - (Phi * dW_dx + W * Phi_grad) \
                  + dPhi * Phi_laplace \
                  + self.Da * (1 - Phi) * common_helper

        return FieldCollection([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt]).data.ravel()

    def fun_numba(self, t, y, progress_proxy, progress_dt, t0):
        """ For tqdm to monitor progress.
        From 
        https://stackoverflow.com/questions/59047892/
        how-to-monitor-the-process-of-scipy-odeint """
        if self.last_t == 0.:
            self.last_t = t0        
        n = int((t - self.last_t) / progress_dt)
        progress_proxy.update(n)
        self.last_t += n * progress_dt        

        """ the numba-accelerated evolution equation """     
        KRat = self.KRat
        m1 = self.m1
        m2 = self.m2
        n1 = self.n1
        n2 = self.n2
        nu1 = self.nu1
        nu2 = self.nu2
        not_too_deep = self.not_too_deep
        not_too_shallow = self.not_too_shallow
        presum = self.presum
        rhorat = self.rhorat
        lambda_ = self.lambda_
        Da= self.Da
        dCa = self.dCa
        dCO3 = self.dCO3
        delta = self.delta
        auxcon = self.auxcon
        delta_x = self.delta_x
        Peclet_min = self.Peclet_min
        Peclet_max = self.Peclet_max
        no_depths = self.Depths.shape[0]
        dPhi_fixed = self.dPhi_fixed
        FV_switch = self.FV_switch

        CA = y[self.CA_sl]
        CC = y[self.CC_sl]
        cCa = y[self.cCa_sl]
        cCO3 = y[self.cCO3_sl]
        Phi = y[self.Phi_sl]

        CA_grad_back_op = self.CA_grad_back_op
        CA_grad_forw_op = self.CA_grad_forw_op
        CC_grad_back_op = self.CC_grad_back_op 
        CC_grad_forw_op = self.CC_grad_forw_op 
        cCa_grad_back_op = self.cCa_grad_back_op 
        cCa_grad_forw_op = self.cCa_grad_forw_op 
        cCa_laplace_op =  self.cCa_laplace_op 
        cCO3_grad_back_op = self.cCO3_grad_back_op
        cCO3_grad_forw_op = self.cCO3_grad_forw_op
        cCO3_laplace_op = self.cCO3_laplace_op
        Phi_grad_back_op = self.Phi_grad_back_op 
        Phi_grad_forw_op = self.Phi_grad_forw_op 
        Phi_laplace_op = self.Phi_laplace_op 

        return LMAHeureuxPorosityDiff.pde_rhs(CA, CC, cCa, cCO3, Phi, 
                                              CA_grad_back_op, CA_grad_forw_op, 
                                              CC_grad_back_op, CC_grad_forw_op,
                                              cCa_grad_back_op, cCa_grad_forw_op, 
                                              cCa_laplace_op, cCO3_grad_back_op, 
                                              cCO3_grad_forw_op, cCO3_laplace_op, 
                                              Phi_grad_back_op, Phi_grad_forw_op, 
                                              Phi_laplace_op, KRat, m1, m2, n1, 
                                              n2, nu1, nu2, not_too_deep, 
                                              not_too_shallow, presum, rhorat, 
                                              lambda_, Da, dCa, dCO3, delta, 
                                              auxcon, delta_x, Peclet_min, 
                                              Peclet_max, no_depths, dPhi_fixed,
                                              FV_switch)

    @staticmethod
    @njit(cache=False)
    def pde_rhs(CA, CC, cCa, cCO3, Phi, CA_grad_back_op, CA_grad_forw_op,
                CC_grad_back_op, CC_grad_forw_op,
                cCa_grad_back_op, cCa_grad_forw_op, cCa_laplace_op, 
                cCO3_grad_back_op, cCO3_grad_forw_op, cCO3_laplace_op, 
                Phi_grad_back_op, Phi_grad_forw_op, Phi_laplace_op, 
                KRat, m1, m2, n1, n2, nu1, nu2, not_too_deep, not_too_shallow, 
                presum, rhorat, lambda_, Da, dCa, dCO3, delta, auxcon, delta_x, 
                Peclet_min, Peclet_max, no_depths, dPhi_fixed, FV_switch):

        CA_grad_back = CA_grad_back_op(CA) 
        CA_grad_forw = CA_grad_forw_op(CA) 
        CC_grad_back = CC_grad_back_op(CC)
        CC_grad_forw = CC_grad_forw_op(CC)
        cCa_grad_back = cCa_grad_back_op(cCa)
        cCa_grad_forw = cCa_grad_forw_op(cCa)
        cCa_laplace = cCa_laplace_op(cCa)
        cCO3_grad_back = cCO3_grad_back_op(cCO3)
        cCO3_grad_forw = cCO3_grad_forw_op(cCO3)
        cCO3_laplace = cCO3_laplace_op(cCO3)
        Phi_grad_back = Phi_grad_back_op(Phi) 
        Phi_grad_forw = Phi_grad_forw_op(Phi)
        Phi_laplace = Phi_laplace_op(Phi)

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
        rate = np.empty(5 * no_depths)
        dPhi = np.empty(no_depths)
        dW_dx = np.empty(no_depths)
        one_minus_Phi = np.empty(no_depths)
        CA_grad = np.empty(no_depths)
        CC_grad = np.empty(no_depths)
        cCa_grad = np.empty(no_depths)
        cCO3_grad = np.empty(no_depths)
        Phi_grad = np.empty(no_depths)

        for i in range(no_depths):
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
            one_minus_Phi[i] = 1 - Phi[i]
            # dPhi[i] = auxcon * F[i] * (Phi[i] ** 3) / one_minus_Phi[i]
            dPhi[i] = dPhi_fixed

            if FV_switch:
                # Fiadeiro-Veronis scheme for equations 42 and 43
                # from l'Heureux. 
                Peclet_cCa = W[i] * delta_x * denominator[i]/ (2. * dCa )
                if np.abs(Peclet_cCa) < Peclet_min:
                    sigma_cCa = 0
                elif np.abs(Peclet_cCa) > Peclet_max:
                    sigma_cCa = np.sign(W[i])
                else:
                     sigma_cCa = np.cosh(Peclet_cCa)/np.sinh(Peclet_cCa) - 1/Peclet_cCa

                Peclet_cCO3 = W[i] * delta_x * denominator[i]/ (2. * dCO3)
                if np.abs(Peclet_cCO3) < Peclet_min:
                    sigma_cCO3 = 0
                elif np.abs(Peclet_cCO3) > Peclet_max:
                    sigma_cCO3 = np.sign(W[i])
                else:
                    sigma_cCO3 = np.cosh(Peclet_cCO3)/np.sinh(Peclet_cCO3) - 1/Peclet_cCO3

                Peclet_Phi = W[i] * delta_x / (2. * dPhi[i])
                if np.abs(Peclet_Phi) < Peclet_min:
                    sigma_Phi = 0
                elif np.abs(Peclet_Phi) > Peclet_max:
                    sigma_Phi = np.sign(W[i])
                else:
                    sigma_Phi = np.cosh(Peclet_Phi)/np.sinh(Peclet_Phi) - 1/Peclet_Phi
            else:
                sigma_cCa = 0
                sigma_cCO3 = 0
                sigma_Phi = 0

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
            rate[i] = - U[i] * CA_grad[i] - Da * ((1 - CA[i]) \
                                    * coA[i] + lambda_ * CA[i] * coC[i])

            # This is dCC_dt
            rate[no_depths + i] = - U[i] * CC_grad[i] + Da * (lambda_ * \
                                    (1 - CC[i]) * coC[i] + CC[i] * coA[i])              

            # This is dcCa_dt
            rate[2 * no_depths + i] =  helper_cCa_grad[i]/Phi[i] - W[i] * \
                                        cCa_grad[i] + Da * one_minus_Phi[i] * \
                                        (delta - cCa[i]) * common_helper3[i] \
                                        /Phi[i]                                 

            # This is dcCO3_dt
            rate[3 * no_depths + i] =  helper_cCO3_grad[i]/Phi[i] - W[i] * \
                                        cCO3_grad[i] + Da * one_minus_Phi[i] * \
                                        (delta - cCO3[i]) * common_helper3[i] \
                                        /Phi[i]                       

            # This is dPhi_dt
            rate[4 * no_depths + i] = - (dW_dx[i] * Phi[i] + W[i] * Phi_grad[i]) \
                                      + dPhi[i] * Phi_laplace[i] + Da * one_minus_Phi[i] \
                                      * common_helper3[i] 

        return rate

    def zeros(self, t, y, progress_proxy, progress_dt, t0):
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        return np.amin(y)

    def zeros_CA(self, t, y, progress_proxy, progress_dt, t0):
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        return np.amin(y[self.CA_sl])

    def zeros_CC(self, t, y, progress_proxy, progress_dt, t0):
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        return np.amin(y[self.CC_sl])

    def ones_CA_plus_CC(self, t, y, progress_proxy, progress_dt, t0): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        CA = y[self.CA_sl]
        CC = y[self.CC_sl]
        return np.amax(CA + CC) - 1

    def ones_Phi(self, t, y, progress_proxy, progress_dt, t0): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        Phi = y[self.Phi_sl]   
        return np.amax(Phi) - 1

    def zeros_U(self, t, y, progress_proxy, progress_dt, t0): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        Phi = y[self.Phi_sl]   
        F = 1 - np.exp(10 - 10 / Phi)
        U = self.presum + self.rhorat * Phi ** 3 * F / (1 - Phi)
        # Assume that U is positive at every depth at the beginning
        # of the integration, so U will become zero first where it
        # is smallest.
        return np.amin(U)

    def zeros_W(self, t, y, progress_proxy, progress_dt, t0): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        Phi = y[self.Phi_sl]   
        F = 1 - np.exp(10 - 10 / Phi)
        W = self.presum - self.rhorat * Phi ** 2 * F
        # Assume that W is negative at every depth at the beginning
        # of the integration, so U will become zero first where it is
        # largest or least negative.
        return np.amax(W)

