""" This returns the 25 elements (from 5 fields) of the Jacobian, 
computed numerically, using the functional form from Derive_jacobian.
There is also the depth dimension. 
In the current implementations of LMAHeureuxPorosityDiff.fun and
LMAHeureuxPorosityDiff.fun_numba, the five fields with each, say, 400 depths,
are treated as 2000 fields, so n = 2000 voor scipy.integrate.solve_ivp.
The solve_ivp documention says that the Jacobian will have shape (n, n),
so (2000, 2000), so calling jac(t, y) should return 4 million numbers!
But most elements will be zero, because derivatives of the five rhs 
expressions (equations 40-43 from l'Heureux) wrt to the five fields will only
be non-zero for equal depths.

Say you want to compute Jacobian element [1,4]. This is 
d f_1 / d y_4. That is the derivate of the right hand side of the second 
equation (equation 41 from l'Heureux) wrt to field number 5, which is Phi. 
We have a functional form for that, from Derive_Jacobian.py, so we can 
compute it for all 400 depths. Now where do these 400 numbers end up in
the (2000, 2000) output matrix? For depth 0, it would be element [400, 1600].
For depth 1, it will be element [401, 1601] and so forth until element
[799, 1999]. So a diagonal within a (400, 400) cell. Ultimately we will fill
25*400 =10000 elements of the 4 million numbers with non-zero values. Actually,
a bit less, because two of the 25 elements of the Jacobian are always zero. 
These correspond with the derivatives of the rhs of equations 40 and 41 wrt
Phi. So we have 23 * 400 = 9200 non-zero elements out of 2000 * 2000 = 4e6
elements. Truly a sparse matrix!"""

import numpy as np
from numba.extending import overload
from numba import njit

@overload(np.heaviside)
def np_heaviside(x1, x2):
    def heaviside_impl(x1, x2):
        if x1 < 0:
            return 0.0
        elif x1 > 0:
            return 1.0
        else:
            return x2

    return heaviside_impl

""" @overload(np.fmax)
def np_fmax(x1, x2):
    def fmax_impl(x1, x2):
        if x1 < x2:
            return x2
        elif x1 > x2:
            return x1
        else:
            return x1

    return fmax_impl """

# @njit(nogil=True, parallel = True)
def Jacobian(y, KRat, m1, m2, n1, n2, nu1, nu2, \
            not_too_deep, not_too_shallow, presum, rhorat, lambda_, Da, dCa, \
            dCO3, delta, auxcon, CA_sl, CC_sl, cCa_sl, cCO3_sl, Phi_sl, \
            no_depths, no_fields):
    """ Retun a sparse (n, n) matrix
    The Jacobian for five field and five right-hand sides, so 25 elements
    of which two are zero, so we need to evaluate 23 expressions. """

    CA = y[CA_sl]
    CC = y[CC_sl]
    cCa = y[cCa_sl]
    cCO3 = y[cCO3_sl]
    Phi = y[Phi_sl]

    selected_depths = not_too_deep * not_too_shallow
    two_factors = cCO3 * cCa

    three_factors = KRat * two_factors
    thr_f_m_one = three_factors - 1
    one_m_thr_f = - thr_f_m_one

    max_thr_f_m_one = np.fmax(0, thr_f_m_one)

    # @njit(nogil=true)
    def jac00():        
        return  -Da * (-CA * (not_too_deep*not_too_shallow * np.fmax(0, -KRat*cCO3*cCa + 1) ** m2 \
                - nu1 * np.fmax(0, KRat*cCO3*cCa - 1)**m1) + CC*lambda_ * \
                (-nu2 * np.fmax(0, -cCO3*cCa + 1)**n2 + np.fmax(0, cCO3*cCa - 1)**n1) + \
                (1 - CA) * (not_too_deep*not_too_shallow * np.fmax(0, -KRat*cCO3*cCa + 1) ** m2 \
                - nu1 * np.fmax(0, KRat*cCO3*cCa - 1)**m1))       
 
    # @njit(nogil=true)
    def jac01():
        return  -CA * Da * lambda_ * (-nu2 * np.fmax(0, -cCO3*cCa + 1)**n2 + \
                 np.fmax(0, cCO3*cCa - 1)**n1)

    # @njit(nogil=true)
    def jac02():
        return  -Da * (CA *CC * lambda_ * (cCO3 * n1 * np.heaviside(cCO3 * cCa - 1, 0) * \
                np.fmax(0, cCO3*cCa - 1) ** (n1 - 1 ) + cCO3 * n2 * nu2 * np.heaviside(-cCO3*cCa + 1, 0) *\
                np.fmax(0, -cCO3*cCa + 1) ** (n2 - 1)) + CA*(1 - CA) * (-KRat * cCO3 * m1 * nu1 *\
                np.heaviside(KRat * cCO3 * cCa - 1, 0) * np.fmax(0, KRat*cCO3*cCa - 1) ** (m1 - 1) - \
                KRat * cCO3 * m2 * not_too_deep * not_too_shallow * np.heaviside(-KRat*cCO3*cCa + 1, 0) * \
                np.fmax(0, -KRat*cCO3*cCa + 1) ** (m2 -  1)))

    # @njit(nogil=true)
    def jac03():
        return -Da * (CA * CC * lambda_ * (cCa * n1 * np.heaviside(cCO3*cCa - 1, 0) * \
                np.fmax(0, cCO3*cCa - 1) ** (n1 - 1) + cCa * n2 * nu2 * np.heaviside(-cCO3*cCa + 1, 0) * \
                np.fmax(0, -cCO3*cCa + 1) ** (n2 - 1)) + CA * (1 - CA) * (-KRat * cCa * m1 * nu1 * \
                np.heaviside(KRat*cCO3*cCa - 1, 0) * np.fmax(0, KRat*cCO3*cCa - 1) ** (m1 - 1) \
                - KRat * cCa * m2 * not_too_deep * not_too_shallow * np.heaviside(-KRat*cCO3*cCa + 1, 0) * \
                np.fmax(0, -KRat*cCO3*cCa + 1) ** (m2 - 1)))

    # @njit(nogil=true)
    def jac04():
        return np.zeros(no_depths)

    # @njit(nogil=True)
    def jac10():
        return  CC * Da * (not_too_deep * not_too_shallow * np.fmax(0, -KRat*cCO3*cCa + 1) ** m2 \
               - nu1 * np.fmax(0, KRat*cCO3*cCa - 1) ** m1)

    # @njit(nogil=true)
    def jac11():
        return  Da * (CA * (not_too_deep * not_too_shallow * np.fmax(0, -KRat*cCO3*cCa + 1) ** m2 - \
                nu1 * np.fmax(0, KRat*cCO3*cCa - 1) ** m1) - CC * lambda_ * \
                (-nu2 * np.fmax(0, -cCO3*cCa + 1) ** n2 + np.fmax(0, cCO3 * cCa - 1) ** n1) + \
                lambda_*(1 - CC)*(-nu2 * np.fmax(0, -cCO3*cCa + 1)**n2 + np.fmax(0, cCO3*cCa - 1)**n1))

    # @njit(nogil=true)
    def jac12():        
        return  Da * (CA * CC * (-KRat * cCO3 * m1 * nu1 * np.heaviside(KRat * cCO3 * cCa - 1, 0) * \
                np.fmax(0, KRat*cCO3*cCa - 1) ** (m1- 1) - KRat * cCO3 * m2 * not_too_deep * \
                not_too_shallow * np.heaviside(-KRat * cCO3 * cCa + 1, 0) * \
                np.fmax(0, -KRat * cCO3 * cCa + 1) ** (m2 - 1)) + CC* lambda_ * \
                (1 - CC) * (cCO3 * n1 * np.heaviside(cCO3 * cCa - 1, 0) * \
                np.fmax(0, cCO3*cCa - 1) ** (n1 - 1) + cCO3 * n2 * nu2 * \
                np.heaviside(-cCO3*cCa + 1, 0) * np.fmax(0, -cCO3 * cCa + 1) ** (n2 - 1)))


    # @njit(nogil=true)
    def jac13():
        return  Da * (CA * CC * (-KRat * cCa * m1 * nu1 * np.heaviside(KRat * cCO3 * cCa - 1, 0) * \
                np.fmax(0, KRat * cCO3 * cCa - 1) ** (m1 - 1) - KRat * cCa * m2 * \
                not_too_deep * not_too_shallow * np.heaviside(-KRat * cCO3 * cCa + 1, 0) * \
                np.fmax(0, -KRat*cCO3*cCa + 1) ** (m2 -1)) + CC * lambda_ * \
                (1 - CC) * (cCa * n1 * np.heaviside(cCO3 * cCa - 1, 0) * \
                np.fmax(0, cCO3*cCa - 1) ** (n1- 1) + cCa * n2 * nu2 * \
                np.heaviside(-cCO3*cCa + 1, 0) * np.fmax(0, -cCO3 * cCa + 1) ** (n2 - 1)))

    # @njit(nogil=true)
    def jac14():
        return np.zeros(no_depths)

    # @njit(nogil=true)
    def jac20():
        return Da * (1 - Phi) * (-cCa + delta) * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1)/Phi

    # @njit(nogil=true)
    def jac21():
        return -Da * lambda_ * (1 - Phi) * (-cCa + delta) * \
               (-nu2 * np.fmax(0, -cCO3 * cCa + 1) ** n2 + \
               np.fmax(0, cCO3 * cCa - 1) ** n1)/Phi

    # @njit(nogil=true)
    def jac22():
        return Da * (1 - Phi) * (-cCa + delta) * (CA* (-KRat * cCO3 * m1 * nu1 * \
               np.heaviside(KRat * cCO3 * cCa - 1, 0) * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** (m1 - 1) - KRat * cCO3 * m2 * \
               not_too_deep  *not_too_shallow * np.heaviside(-KRat * cCO3 * cCa + 1, 0) * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** (m2 - 1)) - CC * lambda_ * \
               (cCO3*n1*np.heaviside(cCO3*cCa - 1, 0) * \
               np.fmax(0, cCO3 * cCa - 1) ** (n1 - 1) + cCO3 * n2 * nu2 * \
               np.heaviside(-cCO3 * cCa + 1, 0) * np.fmax(0, -cCO3 * cCa + 1) ** (n2 - 1)))/Phi - \
               Da * (1 - Phi) * (CA * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1) - CC * lambda_ * \
               (-nu2 * np.fmax(0, -cCO3 * cCa + 1) ** n2 + np.fmax(0, cCO3 * cCa - 1)**n1))/Phi

    # @njit(nogil=true)
    def jac23():
        return Da * (1 - Phi) * (-cCa + delta) * (CA * (-KRat * cCa * m1 * nu1 * \
               np.heaviside(KRat * cCO3 * cCa - 1, 0) * np.fmax(0, KRat * cCO3 * cCa - 1) ** (m1 - 1) - \
               KRat * cCa * m2 * not_too_deep * not_too_shallow * np.heaviside(-KRat*cCO3*cCa + 1, 0) * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** (m2 - 1)) - CC * lambda_ * (cCa * n1 * \
               np.heaviside(cCO3 * cCa - 1, 0) * np.fmax(0, cCO3*cCa - 1) ** (n1 - 1) + \
               cCa * n2 * nu2 * np.heaviside(-cCO3 * cCa + 1, 0) * \
               np.fmax(0, -cCO3 * cCa + 1) ** (n2 - 1)))/Phi

    # @njit(nogil=true)
    def jac24():
        return -Da * (-cCa + delta) * (CA * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1) - CC * lambda_ * (-nu2 * \
               np.fmax(0, -cCO3*cCa + 1) ** n2 + \
               np.fmax(0, cCO3 * cCa - 1) ** n1))/Phi - Da*(1 - Phi)*(-cCa + delta) * \
               (CA*(not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * 
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1) - CC * lambda_ * \
               (-nu2*np.fmax(0, -cCO3 * cCa + 1) ** n2 + np.fmax(0, cCO3 * cCa - 1) ** n1))/Phi**2

    # @njit(nogil=true)
    def jac30():
        return Da * (1 - Phi) * (-cCO3 + delta) * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1)/Phi
    
    # @njit(nogil=true)
    def jac31():
        return -Da * lambda_ * (1 - Phi) * (-cCO3 + delta) * (-nu2 * \
               np.fmax(0, -cCO3 * cCa + 1) ** n2 + np.fmax(0, cCO3 * cCa - 1)**n1)/Phi

    # @njit(nogil=true)
    def jac32():
        return Da * (1 - Phi) * (-cCO3 + delta) * (CA*(-KRat * cCO3 * m1 * nu1 * \
               np.heaviside(KRat*cCO3*cCa - 1, 0) * np.fmax(0, KRat * cCO3 * cCa - 1) ** (m1- 1) - \
               KRat * cCO3 * m2 * not_too_deep * not_too_shallow * \
               np.heaviside(-KRat * cCO3 * cCa + 1, 0) * np.fmax(0, -KRat * cCO3 * cCa + 1) ** (m2 - 1)) - \
               CC * lambda_ * (cCO3 * n1 * np.heaviside(cCO3*cCa - 1, 0) * \
               np.fmax(0, cCO3 * cCa - 1) ** (n1 - 1) + cCO3 * n2 * nu2 * \
               np.heaviside(-cCO3 * cCa + 1, 0) * np.fmax(0, -cCO3*cCa + 1) ** (n2 - 1)))/Phi

    # @njit(nogil=true)
    def jac33():
        return Da * (1 - Phi) * (-cCO3 + delta) * (CA* (-KRat * cCa* m1 * nu1 * \
               np.heaviside(KRat * cCO3 * cCa - 1, 0) * np.fmax(0, KRat*cCO3*cCa - 1) ** (m1 - 1) - \
               KRat * cCa * m2 * not_too_deep * not_too_shallow * \
               np.heaviside(-KRat * cCO3 * cCa + 1, 0) * np.fmax(0, -KRat*cCO3*cCa + 1) ** (m2 - 1)) \
               - CC * lambda_ * (cCa * n1 * np.heaviside(cCO3 * cCa - 1, 0) * \
               np.fmax(0, cCO3*cCa - 1) ** (n1 - 1) + cCa * n2 * nu2 * \
               np.heaviside(-cCO3 * cCa + 1, 0) * np.fmax(0, -cCO3*cCa + 1) ** (n2 - 1)))/Phi - \
               Da * (1 - Phi) * (CA * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1) - CC * lambda_ * (-nu2 * \
               np.fmax(0, -cCO3 * cCa + 1) ** n2 + np.fmax(0, cCO3 * cCa - 1) ** n1))/Phi

    # @njit(nogil=true)
    def jac34():
        return -Da * (-cCO3 + delta) * (CA * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat*cCO3*cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1) - CC*lambda_*(-nu2 * \
               np.fmax(0, -cCO3*cCa + 1) ** n2 + \
               np.fmax(0, cCO3 * cCa - 1) ** n1))/Phi - Da * (1 - Phi) * \
               (-cCO3 + delta) *(CA * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1) - CC * lambda_ * \
               (-nu2 * np.fmax(0, -cCO3 * cCa + 1) ** n2 + \
               np.fmax(0, cCO3 * cCa - 1) ** n1))/Phi**2

    # @njit(nogil=true)
    def jac40():
        return Da * (1 - Phi) * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1)
    
    # @njit(nogil=true)
    def jac41():
        return -Da * lambda_ * (1 - Phi) * (-nu2 * np.fmax(0, -cCO3 * cCa + 1) ** n2 + \
               np.fmax(0, cCO3 * cCa - 1) ** n1)
    
    # @njit(nogil=true)
    def jac42():
        return Da * (1 - Phi) * (CA * (-KRat * cCO3 * m1 * nu1 * \
               np.heaviside(KRat * cCO3 * cCa - 1, 0) * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** (m1 - 1) - KRat * cCO3 * \
               m2 * not_too_deep * not_too_shallow * \
               np.heaviside(-KRat * cCO3 * cCa + 1, 0) * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** (m2 - 1)) - CC * lambda_ * \
               (cCO3 * n1 * np.heaviside(cCO3 * cCa - 1, 0) * \
               np.fmax(0, cCO3 * cCa - 1) ** (n1 - 1) + \
               cCO3 * n2 * nu2 * np.heaviside(-cCO3 * cCa + 1, 0) * \
               np.fmax(0, -cCO3 * cCa + 1) ** (n2 - 1)))

    # @njit(nogil=true)
    def jac43():
        return Da * (1 - Phi) *(CA * (-KRat * cCa * m1 * nu1 * \
               np.heaviside(KRat * cCO3 * cCa - 1, 0) * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** (m1 - 1) - KRat * cCa * m2 * \
               not_too_deep * not_too_shallow * 
               np.heaviside(-KRat * cCO3 * cCa + 1, 0) * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** (m2 - 1)) - CC *lambda_ * \
               (cCa * n1 * np.heaviside(cCO3 * cCa - 1, 0) * \
               np.fmax(0, cCO3 * cCa - 1) ** (n1 - 1) + cCa * n2 * nu2 * \
               np.heaviside(-cCO3 * cCa + 1, 0) * np.fmax(0, -cCO3 * cCa + 1) ** (n2 - 1)))

    # @njit(nogil=true)
    def jac44():
        return -Da * (CA * (not_too_deep * not_too_shallow * \
               np.fmax(0, -KRat * cCO3 * cCa + 1) ** m2 - nu1 * \
               np.fmax(0, KRat * cCO3 * cCa - 1) ** m1) - CC * lambda_ * \
               (-nu2 * np.fmax(0, -cCO3 * cCa + 1) ** n2 + \
               np.fmax(0, cCO3 * cCa - 1) ** n1))
        
    row_indices = np.arange(no_depths)
    col_indices = np.arange(no_depths)
    all_jac_values_rows_and_cols = []
    for i in range(no_fields):
        row = i * no_depths + row_indices
        for j in range(no_fields):
            col = j * no_depths + col_indices
            row_and_col = (row, col)
            if i == 0 and j == 0:
                all_jac_values_rows_and_cols.append((jac00(), row_and_col))
            elif i == 0 and j == 1:
                all_jac_values_rows_and_cols.append((jac01(), row_and_col))     
            elif i == 0 and j == 2:    
                all_jac_values_rows_and_cols.append((jac02(), row_and_col)) 
            elif i == 0 and j == 3:
                all_jac_values_rows_and_cols.append((jac03(), row_and_col))   
            elif i == 0 and j == 4:
                all_jac_values_rows_and_cols.append((jac04(), row_and_col))     
            elif i==1 and j == 0:
                all_jac_values_rows_and_cols.append((jac10(), row_and_col))
            elif i==1 and j == 1:
                all_jac_values_rows_and_cols.append((jac11(), row_and_col))  
            elif i==1 and j == 2:
                all_jac_values_rows_and_cols.append((jac12(), row_and_col)) 
            elif i==1 and j == 3:
                all_jac_values_rows_and_cols.append((jac13(), row_and_col)) 
            elif i==1 and j == 4:    
                all_jac_values_rows_and_cols.append((jac14(), row_and_col))           
            elif i==2 and j == 0:
                all_jac_values_rows_and_cols.append((jac20(), row_and_col)) 
            elif i==2 and j == 1:
                all_jac_values_rows_and_cols.append((jac21(), row_and_col)) 
            elif i==2 and j == 2:
                all_jac_values_rows_and_cols.append((jac22(), row_and_col))  
            elif i==2 and j == 3:
                all_jac_values_rows_and_cols.append((jac23(), row_and_col)) 
            elif i==2 and j == 4:
                all_jac_values_rows_and_cols.append((jac24(), row_and_col))   
            elif i==3 and j == 0:
                all_jac_values_rows_and_cols.append((jac30(), row_and_col))    
            elif i==3 and j == 1:
                all_jac_values_rows_and_cols.append((jac31(), row_and_col))   
            elif i==3 and j == 2:
                all_jac_values_rows_and_cols.append((jac32(), row_and_col))   
            elif i==3 and j == 3:
                all_jac_values_rows_and_cols.append((jac33(), row_and_col))  
            elif i==3 and j == 4:
                all_jac_values_rows_and_cols.append((jac34(), row_and_col))    
            elif i==4 and j == 0:
                all_jac_values_rows_and_cols.append((jac40(), row_and_col))   
            elif i==4 and j == 1:
                all_jac_values_rows_and_cols.append((jac41(), row_and_col))   
            elif i==4 and j == 2:
                all_jac_values_rows_and_cols.append((jac42(), row_and_col))      
            elif i==4 and j == 3:
                all_jac_values_rows_and_cols.append((jac43(), row_and_col))        
            elif i==4 and j == 4:
                all_jac_values_rows_and_cols.append((jac44(), row_and_col))                                                   

    return all_jac_values_rows_and_cols    


    





