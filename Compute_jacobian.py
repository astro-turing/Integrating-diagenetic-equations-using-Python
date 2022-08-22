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
from numba import njit, prange, vectorize

@njit(parallel=True)
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

    def jac00():
        jac00 = -Da * (-CA*not_too_deep*not_too_shallow * np.fmax(0, -KRat*cCO3*cCa + 1)**m2 \
                + CC * lambda_ * (-nu2 * np.fmax(0, -cCO3*cCa + 1) ** n2 + \
                np.fmax(0, cCO3*cCa - 1)**n1) + not_too_deep*not_too_shallow * (1 - CA) * \
                np.fmax(0, -KRat*cCO3*cCa + 1)**m2 + nu1 * np.fmax(0, KRat*cCO3*cCa - 1) ** m1)
        return jac00

    def jac01():
        jac01 = -CA * Da * lambda_*(-nu2 * np.fmax(0, -cCO3*cCa + 1) ** n2 + \
                np.fmax(0, cCO3*cCa - 1)**n1)
        return jac01

    def jac02():
        jac02 = -Da * (CA * CC * lambda_ * (cCO3 * n1 * np.heaviside(cCO3*cCa - 1, 0) \
                * np.fmax(0, cCO3*cCa - 1) ** (n1 - 1) + \
                cCO3 * n2 * nu2 * np.heaviside(-cCO3*cCa + 1, 0) * \
                np.fmax(0, -cCO3*cCa + 1) ** (n2-1)) + \
                (1 - CA)*(-CA * KRat * cCO3 *m2 * not_too_deep * not_too_shallow * \
                np.heaviside(-KRat*cCO3*cCa + 1, 0) * np.fmax(0, -KRat*cCO3*cCa + 1) ** (m2 - 1) - \
                KRat * cCO3 * m1 * nu1 * np.heaviside(KRat*cCO3*cCa - 1, 0) * \
                np.fmax(0, KRat*cCO3*cCa - 1) ** (m1 - 1)))
        return jac02

    def jac03():
        jac03 = -Da * (CA * CC * lambda_ * (cCa * n1* np.heaviside(cCO3*cCa - 1, 0) * \
                np.fmax(0, cCO3*cCa - 1) ** (n1 - 1) + cCa * n2 * nu2 * \
                np.heaviside(-cCO3*cCa + 1, 0) * np.fmax(0, -cCO3*cCa + 1) ** (n2 -1)) + \
                (1 - CA) * (-CA * KRat * cCa * m2 * not_too_deep * not_too_shallow * \
                np.heaviside(-KRat*cCO3*cCa + 1, 0) * np.fmax(0, -KRat*cCO3*cCa + 1) ** (m2 - 1) \
                - KRat * cCa * m1 * nu1 * np.heaviside(KRat*cCO3*cCa - 1, 0) * \
                np.fmax(0, KRat*cCO3*cCa - 1) ** (m1 - 1)))
        return jac03

    def jac04():
        return np.zeros(no_depths)

    def jac10():
        jac10 = CC * Da * not_too_deep * not_too_shallow * np.fmax(0, -KRat*cCO3*cCa + 1) ** m2

                -CA * Da * lambda_*(-nu2 * np.fmax(0, -cCO3*cCa + 1) ** n2 + \
                np.fmax(0, cCO3*cCa - 1)**n1)

        
    row_indices = np.arange(no_depths)
    col_indices = np.arange(no_depths)
    all_jac_values_rows_and_cols = []
    for i in prange(no_fields):
        row = i * no_depths + row_indices
        for j in prange(no_fields):
            col = j * no_depths + col_indices
            row_and_col = (row, col)
            if i == 0 and j == 0:
                all_jac_values_rows_and_cols.append(jac00(), row_and_col)
            elif i == 0 and j == 1:
                all_jac_values_rows_and_cols.append(jac01(), row_and_col)     
            elif i == 0 and j == 2:    
                all_jac_values_rows_and_cols.append(jac02(), row_and_col) 
            elif i == 0 and j == 3:
                all_jac_values_rows_and_cols.append(jac03(), row_and_col)   
            elif i == 0 and j == 4:
                all_jac_values_rows_and_cols.append(jac04(), row_and_col)     
            elif i==1 and j == 0:



    return all_jac_values_rows_and_cols    


    





