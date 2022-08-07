from sympy import symbols, exp, Matrix

from sympy.abc import x

CA, CC, cCa, cCO3, Phi = symbols("CA CC cCa cCO3 Phi")

KRat, m2, not_too_deep, not_too_shallow, nu1, m1 = symbols("KRat m2 not_too_deep not_too_shallow nu1 m1")

n1, nu2, n2, presum, rhorat, Da, lambda_, dCa, delta = symbols("n1 nu2 n2 presum rhorat Da lambda_ dCa delta")

dCO3, auxcon = symbols("dCO3 auxcon")

two_factors = cCa * cCO3

three_factors = two_factors * KRat

coA = CA * ((1 - three_factors) ** m2) * not_too_deep * not_too_shallow - nu1 * (three_factors - 1) ** m1

coC = CC * (((two_factors - 1) ** n1) - nu2 * (1 - two_factors) ** n2)

U = presum + rhorat * Phi ** 3 * (1 - exp(10 - 10 / Phi)) / (1 - Phi)

W = presum - rhorat * Phi ** 2 * (1 - exp(10 - 10 / Phi))

dCA_dt = - U * CA.diff(x) - Da * ((1 - CA) * coA + lambda_ * CA * coC)

dCC_dt = - U * CC.diff(x) + Da * (lambda_ * (1 - CC) * coC + CC * coA)

dcCa_dx = cCa.diff(x)

dcCa_dt = ((Phi * dCa * dcCa_dx).diff(x))/Phi -W * dcCa_dx + \
          + Da * (1 - Phi) * (delta - cCa) * (coA - lambda_ * coC)/Phi

dcCO3_dx = cCO3.diff(x)

dcCO3_dt = (Phi * dCO3 * dcCO3_dx).diff(x)/Phi -W * dcCO3_dx + \
           Da * (1 - Phi) * (delta - cCO3) * (coA - lambda_ * coC)/Phi

F = 1 - exp(10 - 10 / Phi)     

dPhi = auxcon * F * (Phi ** 3) / (1 - Phi)

dPhi_dt = - (W * Phi).diff(x) + dPhi * Phi.diff(x).diff(x) \
          + Da * (1 - Phi) * (coA - lambda_ * coC)

f = Matrix([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt])  

y = Matrix([CA, CC, cCa, cCO3, Phi])

jacob = f.jacobian(y)

print(len(jacob))

for index in range(len(jacob)):
    print(jacob[index])

""" for row in range(5):
    for column in range(5):
        print(f[row*5+column]) """





