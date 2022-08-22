""" This derives a functional form of the Jacobian (25 elements) """

from sympy import symbols, exp, Matrix, Max, log, pprint
from sympy.abc import x
from contextlib import redirect_stdout

CA, CC, cCa, cCO3, Phi = symbols("CA CC cCa cCO3 Phi")

KRat, m2, not_too_deep, not_too_shallow, nu1, m1 = symbols("KRat m2 not_too_deep not_too_shallow nu1 m1")

n1, nu2, n2, presum, rhorat, Da, lambda_, dCa, delta = symbols("n1 nu2 n2 presum rhorat Da lambda_ dCa delta")

dCO3, auxcon = symbols("dCO3 auxcon")

two_factors = cCa * cCO3

three_factors = two_factors * KRat

coA = CA * (Max(1 - three_factors, 0) ** m2 * not_too_deep * not_too_shallow - nu1 * \
      Max(three_factors - 1, 0) ** m1)

coC = CC * (Max(two_factors - 1, 0) ** n1 - nu2 * (Max(1 - two_factors, 0)) ** n2)

U = presum + rhorat * Phi ** 3 * (1 - exp(10 - 10 / Phi)) / (1 - Phi)

W = presum - rhorat * Phi ** 2 * (1 - exp(10 - 10 / Phi))

dCA_dt = - U * CA.diff(x) - Da * ((1 - CA) * coA + lambda_ * CA * coC)

dCC_dt = - U * CC.diff(x) + Da * (lambda_ * (1 - CC) * coC + CC * coA)

dcCa_dx = cCa.diff(x)

# Taking account of equation 6 from l'Heureux
dcCa_dt = (((Phi * dCa * dcCa_dx)/(1 - 2 * log(Phi))).diff(x))/Phi -W * dcCa_dx + \
          + Da * (1 - Phi) * (delta - cCa) * (coA - lambda_ * coC)/Phi

dcCO3_dx = cCO3.diff(x)

# Taking account of equation 6 from l'Heureux
dcCO3_dt = (((Phi * dCO3 * dcCO3_dx)/(1 - 2 * log(Phi))).diff(x))/Phi -W * dcCO3_dx + \
           Da * (1 - Phi) * (delta - cCO3) * (coA - lambda_ * coC)/Phi

F = 1 - exp(10 - 10 / Phi)     

dPhi = auxcon * F * (Phi ** 3) / (1 - Phi)

dPhi_dt = - (W * Phi).diff(x) + dPhi * Phi.diff(x).diff(x) \
          + Da * (1 - Phi) * (coA - lambda_ * coC)

f = Matrix([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt])  

y = Matrix([CA, CC, cCa, cCO3, Phi])

jacob = f.jacobian(y)
print()
print("The Jacobian has {} elements".format(len(jacob)))
print()

for index in range(len(jacob)):
    print()
    print("Index = {}".format(index))
    print(jacob[index])

with open('Jacobian.txt', 'w') as f:
    with redirect_stdout(f):
        for index in range(len(jacob)):
            print("\n")
            print("Index = {}".format(index))
            print("\n")
            print(jacob[index])
            print("\n")





