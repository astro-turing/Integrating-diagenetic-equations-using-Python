from numba import jit, njit, float32
import numpy as np

no_depths = 400
no_fields = 5

@njit(nogil=True)
def jac00():
    return np.zeros((no_depths))

@njit(nogil=True)
def jac01():
    return np.ones((no_depths))

@njit(parallel=True)
def loop_over_all_Jacobian_indices():
    row_indices = np.arange(no_depths)
    col_indices = np.arange(no_depths)
    n = no_fields * no_depths 
    all_jac_values = np.empty((n, n))
    for i in range(no_fields):
        row = i * no_depths + row_indices
        for j in range(no_fields):
            col = j * no_depths + col_indices
            # row_and_col = (row, col)
            if i == 0 and j == 0:
                jac_values = jac00()
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]
            elif i == 0 and j == 1:
                jac_values = jac01()
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]
                    # all_jac_values[row, col] = jac01() 
            else:
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = 2

loop_over_all_Jacobian_indices()
 