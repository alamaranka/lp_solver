import numpy as np

from lp.entity import AlgorithmStatus
from lp.helper import get_c_b


class SimplexSolver:
    """
    This uses Simplex algorithm to solve original LP problem or relaxed MIP problem.

    Parameters
    ==========
    model       : Model class

    Properties
    ==========

    """
    def __init__(self, model):
        self._model = model
        self._is_terminated = False
        pass

    def run(self):
        pass

    def iterate(self):
        c_b = get_c_b(self.basis)
        w = c_b.dot(self.B_inv)
        z_c = {}
        for var in [v for v in self.vars if not v.in_basis]:
            z_c[var] = w.dot(self.A[var]) - var.coeff_c
        entering_var = max(z_c, key=z_c.get)
        if z_c[entering_var] > 0:
            y_k = self.B_inv.dot(self.A[entering_var])
            if np.all(y_k <= 0):
                self.result.status = AlgorithmStatus.UNBOUNDED
                self._is_terminated = True
                return
            rates = {}
            for i in range(len(y_k)):
                if y_k[i] > 0:
                    rates[i] = self.basis[i].value / y_k[i]
            leaving_var_index = min(rates, key=rates.get)
            entering_var.value = rates[leaving_var_index].item()
            entering_var.in_basis = True
            leaving_var = self.basis[leaving_var_index]
            leaving_var.value = 0.0
            leaving_var.in_basis = False
            self.update_basis(leaving_var, entering_var)
            self.update_obj_value()
        else:
            self._is_terminated = True
            self.check_feasibility()


class MIPSolver:
    def __init__(self, model):
        pass
