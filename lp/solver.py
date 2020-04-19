import json

import numpy as np

from lp.entity import AlgorithmStatus, VarNameType, ObjectiveType
from lp.helper import get_c_b


class SimplexSolver:
    """
    This uses Simplex algorithm to solve original LP problem or relaxed MIP problem.

    Parameters
    ==========
    model       : Model class

    Properties
    ==========
    is_terminated : bool whether or not the simplex solver is terminated
    """
    def __init__(self, model):
        self._m = model
        self._is_terminated = False
        pass

    def run(self):
        while not self._is_terminated:
            self.iterate()
            self.update_print_result()

    def iterate(self):
        c_b = get_c_b(self._m.basis)
        w = c_b.dot(self._m.B_inv)
        z_c = {}
        for var in [v for v in self._m.vars if not v.in_basis]:
            z_c[var] = w.dot(self._m.A[var]) - var.coeff_c
        entering_var = max(z_c, key=z_c.get)
        if z_c[entering_var] > 0:
            y_k = self._m.B_inv.dot(self._m.A[entering_var])
            if np.all(y_k <= 0):
                self._m.result.status = AlgorithmStatus.UNBOUNDED
                self._is_terminated = True
                return
            rates = {}
            for i in range(len(y_k)):
                if y_k[i] > 0:
                    rates[i] = self._m.basis[i].value / y_k[i]
            leaving_var_index = min(rates, key=rates.get)
            entering_var.value = rates[leaving_var_index].item()
            entering_var.in_basis = True
            leaving_var = self._m.basis[leaving_var_index]
            leaving_var.value = 0.0
            leaving_var.in_basis = False
            self.update_basis(leaving_var, entering_var)
            self.update_obj_value()
        else:
            self._is_terminated = True
            self.check_feasibility()

    def update_basis(self, leaving_var, entering_var):
        self._m.basis = [entering_var if x == leaving_var else x for x in self._m.basis]
        self._m.B_inv = self.update_B_inv(self._m.basis)
        B_inv_b = self._m.B_inv.dot(self._m.b)
        for v in range(len(self._m.basis)):
            self._m.basis[v].value = B_inv_b[v].item()

    def update_obj_value(self):
        obj_val = 0.0
        for var in self._m.vars:
            obj_val += var.coeff_c * var.value
        self._m.obj.value = obj_val

    def update_B_inv(self, basis_vars):
        n = self._m.n_rows
        basic_matrix = self.get_coeff_matrix(basis_vars, n, n)
        return np.linalg.inv(basic_matrix)

    def check_feasibility(self):
        for var in [v for v in self._m.vars
                    if v.var_name_type == VarNameType.ARTIFICIAL]:
            if var.value > 0.0:
                self._m.result.status = AlgorithmStatus.INFEASIBLE
                return
        if self._is_terminated:
            self._m.result.status = AlgorithmStatus.OPTIMAL
        else:
            self._m.result.status = AlgorithmStatus.FEASIBLE

    def get_coeff_matrix(self, variables, row, col):
        coeffs_a = []
        for var in variables:
            coeffs_a.append(self._m.A[var])
        return np.concatenate(coeffs_a, axis=1).reshape((row, col))

    def update_print_result(self):
        solution = {}
        for var in self._m.vars:
            solution[var.name] = round(var.value, 3)
        self._m.result.solution = solution
        if self._m.obj.obj_type == ObjectiveType.MIN:
            self._m.result.obj_val = round(self._m.obj.value, 3)
        elif self._m.obj.obj_type == ObjectiveType.MAX:
            self._m.result.obj_val = -round(self._m.obj.value, 3)
        print(json.dumps(self._m.result.__dict__))


class MIPSolver:
    def __init__(self, model):
        pass


class InitialBasicSolutionGenerator:
    def __init__(self, model):
        self._m = model
        pass

    def generate(self):
        n_rows = self._m.n_rows
        self._m.B_inv = np.identity(n_rows)
        B_inv_b = self._m.B_inv.dot(self._m.b)
        if np.all(B_inv_b >= 0):
            for v in range(n_rows):
                self._m.basis[v].value = B_inv_b[v].item()
            return True
        return False
