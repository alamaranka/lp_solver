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
    _is_terminated : bool whether or not the simplex solver is terminated
    """
    def __init__(self, model):
        self._model = model
        self._is_terminated = False
        self._E = np.identity(model.n_rows)

    def run(self):
        while not self._is_terminated:
            self.iterate()
        self.prepare_and_print_result()

    def iterate(self):
        c_b = get_c_b(self._model.basis)
        w = c_b.dot(self._E)
        z_c = {}
        for var in [v for v in self._model.vars if not v.in_basis]:
            z_c[var] = w.dot(self._model.A[var]) - var.coeff_c
        entering_var = max(z_c, key=z_c.get)
        if z_c[entering_var] > 0:
            u = self._model.A[entering_var]  # entering variable column
            y_k = self._E.dot(u)
            if np.all(y_k <= 0):
                self._model.result.status = AlgorithmStatus.UNBOUNDED
                self._is_terminated = True
                return
            rates = {}
            for i in range(len(y_k)):
                if y_k[i] > 0:
                    rates[i] = self._model.basis[i].value / y_k[i]
            k = min(rates, key=rates.get)    # leaving variable index
            self.update_basis(k, y_k, entering_var)
            self.update_obj_value()
        else:
            self._is_terminated = True
            self.check_status()

    def update_basis(self, k, y_k, entering_var):
        # update leaving variable
        leaving_var = self._model.basis[k]
        leaving_var.in_basis = False
        leaving_var.value = 0.0
        # update entering variable
        entering_var.in_basis = True
        # update basis matrix
        self._model.basis = [entering_var if x == leaving_var
                             else x for x in self._model.basis]
        # update E
        e = np.identity(self._model.n_rows)
        rep = y_k * (-1 / y_k[k])
        e[:, k] = rep[:, 0]
        e[k, k] = 1 / y_k[k]
        self._E = e.dot(self._E)
        # update variable values in the basis
        b_inv_b = self._E.dot(self._model.b)
        for v in range(len(self._model.basis)):
            self._model.basis[v].value = b_inv_b[v].item()

    def update_obj_value(self):
        obj_val = 0.0
        for var in self._model.vars:
            obj_val += var.coeff_c * var.value
        self._model.obj.value = obj_val

    def check_status(self):
        for var in [v for v in self._model.vars
                    if v.var_name_type == VarNameType.ARTIFICIAL]:
            if var.value > 0.0:
                self._model.result.status = AlgorithmStatus.INFEASIBLE
                return
        if self._is_terminated:
            self._model.result.status = AlgorithmStatus.OPTIMAL
        else:
            self._model.result.status = AlgorithmStatus.FEASIBLE

    def prepare_and_print_result(self):
        solution = {}
        for var in self._model.basis:
            if var.var_name_type == VarNameType.PRIMAL:
                solution[var.name] = round(var.value, 3)
        self._model.result.solution = solution
        if self._model.obj.obj_type == ObjectiveType.MIN:
            self._model.result.obj_val = round(self._model.obj.value, 3)
        elif self._model.obj.obj_type == ObjectiveType.MAX:
            self._model.result.obj_val = -round(self._model.obj.value, 3)
        if self._model.result.status == AlgorithmStatus.OPTIMAL or \
           self._model.result.status == AlgorithmStatus.FEASIBLE:
            print("Solution status: {0}".format(str(self._model.result.status)))
            print("Objective value: {0}".format(self._model.result.obj_val))
            print("Solution: {0}".format(self._model.result.solution))
            # print(json.dumps(self._model.result.__dict__))
        elif self._model.result.status == AlgorithmStatus.INFEASIBLE:
            print("Model is infeasible")
        elif self._model.result.status == AlgorithmStatus.UNBOUNDED:
            print("Model is unbounded")
        else:
            print("Model status is unknown")
