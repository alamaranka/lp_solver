import json
import math
import sys
import numpy as np

from lp.entity import VarNameType, Sense, ObjectiveType, \
    AlgorithmStatus, Result, Variable, Constraint, Objective


class Model:
    def __init__(self):
        self.status = AlgorithmStatus.NONE
        self.obj = None
        self.vars = []
        self.consts = []
        self.basis = []
        self._b = []
        self._b_array = None
        self._slack_count = 0
        self._surplus_count = 0
        self._artificial_count = 0
        self.result = Result()
        self.BIG_M = math.pow(10, 6)
        self.is_terminated = False

    def add_var(self, lb=0, ub=sys.float_info.max, name='',
                variable_type=VarNameType.PRIMAL):
        var = Variable(lb, ub, name, variable_type)
        self.vars.append(var)
        return var

    def add_const(self, expr, sense, rhs):
        const_num = len(self.consts)
        # normalize rhs
        if rhs < 0.0:
            rhs *= -1
            for v in range(len(expr.vals)):
                expr.vals[v] *= -1.0
            sense = Model.set_reverse_sense(sense)
        # construct vector b
        self._b.append(rhs)
        # set variables coeffs_a
        for val, var in zip(expr.vals, expr.vars):
            var.coeffs_a[const_num] = val
        # add slack, surplus, artificial vars
        if sense == Sense.LE:
            self.add_slack_var(const_num)
        elif sense == Sense.GE:
            self.add_surplus_var(const_num)
            self.add_artificial_var(const_num)
        elif sense == Sense.EQ:
            self.add_artificial_var(const_num)
        # add constraint
        const = Constraint(expr, sense, rhs)
        self.consts.append(const)
        return const

    def set_objective(self, expr, obj_type):
        self.obj = Objective(expr, obj_type)
        for e in range(len(self.obj.expr.vars)):
            if obj_type == ObjectiveType.MIN:
                self.obj.expr.vars[e].coeff_c = self.obj.expr.vals[e]
            elif obj_type == ObjectiveType.MAX:
                self.obj.expr.vals[e] *= -1.0
                self.obj.expr.vars[e].coeff_c = self.obj.expr.vals[e]

    def solve(self):
        self.prepare_coefficient_matrices()
        self.solve_lp()

    def solve_lp(self):
        if self.get_basic_feasible_solution():
            while not self.is_terminated:
                self.iterate()
                self.update_print_result()
        else:
            self.status = AlgorithmStatus.INFEASIBLE
            self.update_print_result()

    def iterate(self):
        c_b = self.get_c_b()
        B_inv = self.get_B_inv(self.basis)
        w = c_b.dot(B_inv)
        z_c = {}
        for var in [v for v in self.vars if not v.in_basis]:
            coeffs_a = Model.get_coeff_matrix([var], len(self.consts), 1)
            z_c[var] = w.dot(coeffs_a) - var.coeff_c
        entering_var = max(z_c, key=z_c.get)
        if z_c[entering_var] > 0:
            coeffs_a = Model.get_coeff_matrix([entering_var], len(self.consts), 1)
            y_k = B_inv.dot(coeffs_a)
            if np.all(y_k <= 0):
                self.status = AlgorithmStatus.UNBOUNDED
                self.is_terminated = True
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
            self.is_terminated = True
            self.check_feasibility()

    def add_slack_var(self, c):
        slack = self.add_var(0, sys.float_info.max,
                             's' + str(self._slack_count))
        slack.variable_type = VarNameType.SLACK
        slack.coeff_c = 0.0
        slack.coeffs_a[c] = 1.0
        slack.in_basis = True
        self.basis.append(slack)
        self._slack_count += 1

    def add_surplus_var(self, c):
        surplus = self.add_var(0, sys.float_info.max,
                               'e' + str(self._surplus_count))
        surplus.variable_type = VarNameType.SURPLUS
        surplus.coeff_c = 0.0
        surplus.coeffs_a[c] = -1.0
        self._surplus_count += 1

    def add_artificial_var(self, c):
        artificial = self.add_var(0, sys.float_info.max,
                                  'a' + str(self._surplus_count))
        artificial.variable_type = VarNameType.ARTIFICIAL
        artificial.coeff_c = self.BIG_M
        artificial.coeffs_a[c] = 1.0
        artificial.in_basis = True
        self.basis.append(artificial)
        self._artificial_count += 1

    def prepare_coefficient_matrices(self):
        self._b_array = np.asarray(self._b,
                                   dtype=np.float32)
        self._b_array = self._b_array.reshape(
            (self._b_array.shape[0], 1))

    def get_basic_feasible_solution(self):
        B_inv = self.get_B_inv(self.basis)
        B_inv_b = B_inv.dot(self._b_array)
        if np.all(B_inv_b >= 0):
            for v in range(len(self.basis)):
                self.basis[v].value = B_inv_b[v].item()
            self.update_obj_value()
            self.check_feasibility()
            self.update_print_result()
            return True
        return False

    def check_feasibility(self):
        for var in [v for v in self.vars
                    if v.variable_type == VarNameType.ARTIFICIAL]:
            if var.value > 0.0:
                self.status = AlgorithmStatus.INFEASIBLE
                return
        if self.is_terminated:
            self.status = AlgorithmStatus.OPTIMAL
        else:
            self.status = AlgorithmStatus.FEASIBLE

    def update_print_result(self):
        self.result.status = self.status.name
        solution = {}
        for var in self.vars:
            solution[var.name] = round(var.value, 3)
        self.result.solution = solution
        if self.obj.obj_type == ObjectiveType.MIN:
            self.result.obj_val = round(self.obj.value, 3)
        elif self.obj.obj_type == ObjectiveType.MAX:
            self.result.obj_val = -round(self.obj.value, 3)
        print(json.dumps(self.result.__dict__))

    def get_c_b(self):
        coeffs = []
        for var in self.basis:
            coeffs.append(var.coeff_c)
        return np.asarray(coeffs, dtype=np.float32).reshape((1, len(coeffs)))

    def update_obj_value(self):
        obj_val = 0.0
        for var in self.vars:
            obj_val += var.coeff_c * var.value
        self.obj.value = obj_val

    def update_basis(self, leaving_var, entering_var):
        self.basis = [entering_var if x == leaving_var else x for x in self.basis]
        B_inv = self.get_B_inv(self.basis)
        B_inv_b = B_inv.dot(self._b_array)
        for v in range(len(self.basis)):
            self.basis[v].value = B_inv_b[v].item()

    def get_B_inv(self, basis_vars):
        n = len(self.consts)
        basic_matrix = Model.get_coeff_matrix(basis_vars, n, n)
        if np.linalg.matrix_rank(basic_matrix) == n:
            return np.linalg.inv(basic_matrix)
        else:
            return None

    @staticmethod
    def get_coeff_matrix(variables, row, col):
        matrix = np.zeros((row, col), dtype=np.float32)
        for v in range(len(variables)):
            for key, value in variables[v].coeffs_a.items():
                matrix[key, v] = value
        return matrix.reshape((row, col))

    @staticmethod
    def get_first_or_default(slack_var):
        if not slack_var:
            return None
        return slack_var[0]

    @staticmethod
    def set_reverse_sense(sense):
        if sense == Sense.LE:
            return Sense.GE
        elif sense == Sense.GE:
            return Sense.LE
        else:
            return None

    @staticmethod
    def get_var_index_in_const(var, expr):
        if var in expr.vars:
            return expr.vars.index(var)
        return -1
