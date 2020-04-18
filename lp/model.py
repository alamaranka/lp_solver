import json
import math
import sys
import time

import numpy as np

from lp.entity import VarNameType, Sense, ObjectiveType, \
    AlgorithmStatus, Result, Variable, Constraint, Objective, \
    VarType, UnknownVariableError


class Model:
    """
    The model contains variables, constraints, and objective of the problem.

    Parameters
    ===========
    name            : str defines the name of the model

    Properties
    ===========
    vars            : list of variables in the model
    consts          : list of constraints in the model
    obj             : Objective class contains the objective of the model
    result          : Result class contains the values of variables in the solution if exists
    is_mip          : bool whether or not the problem is MIP
    is_terminated   : bool whether or not the solver is terminated
    A               : dict of coefficient matrix A
    b               : numpy array of b vector
    rhs             : list of right hand side values
    n_rows          : int number of constraints
    n_cols          : int number of variables
    n_slack         : int number of slack variables
    n_surplus       : int number of surplus variables
    n_artificial    : int number of artificial variables
    BIG_M           : double used to model artificial variables in the model
    """
    def __init__(self, name="LP Solver"):
        self.name = name
        self.vars = []
        self.consts = []
        self.obj = None
        self.result = Result()
        self.is_mip = False
        self.is_terminated = False
        self.A = {}
        self.b = None
        self.rhs = []
        self.n_rows = 0
        self.n_cols = 0
        self.n_slack = 0
        self.n_surplus = 0
        self.n_artificial = 0
        self.BIG_M = math.pow(10, 6)

        # TODO: consider moving these to SimplexSolver
        self.basis = []
        self.B_inv = None

    @property
    def __str__(self):
        return self.name

    def solve(self):
        start_time = time.clock()
        self.prepare_coefficient_matrices()
        self.solve_lp()
        end_time = time.clock()
        self.print_algorithm_logs(start_time, end_time)

    def print_algorithm_logs(self, start_time, end_time):
        if self.is_mip:
            problem = 'Mixed-integer programming problem'
        else:
            problem = 'Linear programming problem'
        print('{0} solved in {1:.4f} seconds.'
              .format(problem, end_time - start_time))

    def solve_lp(self):
        if self.get_basic_feasible_solution():
            while not self.is_terminated:
                self.iterate()
                self.update_print_result()
        else:
            self.result.status = AlgorithmStatus.INFEASIBLE
            self.update_print_result()

    def add_var(self, lb=0, ub=sys.float_info.max, name='',
                var_type=VarType.CONTINUOUS,
                var_name_type=VarNameType.PRIMAL):
        if var_type != VarType.CONTINUOUS:
            self.is_mip = True
        var = Variable(lb, ub, name, var_type, var_name_type)
        self.vars.append(var)
        self.n_cols += 1
        return var

    def add_const(self, expr, sense, rhs):
        const_num = self.n_rows
        # normalize rhs
        if rhs < 0.0:
            rhs *= -1
            for v in range(len(expr.vals)):
                expr.vals[v] *= -1.0
            sense = Model.set_reverse_sense(sense)
        # construct vector b
        self.rhs.append(rhs)
        # set variables coeffs_a
        for val, var in zip(expr.vals, expr.vars):
            var.coeffs_a[const_num] = val
        # add slack, surplus, artificial vars
        self.add_extra_vars(sense, const_num)
        # add constraint
        const = Constraint(expr, sense, rhs)
        self.consts.append(const)
        self.n_rows += 1
        return const

    def set_objective(self, expr, obj_type):
        self.obj = Objective(expr, obj_type)
        for e in range(len(self.obj.expr.vars)):
            if obj_type == ObjectiveType.MIN:
                self.obj.expr.vars[e].coeff_c = self.obj.expr.vals[e]
            elif obj_type == ObjectiveType.MAX:
                self.obj.expr.vals[e] *= -1.0
                self.obj.expr.vars[e].coeff_c = self.obj.expr.vals[e]

    def iterate(self):
        c_b = self.get_c_b()
        w = c_b.dot(self.B_inv)
        z_c = {}
        for var in [v for v in self.vars if not v.in_basis]:
            z_c[var] = w.dot(self.A[var]) - var.coeff_c
        entering_var = max(z_c, key=z_c.get)
        if z_c[entering_var] > 0:
            y_k = self.B_inv.dot(self.A[entering_var])
            if np.all(y_k <= 0):
                self.result.status = AlgorithmStatus.UNBOUNDED
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

    def add_extra_vars(self, sense, const_num):
        if sense == Sense.LE:
            self.add_slack_var(const_num)
        elif sense == Sense.GE:
            self.add_surplus_var(const_num)
            self.add_artificial_var(const_num)
        elif sense == Sense.EQ:
            self.add_artificial_var(const_num)

    def add_slack_var(self, c):
        slack = self.add_var(0, sys.float_info.max,
                             's' + str(self.n_slack))
        slack.var_name_type = VarNameType.SLACK
        slack.coeff_c = 0.0
        slack.coeffs_a[c] = 1.0
        slack.in_basis = True
        self.basis.append(slack)
        self.n_slack += 1

    def add_surplus_var(self, c):
        surplus = self.add_var(0, sys.float_info.max,
                               'e' + str(self.n_surplus))
        surplus.var_name_type = VarNameType.SURPLUS
        surplus.coeff_c = 0.0
        surplus.coeffs_a[c] = -1.0
        self.n_surplus += 1

    def add_artificial_var(self, c):
        artificial = self.add_var(0, sys.float_info.max,
                                  'a' + str(self.n_surplus))
        artificial.var_name_type = VarNameType.ARTIFICIAL
        artificial.coeff_c = self.BIG_M
        artificial.coeffs_a[c] = 1.0
        artificial.in_basis = True
        self.basis.append(artificial)
        self.n_artificial += 1

    def prepare_coefficient_matrices(self):
        self.b = np.asarray(self.rhs, dtype=np.float32)
        self.b = self.b.reshape((self.b.shape[0], 1))
        # prepare coeffs_a
        for v in range(self.n_cols):
            matrix = np.zeros((self.n_rows, 1), dtype=np.float32)
            for key, value in self.vars[v].coeffs_a.items():
                matrix[key] = value
            self.A[self.vars[v]] = matrix

    def get_basic_feasible_solution(self):
        self.B_inv = np.identity(len(self.basis))
        B_inv_b = self.B_inv.dot(self.b)
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
                    if v.var_name_type == VarNameType.ARTIFICIAL]:
            if var.value > 0.0:
                self.result.status = AlgorithmStatus.INFEASIBLE
                return
        if self.is_terminated:
            self.result.status = AlgorithmStatus.OPTIMAL
        else:
            self.result.status = AlgorithmStatus.FEASIBLE

    def update_print_result(self):
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
        self.set_B_inv(self.basis)
        B_inv_b = self.B_inv.dot(self.b)
        for v in range(len(self.basis)):
            self.basis[v].value = B_inv_b[v].item()

    def set_B_inv(self, basis_vars):
        n = self.n_rows
        basic_matrix = self.get_coeff_matrix(basis_vars, n, n)
        self.B_inv = np.linalg.inv(basic_matrix)

    def get_coeff_matrix(self, variables, row, col):
        coeffs_a = []
        for var in variables:
            coeffs_a.append(self.A[var])
        return np.concatenate(coeffs_a, axis=1).reshape((row, col))

    def get_value(self, var):
        var_in_vars = self.get_first_or_default([v for v in self.vars if v == var])
        if var_in_vars:
            return var_in_vars.value
        else:
            raise UnknownVariableError('Unknown variable to the solver.')

    @staticmethod
    def get_first_or_default(var):
        if not var:
            return None
        return var[0]

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
