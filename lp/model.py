import math
import sys
import time

import numpy as np

from lp.entity import VarNameType, Sense, ObjectiveType, \
    Result, Variable, Constraint, Objective, \
    VarType, UnknownVariableError, UnknownModelError, SolverParam
from lp.helper import get_first_or_default, set_reverse_sense
from lp.solver import MIPSolver, InitialBasicSolutionGenerator


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
    is_terminated   : bool whether or not the solution is completed
    A               : dict of coefficient matrix A
    b               : numpy array of b vector
    basis           : list of variables in the basis
    B_inv           : numpy array of basis matrix inverse
    rhs             : list of right hand side values
    n_rows          : int number of constraints
    n_cols          : int number of variables
    n_slack         : int number of slack variables
    n_surplus       : int number of surplus variables
    n_artificial    : int number of artificial variables
    BIG_M           : double used to model artificial variables in the model
    solution_time   : double total solution time in seconds
    """

    SOLVER_PARAM = SolverParam()

    def __init__(self, name='Mathematical Model'):
        self.name = name
        self.vars = []
        self.consts = []
        self.obj = None
        self.result = Result()
        self.is_mip = False
        self.is_terminated = False
        self.A = {}
        self.b = None
        self.basis = []
        self.B_inv = None
        self.rhs = []
        self.n_rows = 0
        self.n_cols = 0
        self.n_slack = 0
        self.n_surplus = 0
        self.n_artificial = 0
        self.BIG_M = math.pow(10, 6)
        self.start_time = 0
        self.end_time = 0

    @property
    def __str__(self):
        return self.name

    def solve(self):
        self.start_time = time.clock()
        self.prepare_coefficient_matrices()
        ibsg = InitialBasicSolutionGenerator(self)
        mip_solver = MIPSolver(self)
        if ibsg.generate():
            mip_solver.run()
        else:
            raise UnknownModelError('Unknown model error.')
        self.end_time = time.clock()
        solution_time = self.end_time - self.start_time
        print('Algorithm completed in {0} seconds.'
              .format(round(solution_time, 4)))

    def add_var(self, lb=0, ub=sys.float_info.max, name='',
                var_type=VarType.CONTINUOUS,
                var_name_type=VarNameType.PRIMAL):
        if var_type == VarType.BINARY or \
                var_type == VarType.INTEGER:
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
            sense = set_reverse_sense(sense)
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

    def add_cut(self, expr, sense, rhs):
        matrix = np.zeros((self.n_rows, 1), dtype=np.float32)
        for key, value in self.vars[v].coeffs_a.items():
            matrix[key] = value
        self.A[self.vars[v]] = matrix
        pass

    def get_value(self, var):
        var_in_vars = get_first_or_default([v for v in self.vars if v == var])
        if var_in_vars:
            return var_in_vars.value
        else:
            raise UnknownVariableError('Unknown variable to the solver.')
