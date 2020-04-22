import json
import time

import numpy as np

from lp.entity import AlgorithmStatus, VarNameType, \
    ObjectiveType, VarType, Node
from lp.helper import get_c_b


class InitialBasicSolutionGenerator:
    def __init__(self, model):
        self._model = model

    def generate(self):
        n_rows = self._model.n_rows
        self._model.B_inv = np.identity(n_rows)
        B_inv_b = self._model.B_inv.dot(self._model.b)
        if np.all(B_inv_b >= 0):
            for v in range(n_rows):
                self._model.basis[v].value = B_inv_b[v].item()
            return True
        return False


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
        pass

    def run(self):
        while not self._is_terminated:
            self.iterate()
        self.prepare_and_print_result()

    def iterate(self):
        c_b = get_c_b(self._model.basis)
        w = c_b.dot(self._model.B_inv)
        z_c = {}
        for var in [v for v in self._model.vars if not v.in_basis]:
            z_c[var] = w.dot(self._model.A[var]) - var.coeff_c
        entering_var = max(z_c, key=z_c.get)
        if z_c[entering_var] > 0:
            y_k = self._model.B_inv.dot(self._model.A[entering_var])
            if np.all(y_k <= 0):
                self._model.result.status = AlgorithmStatus.UNBOUNDED
                self._is_terminated = True
                return
            rates = {}
            for i in range(len(y_k)):
                if y_k[i] > 0:
                    rates[i] = self._model.basis[i].value / y_k[i]
            leaving_var_index = min(rates, key=rates.get)
            entering_var.value = rates[leaving_var_index].item()
            entering_var.in_basis = True
            leaving_var = self._model.basis[leaving_var_index]
            leaving_var.value = 0.0
            leaving_var.in_basis = False
            self.update_basis(leaving_var, entering_var)
            self.update_obj_value()
        else:
            self._is_terminated = True
            self.check_status()

    def update_basis(self, leaving_var, entering_var):
        self._model.basis = [entering_var if x == leaving_var
                             else x for x in self._model.basis]
        self._model.B_inv = self.update_B_inv(self._model.basis)
        B_inv_b = self._model.B_inv.dot(self._model.b)
        for v in range(len(self._model.basis)):
            self._model.basis[v].value = B_inv_b[v].item()

    def update_obj_value(self):
        obj_val = 0.0
        for var in self._model.vars:
            obj_val += var.coeff_c * var.value
        self._model.obj.value = obj_val

    def update_B_inv(self, basis_vars):
        n = self._model.n_rows
        basic_matrix = self.get_coeff_matrix(basis_vars, n, n)
        return np.linalg.inv(basic_matrix)

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

    def get_coeff_matrix(self, variables, row, col):
        coeffs_a = []
        for var in variables:
            coeffs_a.append(self._model.A[var])
        return np.concatenate(coeffs_a, axis=1).reshape((row, col))

    def prepare_and_print_result(self):
        solution = {}
        for var in self._model.basis:
            solution[var.name] = round(var.value, 3)
        self._model.result.solution = solution
        if self._model.obj.obj_type == ObjectiveType.MIN:
            self._model.result.obj_val = round(self._model.obj.value, 3)
        elif self._model.obj.obj_type == ObjectiveType.MAX:
            self._model.result.obj_val = -round(self._model.obj.value, 3)
        print(json.dumps(self._model.result.__dict__))


class MIPSolver:
    """
    This uses Branch & Bound algorithm to solve the MIP problem.

    Parameters
    ==========
    model           : Model class

    Properties
    ==========
    _tree          : list of nodes in the solution tree; index 0 is the root node
    _n_nodes       : number of nodes in the solution tree
    _is_terminated : bool whether or not the simplex solver is terminated
    _int_vars      : list of integer and binary variables in the model
    _root_node     : Node class of root
    _mip_gap       : double current mip_gap in the tree
    _solution_time : double total time elapsed in seconds since the solve method called
    """

    def __init__(self, model):
        self._model = model
        self._tree = []
        self._n_nodes = 0
        self._is_terminated = False
        self._int_vars = [v for v in model.vars
                          if (v.var_type == VarType.BINARY) or
                          (v.var_type == VarType.INTEGER)]
        self._root_node = Node(model)
        self._mip_gap = 100.0
        self._solution_time = 0

    def run(self):
        current_node = self._root_node
        self._tree.append(current_node)
        while not self.is_terminated():
            simplex_solver = SimplexSolver(current_node.model)
            simplex_solver.run()
            self._root_node.is_pruned = True
            # TODO: handle pruning and branching
            # if self.is_pruned():
            #   do stuff
            # else:
            #   create 2 deep copy of model and add cuts
            # select current_node

    def is_terminated(self):
        self._solution_time = time.clock() - self._model.start_time
        any_nodes_to_branch = self.any_nodes_to_branch()
        is_mip_gap_reached = self._mip_gap <= self._model.SOLVER_PARAM.MIP_GAP
        is_time_limit_reached = self.solution_time >= self._model.SOLVER_PARAM.TIME_LIMIT
        return (not any_nodes_to_branch) or is_mip_gap_reached or is_time_limit_reached

    def any_nodes_to_branch(self):
        nodes_to_branch = [n for n in self._tree if not n.is_pruned]
        if len(nodes_to_branch) == 0:
            return False
        return True


