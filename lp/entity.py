import enum
import sys


class ProblemInstance:
    def __init__(self, obj, c, var_type, A, b, sense):
        self.obj = obj
        self.c = c
        self.var_type = var_type
        self.A = A
        self.b = b
        self.sense = sense


class Variable:
    def __init__(self, lb, ub, name,
                 var_type, var_name_type):
        self.lb = lb
        self.ub = ub
        self.name = name
        self.var_type = var_type
        self.var_name_type = var_name_type
        self.value = 0.0
        self.coeff_c = 0.0
        self.coeffs_a = {}
        self.in_basis = False


class Expression:
    def __init__(self):
        self.vars = []
        self.vals = []

    def add_term(self, coeff, var):
        self.vars.append(var)
        self.vals.append(coeff)


class Constraint:
    def __init__(self, expr=None,
                 sense=None, rhs=0):
        self.expr = expr
        self.sense = sense
        self.rhs = rhs


class Objective:
    def __init__(self, expr=None, obj_type=None):
        self.expr = expr
        self.obj_type = obj_type
        self.value = 0.0


class Result:
    def __init__(self, status='NONE',
                 obj_val=0, solution=None):
        self.status = status
        self.obj_val = obj_val
        self.solution = solution


class Node:
    def __init__(self, model,
                 is_pruned=False):
        self.model = model
        self.is_pruned = is_pruned


class SolverParam:

    class BranchingAlgorithm(enum.IntEnum):
        NONE = 0
        DFS = 1
        BFS = 2

    MIP_GAP = 0.0
    TIME_LIMIT = sys.float_info.max
    BRANCHING_ALGORITHM = BranchingAlgorithm.DFS


class Sense(enum.IntEnum):
    NONE = 0
    LE = 1
    EQ = 2
    GE = 3


class ObjectiveType(enum.IntEnum):
    NONE = 0
    MIN = 1
    MAX = 2


class VarType(enum.IntEnum):
    NONE = 0
    CONTINUOUS = 1
    BINARY = 2
    INTEGER = 3


class VarNameType(enum.IntEnum):
    NONE = 0
    PRIMAL = 1
    SLACK = 2
    SURPLUS = 3
    ARTIFICIAL = 4


class AlgorithmStatus(enum.IntEnum):
    NONE = 0
    OPTIMAL = 1
    FEASIBLE = 2
    INFEASIBLE = 3
    UNBOUNDED = 4


class UnknownVariableError(Exception):
    def __init__(self, message):
        self.message = message


class UnknownModelError(Exception):
    def __init__(self, message):
        self.message = message
