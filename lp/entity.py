import enum


class Sense(enum.Enum):
    NONE = 0
    LE = 1
    EQ = 2
    GE = 3


class ObjectiveType(enum.Enum):
    NONE = 0
    MIN = 1
    MAX = 2


class VariableType(enum.Enum):
    NONE = 0
    PRIMAL = 1
    SLACK = 2
    SURPLUS = 3
    ARTIFICIAL = 4


class AlgorithmStatus(enum.Enum):
    NONE = 0
    OPTIMAL = 1
    FEASIBLE = 2
    INFEASIBLE = 3
    UNBOUNDED = 4
