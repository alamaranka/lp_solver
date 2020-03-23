import sys
import numpy as np

from lp.entity import Sense, ObjectiveType, Expression
from lp.model import Model


def run():
    model = Model()

    n_cols = 1000
    n_rows = 500
    cols = []
    rows = []
    obj_expr = Expression()

    coeffs = np.random.rand(n_rows, n_cols)
    rhs = np.random.rand(n_rows)
    cost = np.random.rand(n_cols)

    for c in range(n_cols):
        cols.append(model.add_var(lb=0, ub=sys.float_info.max, name='x' + str(c)))
        obj_expr.add_term(cost[c], cols[c])

    for r in range(n_rows):
        expr = Expression()
        for c in range(n_cols):
            expr.add_term(coeffs[r][c], cols[c])
        rows.append(model.add_const(expr, Sense.LE, rhs[r]))

    model.set_objective(obj_expr, ObjectiveType.MAX)

    model.solve()
