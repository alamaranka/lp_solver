import json
import sys

from lp.entity import Sense, ObjectiveType, Expression, ProblemInstance
from lp.model import Model


def run(test_name):
    model = Model()
    obj_expr = Expression()

    with open(test_name + '.json') as json_file:
        data = json.load(json_file,
                         object_hook=lambda d: ProblemInstance(c=d['c'], A=d['A'],
                                                               b=d['b'], sense=d['sense']))

    cost = data.c
    rhs = data.b
    coeffs = data.A
    sense = data.sense
    n_cols = len(data.c)
    n_rows = len(data.b)
    cols = []
    rows = []

    for c in range(n_cols):
        cols.append(model.add_var(lb=0, ub=sys.float_info.max, name='x' + str(c)))
        obj_expr.add_term(cost[c], cols[c])

    for r in range(n_rows):
        expr = Expression()
        for c in range(n_cols):
            expr.add_term(coeffs[r][c], cols[c])
        if sense[r] == '<=':
            rows.append(model.add_const(expr, Sense.LE, rhs[r]))
        elif sense[r] == '>=':
            rows.append(model.add_const(expr, Sense.GE, rhs[r]))
        elif sense[r] == '==':
            rows.append(model.add_const(expr, Sense.EQ, rhs[r]))

    model.set_objective(obj_expr, ObjectiveType.MAX)

    model.solve()
