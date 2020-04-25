import sys

from lp.entity import Sense, ObjectiveType, Expression, VarType
from lp.model import Model


def run(data):
    model = Model()
    obj_expr = Expression()
    var_types = data.var_type
    cost = data.c
    rhs = data.b
    coeffs = data.A
    sense = data.sense
    n_cols = len(data.c)
    n_rows = len(data.b)
    cols = []
    rows = []

    # add variables
    for c in range(n_cols):
        var_type = VarType.NONE
        if var_types[c] == 'c':
            var_type = VarType.CONTINUOUS
        if var_types[c] == 'i':
            var_type = VarType.INTEGER
        elif var_types[c] == 'b':
            var_type = VarType.BINARY
        cols.append(model.add_var(lb=0, ub=sys.float_info.max,
                                  var_type=var_type, name='x' + str(c)))
        obj_expr.add_term(cost[c], cols[c])

    # add constraints
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

    # add objective
    if data.obj == 'max':
        model.set_objective(obj_expr, ObjectiveType.MAX)
    elif data.obj == 'min':
        model.set_objective(obj_expr, ObjectiveType.MIN)

    model.SOLVER_PARAM.MIP_GAP = 0.05
    model.SOLVER_PARAM.TIME_LIMIT = 30.0

    model.solve()
