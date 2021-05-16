from lp.entity import Expression, Sense, ObjectiveType, VarType
from lp.model import Model

r = [100, 150]
a = [[8000, 4000], [15, 30]]
b = [40000, 200]

try:
    # set of indices
    n = range(len(r))

    # create model
    model = Model()

    # create variables
    x = [model.add_var(name='x[%i]' % i, var_type=VarType.INTEGER) for i in n]

    # constraints
    for j in range(2):
        expr = Expression()
        for i in n:
            expr.add_term(a[j][i], x[i])
        model.add_const(expr, Sense.LE, b[j])

    # create objective
    obj_expr = Expression()
    for i in n:
        obj_expr.add_term(r[i], x[i])
    model.set_objective(obj_expr, ObjectiveType.MAX)

    # solve model
    status = model.solve()
except IOError:
    print('Some error!')