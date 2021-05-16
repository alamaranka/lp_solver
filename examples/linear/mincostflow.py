from lp.entity import Expression, Sense, ObjectiveType
from lp.model import Model


c = [
   [0, 4, 4, 0, 0],
   [0, 0, 2, 2, 6],
   [0, 0, 0, 1, 3],
   [0, 0, 0, 0, 2],
   [0, 0, 3, 0, 0],
]

u = [
   [0, 15, 8,  0,  0],
   [0, 0,  20, 4,  10],
   [0, 0,  0,  15, 4],
   [0, 0,  0,  0,  20],
   [0, 0,  5,  0,  0],
]

b = [20, 0, 0, -5, -15]

try:
    # set of indices
    n = range(len(b))

    # create model
    model = Model()

    # create variables
    x = [[model.add_var(name='x[%i,%i]' % (i, j)) for j in n] for i in n]

    # flow constraints
    for i in n:
        expr = Expression()
        for j in n:
            expr.add_term(1.0, x[i][j])
            expr.add_term(-1.0, x[j][i])
        model.add_const(expr, Sense.EQ, b[i])

    # capacity constraints
    for i in n:
        for j in n:
            model.add_const_var(x[i][j], Sense.LE, u[i][j])

    # create objective
    obj_expr = Expression()
    for i in n:
        for j in n:
            obj_expr.add_term(c[i][j], x[i][j])
    model.set_objective(obj_expr, ObjectiveType.MIN)

    # solve model
    status = model.solve()
except IOError:
    print('Some error!')
