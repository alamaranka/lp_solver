from lp.entity import Expression, Sense, ObjectiveType
from lp.model import Model

M = 100  # BIGM value
s = 0
t = 5
c = [
   [0, 4, 2, 0, 0, 0],  # s
   [0, 0, 0, 3, 0, 0],  # a
   [0, 0, 0, 2, 3, 0],  # b
   [0, 0, 1, 0, 0, 2],  # c
   [0, 0, 0, 0, 0, 4],  # d
   [M, 0, 0, 0, 0, 0]   # t
]

try:
    # set of indices
    n = range(len(c))

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
        model.add_const(expr, Sense.EQ, 0)

    # capacity constraints
    for i in n:
        for j in n:
            model.add_const_var(x[i][j], Sense.LE, c[i][j])

    # create objective
    obj_expr = Expression()
    obj_expr.add_term(1.0, x[t][s])
    model.set_objective(obj_expr, ObjectiveType.MAX)

    # solve model
    status = model.solve()
except IOError:
    print('Some error!')