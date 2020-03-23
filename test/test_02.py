from lp.entity import Sense, ObjectiveType, Expression
from lp.model import Model


def run():
    model = Model()

    x = model.add_var(name='x')
    y = model.add_var(name='y')

    expr = Expression()
    expr.add_term(2.0, x)
    expr.add_term(1.0, y)
    model.add_const(expr, Sense.LE, 1.0)

    expr = Expression()
    expr.add_term(1.0, y)
    model.add_const(expr, Sense.GE, 0.5)

    expr = Expression()
    expr.add_term(1.0, x)
    expr.add_term(1.0, y)
    model.add_const(expr, Sense.EQ, 0.75)

    expr = Expression()
    expr.add_term(1.0, x)
    expr.add_term(1.0, y)
    model.set_objective(expr, ObjectiveType.MAX)

    model.solve()
