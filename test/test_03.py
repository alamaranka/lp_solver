from lp.entity import Sense, ObjectiveType, Expression
from lp.model import Model


def run():
    model = Model()

    x = model.add_var(name='x')
    y = model.add_var(name='y')

    expr = Expression()
    expr.add_term(8000.0, x)
    expr.add_term(4000.0, y)
    model.add_const(expr, Sense.LE, 40000.0)

    expr = Expression()
    expr.add_term(15.0, x)
    expr.add_term(30.0, y)
    model.add_const(expr, Sense.LE, 200.0)

    expr = Expression()
    expr.add_term(100.0, x)
    expr.add_term(150.0, y)
    model.set_objective(expr, ObjectiveType.MAX)

    model.solve()

