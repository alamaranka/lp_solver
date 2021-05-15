import sys

from lp.entity import Expression, Sense, ObjectiveType
from lp.model import Model

c = [2.0, 3.5, 8.0, 1.5, 11.0, 1.0]
p = [4.0, 8.0, 7.0, 1.3, 8.0, 9.2]
f = [1.0, 5.0, 9.0, 0.1, 7.0, 1.0]
ch = [15.0, 11.7, 0.4, 22.6, 0.0, 17.0]
cal = [90, 120, 106, 97, 130, 180]


if __name__ == '__main__':
    try:
        # set of indices
        n = range(len(c))

        # create model
        model = Model()

        # create variables
        x = [model.add_var(name='x[%i]' % i) for i in n]

        # constraints
        calorie = Expression()
        protein = Expression()
        fat = Expression()
        carbohydrates = Expression()
        obj_expr = Expression()

        for i in n:
            calorie.add_term(cal[i], x[i])
            protein.add_term(p[i], x[i])
            fat.add_term(f[i], x[i])
            carbohydrates.add_term(ch[i], x[i])
            obj_expr.add_term(c[i], x[i])

        model.add_const(calorie, Sense.GE, 300)
        model.add_const(protein, Sense.LE, 10)
        model.add_const(carbohydrates, Sense.GE, 10)
        model.add_const(fat, Sense.GE, 8)

        fish_expr = Expression()
        fish_expr.add_term(1.0, x[4])
        model.add_const(fish_expr, Sense.GE, 0.5)

        milk_expr = Expression()
        milk_expr.add_term(1.0, x[1])
        model.add_const(milk_expr, Sense.LE, 1.0)

        # create objective
        model.set_objective(obj_expr, ObjectiveType.MIN)

        # solve model
        status = model.solve()
    except IOError:
        print('Some error!')
