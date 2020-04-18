import numpy as np

from lp.entity import Sense


def get_c_b(basis):
    coeffs = []
    for var in basis:
        coeffs.append(var.coeff_c)
    return np.asarray(coeffs, dtype=np.float32).reshape((1, len(coeffs)))


def get_first_or_default(var):
    if not var:
        return None
    return var[0]


def set_reverse_sense(sense):
    if sense == Sense.LE:
        return Sense.GE
    elif sense == Sense.GE:
        return Sense.LE
    else:
        return None


def get_var_index_in_const(var, expr):
    if var in expr.vars:
        return expr.vars.index(var)
    return -1
