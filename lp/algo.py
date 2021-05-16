import copy
import math
import time
from random import randrange

from lp.entity import VarType, Node, Sense
from lp.simplex import SimplexSolver


class Algorithm:
    """
    This uses Branch & Bound algorithm to solve the MIP problem.

    Parameters
    ==========
    model           : Model class

    Properties
    ==========
    _tree          : list of nodes in the solution tree; index 0 is the root node
    _n_nodes       : int number of nodes in the solution tree
    _int_vars      : list of integer and binary variables in the model
    _root_node     : Node class of root
    _mip_gap       : double current mip_gap in the tree
    _solution_time : double total time elapsed in seconds since the solve method called
    """

    def __init__(self, model):
        self._model = model
        self._tree = []
        self._n_nodes = 0
        self._int_vars = [v for v in model.vars
                          if (v.var_type == VarType.BINARY) or
                          (v.var_type == VarType.INTEGER)]
        self._root_node = Node(model)
        self._mip_gap = 100.0
        self._solution_time = 0

    def run(self):
        current_node = self._root_node
        self._tree.append(current_node)
        while not self.is_terminated():
            current_node = [t for t in self._tree if not t.is_pruned][0]
            SimplexSolver(current_node.model).run()
            current_node.is_pruned = True
            self.do_branching(current_node)
            # TODO: handle pruning and branching
            # if self.is_pruned():
            #   do stuff
            # else:
            #   create 2 deep copy of model and add cuts
            # select current_node

    def is_terminated(self):
        self._solution_time = time.perf_counter() - self._model.start_time
        any_nodes_to_branch = self.any_nodes_to_branch()
        is_mip_gap_reached = self._mip_gap <= self._model.SOLVER_PARAM.MIP_GAP
        is_time_limit_reached = self._solution_time >= self._model.SOLVER_PARAM.TIME_LIMIT
        return (not any_nodes_to_branch) or is_mip_gap_reached or is_time_limit_reached

    def do_branching(self, node):
        cond = False
        var = None
        while not cond:
            index = randrange(len(self._int_vars))
            var = self._int_vars[index]
            cond = not var.value.is_integer()

        lower = math.floor(var.value)
        lower_model = copy.deepcopy(node)
        lower_model.is_pruned = False
        lower_model.model.add_const_var(var, Sense.LE, lower)
        self._tree.append(lower_model)

        upper = math.ceil(var.value)
        upper_model = copy.deepcopy(node)
        upper_model.is_pruned = False
        upper_model.model.add_const_var(var, Sense.GE, upper)
        self._tree.append(upper_model)

    def any_nodes_to_branch(self):
        nodes_to_branch = [n for n in self._tree if not n.is_pruned]
        if len(nodes_to_branch) == 0:
            return False
        return True
