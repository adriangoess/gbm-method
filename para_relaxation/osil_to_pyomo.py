import pyomo.environ as pyo

from osil_parser.osil_parser import OSILParser


class OsilPyomoConverter(object):
    """
    Given an osil parser which has parsed an osil file, this one models the respective formulation in pyomo.
    It can further solve it by means of gams

    Keywords:
    osil_parser -- A parsed osil parser with custom type OSILParser
    """
    def __init__(self, osil_parser):
        """storing the osil parser and necessary paramters"""
        assert isinstance(osil_parser, OSILParser)
        self.parser = osil_parser

        self.variable_type_mapping = {"C" : pyo.Reals,
                                      "I" : pyo.Integers,
                                      "B" : pyo.Binary}

    def adjust_parser(self, osil_parser):
        """adjust the initialized parser"""
        assert isinstance(osil_parser, OSILParser)
        self.parser = osil_parser

    def setup_model(self):
        """model the instance in the parser with pyomo"""
        model = pyo.ConcreteModel()

        model = self._init_variables(model)
        model = self._init_objective(model)
        model = self._init_constraints(model)

        return model

    def _init_variables(self, m):
        """init the variables"""
        # internal function to extract bounds
        def get_bounds(model, variable_index):
            var = self.parser.variables[variable_index]
            lb = var.lb
            ub = var.ub
            return (lb, ub)

        # internal function to extract variable type
        def get_var_type(model, variable_index):
            return self.variable_type_mapping[self.parser.variables[variable_index].variable_type]

        def initialize_nonzero(model, variable_index):
            initialize = None
            if self.parser.n_logln > 0:
                var_data = self.parser.variables[variable_index]
                if var_data.ub is None:
                    initialize = 1e-6
                else:
                    initialize = min(var_data.ub, 1e-6)
                if var_data.lb is not None:
                    initialize = max(initialize, var_data.lb)
            return initialize

        # create variables
        variable_indices = list(range(len(self.parser.variables)))
        m.variables = pyo.Var(variable_indices, bounds=get_bounds, domain=get_var_type, initialize=initialize_nonzero)

        return m

    def _init_objective(self, m):
        """init the objective"""
        if len(self.parser.objective) > 0:
            # get offset and sense in objective
            offset = self.parser.objective[0].constant
            sense = pyo.minimize if self.parser.objective[0].direction == "min" else pyo.maximize

            # Index -1 in constraints is assumed to replace the linear objective function
            # TODO: check for validity
            objective_term = self._eval_linear_obj(self.parser.objective[0].coeffs, m.variables)
            objective_term += self._eval_quad_term(self.parser.quad_coeffs[-1], m.variables)
            if -1 in self.parser.nl_constraints.keys():
                objective_term += self.parser.nl_constraints[-1].eval(m.variables)

            m.obj = pyo.Objective(expr=objective_term + offset, sense=sense)
        else:
            m.obj = pyo.Objective(expr=0)
            print(f"Warning; constant objective found in instance {self.parser.name}")
        return m

    def _init_constraints(self, m):
        """init the constraints"""
        # iterate through meta data, construct constraint term, add with lb and ub
        m.constraints = pyo.ConstraintList()
        for constraint_index, (name, lb, ub) in enumerate(self.parser.constraint_infos):
            assert not (lb is None and ub is None), f"lb and ub cannot be None at the same time"

            constraint_term = self._eval_lin_term(self.parser.lin_coeffs[constraint_index], m.variables)
            constraint_term += self._eval_quad_term(self.parser.quad_coeffs[constraint_index], m.variables)
            if constraint_index in self.parser.nl_constraints.keys():
                constraint_term += self.parser.nl_constraints[constraint_index].eval(m.variables)

            if isinstance(constraint_term, (int,)):
                if constraint_term == 0:
                    continue
            if lb == ub:
                m.constraints.add(expr= constraint_term == lb)
            else:
                if lb is not None:
                    m.constraints.add(expr= constraint_term >= lb)
                if ub is not None:
                    m.constraints.add(expr= constraint_term <= ub)
        return m

    @staticmethod
    def _eval_lin_term(lin_terms, variables):
        """ for a list of tuples of (var index, coefficient) construct the linear term """
        term = 0
        # iterate over tuples
        for (variable_index, coefficient) in lin_terms:
            term += coefficient * variables[variable_index]
        return term

    @staticmethod
    def _eval_quad_term(quad_terms, variables):
        """for a list of tuples of (var index, var index, coefficient) construct the quad term"""
        term = 0
        # iterate over tuples
        for (variable_index1, variable_index2, coefficient) in quad_terms:
            term += coefficient * variables[variable_index1] * variables[variable_index2]
        return term

    @staticmethod
    def _eval_linear_obj(coefficients, variables):
        """for a dictionary of (variable index, coefficient) construct a linear term"""
        term = 0
        # iterate over dictionary items
        for (variable_index, coefficient) in coefficients.items():
            term += coefficient * variables[variable_index]
        return term

