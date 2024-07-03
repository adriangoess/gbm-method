import os
import json
import numpy as np
import pyomo.environ as pyo


class ParaboloidModel(object):
    """
        Given settings for a function to approximate, model specifications and a number of paraboloids for the
        approximation, an object of this class is supposed to set up the model from the preprint

        Keyword arguments:
        func_settings_path -- path to a function settings file (string)
        model_settings_path -- path to a model parameters file (string)
        number_paras -- number of paraboloids for the approximation (integer)
        approx_below -- boolean to indicate an approximation from below or above (bool)
        custom_max_n_delta_t, _d -- custom number of discretization points for approximation from above/below, resp. (integer)
    """
    def __init__(self, func_settings_path, model_settings_path, number_paras, approx_below=True,
                 custom_max_n_delta_t=np.inf, custom_max_n_delta_d=np.inf):
        """ initialize the necessary parameters and settings """
        # check for suitability of the function settings path
        assert isinstance(func_settings_path, (str,)), "function settings path must be string"
        assert os.path.exists(func_settings_path), "function settings path must exist"
        assert func_settings_path.endswith(".json"), "function settings path must be json"
        function_settings = json.load(open(func_settings_path))
        assert isinstance(function_settings, (dict,)), "error in parsing function settings json"

        # check for the right keys and initialize
        assert set(function_settings.keys()) == {"name", "dim", "func", "Func", "lb", "ub", "L-constant"}
        self.dim = int(function_settings["dim"])

        # (-1) ** ... flips function if approximation direction from above
        assert function_settings["func"] in ["sin(x1)", "cos(x1)", "exp(x1)", "x^3"], "only sine, cosine, x^3 and exp function implemented to far"
        if function_settings["func"] == "sin(x1)":
            self.f = lambda x: (-1) ** (1 + approx_below) * np.sin(x)
        elif function_settings["func"] == "cos(x1)":
            self.f = lambda x: (-1) ** (1 + approx_below) * np.cos(x)
        elif function_settings["func"] == "x^3":
            self.f = lambda x: (-1) ** (1 + approx_below) * x ** 3
        else:
            self.f = lambda x: (-1) ** (1 + approx_below) * np.exp(x)
        assert function_settings["Func"] in ["-cos(x1)", "sin(x1)", "exp(x1)", "1/4x^4"], ("only -cosine, sine, 1/4 x^3 and exp function as antiderivative implemented so far")
        if function_settings["Func"] == "-cos(x1)":
            self.F = lambda x: (-1) ** (1 + approx_below) * -np.cos(x)
        elif function_settings["Func"] == "sin(x1)":
            self.F = lambda x: (-1) ** (1 + approx_below) * np.sin(x)
        elif function_settings["func"] == "x^3":
            self.F = lambda x: (-1) ** (1 + approx_below) * 0.25 * x ** 4
        else:
            self.F = lambda x: (-1) ** (1 + approx_below) * np.exp(x)

        self.lb = float(function_settings["lb"])

        self.ub = float(function_settings["ub"])
        assert self.lb < self.ub, "lower bound must be strictly less than upper bound"

        self.L = float(function_settings["L-constant"])
        assert self.L > 0, "Lipschitz constant must be positive"

        # check for suitability of model settings path
        assert isinstance(model_settings_path, (str,)), "model settings path must be string"
        assert os.path.exists(model_settings_path), "model settings path must exist"
        assert model_settings_path.endswith(".json"), "model settings path must be json"
        model_parameters = json.load(open(model_settings_path))
        assert isinstance(model_parameters, (dict,)), "error in parsing model settings path json"

        # check for right keys and initialize
        assert set(model_parameters.keys()) == {"name", "eps", "delta"}
        self.eps = float(model_parameters["eps"])
        self.delta = 0
        self.gamma = 0.5 * (self.delta/self.eps) #TODO: adjust

        # initialize delta t as the max TODO: adjust for several dimensions
        max_n_delta_t = int(custom_max_n_delta_t)
        self.delta_t = (self.ub - self.lb)/max_n_delta_t
        self.t_indices = list(range(max_n_delta_t + 1))
        # initialize the slope for the paraboloids TODO: make a adaptive procedure
        self.C = (self.ub - self.lb) * self.L/self.delta_t
        # initialize delta d as the max TODO: adjust for several dimensions;
        max_n_delta_d = int(custom_max_n_delta_d)
        self.delta_d = (self.ub - self.lb)/max_n_delta_d
        self.d_indices = list(range(max_n_delta_d + 1))
        # set values for the big Ms;
        self.M1 = 100
        self.M2 = self.M1
        # set default bounds for all variables;
        self.max_bound = 10_000
        self.default_bounds = [-self.max_bound, self.max_bound]

        # check that number of paraboloids is integer and store
        assert isinstance(number_paras, (int,)), "number of paraboloids must be integer"
        self.n_paras = number_paras
        self.para_indices = list(range(self.n_paras))

    def setup_model(self, print_model=True, initial_variable_values=None):
        """creating the model based on the preprint"""
        # initialize the model
        m = pyo.ConcreteModel()

        # initialize the variables
        m = self._initialize_variables(m, initial_variable_values)

        # initialize the constraints
        m = self._initialize_constraints(m)

        # model the objective function
        m.obj = pyo.Objective(rule=self._minimize_violation, sense=pyo.minimize)

        # model the first parabola to contain the first point for symmetry breaking
        m.symmetry_breaking = pyo.Constraint(expr=m.contain[0, 0] == 1)

        if print_model:
            m.pprint()
        return m

    def extract_results(self, model):
        """ extract variable values for solved model """
        quadratic_coefficients = self._extract_values_per_variable(self.para_indices, model.quad)
        linear_coefficients = self._extract_values_per_variable(self.para_indices, model.lin)
        constant_coefficients = self._extract_values_per_variable(self.para_indices, model.cons)

        return quadratic_coefficients, linear_coefficients, constant_coefficients

    def _initialize_variables(self, model, initial_values):
        """ initializing the variables of the model """
        # initialize variables for the coefficients of the parabolas
        model.quad = pyo.Var(self.para_indices, bounds=self.default_bounds)
        model.lin = pyo.Var(self.para_indices, bounds=self.default_bounds)
        model.cons = pyo.Var(self.para_indices, bounds=self.default_bounds)
        # if available, initialize as much variable as possible
        if initial_values is not None:
            assert len(initial_values) == 3, f"initial variable values should contain quads, lins and cons"
            quads, lins, cons = initial_values
            assert len(quads) == len(lins) == len(cons), f"lengths of variable value lists must be identical"
            for i in range(min(len(quads), len(self.para_indices))):
                model.quad[i] = quads[i][0]
                model.lin[i] = lins[i][0]
                model.cons[i] = cons[i][0]

        # initialize variables to track containment of the error bounds, i.e. s^l_t
        model.contain = pyo.Var(self.para_indices, self.t_indices, within=pyo.Binary)
        # initialize variables to track the violation in terms of the integral between consecutive points
        model.viol = pyo.Var(self.para_indices, self.d_indices, within=pyo.NonNegativeReals)

        return model

    def _initialize_constraints(self, model):
        """ initializing the constraints of the model """
        # (5b): model lower bound of parabolas as big M formulation; p^l(t) >= f(t) - del - M1 * (1-s^l_t)
        model.approx_bound = pyo.Constraint(self.para_indices, self.t_indices, rule=self._lower_parabola_bound)
        # (5c): enforcing one of the containment binaries to be 1; sum_l s^l_t >= 1
        model.containment = pyo.Constraint(self.t_indices, rule=self._containment_sum)

        # (5e): model upper bound of parabolas; p^l(t) <= f(d) - gamma * eps
        model.upper_bound = pyo.Constraint(self.para_indices, self.d_indices, rule=self._upper_parabola_bound)
        # (5f): track violation of integral between parabola and function; v^l_d >= integral p - (f - gamma eps)
        model.violation_tracking = pyo.Constraint(self.para_indices, self.d_indices[:-1],
                                                  rule=self._violation_tracking)

        return model

    def _lower_parabola_bound(self, model, para_index, t_index):
        """
            modeling the lower bound of each parabola as the function to approximate, eps, and a big m formulation;
            constraint (5b)
        """
        # initialize the discretization point t; TODO: adjust to multi-dimensional
        t = self.lb + t_index * self.delta_t
        # initialize the parabola w.r.t. t
        parabola = model.quad[para_index] * t ** 2 + model.lin[para_index] * t + model.cons[para_index]
        # evaluate the function to approximate at t and compute the big M formula
        func_eval = self.f(t)
        big_m_formula = self.M1 * (1 - model.contain[para_index, t_index])

        # return constraint as modelled in (5b) of draft + custom epsilon
        return parabola >= func_eval - 0.9 * self.eps - big_m_formula # -self.delta

    def _containment_sum(self, model, t_index):
        """ second part of big M formulation such that one containment variable has to be nonzero; constraint (5c) """
        return sum(model.contain[para_index, t_index] for para_index in self.para_indices) >= 1

    def _containment_slope_bounds(self, model):
        """
            slope bound at neighboring points for all discretization points t, i.e.
            |d/dxi p^l(t')| <= 2L + M2 * (1-s^l_t) = rhs <-> d/dxi p^l(t') <= rhs and d/dxi p^l(t') >= -rhs;
            constraints (5d)
        """
        # initialize the constraint lists
        model.upper_slope_bounds = pyo.ConstraintList()
        model.lower_slope_bounds = pyo.ConstraintList()
        # TODO: make multi-dimensional
        for para_index in self.para_indices:
            for t_index in self.t_indices:
                # compute all possible neighbouring t' to t
                neighbors = []
                t = self.lb + t_index * self.delta_t
                # TODO make multi-dimensional
                t_prime1 = t - self.delta_t
                if t_prime1 >= self.lb:
                    neighbors.append(t_prime1)
                t_prime2 = t + self.delta_t
                if t_prime2 <= self.ub:
                    neighbors.append(t_prime2)

                # add constraints for each t', rhs = 2L + M2 * (1-s^l_t) is computed a-priori
                rhs = 2 * self.L + self.M2 * (1 - model.contain[para_index, t_index])
                for t_prime in neighbors:
                    # compute the derivative of p^l at t' = 2 * quad * t' + 2 * lin
                    p_derivative = 2 * model.quad[para_index] * t_prime + model.lin[para_index]
                    # add upper and lower bound constraints as in description
                    model.upper_slope_bounds.add(expr=p_derivative <= rhs)
                    model.lower_slope_bounds.add(expr=p_derivative >= -rhs)

        return model

    def _upper_parabola_bound(self, model, para_index, d_index):
        """
            method to model the function to approximate itself as an upper bound to the approximation; constraint (5e)
        """
        # initialize the discretization point d; TODO: adjust to multi-dimensional
        d = self.lb + d_index * self.delta_d
        # initialize the parabola w.r.t. d
        parabola = model.quad[para_index] * d ** 2 + model.lin[para_index] * d + model.cons[para_index]
        # evaluate the function to approximate at d
        func_eval = self.f(d)

        # return constraint as modelled in (5e) of draft manipulated by a fixed eps
        return parabola <= func_eval - 0.25 * self.eps #self.gamma * self.eps

    def _violation_tracking(self, model, para_index, d_index):
        """
            violation variables shall track a positivity of the integral between parabola and function on an interval;
            constraint (5f)
        """
        # initialize lower and upper bound of the integral; TODO: make multi-dimensional
        d = self.lb + d_index * self.delta_d
        d_next = d + self.delta_d
        assert d_next <= self.ub + 1e9, f"right boundary of integral cannot exceed upper bound"

        # compute the anti-derivative of the parabola at d and at d_next
        cubic_d = (1 / 3 * d ** 3 * model.quad[para_index]
                   + 1 / 2 * d ** 2 * model.lin[para_index]
                   + d * model.cons[para_index])
        cubic_d_next = (1 / 3 * d_next ** 3 * model.quad[para_index]
                        + 1 / 2 * d_next ** 2 * model.lin[para_index] +
                        d_next * model.cons[para_index])

        # compute the integral with the anti-derivative of the function F by first computing boundaries
        integral_upper_boundary = cubic_d_next - (self.F(d_next) - self.gamma * self.eps * d_next)
        integral_lower_boundary = cubic_d - (self.F(d) - self.gamma * self.eps * d)
        integral = integral_upper_boundary - integral_lower_boundary

        # return the violation tracking as variable >= integral
        return model.viol[para_index, d_index] >= integral

    def _minimize_violation(self, model):
        """ objective minimizes the sum of all violation variables """
        # TODO: make multi-dimensional
        return sum(model.viol[para_index, d_index] for para_index in self.para_indices
                   for d_index in self.d_indices[:-1])

    def _extract_values_per_variable(self, index1, variables, index2=None, print_values=False, print_null=False):
        """
            extract lists of variable values given the variables and possible index lists 1 and 2
        """
        # assert the expected types
        assert isinstance(index1, (list,)), f"index1 must be a list to extract values from it"
        # maximal number of letters per line for printing
        n_max_chars = 50
        # maximal number of digits to round to
        n_digits = 5
        # initial list of values
        values = []
        # depending on the availability of one or two index sets, extract the value and print if desired
        if index2 is None:
            for element in index1:
                variable = variables[element]
                value = None if variable.value is None else np.round(variable.value, n_digits)
                values.append([value])
                if print_values:
                    self._print_value(variable, value, n_max_chars, print_null)
        else:
            for ind1 in index1:
                for ind2 in index2:
                    variable = variables[ind1, ind2]
                    value = None if variable.value is None else np.round(variable.value, n_digits)
                    values.append(value)
                    if print_values:
                        self._print_value(variable, value, n_max_chars, print_null)
        return values

    @staticmethod
    def _print_value(variable, value, n_max_chars, print_null):
        """ printing values for a maximal number of chars and if null """
        n_blanks = max(n_max_chars - len(str(variable)), 0)
        if (value is not None and value != 0) or print_null:
            print(f"{str(variable)}" + " " * n_blanks + f"\t| {value}")

