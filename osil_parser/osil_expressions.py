import numpy as np
import pyomo.environ as pyo
from osil_parser.osil_var import OSILVariable

"""
Storage of different expressions apart from variables and objectives
Contained here are
- OSILSummand (tuple of variable index (if), coefficient, level in expression tree, bounds)
- OSILSum (list of objects to sum, level, bound)
- OSILFactor (tuple of variable index, coefficient (one of these two =None), level in expression tree, bounds)
- OSILProduct (list of objects to multiply, level, bounds)
- OSILSquare (expression, level, bounds)
- OSILPower (expression, coefficient, level, bounds)
- OSILCosine (expression, level, bounds)
- OSILSine (variable index, level, bounds)
- OSILNegate (expression, level, bounds)
- OSILDivide (numerator, denominator, level, bounds)
- OSILSquareroot (expression, level, bounds)
- OSILExp (expression, level, bounds)
- OSILAbs (expression, level, bounds)
- OSILLn (expression, level, bounds)
- OSILLog10 (expression, level, bounds)
"""


class OSILSummand(object):
    def __init__(self, variable_index, coefficient, level):
        """initialize summand object (variable index if available, coefficient, level in expression tree, bounds)"""
        # index None is equivalent to constant
        assert (isinstance(variable_index, (int,)) and variable_index >= 0) or isinstance(variable_index, type(None))
        self.variable_index = variable_index
        assert isinstance(coefficient, (int, float))
        self.coefficient = coefficient
        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

    def compute_bounds(self, variables):
        """compute the lower and upper bound of the summand object, given a list of OSILVariables"""
        if self.variable_index is None:
            self.lower_bound = self.upper_bound = self.coefficient
        else:
            var = variables[self.variable_index]
            assert isinstance(var, (OSILVariable,)), f"Variables must be list of OSILVariable objects"
            if var.lb is None:
                lb = -np.inf
            else:
                lb = self.coefficient * var.lb
            if var.ub is None:
                ub = np.inf
            else:
                ub = self.coefficient * var.ub

            self.lower_bound = min(lb, ub)
            self.upper_bound = max(lb, ub)

            self.lower_bound = None if np.isinf(-self.lower_bound) else self.lower_bound
            self.upper_bound = None if np.isinf(self.upper_bound) else self.upper_bound

        return self.lower_bound, self.upper_bound

    def update_coefficient(self, coefficient):
        """update coefficient of summand"""
        assert isinstance(coefficient, (int, float))
        self.coefficient = coefficient

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return coefficient times variable or just coefficient"""
        if self.variable_index is None:
            return self.coefficient
        else:
            return self.coefficient * variables[self.variable_index]


class OSILSum(object):
    def __init__(self, sum_entities, level):
        """initialize sum object as list of summand objects + other nonlinear expressions & level in expression tree"""
        assert isinstance(sum_entities, (list, )) and len(sum_entities) > 0
        self.sum_entities = sum_entities
        # TODO: assertion for entity types
        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

        self.arg_lbs = []
        self.arg_ubs = []

    def compute_bounds(self, variables):
        """computing the bounds for the sum object with a list of OSILSummands and other non linearities"""
        self.arg_lbs = []
        self.arg_ubs = []
        for summand in self.sum_entities:
            # compute bounds of sum entity and store
            lb, ub = summand.compute_bounds(variables)
            self.arg_lbs.append(lb)
            self.arg_ubs.append(ub)

        # if None in a list, keep bound None, otherwise compute bound as sum
        if None not in self.arg_lbs:
            self.lower_bound = sum(self.arg_lbs)
        if None not in self.arg_ubs:
            self.upper_bound = sum(self.arg_ubs)

        return self.lower_bound, self.upper_bound

    def add_sum_entity(self, entity):
        """addition of summation entity like summand object or other nonlinear expression"""
        # TODO: add other possible entities like nonlinear expressions
        assert isinstance(entity, (OSILSummand, OSILProduct))
        self.sum_entities.append(entity)

    def remove_sum_entity(self, index):
        """removal of summation entity"""
        assert isinstance(index, (int,))
        self.sum_entities = self.sum_entities[:index] + self.sum_entities[(index + 1):]

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return sum of summation entities in evaluated form """
        summing = 0
        for entity in self.sum_entities:
            summing += entity.eval(variables)
        return summing


class OSILFactor(object):
    def __init__(self, variable_index, coefficient, level):
        """initialize factor object (variable index if available, coefficient, level in expression tree, bounds)"""
        assert (isinstance(variable_index, (int,)) and variable_index >= 0) or variable_index is None
        self.variable_index = variable_index
        assert isinstance(coefficient, (int, float))
        self.coefficient = coefficient
        # avoid variable index None and coefficient None
        assert not (self.variable_index is None and self.coefficient is None)
        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

    def compute_bounds(self, variables):
        """compute the lower and upper bound of the factor object, given a list of OSILVariables"""
        if self.variable_index is None:
            self.lower_bound = self.upper_bound = self.coefficient
        else:
            var = variables[self.variable_index]
            assert isinstance(var, (OSILVariable,)), f"Variables must be list of OSILVariable objects"
            if var.lb is None:
                lb = -np.inf
            else:
                lb = self.coefficient * var.lb
            if var.ub is None:
                ub = np.inf
            else:
                ub = self.coefficient * var.ub
            self.lower_bound = min(lb, ub)
            self.upper_bound = max(lb, ub)

            self.lower_bound = None if np.isinf(-self.lower_bound) else self.lower_bound
            self.upper_bound = None if np.isinf(self.upper_bound) else self.upper_bound

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return variable or coefficient"""
        if self.variable_index is None:
            return self.coefficient
        else:
            return self.coefficient * variables[self.variable_index]


class OSILProduct(object):
    def __init__(self, factors, level):
        """initialize product object as list of factor objects or other nonlinear expressions, level in expr. tree, and
        bounds"""
        assert isinstance(factors, (list,)) and len(factors) > 0
        self.factors = factors
        # TODO: assertion for entity types
        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

        self.arg_lbs = []
        self.arg_ubs = []

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        current_lb, current_ub = 1, 1
        # iterate over factors, compute bounds of factor, adjust current lower and upper bound accordingly
        for factor in self.factors:
            lb, ub = factor.compute_bounds(variables)
            # store lb and ub
            self.arg_lbs.append(lb)
            self.arg_ubs.append(ub)
            # if a bound is None, compute with +- inf
            lb = -np.inf if lb is None else lb
            ub = np.inf if ub is None else ub
            if current_lb <= 0:
                temp_lb = current_lb * ub
            else:
                temp_lb = current_lb * lb

            if current_ub <= 0:
                temp_ub = current_ub * lb
            else:
                temp_ub = current_ub * ub

            current_lb = min(temp_lb, temp_ub)
            current_ub = max(temp_lb, temp_ub)

            # temp_lb = min(current_lb * lb, current_lb * ub)
            # temp_ub = max(current_ub * ub, current_ub * lb)
            # current_lb = min(temp_lb, temp_ub)
            # current_ub = max(temp_lb, temp_ub)
            #current_lb = min(current_lb * lb, current_lb * ub, current_ub * lb, current_ub * ub)
            #current_ub = max(current_lb * lb, current_lb * ub, current_ub * lb, current_ub * ub)

        if current_lb > -np.inf:
            self.lower_bound = current_lb
        if current_ub < np.inf:
            self.upper_bound = current_ub

        return self.lower_bound, self.upper_bound

    def add_factor(self, factor):
        """addition of product entity like factor object or other nonlinear expression"""
        # TODO: add other possible entities
        assert isinstance(factor, (OSILFactor, OSILProduct))
        self.factors.append(factor)

    def remove_factor(self, index):
        """removal of product entity"""
        assert isinstance(index, (int,))
        self.factors = self.factors[:index] + self.factors[(index + 1):]

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return product of factor entities in evaluated form """
        product = 1
        for entity in self.factors:
            product *= entity.eval(variables)
        return product


class OSILSquare(object):
    def __init__(self, expression, level, coefficient=1.0):
        """initialize square object (variable index, level in expression tree, variable coefficient, bounds)"""
        if isinstance(expression, (int,)):
            assert expression >= 0, f"Invalid variable index {expression}!"
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        assert isinstance(coefficient, (float,))
        self.coefficient = coefficient

        self.lower_bound = None
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        if isinstance(self.expression, (int,)):
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            current_lb = var.lb
            current_ub = var.ub
        else:
            current_lb, current_ub = self.expression.compute_bounds(variables)

        # set None as +- infinity for computations and multiply with coefficient
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, self.coefficient)

        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # if 0 is between lb and ub, set as lower bound
        if lb <= 0 <= ub:
            self.lower_bound = 0
        else:
            lb = min(lb ** 2, ub ** 2)
            self.lower_bound = None if np.isinf(-lb) else lb
        ub = max(lb ** 2, ub ** 2)
        self.upper_bound = None if np.isinf(ub) else ub

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return variable squared"""
        # (maybe) TODO: adjust to variables[self.variable_index]*variables[self.variable_index]
        if isinstance(self.expression, (int,)):
            return (self.coefficient * variables[self.expression])**2
        else:
            return (self.coefficient * self.expression.eval(variables))**2


class OSILPower(object):
    def __init__(self, expression, exponent, base_coefficient, exponent_coefficient, level):
        """initialize power object (expression, coefficient, level in expression tree, bounds)"""
        # Check that expression is variable index (int), number (float) or known nl expression
        if isinstance(expression, (int, float)):
            # expression is variable index or base number
            assert expression >= 0, f"Variable index or base number has to be greater or equal than 0"
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        # Check that exponent is variable index (int), number (float) or known nl expression
        if isinstance(exponent, (int,)):
            # expression is variable index
            assert exponent >= 0, f"Variable index has to be greater or equal than 0"
        elif isinstance(exponent, (float,)):
            assert exponent != 0, f"Exponent must be different from zero"
        else:
            assert isinstance(exponent, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine, OSILNegate,
                                         OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.exponent = exponent

        assert isinstance(base_coefficient, (int, float))
        self.base_coefficient = base_coefficient
        assert isinstance(exponent_coefficient, (int, float))
        self.exponent_coefficient = exponent_coefficient

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

        self.base_lb = None
        self.base_ub = None
        self.exp_lb = None
        self.exp_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the lower and upper bounds of base and exponent (times coefficient)
        bounds = []
        for argument, coefficient in zip([self.expression, self.exponent],
                                         [self.base_coefficient, self.exponent_coefficient]):
            if isinstance(argument, (int,)):
                assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
                var = variables[argument]
                curr_lb = var.lb
                curr_ub = var.ub
            elif isinstance(argument, (float,)):
                curr_lb = curr_ub = argument
            else:
                curr_lb, curr_ub = argument.compute_bounds(variables)

            # set None as +- infinity for computations and multiply with coefficient
            lb, ub = bounds_convert_and_multiply(curr_lb, curr_ub, coefficient)
            bounds += [lb, ub]

        lb_base, ub_base, lb_exponent, ub_exponent = bounds
        # store bounds
        self.base_lb = lb_base
        self.base_ub = ub_base
        self.exp_lb = lb_exponent
        self.exp_ub = ub_exponent

        # for constant and integral exponent, we apply a case distinction
        if lb_exponent == ub_exponent == int(ub_exponent):
            # if exponent is even, lower bound is either zero (if feasible) or regarding upper/lower bound
            if int(ub_exponent) % 2 == 0:
                if lb_base <= 0.0 <= ub_base:
                    self.lower_bound = 0
                else:
                    self.lower_bound = min(lb_base ** ub_exponent, ub_base ** ub_exponent)
                self.upper_bound = max(lb_base ** ub_exponent, ub_base ** ub_exponent)
            # if exponent is odd, function is monotonously increasing
            else:
                self.lower_bound = lb_base ** ub_exponent
                self.upper_bound = ub_base ** ub_exponent

        # if base can be negative, we apply no bounds, otherwise lb is minimum of smallest base < 1 and largest exponent
        # and largest base with smallest exponent; ub vice versa
        elif lb_base < 0 and lb_exponent != ub_exponent:
            print(f"possible negative base for power constraint detected")
            return self.lower_bound, self.upper_bound
        else:
            self.lower_bound = min(lb_base**ub_exponent, ub_base**lb_exponent)
            self.upper_bound = max(lb_base**lb_exponent, ub_base**ub_exponent)

        if np.isinf(-self.lower_bound):
            self.lower_bound = None
        if np.isinf(self.upper_bound):
            self.upper_bound = None

        return self.lower_bound, self.upper_bound

    # def update_exponent(self, exponent):
    #     """update exponent of power"""
    #     assert isinstance(exponent, (int, float))
    #     self.exponent = exponent

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return variable or expression to the power of exponent """
        # compute base
        if isinstance(self.expression, (int,)):
            base = variables[self.expression]
        elif isinstance(self.expression, (float,)):
            base = self.expression
        else:
            base = self.expression.eval(variables)
        base *= self.base_coefficient

        # compute exponent
        if isinstance(self.exponent, (int,)):
            exponent = variables[self.exponent]
        elif isinstance(self.exponent, (float,)):
            exponent = self.exponent
        else:
            exponent = self.exponent.eval(variables)
        exponent *= self.exponent_coefficient

        return base**exponent


class OSILCosine(object):
    def __init__(self, expression, level, coefficient=1.0):
        """initialize cosine object (expression, level in expression tree, bounds)"""
        if isinstance(expression, (int,)):
            # expression is supposed to be variable index
            assert expression >= 0
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(coefficient, (float,))
        self.coefficient = coefficient

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = -1
        self.upper_bound = 1

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        if isinstance(self.expression, (int,)):
            # extract lower and upper bound from variables
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            curr_lb = var.lb
            curr_ub = var.ub
        else:
            curr_lb, curr_ub = self.expression.compute_bounds(variables)
        lb, ub = bounds_convert_and_multiply(curr_lb, curr_ub, self.coefficient)
        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # TODO: check calculations below
        # if -pi + multiple of 2 * pi is between lb and ub, lb is -1, else lb is minimum of cos(lb) and cos(ub)
        # -pi + k * (2pi) in [lb, ub] <-> k * (2pi) in [lb + pi, ub + pi] <-> k in [(lb + pi)/(2pi), (ub + pi)/(2pi)]
        temp_lb = (lb + np.pi) / (2*np.pi)
        temp_ub = (ub + np.pi) / (2*np.pi)
        next_int = np.ceil(temp_lb)
        if next_int <= temp_ub:
            self.lower_bound = -1
        else:
            self.lower_bound = min(np.cos(lb), np.cos(ub))
        # if a multiple of 2 * pi is between lb and ub, ub is 1, else ub is maximum of cos(lb) and cos(ub)
        # k * (2pi) in [lb, ub] <-> k in [lb / (2pi), ub / (2pi)]
        temp_lb = lb / (2*np.pi)
        temp_ub = ub / (2*np.pi)
        next_int = np.ceil(temp_lb)
        if next_int <= temp_ub:
            self.upper_bound = 1
        else:
            self.upper_bound = max(np.cos(lb), np.cos(ub))

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return pyo cosine of the variable"""
        if isinstance(self.expression, (int,)):
            argument = variables[self.expression]
        else:
            argument = self.expression.eval(variables)
        argument *= self.coefficient

        return pyo.cos(argument)


class OSILSine(object):
    def __init__(self, expression, level, coefficient=1.0):
        """initialize sine object (expression, level in expression tree, bounds)"""
        if isinstance(expression, (int,)):
            assert expression >= 0
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(coefficient, (float,))
        self.coefficient = coefficient

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = -1
        self.upper_bound = 1

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        if isinstance(self.expression, (int,)):
            # extract lower and upper bound from variables
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            curr_lb = var.lb
            curr_ub = var.ub
        else:
            curr_lb, curr_ub = self.expression.compute_bounds(variables)
        lb, ub = bounds_convert_and_multiply(curr_lb, curr_ub, self.coefficient)

        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # TODO: check calculations below
        # if -1/2 pi + multiple of 2 * pi is between lb and ub, lb is -1, else lb is minimum of sin(lb) and sin(ub)
        # -pi/2 + k * (2pi) in [lb, ub] <-> k * (2pi) in [lb + pi/2, ub + pi/2] <->
        # k in [(lb + pi/2)/(2pi), (ub + pi/2)/(2pi)]
        temp_lb = (lb + np.pi/2) / (2 * np.pi)
        temp_ub = (ub + np.pi/2) / (2 * np.pi)
        next_int = np.ceil(temp_lb)
        if next_int <= temp_ub:
            self.lower_bound = -1
        else:
            self.lower_bound = min(np.sin(lb), np.sin(ub))
        # if -3/2 pi + multiple of 2 * pi is between lb and ub, ub is 1, else ub is maximum of sin(lb) and sin(ub)
        # -3pi/2 + k * (2pi) in [lb, ub] <-> k * (2pi) in [lb + 3pi/2, ub + 3pi/2] <->
        # k in [(lb + 3pi/2)/(2pi), (ub + 3pi/2)/(2pi)]
        temp_lb = (lb + 3 * np.pi / 2) / (2 * np.pi)
        temp_ub = (ub + 3 * np.pi / 2) / (2 * np.pi)
        next_int = np.ceil(temp_lb)
        if next_int <= temp_ub:
            self.upper_bound = 1
        else:
            self.upper_bound = max(np.sin(lb), np.sin(ub))

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return pyo sine of the variable"""
        if isinstance(self.expression, (int,)):
            argument = variables[self.expression]
        else:
            argument = self.expression.eval(variables)
        argument *= self.coefficient

        return pyo.sin(argument)


class OSILNegate(object):
    def __init__(self, expression, level):
        """initialize negate object (= multiply by -1) with (expression and level in expression tree, bounds)"""
        assert isinstance(expression, (int, OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                       OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression
        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        if isinstance(self.expression, (int,)):
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            current_lb = var.lb
            current_ub = var.ub
        else:
            current_lb, current_ub = self.expression.compute_bounds(variables)

        # set None as +- infinity for computations and multiply with coefficient -1 as negate would
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, -1.0)

        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        self.lower_bound = None if np.isinf(-lb) else lb
        self.upper_bound = None if np.isinf(ub) else ub

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """either return negative of variable or negative of eval of expression"""
        if isinstance(self.expression, (int,)):
            return -1 * variables[self.expression]
        else:
            return -1 * self.expression.eval(variables)


class OSILDivide(object):
    def __init__(self, numerator, denominator, level, numerator_is_constant=False, numerator_coeff=1.0,
                 denominator_coeff=1.0):
        """initialize divide object with (numerator, denominator, and level in expression tree, bounds)"""
        if numerator_is_constant:
            assert isinstance(numerator, (float,))
        else:
            assert isinstance(numerator, (int, OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                          OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.numerator = numerator
        self.numerator_constant = numerator_is_constant
        assert isinstance(numerator_coeff, (float,))
        self.numerator_coefficient = numerator_coeff

        assert isinstance(denominator, (int, OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                        OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.denominator = denominator
        assert isinstance(denominator_coeff, (float,))
        self.denominator_coefficient = denominator_coeff

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

        self.num_lb = None
        self.num_ub = None
        self.den_lb = None
        self.den_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the lower and upper bounds of base and exponent (times coefficient)
        bounds = []
        for argument, coefficient in zip([self.numerator, self.denominator],
                                         [self.numerator_coefficient, self.denominator_coefficient]):
            if isinstance(argument, (int,)):
                assert argument < len(variables), f"Variable index must occur in given list of OSILVariables"
                var = variables[argument]
                curr_lb = var.lb
                curr_ub = var.ub
            elif isinstance(argument, (float,)):
                curr_lb = curr_ub = argument
            else:
                curr_lb, curr_ub = argument.compute_bounds(variables)

            # set None as +- infinity for computations and multiply with coefficient
            lb, ub = bounds_convert_and_multiply(curr_lb, curr_ub, coefficient)
            bounds += [lb, ub]

        # if denominator contains zero in bounds, return None bounds
        lb_num, ub_num, lb_den, ub_den = bounds

        # store argument bounds
        self.num_lb = lb_num
        self.num_ub = ub_num
        self.den_lb = lb_den
        self.den_ub = ub_den

        if lb_den <= 0 <= ub_den:
            return self.lower_bound, self.upper_bound
        possible_bounds = [lb_num/lb_den, lb_num/ub_den, ub_num/lb_den, ub_num/ub_den]
        self.lower_bound = min(possible_bounds)
        self.upper_bound = max(possible_bounds)
        if np.isinf(-self.lower_bound):
            self.lower_bound = None
        if np.isinf(self.upper_bound):
            self.upper_bound = None

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return division of numerator/denominator for evals/variables"""
        if self.numerator_constant:
            numerator = self.numerator
        elif isinstance(self.numerator, (int,)):
            numerator = variables[self.numerator]
        else:
            numerator = self.numerator.eval(variables)
        numerator *= self.numerator_coefficient
        if isinstance(self.denominator, (int,)):
            denominator = variables[self.denominator]
        else:
            denominator = self.denominator.eval(variables)
        denominator *= self.denominator_coefficient
        return numerator/denominator


class OSILSquareroot(object):
    def __init__(self, expression, level):
        """initialize square root object (variable index, level in expression tree, bounds)"""
        if isinstance(expression, (int,)):
            assert expression >= 0, f"Invalid variable index {expression}!"
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = 0
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        if isinstance(self.expression, (int,)):
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            current_lb = var.lb
            current_ub = var.ub
        else:
            current_lb, current_ub = self.expression.compute_bounds(variables)

        # set None as +- infinity for computations
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, 1.0)

        self.arg_lb = lb
        self.arg_ub = ub

        if lb > 0:
            self.lower_bound = np.sqrt(lb)
        if not np.isinf(ub):
            self.upper_bound = np.sqrt(ub)

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return variable squared"""
        if isinstance(self.expression, (int,)):
            return pyo.sqrt(variables[self.expression])
        else:
            return pyo.sqrt(self.expression.eval(variables))


class OSILExp(object):
    def __init__(self, expression, level, coefficient=1.0):
        """initialize exp function object (variable index, level in expression tree, bounds)"""
        if isinstance(expression, (int,)):
            assert expression >= 0, f"Invalid variable index {expression}!"
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(coefficient, (float,))
        self.coefficient = coefficient

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = 0
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        if isinstance(self.expression, (int,)):
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            current_lb = var.lb
            current_ub = var.ub
        else:
            current_lb, current_ub = self.expression.compute_bounds(variables)

        # set None as +- infinity for computations and multiply with coefficient
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, self.coefficient)

        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # if neither lb nor ub is -inf, set lower bound to exp(min(lb, ub))
        if not np.isinf(-lb) and not np.isinf(-ub):
            self.lower_bound = np.exp(min(lb, ub))
        # if neither lb nor ub is inf, set upper bound to exp(max(lb, ub))
        if not np.isinf(lb) and not np.isinf(ub):
            self.upper_bound = np.exp(max(lb, ub))

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return exp function of variable or nl """
        if isinstance(self.expression, (int,)):
            return pyo.exp(self.coefficient * variables[self.expression])
        else:
            return pyo.exp(self.coefficient * self.expression.eval(variables))


class OSILAbs(object):
    def __init__(self, expression, level, coefficient=1.0):
        """initialize absolute value object (variable index, level in expression tree, coefficient of var, bounds)"""
        if isinstance(expression, (int,)):
            assert expression >= 0, f"Invalid variable index {expression}!"
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        assert isinstance(coefficient, (float,))
        self.coefficient = coefficient

        self.lower_bound = 0
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        if isinstance(self.expression, (int,)):
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            current_lb = var.lb
            current_ub = var.ub
        else:
            current_lb, current_ub = self.expression.compute_bounds(variables)

        # set None as +- infinity for computations and multiply with coefficient
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, self.coefficient)

        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # if 0 is not contained in the argument bounds, lb is computed as minimum of absolute values
        if not (lb <= 0 <= ub):
            self.lower_bound = min(np.abs(lb), np.abs(ub))
        self.upper_bound = max(np.abs(lb), np.abs(ub))
        if np.isinf(self.upper_bound):
            self.upper_bound = None

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return absolute value of variable or expression"""
        if isinstance(self.expression, (int,)):
            return abs(self.coefficient * variables[self.expression])
        else:
            return abs(self.expression.eval(variables))


class OSILLn(object):
    def __init__(self, expression, level, coefficient=1.0):
        """initialize ln function object (variable index, level in expression tree, bounds)"""
        if isinstance(expression, (int,)):
            assert expression >= 0, f"Invalid variable index {expression}!"
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        assert isinstance(coefficient, (float,))
        self.coefficient = coefficient

        self.lower_bound = None
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        if isinstance(self.expression, (int,)):
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            current_lb = var.lb
            current_ub = var.ub
        else:
            current_lb, current_ub = self.expression.compute_bounds(variables)

        # set None as +- infinity for computations and multiply with coefficient
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, self.coefficient)

        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # if lb > 0, compute lower bound as log(lb)
        if lb > 0:
            self.lower_bound = np.log(lb)
        if ub < np.inf:
            self.upper_bound = np.log(ub)

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return ln function of variable or expression"""
        if isinstance(self.expression, (int,)):
            return pyo.log(self.coefficient * variables[self.expression])
        else:
            return pyo.log(self.coefficient * self.expression.eval(variables))


class OSILLog10(object):
    def __init__(self, expression, level):
        """initialize log10 function object (variable index, level in expression tree)"""
        if isinstance(expression, (int,)):
            assert expression >= 0, f"Invalid variable index {expression}!"
        else:
            assert isinstance(expression, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine,
                                           OSILNegate, OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10))
        self.expression = expression

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        if isinstance(self.expression, (int,)):
            assert self.expression < len(variables), f"Variable index must occur in given list of OSILVariables"
            var = variables[self.expression]
            current_lb = var.lb
            current_ub = var.ub
        else:
            current_lb, current_ub = self.expression.compute_bounds(variables)

        # set None as +- infinity for computations and multiply with coefficient
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, 1.0)

        # store argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # if lb > 0, compute lower bound as log(lb)
        if lb > 0:
            self.lower_bound = np.log10(lb)
        if ub < np.inf:
            self.upper_bound = np.log10(ub)

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return log10 function of variable or expression"""
        if isinstance(self.expression, (int,)):
            return pyo.log10(variables[self.expression])
        else:
            return pyo.log10(self.expression.eval(variables))


class OSILSignPower(object):
    def __init__(self, base, exponent, level):
        """initialize power object (base, exponent, base coefficient, level in expression tree, bounds)"""
        # Check that expression is variable index (int) or number (float) TODO: or known nl expression
        assert isinstance(base, (int,))
        # expression is variable index or base number
        assert base >= 0, f"Variable index has to be greater or equal than 0"

        self.base = base

        # Check that exponent is number bigger than 1 TODO:variable index (int), number (float) or known nl expression
        assert isinstance(exponent, (float,))
        # exponent is number
        assert exponent > 1, f"Exponent must be greater than one"

        self.exponent = exponent

        assert isinstance(level, (int,)) and level >= 0
        self._level = level

        self.lower_bound = None
        self.upper_bound = None

        self.arg_lb = None
        self.arg_ub = None

    def compute_bounds(self, variables):
        """computing the bounds given a list of OSILVariables"""
        # compute the bounds of the argument
        assert self.base < len(variables), f"Variable index must occur in given list of OSILVariables"
        var = variables[self.base]
        current_lb = var.lb
        current_ub = var.ub

        # set None as +- infinity for computations and multiply with coefficient
        lb, ub = bounds_convert_and_multiply(current_lb, current_ub, 1.0)

        # store the argument bounds
        self.arg_lb = lb
        self.arg_ub = ub

        # if lb > -infinity, lower bound is lb * abs(lb)**(exp - 1)
        if lb > -np.inf:
            self.lower_bound = lb * np.abs(lb)**(self.exponent - 1)
        if ub < np.inf:
            self.upper_bound = ub * np.abs(ub)**(self.exponent - 1)

        return self.lower_bound, self.upper_bound

    def get_level(self):
        """return level in expression tree"""
        return self._level

    def eval(self, variables):
        """return variable * abs(variable)^(exponent - 1) in order to account for signpower """
        # compute expression and return
        return variables[self.base] * abs(variables[self.base])**(self.exponent - 1)


def bounds_convert_and_multiply(curr_lb, curr_ub, coefficient):
    """ recalculating None to +- infinity, multiplying with coefficient and adjust lb/ub accordingly """
    # set None as +- infinity for computations and multiply with coefficient
    curr_lb = -np.inf if curr_lb is None else curr_lb
    curr_ub = np.inf if curr_ub is None else curr_ub

    lb = min(coefficient * curr_lb, coefficient * curr_ub)
    ub = max(coefficient * curr_lb, coefficient * curr_ub)

    return lb, ub
