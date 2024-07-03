import copy

from osil_parser.osil_var import OSILVariable
from osil_parser.osil_obj import OSILObjective
from osil_parser.osil_expressions import *
from osil_parser.osil_parser import *


def reformulate_osil_parser_to_1d(orig_parser):
    """
    This method is dedicated to take an osil parser object and reformulate all its nonlinear constraints
    into linear constraints or one-dimensional reformulations

    :param orig_parser: an osil parser which has parsed an osil file
    :return: the same osil parser with adjusted constraints and variables
    """
    # check for suitable input
    assert check_reformulate_input(orig_parser)
    parser = copy.deepcopy(orig_parser)

    # initialize necessary parameters
    n_new_variables = 0
    # extract the initial indices of the non-linear functions
    nl_indices = list(parser.nl_constraints.keys())

    # iterate over nonlinear constraints, reformulate nonlinear constraints as long as there are some
    for index in nl_indices:
        # extract current non-linearity
        nl = parser.nl_constraints[index]

        # sanity check for already handled objects
        assert isinstance(nl, (OSILSum, OSILProduct, OSILSquare, OSILPower, OSILCosine, OSILSine, OSILNegate,
                               OSILDivide, OSILSquareroot, OSILExp, OSILAbs, OSILLn, OSILLog10, OSILSignPower)), \
            f"non-linearity was not implemented yet"

        if isinstance(nl, (OSILSquare, OSILCosine, OSILSine, OSILNegate, OSILSquareroot, OSILExp, OSILAbs, OSILLn,
                           OSILLog10)):
            # check if its argument expression is a variable or a coefficient; TODO: make it dependent on type of nl
            if isinstance(nl.expression, (float, int, OSILVariable)):
                # if argument is coefficient or variable -> nothing needs to be re-formulated
                continue
            else:
                new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                       nl.expression, nl_indices)
                # manipulate the current non-linearity such that it is non-linearity(new variable)
                nl.expression = new_variable_index

        elif isinstance(nl, (OSILSum,)):
            # iterate through every summand and apply above
            for entity_index, entity in enumerate(nl.sum_entities):
                # if entity is Summand, it is variable and/or coefficient, so nothing to do
                if isinstance(entity, (OSILSummand,)):
                    continue
                else:
                    new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                           entity, nl_indices)
                    new_summand = OSILSummand(new_variable_index, 1.0, 1)
                    new_summand.compute_bounds(parser.variables)
                    nl.sum_entities[entity_index] = new_summand

        elif isinstance(nl, (OSILProduct,)):
            # first substitute all general non-linear expressions with variables
            variable_factor_indices = []
            for factor_index, factor in enumerate(nl.factors):
                # if factor is OSILFactor, it is variable and/or coefficient, so just count variable here
                if isinstance(factor, (OSILFactor,)):
                    if factor.variable_index is not None:
                        variable_factor_indices.append(factor_index)
                else:
                    # create a new variable for the expression and add it to parser as well as to product as OSILFactor
                    new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                           factor, nl_indices)
                    new_factor = OSILFactor(new_variable_index, 1.0, 1)
                    new_factor.compute_bounds(parser.variables)
                    nl.factors[factor_index] = new_factor
                    variable_factor_indices.append(factor_index)

            # second create a new variable for each pair such that we only end up with z = x * y constraints
            while len(variable_factor_indices) > 2:
                # take the last two indices and extract factor elements
                factor_1_index = variable_factor_indices.pop()#[-1]
                factor_2_index = variable_factor_indices.pop()#[-2]
                factor_1 = nl.factors[factor_1_index]
                factor_2 = nl.factors[factor_2_index]
                # create a new product element
                sub_product = OSILProduct([factor_1, factor_2], 2)
                sub_product.compute_bounds(parser.variables)
                # create the new constraint
                new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                       sub_product, nl_indices)
                # create a factor object for the newly created variable
                new_factor = OSILFactor(new_variable_index, 1.0, 1)
                new_factor.compute_bounds(parser.variables)
                # delete the old factors
                del nl.factors[factor_1_index]
                del nl.factors[factor_2_index]
                # add new factor and respective index
                new_factor_index = len(nl.factors)
                nl.factors.append(new_factor)
                variable_factor_indices.append(new_factor_index)

        elif isinstance(nl, (OSILPower,)):
            # check if base is a variable or a coefficient
            if not isinstance(nl.expression, (float, int)):
                new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                       nl.expression, nl_indices)
                # manipulate the current non-linearity such that it is non-linearity(new variable); TODO:check!!
                nl.expression = new_variable_index

            # check if the exponent is a variable or a coefficient
            if not isinstance(nl.exponent, (float, int)):
                new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                       nl.exponent, nl_indices)
                # manipulate the current non-linearity such that it is non-linearity(new variable); TODO: check!!
                nl.exponent = new_variable_index

        elif isinstance(nl, (OSILDivide,)):
            # replace numerator with variable if not a variable or a coefficient
            if not isinstance(nl.numerator, (float, int)):
                new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                       nl.numerator, nl_indices)
                # manipulate the current non-linearity such that it is non-linearity(new variable)
                nl.numerator = new_variable_index

            # replace denominator with variable if not a variable or a coefficient
            if not isinstance(nl.denominator, (float, int)):
                new_variable_index, n_new_variables, nl_indices = single_reformulation(parser, n_new_variables,
                                                                                       nl.denominator, nl_indices)
                # manipulate the current non-linearity such that it is non-linearity(new variable)
                nl.denominator = new_variable_index

            # if the denominator is not just a coefficient, replace the fraction x/y by a variable z and re-arrange to
            # a new non-linearity z * y == x
            n_new_variables = reformulate_fraction(parser, n_new_variables, index, nl)

        elif isinstance(nl, (OSILSignPower,)):
            # signpower is only implemented for variables and coefficients
            continue
        else:
            assert f"UNKNOWN OSIL EXPRESSION"

    return n_new_variables, parser


def check_reformulate_input(parser):
    """
    Checking if the argument is indeed an osil parser object and certain parameters are given

    :param parser: osil parser which has parsed an osil file
    :return: True if valid, else False
    """
    assert isinstance(parser, OSILParser), f"Given parser is no OSILParser; leaving."

    assert parser._parsed, f"Parser must have been parsed a file before; leaving."

    return True


def single_reformulation(parser, n_new_variables, argument, nl_indices):
    """
    creates a variable, say y, and adds it to the parser alongside a constraint saying y == argument, where 'argument'
    is the argument of the current non-linearity

    :param parser: osil parser element to operate on
    :param n_new_variables: current number of newly added variables
    :param argument: the argument of the non-linearity to re-model
    :param nl_indices: current indices of non-linear constraints
    :return: the index of the newly added variable, and the total amount of new variables
    """
    # create a new variable and add it to the parser
    n_new_variables += 1
    new_variable_name = f"aux{n_new_variables}"
    new_variable = OSILVariable(new_variable_name, argument.lower_bound, argument.upper_bound)
    new_variable_index = len(parser.variables)
    parser.variables.append(new_variable)

    # add the new constraint of new_variable == expression
    n_constraints = len(parser.constraint_infos)
    new_constraint_name = f"e{n_constraints + 1}"
    # constraint info consists of name, lower bound, upper bound
    new_constraint_info = (new_constraint_name, 0.0, 0.0)
    parser.constraint_infos.append(new_constraint_info)
    # add new variable with coefficient -1, such that -new_variable + expression == 0
    new_lin_coef = (new_variable_index, -1.0)
    parser.lin_coeffs[n_constraints] = [new_lin_coef]
    parser.quad_coeffs[n_constraints] = []
    # add the remaining expression of the current nl as a new nl
    parser.nl_constraints[n_constraints] = argument
    nl_indices.append(n_constraints)

    return new_variable_index, n_new_variables, nl_indices


def reformulate_fraction(parser, n_new_variables, nl_index, divide_nl):
    """
        creates a variable, say z, and adds it to the parser, where the divide non-linearity x/y has been.
        Additionally, a constraint z * y == x is added.

        :param parser: osil parser element to operate on -> OSILparser
        :param n_new_variables: current number of newly added variables -> integer
        :param nl_index: the index of the divide non-linearity -> integer
        :param divide_nl: the OSILdivide expression to re-model
        :return: the total amount of new variables
        """
    # create a new variable, say z, and add it to the parser
    n_new_variables += 1
    new_variable_name = f"aux{n_new_variables}"
    new_variable = OSILVariable(new_variable_name, divide_nl.lower_bound, divide_nl.upper_bound)
    new_variable_index = len(parser.variables)
    parser.variables.append(new_variable)

    # add the newly created variable linearly in the current constraint
    replacement_lin_coef = (new_variable_index, 1.0)
    parser.lin_coeffs[nl_index].append(replacement_lin_coef)
    # delete the non-linearity from the current constraint index
    del parser.nl_constraints[nl_index]

    # add the new constraint of [new_variable z] * [denominator y]  == [numerator x]]
    # add respective constraint infos
    n_constraints = len(parser.constraint_infos)
    new_constraint_name = f"e{n_constraints + 1}"
    # constraint info consists of name, lower bound, upper bound
    if divide_nl.numerator_constant:
        bound = divide_nl.numerator
        assert divide_nl.numerator_coefficient == 1.0, f"If numerator is constant, coefficient must equal 1.0"
    else:
        bound = 0
    new_constraint_info = (new_constraint_name, bound, bound)
    parser.constraint_infos.append(new_constraint_info)

    # if numerator not constant,
    # add the numerator with coefficient -1 * old coefficient, such that -[numerator] + [z * denominator] == 0
    if divide_nl.numerator_constant:
        parser.lin_coeffs[n_constraints] = []
    else:
        new_lin_coef = (divide_nl.numerator, -1.0 * divide_nl.numerator_coefficient) #TODO: add assertions
        parser.lin_coeffs[n_constraints] = [new_lin_coef]

    # add the [new variable z] * [denominator] as quadratic coefficients
    new_quad_coef = (new_variable_index, divide_nl.denominator, divide_nl.denominator_coefficient) #TODO: add assertions
    parser.quad_coeffs[n_constraints] = [new_quad_coef]

    return n_new_variables
