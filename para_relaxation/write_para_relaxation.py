from definitions import *
from osil_parser.osil_parser import OSILParser
from osil_parser.osil_1Dreformulation import reformulate_osil_parser_to_1d, single_reformulation
from osil_parser.osil_expressions import *
from osil_to_pyomo import OsilPyomoConverter
from utilities.pyomo_solver import PyomoSolver
from instance_extractor import attributes, get_instances_by_attribute

import numpy as np
import time
import copy
import json

# test set name
test_set = "minlplib"
# data format
data_suffix = "osil"

# solver choose from scip or gurobi
solver = 'scip'

# epsilon for approximation guarantee
eps = 1e-2

# choice of attributes
selected_attributes = {
    "name" : ['lnts50', 'lnts100', 'lnts200', 'lnts400'],
    #"adddate" : ["2000-01-30", None],
    #"convex" : False,
    "dualbound" : [-np.inf, np.inf], # allows for deleting dualbound is None
    "ncons" : [0, 10000],
    "nvars" : [0, 10000],
    #"probtype" : ["INLP"]
}

solver_output = True


def substitute_paraboloids(orig_parser, n_new_vars):
    re_parser = copy.deepcopy(orig_parser)

    n_para_below = 0
    n_para_above = 0

    nl_indices_to_remove = []
    for nl_index, nl in re_parser.nl_constraints.items():
        # substitute only sine/cosine functions
        if isinstance(nl, (OSILSine, OSILCosine)):
            # extract coefficient and variable index
            argument_coefficient = nl.coefficient
            argument_index = nl.expression
            assert isinstance(argument_index, (int,)), f"argument must be variable index in expression tree"
            argument_lb = re_parser.variables[argument_index].lb
            argument_ub = re_parser.variables[argument_index].ub
            # recompute the bounds due to the coefficient, i.e., (x'=) 5x for x in [-1, 1] -> x' in [-5, 5]
            argument_lb *= argument_coefficient
            argument_ub *= argument_coefficient

            # remove the nl from the parser and add variable with respective bounds instead
            # save current nl index for removal
            nl_indices_to_remove.append(nl_index)
            # if the linear coefficients for these constraints are 1 and no quadratic coefficients, save variable index
            if len(re_parser.lin_coeffs[nl_index]) == 1 and len(re_parser.quad_coeffs[nl_index]) == 0:
                var_index = re_parser.lin_coeffs[nl_index][0][0]
                assert re_parser.lin_coeffs[nl_index][0][1] == -1.0
                # delete the equality constraint, as we take the variable as is
                re_parser.lin_coeffs[nl_index] = []
                assert re_parser.variables[var_index].lb == nl.lower_bound
                assert re_parser.variables[var_index].ub == nl.upper_bound
            # else create a new variable and replace the occurrence of (co)sine with it
            else:
                n_new_vars += 1
                new_variable_name = f"aux{n_new_vars}"
                new_variable = OSILVariable(new_variable_name, nl.lower_bound, nl.upper_bound)
                var_index = len(re_parser.variables)
                parser.variables.append(new_variable)
                new_lin_coef = (var_index, 1.0)
                re_parser.lin_coeffs[nl_index].append(new_lin_coef)

            # depending on the type of (in)equality, add approximations from above and/or below
            # check for the type of (in)equality
            constraint_infos = re_parser.constraint_infos[nl_index]
            # get the approximation infos
            json_file = open(ROOT_DIR + "/para_computation/para_parameters.json", "r")
            para_params = json.load(json_file)
            function_string = "sin" if isinstance(nl, (OSILSine,)) else "cos"

            # iterate over lower and upper bound (indexed 1 and 2 in constraint infos)
            for bound_index in [1, 2]:
                # add paraboloid approximations if the constraint bound is not None
                if constraint_infos[bound_index] is not None:
                    # extract suitable paraboloid parameters
                    approx_string = "below" if bound_index == 1 else "above"
                    quads, lins, cons = para_params[str(eps)][function_string][str(np.round(nl.arg_lb, 6))][str(np.round(nl.arg_ub, 6))][approx_string]
                    # add quadratic and linear coefficients for the arguments variable index
                    # the constant parameter is added as rhs/lhs, -constant >= paraboloid - constant - y respective
                    # -constant <= paraboloid - constant - y for upper bounds
                    for quad, lin, con in zip(quads, lins, cons):
                        # number of current constraints gives new constraint index
                        n_constraints = len(re_parser.constraint_infos)
                        # depending on constraint type, approximate from below or above
                        if bound_index == 1:
                            constraint_info = (f"para_below{n_para_below}", None, -con)
                            n_para_below += 1
                        else:
                            constraint_info = (f"para_above{n_para_above}", -con, None)
                            n_para_above += 1

                        # for the argument, say x, and the auxiliary variable, say y, include the linear coefficient
                        # and -1, respectively, as linear coefficients
                        lin_coefs = [(argument_index, lin), (var_index, -1.0)]
                        # only the argument needs its quadratic coefficient
                        quad_coefs = [(argument_index, argument_index, quad)]
                        # add the linear and quadratic coefficients to the parser as well as the constraint infos,
                        # creating a new constraint
                        re_parser.lin_coeffs[n_constraints] = lin_coefs
                        re_parser.quad_coeffs[n_constraints] = quad_coefs
                        re_parser.constraint_infos.append(constraint_info)

    for nl_index in nl_indices_to_remove:
        del re_parser.nl_constraints[nl_index]

    return re_parser


if __name__ == "__main__":
    # extract list of instances as defined by the chosen attributes
    instances, primal_bounds, dual_bounds = get_instances_by_attribute(test_set, selected_attributes)

    primal_values = []
    deviating_instances = []

    # for each instance, parse and reformulate_1d. For both then convert to pyomo, solve, and check for identical values
    # for the primal bound
    for index, inst in enumerate(instances):
        # set up parser
        instance_path = os.path.join(ROOT_DIR, test_set, data_suffix, inst + f".{data_suffix}")
        parser = OSILParser(instance_path)
        # parse
        parser.parse()

        # convert parser to parser with 1-d reformulation
        n_new_variables, parser_1d = reformulate_osil_parser_to_1d(parser)

        # exchange exp functions with parabolas
        parser_para = substitute_paraboloids(parser_1d, n_new_variables)

        # initialize osil to pyomo
        converter_1d = OsilPyomoConverter(parser_1d)
        # set up model
        model_1d = converter_1d.setup_model()

        # same for paraboloid parser
        converter_para = OsilPyomoConverter(parser_para)
        model_para = converter_para.setup_model()
        # if model should be written
        model_para.write(f"{inst}_para.gms")



