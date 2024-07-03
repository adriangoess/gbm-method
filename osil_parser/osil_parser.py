import xml.etree.ElementTree as ET

from osil_parser.osil_var import OSILVariable
from osil_parser.osil_obj import OSILObjective
from osil_parser.osil_expressions import *


class OSILParser(object):
    """
    Parse a string to an osil instance and store variables, objectives, constraints.
    The latter is stored as an expression tree which is the canonical form originating from osil format.
    All elements are stored as objects of separate classes.

    Keyword arguments:
    osil_file_path -- path to the osil file to parse (string)
    """

    def __init__(self, osil_file_path):
        self.prefix = "{os.optimizationservices.org}"

        tree = ET.parse(osil_file_path)
        self.root = tree.getroot()
        assert self._strip(self.root.tag) == "osil"

        self.name = ""
        self.objective = []
        self.variables = []
        self.constraint_infos = []
        self.lin_coeffs = {}
        self.quad_coeffs = {}
        self.nl_constraints = {}
        self._parsed = False

        # initialize counting variables for statistical reasons
        self.n_cos = 0
        self.n_sin = 0
        self.n_sqrt = 0
        self.n_exp = 0
        self.n_logln = 0

    def parse(self):
        """main method for parsing the initialized string"""
        # TODO: Replace for-loop by self.root.find(..) if it is clear that other objects do not occur
        for child in self.root:
            stripped_tag = self._strip(child.tag)
            if stripped_tag == "instanceHeader":
                self._parse_header(child)
            elif stripped_tag == "instanceData":
                self._parse_data(child)
            else:
                print(f"Error encountered; Tag {stripped_tag} on level 1 unknown")
                exit()
        self._parsed = True

    def _parse_header(self, instance_header):
        """parse the header node for reading the instance name"""
        # TODO: Replace for-loop by instance_header.find(..) if it is clear that other objects do not occur
        for child in instance_header:
            stripped_tag = self._strip(child.tag)
            if stripped_tag == "name":
                self.name = child.text
            else:
                print(f"Warning; Instance Header Child Tag {stripped_tag} unknown")
        return 0

    def _parse_data(self, instance_data):
        """parse variables, objectives, constraints and save"""
        # parse variables first
        variable_node = instance_data.find(self.prefix + "variables")
        assert variable_node is not None
        self._parse_variables(variable_node)

        # parse the objective function
        objective_node = instance_data.find(self.prefix + "objectives")
        if objective_node is None:
            print(f"Warning; no objective found in instance {self.name}")
        else:
            self._parse_objective(objective_node)

        # parse the constraints names and bounds
        constraints_node = instance_data.find(self.prefix + "constraints")
        if constraints_node is not None:
            self._parse_constraints_node(constraints_node)

        # initialize the linear, quadratic and nl terms dictionary with empty lists
        self._init_term_dicts()

        # parse the linear constraints
        linear_coefficients_node = instance_data.find(self.prefix + "linearConstraintCoefficients")
        if linear_coefficients_node is not None:
            self._parse_linear_coefficients(linear_coefficients_node)

        # parse the quadratic constraints
        quadratic_coefficients_node = instance_data.find(self.prefix + "quadraticCoefficients")
        if quadratic_coefficients_node is not None:
            self._parse_quadratic_coefficients(quadratic_coefficients_node)

        # parse the nonlinear expression constraints
        nonlinear_expressions_node = instance_data.find(self.prefix + "nonlinearExpressions")
        if nonlinear_expressions_node is not None:
            self._parse_nonlinear_expressions(nonlinear_expressions_node)

        # check for unchecked tags
        for child in instance_data:
            stripped_tag = self._strip(child.tag)
            if stripped_tag in ["variables", "objectives", "nonlinearExpressions", "quadraticCoefficients",
                                "linearConstraintCoefficients", "constraints"]:
                continue
            else:
                print(f"Warning; Instance Data Child Tag {stripped_tag} unknown")
        return 0

    def _parse_variables(self, node):
        """Parse and save the variables"""
        n_vars = int(node.attrib["numberOfVariables"])
        for var in node:
            # assert when new attributes appear!
            assert set(var.attrib.keys()) - {"name", "lb", "ub", "type"} == set()
            name = var.attrib["name"]
            lb = var.attrib.get("lb")
            # assign default lower bound of 0
            lb = float(lb) if lb is not None else 0
            ub = var.attrib.get("ub")
            ub = float(ub) if ub is not None else ub
            # extract type and store
            var_type = var.attrib.get("type")
            assert var_type in [None, "I", "B", "C"]
            self.variables.append(OSILVariable(name, lb, ub, var_type))
        assert len(self.variables) == n_vars
        return 0

    def _parse_objective(self, node):
        """Parse and save the objective"""
        # TODO: check for several objectives
        for obj in node:
            # save meta data for objective
            assert self._strip(obj.tag) == "obj"
            name = obj.attrib["name"]
            direction = obj.attrib["maxOrMin"]
            n_coeffs = int(obj.attrib["numberOfObjCoef"])
            # create a dictionary variable index <-> linear coefficient in objective
            coef_dict = {}
            for coef_node in obj:
                var_index = int(coef_node.attrib["idx"])
                # check variable index is not yet in dict
                assert var_index not in coef_dict.keys()
                coef_dict[var_index] = float(coef_node.text)
            assert len(coef_dict.keys()) == n_coeffs

            constant = obj.attrib.get("constant")
            constant = 0 if constant is None else float(constant)
            self.objective.append(OSILObjective(name, direction, n_coeffs, coef_dict, constant))
        assert len(self.objective) == 1, f"Unknown handling of several objectives"
        return 0

    def _parse_constraints_node(self, node):
        """ construct a list of (constraint name, lb, ub) for each entry in constraints tag """
        # store number of constraints for assertion
        n_constraints = int(node.attrib["numberOfConstraints"])

        # init constraint list
        constraints = list()

        # iterate over children and extract name, lb, ub
        for child in node:
            name = child.attrib["name"]
            lb = child.attrib.get("lb")
            lb = lb if lb is None else float(lb)
            ub = child.attrib.get("ub")
            ub = ub if ub is None else float(ub)
            constraints.append((name, lb, ub))

            # ensure no keys in attributes are untracked
            assert set(child.attrib.keys()) - {"name", "lb", "ub"} == set()

        # assert if necessary
        assert n_constraints == len(constraints), f"error when parsing constraint meta info"

        self.constraint_infos = constraints
        return 0

    def _init_term_dicts(self):
        """ for each constraint index, initialize a linear, quadratic and nl empty list """
        self.quad_coeffs[-1] = []
        for constraint_index in range(len(self.constraint_infos)):
            self.lin_coeffs[constraint_index] = []
            self.quad_coeffs[constraint_index] = []

    def _parse_linear_coefficients(self, node):
        """ construct a dictionary with constraint indices as keys and list of tuples of (var index, coefficient)"""
        # store the number of nonzeros for assertion
        n_lin_terms = int(node.attrib["numberOfValues"])
        assert self._strip(node[0].tag) == "start"
        assert self._strip(node[2].tag) == "value"
        # parse start and value child via default method
        start_indices = self._parse_el_elements(node[0])
        values = self._parse_el_elements(node[2])

        # init counter for linear expressions
        count_lin_expr = 0
        if node[1].tag == self.prefix + 'colIdx':
            # parse column indices
            column_indices = self._parse_el_elements(node[1])
            assert len(column_indices) == len(values), f"number of indices must equal number of values in linear coefs"
            # start values (from above) give start and end index in the column indices list for the current row
            for row_index, (curr_row_from, curr_row_to) in enumerate(zip(start_indices[:-1], start_indices[1:])):
                for index in range(curr_row_from, curr_row_to):
                    self.lin_coeffs[row_index].append((column_indices[index], values[index]))
                    count_lin_expr += 1
        else:
            assert node[1].tag == self.prefix + "rowIdx", f"neither column nor row indices found in linear constraints"
            # parse row indices
            row_indices = self._parse_el_elements(node[1])
            assert len(row_indices) == len(values), f"number of indices must equal number of values in linear coefs"
            # start values (from above) give start and end index in the row indices list for the current column
            for column_index, (curr_col_from, curr_col_to) in enumerate(zip(start_indices[:-1], start_indices[1:])):
                for index in range(curr_col_from, curr_col_from):
                    row_index = row_indices[index]
                    self.lin_coeffs[row_index].append((column_index, values[index]))
                    count_lin_expr += 1

        assert n_lin_terms == count_lin_expr, f"Error in extracting linear expressions"
        return 0

    def _parse_el_elements(self, node):
        """parsing all el child notes and computing (if necessary) the respective values"""
        assert self._strip(node.tag) in ["start", "value", "rowIdx", "colIdx"]
        values = []
        for el in node:
            assert self._strip(el.tag) == "el"
            assert set(el.attrib.keys()) - {"mult", "incr"} == set()
            mult = el.attrib.get("mult")
            mult = 1 if mult is None else int(mult)
            incr = el.attrib.get("incr")
            incr = 0 if incr is None else int(incr)
            start_value = float(el.text) if self._strip(node.tag) == "value" else int(el.text)
            for m in range(mult):
                values.append(start_value + incr * m)
        return values

    def _parse_quadratic_coefficients(self, node):
        """construct a dictionary with constraint indices as keys and list of tuples of
        (var index, var index, coefficient)"""
        # store the number of quadratic terms for assertion
        n_quad_terms = int(node.attrib["numberOfQuadraticTerms"])
        # iterate over quadratic terms and count
        i = 0
        for qterm_node in node:
            # extract necessary info
            constraint_index = int(qterm_node.attrib["idx"])
            variable_index1 = int(qterm_node.attrib["idxOne"])
            variable_index2 = int(qterm_node.attrib["idxTwo"])
            # check if there is always a coefficient given
            assert "coef" in qterm_node.attrib.keys()
            coefficient = float(qterm_node.attrib["coef"])

            assert qterm_node.attrib.keys() - {"idx", "idxOne", "idxTwo", "coef"} == set(), \
                f"more than the expected keys in quad term parsing"
            # append to list for current constraint index
            self.quad_coeffs[constraint_index].append((variable_index1, variable_index2, coefficient))

            i += 1

        assert n_quad_terms == i

        return 0

    def _parse_nonlinear_expressions(self, node):
        """Parse the nonlinear expression node and all its components"""
        # store number of nonlinear constraints for assertion
        n_nonlinear_expr = int(node.attrib["numberOfNonlinearExpressions"])

        count_nl_expr = 0
        for nl in node:
            # parse single nonlinear constraint at once and recursively
            idx = int(nl.attrib["idx"])
            # assert if the highest nl contains more than one nonlinear expression
            assert len(nl) == 1
            self.nl_constraints[idx] = self._parse_single_nl(nl[0], self._strip(nl[0].tag), 0)
            self.nl_constraints[idx].compute_bounds(self.variables)
            count_nl_expr += 1

        assert n_nonlinear_expr == count_nl_expr, f"More/less nonlinear constraints than expected"
        return 0

    def _parse_single_nl(self, nl, kind, level):
        """Parse the current nonlinear expression on its level, return a respective object and its bounds"""
        # assertion for current implementation
        assert kind in ["sum", "product", "square", "power", "cos", "sin", "negate", "divide", "sqrt", "exp", "abs",
                        "ln", "log10", "signpower"], \
            f"Invalid kind of nonlinear constraint: {kind}"

        # assertion if general non-linearity has coefficient
        assert nl.attrib.get("coef") is None, f"coefficient for non-linearity is not implemented yet"

        # check for kind of nonlinear constraint
        if kind == "sum":
            # save the entities 'variable', 'constant/number', 'nonlinear constraint' in sum object
            sum_entities = []
            for expr in nl:
                stripped_tag = self._strip(expr.tag)
                if stripped_tag == "variable":
                    # variable entities are constructed by name (=idx) and coefficient
                    idx = int(expr.attrib["idx"])
                    coefficient = expr.attrib.get("coef")
                    coefficient = 1.0 if coefficient is None else float(coefficient)
                    sum_entities.append(OSILSummand(idx, coefficient, level + 1))
                    # assert if more attributes are included
                    assert set(expr.attrib.keys()) - {"idx", "coef"} == set(), "Unknown attribute when handling sums"
                elif stripped_tag == "number":
                    assert len(expr.attrib.keys()) == 1, f"More/less than 1 argument in number"
                    # number/constant entities are constructed by their value
                    value = float(expr.attrib["value"])
                    sum_entities.append(OSILSummand(None, value, level + 1))
                else:
                    # other entities are nonlinear constraints to be solved as such
                    # TODO: check for linear/quadratic constraints
                    sum_entities.append(self._parse_single_nl(expr, stripped_tag, level + 1))
            return OSILSum(sum_entities, level)
        elif kind == "product":
            # save the entities 'variable', 'constant/number', 'nonlinear constraint' in product object
            factors = []
            for expr in nl:
                stripped_tag = self._strip(expr.tag)
                if stripped_tag == "variable":
                    # variable entities can be constructed by name only, when coefficients in product are saved extra
                    idx = int(expr.attrib["idx"])
                    coefficient = expr.attrib.get("coef")
                    coefficient = float(coefficient) if coefficient is not None else 1
                    factors.append(OSILFactor(idx, coefficient, level + 1))
                    # assert if more attributes are included
                    assert set(expr.attrib.keys()) - {"idx", "coef"} == set(), "Unknown attribute when handling product"
                elif stripped_tag == "number":
                    # number/constant entities are constructed by their value (which is supposed to be unique)
                    value = float(expr.attrib["value"])
                    assert len(expr.attrib.keys()) == 1
                    factors.append(OSILFactor(None, value, level + 1))
                else:
                    # other entities are nonlinear constraints to be solved as such
                    # TODO: check for linear/quadratic constraints
                    factors.append(self._parse_single_nl(expr, stripped_tag, level + 1))
            return OSILProduct(factors, level)
        elif kind == "square":
            # return a square object with the variable tag or a general nl
            assert len(nl) == 1, f"More/less objects than allowed in square creation: {len(nl)} != 1"
            node = nl[0]
            # check kind of node
            stripped_tag = self._strip(node.tag)
            assert stripped_tag != "number", f"Number tag not allowed in square nl"

            if stripped_tag == "variable":
                coefficient = node.attrib.get("coef")
                coefficient = 1.0 if coefficient is None else float(coefficient)
                # assert if more attributes are included
                assert set(node.attrib.keys()) - {"idx", "coef"} == set(), "Unknown attribute when handling in sqrt"
                return OSILSquare(int(node.attrib["idx"]), level, coefficient)
            else:
                return OSILSquare(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "power":
            # return a power object with the variable tag and the coefficient or assert otherwise
            assert len(nl) == 2, f"More/less objects than allowed in power creation: {len(nl)}"

            # extract relevant information for base (nl[0]) and power (nl[1])
            coefficients = [base_coefficient, power_coefficient] = [1.0, 1.0]
            elements = [base, power] = [None, None]
            for i, expr in enumerate(nl):
                stripped_tag = self._strip(expr.tag)
                if stripped_tag == "variable":
                    # handle variable coefficients
                    coefficient = expr.get("coef")
                    coefficients[i] = 1.0 if coefficient is None else float(coefficient)

                    elements[i] = int(expr.attrib["idx"])
                    # assert if more attributes are included
                    assert set(expr.attrib.keys()) - {"idx", "coef"} == set(), \
                        "Unknown attribute when handling variable in power"
                elif stripped_tag == "number":
                    elements[i] = float(expr.attrib["value"])
                    # assert if more attributes are included
                    assert len(expr.attrib.keys()) == 1, "More attributes than needed in power creation"
                else:
                    elements[i] = self._parse_single_nl(expr, stripped_tag, level + 1)
            return OSILPower(elements[0], elements[1], coefficients[0], coefficients[1], level)

        elif kind == "cos":
            # return a cosine object with variable tag or a general nl
            assert len(nl) == 1, f'More/less objects than allowed in cosine creation: {len(nl)} != 1'
            assert self._strip(nl[0].tag) != "number", f"so far, cosine does not support number as argument"
            node = nl[0]
            stripped_tag = self._strip(node.tag)

            self.n_cos += 1
            if stripped_tag == "variable":
                coefficient = nl[0].get("coef")
                coefficient = 1.0 if coefficient is None else float(coefficient)
                # assert if unhandled attribute is available
                assert set(nl[0].attrib.keys()) - {"idx", "coef"} == set(), f"unknown attribute in argument of cos"

                return OSILCosine(int(node.attrib["idx"]), level, coefficient)
            else:
                return OSILCosine(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "sin":
            # return sine object with variable tag or a general nl
            assert len(nl) == 1, f'More/less objects than allowed in sine creation: {len(nl)} != 1'
            assert self._strip(nl[0].tag) != "number", f"so far, sine does not support number as argument"
            node = nl[0]
            stripped_tag = self._strip(node.tag)

            self.n_sin += 1
            if stripped_tag == "variable":
                coefficient = nl[0].get("coef")
                coefficient = 1.0 if coefficient is None else float(coefficient)
                # assert if unhandled attribute is available
                assert set(nl[0].attrib.keys()) - {"idx", "coef"} == set(), f"unknown attribute in argument of sin"

                return OSILSine(int(node.attrib["idx"]), level, coefficient)
            else:
                return OSILSine(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "negate":
            # return negated object with general nonlinear expression or variable
            assert len(nl) == 1, f'More/less objects than allowed in negate creation: {len(nl)} != 1'
            stripped_tag = self._strip(nl[0].tag)
            if stripped_tag == "variable":
                idx = int(nl[0].attrib("idx"))
                # assert if more attributes are included
                assert set(nl[0].attrib.keys()) - {"idx"} == set(), "Unknown attribute when handling negation"
                return OSILNegate(idx, level)
            else:
                # other entities are nonlinear constraints to be solved as such
                # TODO: check for linear/quadratic constraints
                return OSILNegate(self._parse_single_nl(nl[0], stripped_tag, level + 1), level)

        elif kind == "divide":
            # return divide (or fraction) object with general nonlinear expressions and/or variables
            assert len(nl) == 2, f'More/less objects than allowed in divide creation: {len(nl)} != 2'
            numerator_strip_tag = self._strip(nl[0].tag)
            denominator_strip_tag = self._strip(nl[1].tag)

            numerator_is_constant = numerator_strip_tag == "number"
            numerator_coefficient = 1.0
            denominator_coefficient = 1.0
            if numerator_strip_tag == "number":
                assert len(nl[0].attrib.keys()) == 1
                numerator = float(nl[0].attrib["value"])
            elif numerator_strip_tag == "variable":
                # only save the variable index as numerator if variable
                numerator = int(nl[0].attrib["idx"])
                numerator_coefficient = nl[0].get("coef")
                numerator_coefficient = 1.0 if numerator_coefficient is None else float(numerator_coefficient)
                # assert if unhandled attribute is available
                assert set(nl[0].attrib.keys()) - {"idx", "coef"} == set(), f"unknown attribute in numerator of divide"
            else:
                numerator = self._parse_single_nl(nl[0], numerator_strip_tag, level + 1)
            if denominator_strip_tag == "variable":
                # only save the variable index as numerator if variable
                denominator = int(nl[1].attrib["idx"])
                denominator_coefficient = nl[1].attrib.get("coef")
                denominator_coefficient = 1.0 if denominator_coefficient is None else float(denominator_coefficient)
                # assert if unhandled attribute is available
                assert set(nl[1].attrib.keys()) - {"idx", "coef"} == set(), f"unknown attribute in denominator of divide"
            else:
                denominator = self._parse_single_nl(nl[1], denominator_strip_tag, level + 1)

            return OSILDivide(numerator, denominator, level, numerator_is_constant, numerator_coefficient,
                              denominator_coefficient)

        elif kind == "sqrt":
            # return a square root object with the variable tag or a general nl
            assert len(nl) == 1, f"More/less objects than allowed in square root creation: {len(nl)} != 1"
            node = nl[0]
            # check kind of node
            stripped_tag = self._strip(node.tag)
            assert stripped_tag != "number", f"Number tag not allowed in square root nl"

            self.n_sqrt += 1
            if stripped_tag == "variable":
                # assert if more attributes are included
                assert set(node.attrib.keys()) - {"idx"} == set(), "Unknown attribute when handling squareroot"
                return OSILSquareroot(int(node.attrib["idx"]), level)
            else:
                return OSILSquareroot(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "exp":
            # return an exponential function object with the variable tag or a general nl
            assert len(nl) == 1, f"More/less objects than allowed in exp creation: {len(nl)} != 1"
            node = nl[0]
            # check kind of node
            stripped_tag = self._strip(node.tag)
            assert stripped_tag != "number", f"Number tag not allowed in exp nl"

            self.n_exp += 1
            # depending on tag, create exp object with variable and coef or with nl
            if stripped_tag == "variable":
                coefficient = nl[0].get("coef")
                coefficient = 1.0 if coefficient is None else float(coefficient)
                # assert if unhandled attribute is available
                assert set(nl[0].attrib.keys()) - {"idx", "coef"} == set(), f"unknown attribute in argument of exp"

                return OSILExp(int(node.attrib["idx"]), level, coefficient)
            else:
                return OSILExp(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "abs":
            # return an absolute value object with the variable tag or a general nl
            assert len(nl) == 1, f"More/less objects than allowed in abs creation: {len(nl)} != 1"
            node = nl[0]
            # check kind of node
            stripped_tag = self._strip(node.tag)
            assert stripped_tag != "number", f"Number tag not allowed in abs nl"

            if stripped_tag == "variable":
                coefficient = node.attrib.get("coef")
                coefficient = 1.0 if coefficient is None else float(coefficient)
                # assert if more attributes are included
                assert set(node.attrib.keys()) - {"idx", "coef"} == set(), "Unknown attribute when handling abs"
                return OSILAbs(int(node.attrib["idx"]), level, coefficient)
            else:
                return OSILAbs(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "ln":
            # return a ln function object with the variable tag or a general nl
            assert len(nl) == 1, f"More/less objects than allowed in ln creation: {len(nl)} != 1"
            node = nl[0]
            # check kind of node
            stripped_tag = self._strip(node.tag)
            assert stripped_tag != "number", f"Number tag not allowed in ln nl"

            self.n_logln += 1
            if stripped_tag == "variable":
                coefficient = node.attrib.get("coef")
                coefficient = 1.0 if coefficient is None else float(coefficient)
                # assert if more attributes are included
                assert set(node.attrib.keys()) - {"idx", "coef"} == set(), "Unknown attribute when handling ln"
                return OSILLn(int(node.attrib["idx"]), level, coefficient)
            else:
                return OSILLn(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "log10":
            # return a log10 function object with the variable tag or a general nl
            assert len(nl) == 1, f"More/less objects than allowed in log10 creation: {len(nl)} != 1"
            node = nl[0]
            # check kind of node
            stripped_tag = self._strip(node.tag)
            assert stripped_tag != "number", f"Number tag not allowed in log10 nl"

            self.n_logln += 1
            if stripped_tag == "variable":
                # assert if more attributes are included
                assert set(node.attrib.keys()) - {"idx"} == set(), "Unknown attribute when handling log10"
                return OSILLog10(int(node.attrib["idx"]), level)
            else:
                return OSILLog10(self._parse_single_nl(node, stripped_tag, level + 1), level)

        elif kind == "signpower":
            # return a sigpower object, i.e., sign(x) * |x|**c
            assert len(nl) == 2, f"More/less objects than allowed in power creation: {len(nl)}"

            # check that first nl is variable, second one is constant
            assert self._strip(nl[0].tag) == "variable", f"no other than variable allowed for base in signpower creation"
            assert self._strip(nl[1].tag) == "number", f"no other than number allowed for exponent in signpower creation"
            # extract relevant information for base (nl[0]) and power (nl[1])
            base = int(nl[0].attrib["idx"])
            # assert if more attributes are included
            assert set(nl[0].attrib.keys()) - {"idx"} == set(), "Unknown attribute when handling variable in signpower"

            power = float(nl[1].attrib["value"])
            # assert if more attributes are included
            assert len(nl[1].attrib.keys()) == 1, "More attributes than needed in signpower creation"
            return OSILSignPower(base, power, level)

    def _strip(self, tag_name):
        """replace the default string"""
        return tag_name.replace(self.prefix, "")
