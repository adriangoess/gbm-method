import pyomo.environ as pyo
from utilities.pyomo_solver import PyomoSolver


def check_function_violation(function_string, quads, lins, conss, domain, dim=1, print_result=True, approx_below=True):
    """
    This function checks whether our function to approximate is so from below.
    In particular, if f >= max_{i} p_i on the entire domain.
    This is conducted by computing the minimum of f - p_i for each i and return false if this value is negative for an i
    For approximation from above, we only have to adapt the sign of f in the objective, as the paramodeller does so.
    :param function_string: function to approximate -> string 'sin' or 'exp'
    :param quads: quadratic coefficients for each paraboloid -> list
    :param lins: linear coefficients for each paraboloid -> list
    :param conss: constants for each paraboloid -> list
    :param domain: upper and lower bound of variable
    :param dim: dimension for f, the paraboloids and the domain (default = 1)
    :return: True/False
    """
    minimal_f_p_distances = []
    violation = False
    # check for validity of parameter types and initialize
    assert function_string in ["sin", "cos", "exp", "x^3"], f"only sin, cos, x^3 and exp implemented for function to approximate"
    if function_string == "sin":
        f = pyo.sin
    elif function_string == "cos":
        f = pyo.cos
    elif function_string == "x^3":
        f = lambda x: x ** 3
    else:
        f = pyo.exp
    assert check_coefficient_validity(quads, lins, conss, dim)
    assert len(domain) == 2, f"domain must have two entries, lb and ub"
    assert len(domain[0]) == dim, f"domain entries must have identical dimension as the value of dim"
    assert len(domain[1]) == dim, f"domain entries must have identical dimension as the value of dim"
    assert domain[0] < domain[1], f"ub as to exceed lb"

    # for each paraboloid, model min f - p_i over domain
    for i, (quadratics, linears, constants) in enumerate(zip(quads, lins, conss)):
        # initialize the model
        m = pyo.ConcreteModel()

        # initialize the variable
        def get_var_bounds(model, var_index):
            return domain[0][var_index], domain[1][var_index]

        m.x = pyo.Var(list(range(dim)), within=pyo.Reals, bounds=get_var_bounds)

        # initialize the paraboloid function
        paraboloid = 0
        for j in range(dim):
            paraboloid += quadratics[j] * m.x[j]**2 + linears[j] * m.x[j] + constants[j]
        m.obj = pyo.Objective(expr=(-1) ** (1 + approx_below) * f(m.x[0]) - paraboloid, sense=pyo.minimize)

        # solve the model
        solver = PyomoSolver("scip")
        solver.solve_model(m, False)
        minimal_f_p_distances.append(m.obj())
        if m.obj() < 0:
            violation = True

    if min(minimal_f_p_distances) < 0 and print_result:
        print(f"Function violation: {min(minimal_f_p_distances)}")
    elif print_result:
        print(f"No function violation!")
    return violation, minimal_f_p_distances


def check_approximation_violation(function_string, quads, lins, conss, domain, eps, dim=1, print_result=True,
                                  approx_below=True):
    """
    This function checks whether the approximation by paraboloids fulfills the epsilon approximation guarantee.
    In particular, if max_{i} p_i >= function - epsilon on the entire domain.
    This is conducted by computing the minimum of (y - f + eps) where y >= p_i for each i and stop if this value is negative.
    For approx from above, we just reverse the sign of f, as the para modeller does so.
    :param function_string: function to approximate -> string 'sin' or 'exp'
    :param quads: quadratic coefficients for each paraboloid -> list
    :param lins: linear coefficients for each paraboloid -> list
    :param conss: constants for each paraboloid -> list
    :param domain: upper and lower bound of variable
    :param eps: approximation guarantee, float >= 0
    :param dim: dimension for f, the paraboloids and the domain (default = 1)
    :return: True/False
    """
    # check for validity of parameter types and initialize
    assert function_string in ["sin", "cos", "exp", "x^3"], f"only sin, cos, x^3, and exp implemented for function to approximate"
    if function_string == "sin":
        f = pyo.sin
    elif function_string == "cos":
        f = pyo.cos
    elif function_string == "x^3":
        f = lambda x: x ** 3
    else:
        f = pyo.exp
    assert check_coefficient_validity(quads, lins, conss, dim)
    assert len(domain) == 2, f"domain must have two entries, lb and ub"
    assert len(domain[0]) == dim, f"domain entries must have identical dimension as the value of dim"
    assert len(domain[1]) == dim, f"domain entries must have identical dimension as the value of dim"
    assert domain[0] < domain[1], f"ub has to exceed lb"
    assert isinstance(eps, (float,)), f"approximation guarantee eps must be float"

    # initialize the model
    m = pyo.ConcreteModel()

    # initialize the variables
    def get_var_bounds(model, var_index):
        return domain[0][var_index], domain[1][var_index]

    m.x = pyo.Var(list(range(dim)), within=pyo.Reals, bounds=get_var_bounds)
    m.y = pyo.Var("y", within=pyo.Reals)

    # for each paraboloid, model y >= p_i over domain
    m.constraints = pyo.ConstraintList()
    for i, (quadratics, linears, constants) in enumerate(zip(quads, lins, conss)):
        # initialize the paraboloid function
        paraboloid = 0
        for j in range(dim):
            paraboloid += quadratics[j] * m.x[j]**2 + linears[j] * m.x[j] + constants[j]
        m.constraints.add(expr=m.y["y"] >= paraboloid)

    sign_of_f = -1 if approx_below else 1
    m.obj = pyo.Objective(expr=m.y["y"] + sign_of_f * f(m.x[0]) + eps, sense=pyo.minimize)

    # solve the model
    solver = PyomoSolver("scip")
    solver.solve_model(m, False)
    if m.obj() < 0:
        if print_result:
            print(f"Approximation violation: {m.obj()}")
        return True, m.obj()

    if print_result:
        print("No approximation violation!")

    return False, m.obj()


def check_coefficient_validity(quads, lins, conss, dim=1):
    """
    Check the validity of paraboloid coefficients for other functions in this file
    :param quads: list of floats (dim == 1) or list of tuples of floats
    :param lins: list of floats (dim == 1) or list of tuples of floats
    :param conss: list of floats (dim == 1) or list of tuples of floats
    :param dim: dimension, integer
    :return: True/False
    """
    assert isinstance(dim, (int,)), "dim(ension) must be integer"
    assert dim > 0, "dim(ension) must be positive"

    # initialize namings and iterate zipped lists
    names = ["quadratic coefficients", "linear coefficients", "constant coefficients"]
    for name, coefs in zip(names, [quads, lins, conss]):
        assert isinstance(coefs, (list,)), f"{name} needs to be a list of respective parameters"
        for entry in coefs:
            assert isinstance(entry, (tuple, list)), f"{name} entries must be tuples/lists of floats"
            assert len(entry) == dim, f"{name} entries have to be tuples with dimension={dim}"
            for sub_entry in entry:
                assert isinstance(sub_entry, (float,)), f"{name} entries for one paraboloid must be floats"

    return True


