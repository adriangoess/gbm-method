class OSILObjective(object):
    def __init__(self, name, direction, n_coeffs, coeffs, constant):
        assert isinstance(name, (str,))
        self.name = name

        assert direction in ["min", "max"]
        self.direction = direction

        assert isinstance(n_coeffs, (int, ))
        assert n_coeffs >= 0
        self.n_coeffs = n_coeffs

        assert isinstance(coeffs, (dict, ))
        for k, v in coeffs.items():
            # check keys for variable indices (int) and values for floats (coefficients)
            assert isinstance(k, (int,))
            assert isinstance(v, (float,))
        self.coeffs = coeffs

        assert isinstance(constant, (int, float))
        self.constant = constant

    def update_name(self, name):
        assert isinstance(name, (str,))
        self.name = name

    def update_direction(self, direction):
        assert direction in ["min", "max"]
        self.direction = direction

    def update_n_coeffs(self, n_coeffs):
        assert isinstance(n_coeffs, (int,))
        assert n_coeffs >= 0
        self.n_coeffs = n_coeffs

    def update_coeff(self, variable_index, coefficient):
        assert isinstance(variable_index, (int,))
        assert isinstance(coefficient, (float,))

        if variable_index in self.coeffs.keys():
            self.coeffs[variable_index] = coefficient
        else:
            print(f"WARNING; variable index {variable_index} not in objective coefficients")

    def update_constant(self, constant):
        assert isinstance(constant, (int, float))
        self.constant = constant

    def eval(self, variables):
        # construct and evaluate the linear objective
        obj = 0
        for variable_index, coefficient in self.coeffs.items():
            obj += coefficient * variables[variable_index]
        return obj

