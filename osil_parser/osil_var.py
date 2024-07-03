variable_types = ["C", "I", "B"]


class OSILVariable(object):
    def __init__(self, name="", lb=None, ub=None, variable_type="C"):
        assert isinstance(name, (str,))
        self.name = name
        assert isinstance(lb, (int, float, type(None)))
        self.lb = lb
        assert isinstance(ub, (int, float, type(None)))
        self.ub = ub
        assert variable_type in variable_types or variable_type is None
        self.variable_type = "C" if variable_type is None else variable_type

    def update_name(self, name):
        assert isinstance(name, (str, ))
        self.name = name

    def update_lb(self, lb):
        assert isinstance(lb, (int, float, type(None)))
        self.lb = lb

    def update_ub(self, ub):
        assert isinstance(ub, (int, float, type(None)))
        self.ub = ub

    def update_variable_type(self, variable_type):
        assert variable_type in variable_types
        self.variable_type = variable_type




