import os
import numpy as np
import pandas as pd
from definitions import *

attributes = [
    'name',
    'adddate',
    'convex',
    'dualbound',
    'gap',
    'initinfeasibility',
    'nbinvars',
    'ncons',
    'ncontvars',
    'nintvars',
    'nvars',
    'primalbound',
    'probtype'
]


def get_instances_by_attribute(set_name, attr_dict={}, feasible=True):
    """return a list of instance names of set set_name which fulfill the specified attributes attr_dict

    Keyword arguments:
    set_name -- the name of the test set; there must exist a folder and a [set_name]-overview.csv file in [set_name]/
    attr_dict -- a dictionary consisting of optional keys equivalent to the attributes above, i.e., 'name', 'adddate',
                 'convex', 'dualbound', 'gap', 'initinfeasibility', 'nbinvars', 'ncons', 'ncontvars', 'nintvars',
                 'nvars', 'primalbound' or 'probtype'.
                 For 'name' and 'probtype' a string or a list of strings is expected
                 For 'adddate' a tuple two date-strings in the form YYYY-MM-DD is expected, symbolizing 'from' and 'to'
                 For 'convex' a boolean value is expected
                 For 'dualbound', 'gap', 'primalbound', a tuple of two floats is expected
                 For the rest, a tuple with integer lower and upper bound on the respective attribute is required
                 In general, None = +-inf
    """

    # initialize the paths and read the data
    cwd = ROOT_DIR
    set_folder = os.path.join(cwd, set_name)
    assert os.path.exists(set_folder)
    overview_file = os.path.join(set_folder, set_name + "-overview.csv")
    assert os.path.exists(overview_file)

    df = pd.read_csv(overview_file, delimiter=";")
    n_rows = df.shape[0]

    # re-format "name" and "probtype" to string, "adddate" to date, "convex" to boolean
    df["name"] = df["name"].astype("string")
    df["probtype"] = df["probtype"].astype("string")
    df["adddate"] = pd.to_datetime(df["adddate"])
    df["convex"] = df["convex"].astype(bool)

    # initialize to respect everything
    condition = np.ones(n_rows).astype(int)

    # iterate for through the attributes, adjust initial condition and skip unknowns
    for key in attr_dict.keys():
        if key in attributes:
            attr_index = list(df.columns).index(key)
        else:
            print(f"Key {key} not in attributes. Skipping.")
            continue

        attr_vals = attr_dict[key]
        if key in ["name", "probtype"]:
            if isinstance(attr_vals, str):
                new_condition = (df.iloc[:, attr_index] == attr_vals).astype(int)
            else:
                try:
                    new_condition = np.isin(df.iloc[:, attr_index], attr_vals).astype(int)
                except TypeError:
                    raise AssertionError(f"For key {key} attributes should be a string or a list of strings")
            condition = condition & new_condition
        elif key == "adddate":
            assert len(attr_vals) == 2
            try:
                if attr_vals[0] not in ["", None]:
                    condition = condition & (attr_vals[0] <= df.iloc[:, attr_index])
                if attr_vals[1] not in ["", None]:
                    condition = condition & (df.iloc[:, attr_index] <= attr_vals[1])
            except TypeError:
                raise AssertionError(f"For key {key} attributes should be date strings in format YYYY-MM-DD")
        elif key == "convex":
            assert isinstance(attr_vals, (bool, int))
            try:
                new_condition = (df.iloc[:, attr_index] == attr_vals).astype(int)
            except TypeError:
                raise AssertionError(f"For key {key} attributes should be 0/1 integers or boolean")
            condition = condition & new_condition
        else:
            try:
                lb, ub = attr_vals
            except IndexError:
                raise AssertionError(f"For key {key} attributes should be a double of lower and upper bound")
            if lb is not None:
                try:
                    condition = condition & (df.iloc[:, attr_index] >= lb)
                except TypeError:
                    raise AssertionError(f"For key {key} attributes should be a double of lower and upper bound")
            if ub is not None:
                try:
                    condition = condition & (df.iloc[:, attr_index] <= ub)
                except TypeError:
                    raise AssertionError(f"For key {key} attributes should be a double of lower and upper bound")

    name_index = list(df.columns).index("name")
    primal_index = list(df.columns).index("primalbound")
    dual_index = list(df.columns).index("dualbound")
    return (list(df.loc[condition.astype(bool)].iloc[:, name_index].values),
            list(df.loc[condition.astype(bool)].iloc[:, primal_index].values),
            list(df.loc[condition.astype(bool)].iloc[:, dual_index].values))
