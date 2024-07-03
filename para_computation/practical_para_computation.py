import os
import time
import json
import numpy as np
from pyomo.opt import TerminationCondition
from para_inexact_modeler import ParaboloidModel
from utilities.pyomo_solver import PyomoSolver
from result_checker import check_function_violation, check_approximation_violation

# choice between logarithmic and linear (+1) search for parabolas
log_search = True

# define a maximal limit for solving each instance
max_time_limit = 60 * 60

approx_dirs_below = [True, False]

function_setting_names = ["exp_setting1.json", "sine_setting0.json", "x3_setting1.json"]

model_setting_names = [f"setting_eps-{i}.json" for i in range(3)]


if __name__ == "__main__":
    # current work directory for path creation
    current_path = os.getcwd()

    times = {}

    # loop through approximation directory, True for "from below"
    for approx_below in approx_dirs_below:
        times[approx_below] = {}
        # loop through necessary function settings and model settings
        for func_setting_name in function_setting_names:
            times[approx_below][func_setting_name] = {}
            # create path, extract settings and relevant values
            func_setting_path = os.path.join(current_path, "func_settings", func_setting_name)
            function_settings = json.load(open(func_setting_path))
            func_lb = float(function_settings["lb"])
            func_ub = float(function_settings["ub"])
            func_int_length = func_ub - func_lb
            func_maxL = float(function_settings["L-constant"])

            # by starting to approx a function setting, we start with n_paraboloids = 1 and increase according to the result
            # it is assumed that the model settings are sorted by decreasing eps!
            n_paraboloids = 1
            log_lb = 1

            for model_setting_name in model_setting_names:
                times[approx_below][func_setting_name][model_setting_name] = [0, 0, 0]
                print("-"*100)

                model_setting_path = os.path.join(current_path, "model_settings", model_setting_name)

                # outer loop finds amount of paraboloids
                unsatisfactory = True
                log_ub = None

                # initialize discretization steps for t (approx to f) and d (approx in eps range) based on settings
                model_settings = json.load(open(model_setting_path))
                model_eps = float(model_settings["eps"])

                n_delta_t_increase = np.ceil(func_int_length / model_eps * func_maxL)
                n_delta_d_increase = np.ceil(func_int_length / model_eps * func_maxL)
                n_max_delta_t = np.ceil(func_int_length / model_eps * func_maxL / 10)
                n_max_delta_d = np.ceil(func_int_length / model_eps * func_maxL / 10)

                # values to set for respecting solutions and to increase the number of paraboloids when solutions are unsatisfying
                objective_tolerance = 10.0
                n_function_violations = 0
                n_approximation_violations = 0

                # iteration counter and initialization for quadratic, linear and constant coefficients
                n_iterations = 0
                quads, lins, cons = [], [], []

                while unsatisfactory:
                    # track time
                    start_modelling_time = time.time()
                    if n_iterations > 0:
                        times[approx_below][func_setting_name][model_setting_name][2] += start_modelling_time - start_eval_time


                    print(f"Iterations: {n_iterations}; Paraboloids: {n_paraboloids}")
                    n_iterations += 1
                    # initialize paraboloid finding model
                    para_modeler = ParaboloidModel(func_setting_path, model_setting_path, n_paraboloids, approx_below,
                                                   n_max_delta_t, n_max_delta_d)
                    # set up the model
                    model = para_modeler.setup_model(False, [quads, lins, cons])
                    #model.pprint()
                    #model.write(f"para_model_{model_setting_name}_{func_setting_name}_p{n_paraboloids}.gms")

                    # track time
                    start_solving_time = time.time()
                    times[approx_below][func_setting_name][model_setting_name][0] += start_solving_time - start_modelling_time

                    # solve model
                    solver = PyomoSolver("gurobi", time_limit=max_time_limit, abs_gap=0)
                    objective, results = solver.solve_model(model, tee=False)

                    print(f"Finished solving; Starting validation")

                    # track time
                    start_eval_time = time.time()
                    times[approx_below][func_setting_name][model_setting_name][1] += start_eval_time - start_solving_time

                    print(f"Time needed for solving: {np.round(results.solver.user_time, 1)}")

                    time_limit_reached = results.solver.termination_condition == TerminationCondition.maxTimeLimit
                    infeasible = results.solver.termination_condition == TerminationCondition.infeasible

                    # increase the amount of paraboloids if they do not satisfy the requirements
                    if objective > objective_tolerance or infeasible or (time_limit_reached and np.isnan(results.problem.upper_bound)):
                        if objective > objective_tolerance:
                            print(f"Increase of number of paraboloids due to objective tolerance!")
                        elif infeasible:
                            print(f"Increase of number of paraboloids due to infeasibility!")
                        else:
                            print(f"Increase of number of paraboloids due to time limit!")
                        if log_search:
                            log_lb = n_paraboloids
                            if log_ub is None:
                                n_paraboloids *= 2
                            else:
                                if log_ub <= log_lb + 1:
                                    break
                                n_paraboloids = log_lb + int((log_ub - log_lb)/2)
                        else:
                            n_paraboloids += 1
                        n_function_violations = 0
                        n_approximation_violations = 0
                        continue

                    # extract the parameters
                    quads, lins, cons = para_modeler.extract_results(model)

                    # check validity for testing
                    assert "sine" in func_setting_name or "exp" in func_setting_name or "x3" in func_setting_name, \
                        f"so far only implemented for sine, cosine, x^3 and exp"
                    domain = [[para_modeler.lb], [para_modeler.ub]]
                    if func_setting_name.startswith("sine"):
                        function_string = "sin"
                    elif "cosine" in func_setting_name:
                        function_string = "cos"
                    elif "x3" in func_setting_name:
                        function_string = "x^3"
                    else:
                        function_string = "exp"

                    # check if a paraboloid exceeds the function
                    function_violation, f_p_distances = check_function_violation(function_string, quads, lins, cons, domain,
                                                                                        dim=1, print_result=True, approx_below=approx_below)

                    # compute manipulated constant coefficients of the paraboloids which result from shifting the current ones
                    # exactly below function f
                    manip_cons = cons.copy()
                    # decrease f to p distances by a constant of 1e-6
                    f_p_distances = [max([0, dist - 1e-6]) if not function_violation else dist - 1e-6 for dist in f_p_distances]
                    for k, para_constants in enumerate(cons):
                        for l, entry in enumerate(para_constants):
                            manip_cons[k][l] += f_p_distances[k]

                    # sanity check:
                    shifted_function_violation, f_p_distances = check_function_violation(function_string, quads, lins, cons,
                                                                                                domain, para_modeler.dim, False, approx_below)
                    assert not shifted_function_violation, f"function violation can not be false if shifted paras are considered"

                    # if a paraboloid exceeds the function, increase the amount of discretization points
                    if function_violation:
                        # increase the amount of discretization points linearly on base value
                        n_max_delta_d += n_delta_d_increase
                        n_function_violations += 1
                        if n_function_violations == 20:
                            n_function_violations = 0
                            n_paraboloids += 1
                            continue

                    # check if the eps approximation is violated with the shifted constants
                    approximation_violation, approximation_violation_distance = check_approximation_violation(
                        function_string, quads, lins, manip_cons, domain, eps=para_modeler.eps, dim=1, print_result=True, approx_below=approx_below)

                    # if epsilon approximation is violated increase the amount of segments
                    if approximation_violation:
                        n_max_delta_t += n_delta_t_increase
                        n_approximation_violations += 1
                        if n_approximation_violations == 20:
                            n_approximation_violations = 0
                            n_paraboloids += 1

                    unsatisfactory = function_violation or approximation_violation
                    if not unsatisfactory and log_search:
                        # as the current number of paraboloids suffices, safe it as upper bound and decrease (when searching log)
                        log_ub = n_paraboloids
                        if log_ub <= log_lb + 1:
                            print(f"Logarithmic upper bound less/equal than lower bound. Exiting loop.")
                        else:
                            increase = int((log_ub - log_lb)/2)
                            n_paraboloids = log_lb + increase
                            unsatisfactory = True

                        # get necessary parameters for storage
                        eps = str(para_modeler.eps)
                        lb = str(np.round(para_modeler.lb, 6))
                        ub = str(np.round(para_modeler.ub, 6))
                        approx_string = "below" if approx_below else "above"
                        if para_modeler.dim == 1:
                            out_quads = [(-1) ** (1 + approx_below) * quad[0] for quad in quads]
                            out_lins = [(-1) ** (1 + approx_below) * lin[0] for lin in lins]
                            out_cons = [(-1) ** (1 + approx_below) * con[0] for con in cons]
                        else:
                            out_quads, out_lins, out_cons = quads, lins, cons

                    if n_paraboloids > n_max_delta_d:
                        exit("Problem can not be solved in given time limit")

                # save found parameters in the respective json; the keys are [eps][func][lb][ub][below] where eps is float,
                # func is in ("sin", "cos", "exp"), lb/ub are float, below/above is string, and give (quads, lins, cons)
                with open('para_parameters.json', 'r+') as file:
                    params = json.load(file)
                    func_entries = params.get(eps)
                    if func_entries is None:
                        params[eps] = {}
                    bound_entries = params[eps].get(function_string)
                    if bound_entries is None:
                        params[eps][function_string] = {}
                    ub_entries = params[eps][function_string].get(lb)
                    if ub_entries is None:
                        params[eps][function_string][lb] = {}
                    approx_entries = params[eps][function_string][lb].get(ub)
                    if approx_entries is None:
                        params[eps][function_string][lb][ub] = {}
                    param_triple = params[eps][function_string][lb][ub].get(approx_string)
                    if param_triple is None:
                        params[eps][function_string][lb][ub][approx_string] = (out_quads, out_lins, out_cons)
                    else:
                        if len(param_triple[0]) > len(quads):
                            params[eps][function_string][lb][ub][approx_string] = (out_quads, out_lins, out_cons)

                    file.seek(0)
                    json.dump(params, file, indent=4)
                    file.truncate()


    print("finished\n")
