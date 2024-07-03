import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


class PyomoSolver(object):
    def __init__(self, solver="scip", rel_gap=0, time_limit=30, abs_gap=0.0):
        assert solver in ["antigone", "baron", "gurobi", "scip", "lindo", "lindoglobal"]
        self.solver = solver

        self.Options = []
        if solver == 'scip':
            self.Options = ['GAMS_MODEL.optfile = 1;', '$onecho > scip.opt',
                            f'limits/gap={rel_gap}',
                            f'limits/absgap={abs_gap}',
                            f'limits/time={time_limit}',
                            #f'heuristics/emphasis="aggressive"',
                            '$offecho']
        elif solver == 'baron':
            self.Options = ['GAMS_MODEL.optfile=1;', '$onecho > baron.opt',
                            f'epsr {rel_gap}',
                            f'epsa {abs_gap}',
                            f'maxtime {time_limit}',
                            f'AbsConFeasTol 1e-5',
                            '$offecho',
                            ]
        elif solver == 'gurobi':
            self.Options = ['GAMS_MODEL.optfile=1;', '$onecho > gurobi.opt',
                            f'TimeLimit = {time_limit}',
                            f'mipgapabs = {abs_gap}',
                            f'mipgap = {rel_gap}',
                            f'mipfocus = 1',
                            '$offecho']

    def solve_model(self, model, tee=True):
        assert isinstance(model, pyo.ConcreteModel)

        optimizer = pyo.SolverFactory("gams")

        try:
            results = optimizer.solve(model, solver=self.solver, tee=tee, add_options=self.Options, warmstart=True)
            if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
                print(f"Time Limit reached")
            infeasible = results.solver.termination_condition == TerminationCondition.infeasible
        except ValueError:
            print("baron is unable to solve")
            return 0.0, None

        try:
            obj = model.obj()
        except ValueError:
            print(f"failing to retrieve objective, taking lower bound; CHECK IF MINIMIZING")
            obj = results.problem.lower_bound
        return obj, results
