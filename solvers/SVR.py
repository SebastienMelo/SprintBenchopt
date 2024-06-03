from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.svm import SVR

class Solver(OSolver):

    name = 'SVR'
    requirements = ["pip:optuna"]

    def get_model(self):
        return SVR()

    def sample_parameters(self, trial):
        params = {}
        params['C'] = trial.suggest_float("C", 1e-1, 1e1, log=True)
        return params