from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.neighbors import KNeighborsRegressor

class Solver(OSolver):

    name = 'KNeighborsRegressor'
    requirements = ["pip:optuna"]

    def get_model(self):
        return KNeighborsRegressor()

    def sample_parameters(self, trial):
        params = {}
        params['n_neighbors'] = trial.suggest_float("n_neighbors", 5, 20, step=5)
        return params