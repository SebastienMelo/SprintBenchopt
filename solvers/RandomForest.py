from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.ensemble import RandomForestRegressor


class Solver(OSolver):

    name = 'RandomForestRegressor'
    requirements = ["pip:optuna"]

    def get_model(self):
        return RandomForestRegressor()

    def sample_parameters(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200, step=10)
        return dict(
            n_estimators=n_estimators
        )