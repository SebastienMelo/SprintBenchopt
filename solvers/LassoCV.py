from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.linear_model import Lasso


class Solver(OSolver):

    name = 'Lasso'
    requirements = ["pip:optuna"]

    def get_model(self):
        return Lasso()

    def sample_parameters(self, trial):
        alpha = trial.suggest_float("alpha", 1e-2, 5e-1)
        return dict(
            alpha = alpha
        )