from benchopt import BaseObjective, safe_import_context
from sklearn.model_selection import train_test_split
# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    
    # Name to select the objective in the CLI and to display the results.
    name = "Coverages"

    # URL of the main repo for this benchmark.from sklearn.model_selection import train_test_split
    url = "https://github.com/SebastienMelo/SprintBenchopt"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'whiten_y': [False, True],
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(
                self, X, y,
                categorical_indicator
        ):
            # The keyword arguments of this function are the keys of the dictionary
            # returned by `Dataset.get_data`. This defines the benchmark's
            # API to pass data. This is customizable for each benchmark.
            rng = np.random.RandomState(self.seed)
            X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.33)
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train_full, y_train_full, test_size=0.5
            )

            self.X_train, self.y_train = X_train, y_train
            self.X_cal, self.y_cal = X_cal, y_cal
            self.X_test, self.y_test = X_test, y_test
            self.categorical_indicator = categorical_indicator

    def evaluate_result(self, model, quantile):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.   

        y_pred = model.predict(self.X_test)
        coverage = np.sum(((y_pred - quantile) <= self.y_test)
        * (self.y_test <= (y_pred + quantile))
        ) / len(self.y_test)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            coverage=coverage, interval_size=2 * quantile
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(coverage=0, interval_size=1)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_cal=self.X_cal,
            y_cal=self.y_cal,
        )
