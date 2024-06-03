from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import openml

datasets_id = [
    361072, 361073, 361074, 361076, 361077, 361279, 361078, 361079,
    361080, 361081, 361082, 361280, 361084, 361085,
    361086, 361087, 361088
]

names_datasets = [f"names_{id}" for id in datasets_id]

DATASETS = dict(zip(names_datasets, datasets_id))

class Dataset(BaseDataset):

    name = 'openml'

    install_cmd = 'conda'
    requirements = ["pip:openml", "pip:chardet"]

    parameters = {
        "dataset": list(DATASETS),
    }

    def get_data(self):
        dataset = openml.tasks.get_task(
            self.dataset
        )
        X, y = dataset.get_X_and_y()

        return dict(
            X=X,
            y=y,
        )
