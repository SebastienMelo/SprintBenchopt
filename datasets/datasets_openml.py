from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import openml


DATASETS = {
    'elevators': 45031,
    'pol': 44122,
    'rl': 44160,
    'road_safety': 44161,
    'wine': 44091,
}


class Dataset(BaseDataset):

    name = 'openml'

    install_cmd = 'conda'
    requirements = ["pip:openml", "pip:chardet"]

    parameters = {
        "dataset": list(DATASETS),
    }

    def get_data(self):
        dataset = openml.datasets.get_dataset(
            self.dataset
        )
        X, y, cat_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )

        return dict(
            X=X,
            y=y,
        )
