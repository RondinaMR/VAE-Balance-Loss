import numpy as np

from sklearn.preprocessing import OneHotEncoder

from clearbox_engine import Dataset, Preprocessor, TabularEngine


class ReconstructionError:
    original_dataset: Dataset
    synthetic_dataset: Dataset
    preprocessor: Preprocessor

    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        engine: TabularEngine,
        preprocessor: Preprocessor = None,
    ):
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )
        self.engine = engine

    def get(self):
        if self.original_dataset.target_column is None:
            original_ds = self.preprocessor.transform(self.original_dataset.get_x())
            original_reconstruction_error = self.engine.reconstruction_error(
                original_ds
            )

            synthetic_ds = self.preprocessor.transform(self.synthetic_dataset.get_x())
            synthetic_reconstruction_error = self.engine.reconstruction_error(
                synthetic_ds
            )
        else:
            if not self.original_dataset.regression:
                y_encoder = OneHotEncoder(handle_unknown="ignore")
                y_encoder.fit(self.original_dataset.get_y().to_numpy().reshape(-1, 1))

                original_y = y_encoder.transform(
                    self.original_dataset.get_y().to_numpy().reshape(-1, 1)
                ).toarray()

                synthetic_y = y_encoder.transform(
                    self.synthetic_dataset.get_y().to_numpy().reshape(-1, 1)
                ).toarray()
            else:
                original_y = self.original_dataset.get_normalized_y()
                synthetic_y = self.synthetic_dataset.get_normalized_y()

            original_ds = self.preprocessor.transform(self.original_dataset.get_x())

            original_reconstruction_error = self.engine.reconstruction_error(
                original_ds, original_y
            )

            synthetic_ds = self.preprocessor.transform(self.synthetic_dataset.get_x())
            synthetic_reconstruction_error = self.engine.reconstruction_error(
                synthetic_ds, synthetic_y
            )

        hist, bin_edges = np.histogram(
            original_reconstruction_error,
            bins=100,
            range=(
                min(
                    original_reconstruction_error.min(),
                    synthetic_reconstruction_error.min(),
                ),
                max(
                    original_reconstruction_error.max(),
                    synthetic_reconstruction_error.max(),
                ),
            ),
            density=True,
        )

        bins = []
        for i in range(1, len(bin_edges)):
            mean_value = (bin_edges[i] + bin_edges[i - 1]) / 2
            bins.append(round(float(mean_value), 4))

        train_hist = []
        for value in hist:
            train_hist.append(round(float(value), 4))

        hist, _ = np.histogram(
            synthetic_reconstruction_error, bins=bin_edges, density=True
        )

        synthetic_hist = []
        for value in hist:
            synthetic_hist.append(round(float(value), 4))

        histogram = {
            "bin_edges": bins,
            "original_hist": train_hist,
            "synthetic_hist": synthetic_hist,
        }

        return histogram
