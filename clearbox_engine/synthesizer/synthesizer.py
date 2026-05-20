import abc

import numpy as np
rng = np.random.default_rng(42)
import pandas as pd
import scipy

from clearbox_engine.dataset.dataset import Dataset
from clearbox_engine.engine.tabular_engine import TabularEngine
from clearbox_engine.preprocessor.preprocessor import Preprocessor


class Synthesizer(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "generate") and callable(subclass.fit)

    def __init__(
        self, dataset: Dataset, engine: TabularEngine, preprocessor: Preprocessor = None
    ):
        self.dataset = dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(dataset)
        )
        self.engine = engine
        self.sampled_indexes = None

    def _sample_vae(self, x, recon_x):

        preprocessed_x = self.preprocessor.transform(x)

        n_ordinal_features = self.preprocessor.get_features_sizes(
        )[0][0] if self.preprocessor.get_features_sizes()[0] else 0
        categorical_features_sizes = self.preprocessor.get_features_sizes()[1]

        ordinal_features_sampled = np.zeros(
            (preprocessed_x.shape[0], n_ordinal_features))

        for i in range(n_ordinal_features):
            if isinstance(preprocessed_x[:, i], scipy.sparse.csr_matrix):
                converted_input = preprocessed_x[:, i].toarray().reshape(1, -1)[0]
            else:
                converted_input = preprocessed_x[:, i]

            ordinal_features_sampled[:, i] = \
                recon_x[:, i] + self.engine.search_params['gauss_s'] * rng.standard_normal(recon_x.shape[0])

        categorical_features_sampled = np.zeros(
            (preprocessed_x.shape[0], preprocessed_x.shape[1] - n_ordinal_features))
        view_decoded = recon_x[:, n_ordinal_features:]

        for i in range(preprocessed_x.shape[0]):
            w2 = 0  # index categorical label in preprocessed space
            w3 = 0  # index categorical feature
            features = (preprocessed_x[i, n_ordinal_features:] > 0)
            if isinstance(features, scipy.sparse.csr_matrix):
                features = features.toarray().reshape(1, -1)[0]

            for w in categorical_features_sizes:
                if (features[w2:w2 + w]).sum() == 0:
                    # it means that there's a NaN or an unknown
                    categorical_features_sampled[i, w3] = 0.0
                else:

                    distribution = view_decoded[i, w2:w2 + w]

                    distribution = np.asarray(distribution).astype('float64')
                    distribution/=distribution.sum()
                    pick = rng.choice(w, p=distribution)
                    categorical_features_sampled[i, w2 + pick] = 1.
                w2 += w
                w3 += 1

        e = np.hstack([ordinal_features_sampled, categorical_features_sampled])
        ip = self.preprocessor._setup_inverse_preprocessor()

        return ip(e)

    def _force_temporal_precedence(self, synthetic_dataset: pd.DataFrame):
        datetime_features = [
            column
            for column in self.dataset.column_types.keys()
            if self.dataset.column_types[column] == "datetime"
        ]
        if datetime_features:
            sample = self.dataset.data[datetime_features].head(1)
            column_names = sample.columns.tolist()
            values = sample.values.tolist()
            sorted_columns = [x for _, x in sorted(zip(values, column_names))]

            for index, row in synthetic_dataset[datetime_features].iterrows():
                column_names = row.to_frame().T.columns.tolist()
                values = row.values.tolist()
                if [x for _, x in sorted(zip(values, column_names))] != sorted_columns:
                    sorted_values = sorted(values)
                    for i, col in enumerate(column_names):
                        synthetic_dataset.loc[index, col] = sorted_values[i]

    @abc.abstractmethod
    def generate(self, has_header: bool = None):
        """Generate new data"""
        raise NotImplementedError
