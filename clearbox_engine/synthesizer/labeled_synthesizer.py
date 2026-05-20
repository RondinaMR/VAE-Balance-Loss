from .synthesizer import Synthesizer

import os
from typing import List
import numpy as np
rng = np.random.default_rng(42)
import pandas as pd


class LabeledSynthesizer(Synthesizer):
    def _generate_instance(
        self,
        new_samples: pd.DataFrame,
        encoded: np.ndarray,
        X: np.ndarray,
        reshuffle_indexes: np.ndarray,
        index: int,
        sampled_index: int,
        n_sampling_points: int = 5,
        hybrid_columns: List = [],
    ):
        encoded_istance = encoded[sampled_index, :]
        distances = ((encoded - encoded_istance) ** 2).sum(axis=1) ** 0.5

        idx1 = np.argpartition(np.array(distances), n_sampling_points)[
            0:n_sampling_points
        ]

        discarded_columns = [i for i in self.preprocessor.discarded[0]]

        columns_to_shuffle = [
            col
            for col in self.dataset.data.columns.tolist()
            if (
                col != self.dataset.target_column
                and col not in discarded_columns + hybrid_columns
            )
        ]

        for i, j in enumerate(columns_to_shuffle):
            new_samples[j].at[index] = X[j].at[idx1[reshuffle_indexes[index, i]]]

    def generate(
        self,
        has_header=None,
        points=None,
        n_sampling_points=5,
        hybrid_columns=[],
        latent_noise=0.0,
        Y = None,
    ):

        X = self.dataset.get_x()

        if Y is None:
            if self.dataset.regression:
                Y = self.dataset.get_normalized_y()
            else:
                Y = self.dataset.get_one_hot_encoded_y()

        if len(hybrid_columns) == 0:
            hybrid_columns = list(X.columns)

        if points is None:
            n_samples = min(1500000, self.dataset.data.shape[0])

            sampled_indexes = rng.choice(
                range(X.shape[0]), n_samples, replace=False
            )

        else:
            n_samples = len(points)
            sampled_indexes = rng.choice(points, n_samples, replace=False)

        self.sampled_indexes = sampled_indexes
        data = self.dataset.get_x()
        preprocessed_data = self.preprocessor.transform(data)

        _, encoded, _ = self.engine.apply(preprocessed_data, Y)
        # encoded = ((1.-latent_noise) * encoded + latent_noise * encoded.std() * rng.random.randn(*encoded.shape))
        encoded = encoded + latent_noise * rng.standard_normal(encoded.shape)
        new_samples = pd.DataFrame(
            index=range(n_samples),
            columns=list(data.columns) + [self.dataset.target_column],
        )

        #NOTE Is it ok?
        # np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))

        discarded_columns = [i for i in self.preprocessor.discarded[0]]
        columns_to_shuffle = [
            col
            for col in self.dataset.data.columns.tolist()
            if (col != self.dataset.target_column and col not in discarded_columns)
        ]

        reshuffle_indexes = np.zeros((n_samples, len(columns_to_shuffle)))

        for i in range(len(columns_to_shuffle)):
            reshuffle_indexes[:, i] = rng.choice(
                np.arange(0, n_sampling_points), reshuffle_indexes.shape[0]
            )

        reshuffle_indexes = reshuffle_indexes.astype(int)

        columns_to_shuffle = [
            col
            for col in self.dataset.data.columns.tolist()
            if (
                col != self.dataset.target_column
                and col not in discarded_columns + hybrid_columns
            )
        ]
        if len(columns_to_shuffle) > 0:
            for i in range(n_samples):
                self._generate_instance(
                    new_samples,
                    encoded,
                    X,
                    reshuffle_indexes,
                    i,
                    sampled_indexes[i],
                    n_sampling_points,
                    hybrid_columns,
                )

        # preprocessed_new_samples = self.preprocessor.transform(
        #     new_samples.drop([self.dataset.target_column], axis=1)
        # )

        if len(hybrid_columns) > 0:
            recon_x = self.engine.decode(encoded, Y)
            recon_x = np.asarray(recon_x)
            out = self._sample_vae(
                data.iloc[sampled_indexes, :], recon_x[sampled_indexes, :]
            )

            vaedf = out  # .iloc[sampled_indexes]
            vaedf.index = new_samples.index

            to_fill = [i for i in hybrid_columns if i not in discarded_columns]
            for i in to_fill:
                new_samples[i] = vaedf[i]
        #
        # recon_error = self.engine.reconstruction_error(
        #     preprocessed_new_samples, Y[y_engine.astype(int)]
        # )
        if self.dataset.get_y() is not None:
            new_samples[self.dataset.target_column] = (
                self.dataset.get_y().iloc[sampled_indexes].values
            )

        # if self.dataset.regression is True:
        #     y_regression = new_samples[self.dataset.target_column]
        #     y_regression = np.clip(y_regression + y_regression.std() * rng.random.randn(*y_regression.shape),
        #                            y_regression.min(),
        #                            y_regression.max())
        #     new_samples[self.dataset.target_column] = y_regression

        for i in discarded_columns:
            new_samples[i] = X[i].iloc[sampled_indexes].values

        # if recon_sample is False:
        # sampled_indexes = rng.choice(
        #     np.arange(n_samples),
        #     size=n_samples,
        #     replace=False,
        # )
        # else:
        # sampled_indexes = rng.choice(
        #     np.arange(n_samples),
        #     size=n_samples,
        #     replace=False,
        #     p=(recon_error.max() - recon_error) / (recon_error.max() - recon_error).sum(),
        # )

        for i in self.preprocessor.discarded[0]:
            if X[i].iloc[sampled_indexes].nunique() > 1:
                new_samples[i] = "*"

        dtypes = self.dataset.data.dtypes.to_dict()
        for i in dtypes:
            if dtypes[i] != "bool":
                new_samples[i] = new_samples[i].astype(dtypes[i])

        # for i in self.preprocessor.discarded[1]:
        #     for _ in i[1]:
        #         dtypes[i[0]] = np.dtype("O")
        #         new_samples[i[0]] = new_samples[i[0]].replace(i[1], "*")

        # new_samples = new_samples.astype(dtypes)

        # new_samples = new_samples.sample(frac=1.0)

        cat_cols = new_samples.columns[new_samples.dtypes == 'object']
        for i in cat_cols:
            new_samples[i] = new_samples[i].replace('nan', np.nan)

        if self.engine.privacy_budget == 0.:
            num_cols = new_samples.columns[new_samples.dtypes != 'object']

            for i in num_cols:
                a = new_samples[i].dropna().unique()
                a.sort()
                if a.shape[0] > 1:
                    dt = np.absolute(a[1:] - a[:-1]).min()
                else:
                    dt = 0.
                new_samples[i] = new_samples[i] + dt * rng.choice([-2, -1, 0, 1, 2], new_samples.shape[0])

        if not has_header:
            new_samples = new_samples.rename(columns=new_samples.iloc[0]).drop(
                new_samples.index[0]
            )
            return new_samples

        # try:
        #     self._force_temporal_precedence(new_samples)
        # except Exception as e:
        #     print(e)
        #     pass

        return new_samples.reindex(columns=self.dataset.columns())
