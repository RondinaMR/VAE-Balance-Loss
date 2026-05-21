import math
import threading

import numpy as np

from typing import List

from cbx_engine import TabularEngine


def learning_rule(training_rows_size, num_cols=30, task='regression'):

    if training_rows_size < 1000:
        model_epochs = 1000
        model_batch_size = 16
    elif training_rows_size < 10000:
        model_epochs = 500
        model_batch_size = 32
    elif training_rows_size < 50000:
        model_epochs = 300
        model_batch_size = 128
    else:
        model_epochs = 100
        model_batch_size = 256

    model_learning_rate = 0.001

    return model_learning_rate, model_epochs, model_batch_size


class Autoconfig:
    train_ds: np.ndarray
    y_train_ds: np.ndarray = None
    ordinal_features_sizes: int
    categorical_features_sizes: List

    def __init__(
        self,
        train_ds: np.ndarray,
        ordinal_features_sizes: int,
        categorical_features_sizes: List,
        y_train_ds: np.ndarray = None,
    ):
        splitted_train_ds = np.split(
            train_ds, [math.ceil(train_ds.shape[0] * 0.8)], axis=0
        )
        self.train_ds = splitted_train_ds[0]
        self.test_ds = splitted_train_ds[1]
        if y_train_ds is not None:
            splitted_y_train_ds = np.split(
                y_train_ds, [math.ceil(y_train_ds.shape[0] * 0.8)], axis=0
            )
            self.y_train_ds = splitted_y_train_ds[0]
            self.y_test_ds = splitted_y_train_ds[1]
        else:
            self.y_train_ds = None
            self.y_test_ds = None
        self.ordinal_features_sizes = ordinal_features_sizes
        self.categorical_features_sizes = categorical_features_sizes

    def grid_search(self):
        features_size = self.train_ds.shape[1]
        rows_number = self.train_ds.shape[0]

        if features_size < 16:
            architectures = [
                [min(2, math.ceil(features_size / 4))],
            ]
        elif features_size < 64:
            architectures = [
                [math.ceil(features_size / 2), 4],
            ]
        else:
            architectures = [
                [math.ceil(features_size / 2), 8],
            ]

        batch_sizes = [128, 256]
        if rows_number < 512:
            batch_sizes = [16]

        grid_search = []
        for architecture in architectures:
            for batch in batch_sizes:
                grid_search.append([architecture, batch])

        processes = []
        engines = []
        losses = []

        for i, (architecture, batch_size) in enumerate(grid_search):
            engines.append(
                TabularEngine(
                    layers_size=architecture,
                    x_shape=self.train_ds[0].shape,
                    y_shape=self.y_train_ds[0].shape
                    if self.y_train_ds is not None
                    else [0],
                    ordinal_feature_sizes=self.ordinal_features_sizes,
                    categorical_feature_sizes=self.categorical_features_sizes,
                )
            )

            p = threading.Thread(
                target=engines[i].fit,
                args=(
                    self.train_ds,
                    self.y_train_ds if self.y_train_ds is not None else None,
                    5,
                    batch_size,
                    1e-2,
                ),
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for i, engine in enumerate(engines):
            losses.append(
                engine.evaluate(
                    self.test_ds, self.y_test_ds if self.y_test_ds is not None else None
                )["mean_reconstruction_loss"]
            )

        del engines

        _, idx = min((val, idx) for (idx, val) in enumerate(losses))

        return grid_search[idx]
