import json
import optax
import numpy as np

from typing import Dict, Sequence, Tuple

import pandas as pd
from jax import random
from flax.core.frozen_dict import FrozenDict
from flax import serialization
from flax.training import train_state
from tqdm import trange

from clearbox_engine.VAE.tabular_vae_weight import TabularVAEWeight, eval, train_step
from .engine import EngineInterface

import logging

# create logger
module_logger = logging.getLogger('engine.tabular_engine_weight')


class TabularEngineWeight(EngineInterface):
    model: TabularVAEWeight
    params: FrozenDict
    search_params: Dict
    architecture = Dict
    hashed_architecture = str

    def __init__(
            self,
            layers_size: Sequence[int],
            ordinal_feature_sizes: Sequence[int],
            categorical_feature_sizes: Sequence[int],
            x_shape: Sequence[int],
            y_shape: Sequence[int] = [0],
            params: FrozenDict = None,
            train_params: Dict = None,
            privacy_budget: float = 1.0,
            folder_path: str = None
    ):
        self.folder_path = folder_path
        key = random.key(42)
        if x_shape[0] > 300:
            layers_size = [int(x_shape[0]/1.4)]

        if train_params is None:
            train_params = {
                "l2_reg": 0.000,
                "beta": 0,
                "alpha": .1,
                "gauss_s": 0.01,
                "gauss_s_c": 0.1,
                "weight_decay": 0.000,
                "prob_clip": 0.99,
                "simpson_weight": 1,
                "simpson_magnitude": 1.0
            }

        self.privacy_budget = privacy_budget
        self.search_params = train_params
        self.model = TabularVAEWeight(
            encoder_widths=layers_size,
            decoder_widths=layers_size[::-1],
            x_shape=x_shape,
            y_shape=y_shape,
            ordinal_feature_sizes=ordinal_feature_sizes,
            categorical_feature_sizes=categorical_feature_sizes,
            search_params=train_params
        )

        key, subkey = random.split(key)
        x = random.uniform(subkey, [np.prod(x_shape)])

        if y_shape != [0]:
            key, subkey = random.split(key)
            y = random.uniform(subkey, [np.prod(y_shape)])
        else:
            y = None

        if params:
            self.params = params
        else:
            key, subkey = random.split(key)
            self.params = self.model.init(key, x, y)["params"]
        self.search_params = train_params
        self.architecture = dict()
        self.architecture["layers_size"] = layers_size
        self.architecture["x_shape"] = x_shape
        self.architecture["y_shape"] = y_shape
        self.architecture["ordinal_feature_sizes"] = ordinal_feature_sizes
        self.architecture["categorical_feature_sizes"] = categorical_feature_sizes

        self.hashed_architecture = json.dumps(self.architecture)
        self.logger = logging.getLogger('engine.tabular_engine_weight.TabularEngineWeight')
        self.logger.info('creating an instance of TabularEngineWeight')
        self.logger.info('ordinal_feature_sizes: ' + str(ordinal_feature_sizes))
        self.logger.info('categorical_feature_sizes: ' + str(categorical_feature_sizes))

    def apply(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        return self.model.apply({"params": self.params}, x, y)

    def encode(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.model.apply({"params": self.params}, x, y, method=self.model.encode)

    def decode(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.model.apply({"params": self.params}, z, y, method=self.model.decode)

    def fit(
            self,
            train_ds: np.ndarray,
            y_train_ds: np.ndarray = None,
            epochs: int = 20,
            batch_size: int = 128,
            learning_rate: float = 1e-2
    ):
        weight_decay = self.search_params["weight_decay"]

        self.logger.info('Creating TrainState')
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        )

        if y_train_ds is not None:
            train_loader = np.hstack([train_ds, y_train_ds])
        else:
            train_loader = np.hstack([train_ds])

        splits = np.arange(batch_size, train_loader.shape[0], batch_size)

        self.logger.info('Starting fit loop')
        i = 1
        # data structure to record batch and epoch losses
        losses = []
        for _ in trange(epochs, desc="Engine fitting in progress", unit="epoch"):
            epoch_loss = 0
            batch_num = 0
            for batch in np.array_split(train_loader, splits, axis=0):
                state, loss_value = train_step(self.hashed_architecture, state, batch, self.search_params, tuple(self.search_params["balance_mask"]))
                batch_num += 1
                epoch_loss += loss_value
                losses.append({'type': 'batch', 'this_epoch': i, 'total_epochs': epochs, 'batch_num': batch_num,
                               'loss': loss_value})
                self.logger.debug(f"[E:{i}/{epochs} - B:{batch_num}] batch loss: {loss_value}")
            epoch_loss = epoch_loss / batch_num
            losses.append({'type': 'epoch', 'this_epoch': i, 'total_epochs': epochs, 'batch_num': batch_num, 'loss': epoch_loss})
            self.logger.debug(f"[E:{i}/{epochs} - B:{batch_num}] average epoch loss: {epoch_loss}")
            i += 1
        losses_df = pd.DataFrame(losses)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.logger.info(f"\n{losses_df[losses_df['type'] == 'epoch']}")
        losses_df[losses_df['type'] == 'epoch'].to_csv(f'{self.folder_path}epoch-losses.csv')
        self.params = state.params
        self.logger.info(f"self.params = \n{self.params}")
        self.logger.info('Fitting complete')

    def evaluate(
            self,
            test_ds: np.ndarray,
            y_test_ds: np.ndarray = None,
    ):
        if y_test_ds is not None:
            test_loader = np.hstack([test_ds, y_test_ds])
        else:
            test_loader = np.hstack([test_ds])

        metrics = eval(
            self.hashed_architecture, self.params, test_loader, self.search_params
        )

        return metrics

    def reconstruction_error(
            self,
            x: np.ndarray,
            y: np.ndarray = None,
    ):
        if y is not None:
            instances = np.hstack([x, y])
        else:
            instances = x

        reconstruction_error = np.empty(shape=instances.shape[0])
        i = 0
        for batch in np.array_split(instances, min(256, instances.shape[0]), axis=0):
            batch_reconstruction_error = eval(
                self.hashed_architecture, self.params, batch, self.search_params
            )["reconstruction_loss"]
            for reconstruction_error_i in batch_reconstruction_error:
                reconstruction_error[i] = reconstruction_error_i
                i += 1

        return np.asarray(reconstruction_error)

    def sample_from_latent_space(
            self,
            x: np.ndarray,
            ds: np.ndarray,
            y: np.ndarray = None,
            y_ds: np.ndarray = None,
            n_samples: int = 100,
    ):
        n_samples = min(n_samples, ds.shape[0] - 1)

        encoded_ds = self.encode(ds, y_ds)[0]
        encoded_x = self.encode(x, y)[0]

        distances = ((encoded_ds - encoded_x) ** 2).sum(axis=1) ** 0.5

        idx = np.argpartition(np.asarray(distances), n_samples)[0:n_samples]
        encoded_samples = encoded_ds[idx]

        return encoded_samples, idx

    def save(self, architecture_filename: str, sd_filename: str):
        state_dict = serialization.to_state_dict(self.params)
        np.save(sd_filename, state_dict)

        with open(architecture_filename, "w") as f:
            json.dump(self.architecture, f)
