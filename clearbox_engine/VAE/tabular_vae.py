import json
import jax
import jax.numpy as jnp
import numpy as np
rng = np.random.default_rng(42)
from functools import partial
from flax import linen as nn
from typing import Sequence, Tuple, Dict

from .vae import VAEInterface
import logging

# create logger
module_logger = logging.getLogger('cbx-engine.tabular_vae')

def jax_log_debug(fmt: str, *args, **kwargs):
    jax.debug.callback(
        lambda *args, **kwargs: module_logger.debug(fmt.format(*args, **kwargs)),
        *args, **kwargs, ordered=True)

def jax_log_info(fmt: str, *args, **kwargs):
    jax.debug.callback(
        lambda *args, **kwargs: module_logger.info(fmt.format(*args, **kwargs)),
        *args, **kwargs, ordered=True)

class Encoder(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        if y is not None:
            x = jnp.hstack([x, y])

        for i, feat in enumerate(self.features[:-1]):
            x = nn.sigmoid(nn.Dense(feat, name=f"layers_{i}")(x))
        mean_x = nn.Dense(self.features[-1], name="layers_mean")(x)
        log_var_x = nn.Dense(self.features[-1], name="layers_logvar")(x)

        return mean_x, log_var_x


class Decoder(nn.Module):
    features: Sequence[int]
    ordinal_feature_sizes: Sequence[int]
    categorical_feature_sizes: Sequence[int]

    @nn.compact
    def __call__(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if y is not None:
            z = jnp.hstack([z, y])

        for i, feat in enumerate(self.features[1:-1]):
            z = nn.sigmoid(nn.Dense(feat, name=f"layers_{i}")(z))
        z = nn.Dense(self.features[-1], name=f"layers_{len(self.features)-1}")(z)

        features_splitting_points = np.cumsum(
            self.ordinal_feature_sizes + self.categorical_feature_sizes
        )

        splitted_z = np.split(z, features_splitting_points, axis=(z.ndim - 1))[:-1]

        activations = []
        if len(self.ordinal_feature_sizes) > 0:
            activations.append(splitted_z[0])

        for categorical_tensor in splitted_z[len(self.ordinal_feature_sizes) :]:
            activations.append(nn.softmax(categorical_tensor, axis=(z.ndim - 1)))

        return jnp.hstack(activations)


class TabularVAE(VAEInterface, nn.Module):
    encoder_widths: Sequence[int]
    decoder_widths: Sequence[int]
    x_shape: Sequence[int]
    y_shape: Sequence[int]
    ordinal_feature_sizes: Sequence[int]
    categorical_feature_sizes: Sequence[int]
    search_params: Dict

    def setup(self):
        input_dim = np.prod(self.x_shape) + np.prod(self.y_shape)
        self.encoder = Encoder(self.encoder_widths)
        self.decoder = Decoder(
            self.decoder_widths + (input_dim,),
            self.ordinal_feature_sizes,
            self.categorical_feature_sizes,
        )
        self.logger = logging.getLogger('cbx-engine.tabular_vae.TabularVAE')

    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> Tuple:
        mean, logvar = self.encoder(x, y)
        latent = latent_space_sampling(mean, logvar)
        recon_x = self.decoder(latent, y)
        return recon_x, mean, logvar

    def encode(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        assert x.shape[1:] == self.x_shape

        x = jnp.reshape(x, (x.shape[0], -1))
        if y is not None:
            y = jnp.reshape(y, (y.shape[0], -1))

        return self.encoder(x, y)

    def decode(self, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        x = self.decoder(z, y)
        # x = jnp.reshape(x, (x.shape[0],) + self.x_shape + self.y_shape)
        return x


def latent_space_sampling(mean, log_variance):
    standard_deviation = jnp.exp(log_variance / 2.0)
    epsilon = rng.standard_normal(standard_deviation.shape)
    return mean + epsilon * standard_deviation


def compute_kernel(X, Y):
    X_size = X.shape[0]
    Y_size = Y.shape[0]
    dim = X.shape[1]
    X = jnp.expand_dims(X, axis=1)
    Y = jnp.expand_dims(Y, axis=0)
    tiled_X = jnp.broadcast_to(X, (X_size, Y_size, dim))
    tiled_Y = jnp.broadcast_to(Y, (X_size, Y_size, dim))
    kernel_input = jnp.power(tiled_X - tiled_Y, 2).mean(2) / float(dim)
    return jnp.exp(-kernel_input)


def compute_mmd(X, Y):
    XX = compute_kernel(X, X)
    YY = compute_kernel(Y, Y)
    XY = compute_kernel(X, Y)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


@jax.vmap
def cross_entropy_loss(logs, targets):
    nll = jnp.take_along_axis(logs, jnp.expand_dims(targets, axis=0), axis=0)
    ce = -jnp.mean(nll)
    return ce


def compute_metrics(architecture, recon_x, x, mean, logvar, search_params):
    features_splitting_points = np.cumsum(
        architecture["ordinal_feature_sizes"]
        + architecture["categorical_feature_sizes"]
    )

    span_numerical = len(architecture["ordinal_feature_sizes"])
    splitted_z = jnp.split(recon_x, features_splitting_points, axis=1)[:-1]
    splitted_x = jnp.split(x, features_splitting_points, axis=1)[:-1]

    if len(architecture["ordinal_feature_sizes"]) > 0:
        gauss_sigmas = jnp.zeros(
            sum(architecture["ordinal_feature_sizes"])
        ) + jnp.array([search_params["gauss_s"]])

        loss_ordinal = jnp.square(splitted_z[0] - splitted_x[0])
        #loss_ordinal = jnp.sum(jnp.divide(loss_ordinal, gauss_sigmas), 1)
        loss_ordinal = jnp.sum(jnp.mean(jnp.divide(loss_ordinal, gauss_sigmas), 0))
    else:
        loss_ordinal = 0

    CE_categorical = 0

    if len(architecture["categorical_feature_sizes"]) > 0:
        for original_categorical, reconstructed_categorical in zip(
            splitted_x[span_numerical:], splitted_z[span_numerical:]
        ):
            targets = jnp.argmax(original_categorical, axis=1)
            CE_categorical += cross_entropy_loss(
                jnp.log(reconstructed_categorical + 1e-6), targets
            )

    normal_samples = rng.standard_normal(mean.shape)
    mmd_regularizer = compute_mmd(normal_samples, mean)

    loss_kld = -0.5 * jnp.mean(1 + logvar - mean ** 2 - jnp.exp(logvar))

    loss = (
        jnp.mean(loss_ordinal + CE_categorical)
        + search_params["alpha"] * loss_kld
        + search_params["beta"] * mmd_regularizer
    )

    reconstruction_loss = loss_ordinal + CE_categorical

    return {
        "loss": loss,
        "mean_reconstruction_loss": jnp.mean(reconstruction_loss),
        "reconstruction_loss": reconstruction_loss,
        "loss_ordinal": loss_ordinal,
        "loss_cat": CE_categorical,
    }

@partial(jax.jit, static_argnums=(0,))
def train_step(hashed_architecture, state, batch, search_params):
    # module_logger.info("Start train step")
    architecture = json.loads(hashed_architecture)

    y_batch = None
    if architecture["y_shape"] != [0]:
        y_batch = jnp.split(
            batch,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(batch.ndim - 1),
        )[1]

        x_batch = jnp.split(
            batch,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(batch.ndim - 1),
        )[0]
    else:
        x_batch = batch

    def loss_fn(params):
        # module_logger.info("Start loss_fn")
        VAE = TabularVAE(
            encoder_widths=architecture["layers_size"],
            decoder_widths=architecture["layers_size"][::-1],
            x_shape=architecture["x_shape"],
            y_shape=architecture["y_shape"],
            ordinal_feature_sizes=architecture["ordinal_feature_sizes"],
            categorical_feature_sizes=architecture["categorical_feature_sizes"],
            search_params=search_params,
        )
        recon_x, mean, logvar = VAE.apply({"params": params}, x_batch, y_batch)
        span_numerical = len(architecture["ordinal_feature_sizes"])

        features_splitting_points = np.cumsum(
            architecture["ordinal_feature_sizes"]
            + architecture["categorical_feature_sizes"]
        )

        splitted_z = jnp.split(recon_x, features_splitting_points, axis=1)[:-1]
        splitted_x = jnp.split(x_batch, features_splitting_points, axis=1)[:-1]
        # jax_log_debug("splitted_z.len: {x}", x=len(splitted_z))
        # jax_log_debug("splitted_x.len: {x}", x=len(splitted_x))

        if len(architecture["ordinal_feature_sizes"]) > 0:
            gauss_sigmas = jnp.zeros(
                sum(architecture["ordinal_feature_sizes"])
            ) + jnp.array([2.0 * search_params["gauss_s"] * search_params["gauss_s"]])

            loss_ordinal = jnp.square(splitted_z[0] - splitted_x[0])
            loss_ordinal = jnp.sum(jnp.divide(loss_ordinal, gauss_sigmas), 1)
        else:
            loss_ordinal = 0

        CE_categorical = 0
        module_logger.debug(
            f'len(architecture["categorical_feature_sizes"]) = {len(architecture["categorical_feature_sizes"])}')
        if len(architecture["categorical_feature_sizes"]) > 0:
            for original_categorical, reconstructed_categorical in zip(
                splitted_x[span_numerical:], splitted_z[span_numerical:]
            ):
                # targets = jnp.argmax(original_categorical, axis=1)
                # CE_categorical += cross_entropy_loss(
                #     jnp.log(reconstructed_categorical + 1e-6), targets
                # )
                gauss_sigmas = jnp.zeros(
                    reconstructed_categorical.shape[1]) + jnp.array([2.0 * search_params["gauss_s_c"] * search_params["gauss_s_c"]])
                jax_log_debug("reconstructed_categorical.shape: {x}", x=reconstructed_categorical.shape)
                # jax_log_debug("original_categorical.shape: {x}", x=original_categorical.shape)
                jax_log_debug("reconstructed_categorical[0]: {x}", x=reconstructed_categorical[0])
                # jax_log_debug("original_categorical[0]: {x}", x=original_categorical[0])

                loss_cat = jnp.square(reconstructed_categorical - original_categorical)
                loss_cat = jnp.sum(jnp.mean(jnp.divide(loss_cat, gauss_sigmas), 0))

                # loss_cat_noise = jnp.sum(jnp.mean(jnp.clip(reconstructed_categorical-search_params["prob_clip"], 0.), 0))
                loss_cat_noise = jnp.sum(jnp.clip(reconstructed_categorical - search_params["prob_clip"], 0.))
                
                jax_log_debug("reconstructed_categorical: {rc}", rc=reconstructed_categorical)
                jax_log_debug("loss_cat:{lc}", lc=loss_cat)
                
                CE_categorical += loss_cat + loss_cat_noise

        # normal_samples = np.random.randn(*mean.shape)
        # mmd_regularizer = compute_mmd(normal_samples, mean)

        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params])
        weight_penalty = search_params["l2_reg"] * 0.5 * weight_l2
        # loss_PCE = jnp.mean(mean ** 2) + jnp.mean((jnp.exp(logvar / 2.0)-1.0)**2)
        loss_kld = -0.5 * jnp.mean(1 + logvar - mean ** 2 - jnp.exp(logvar))

        loss = (
            jnp.mean(loss_ordinal + CE_categorical)
            + weight_penalty
            + search_params["alpha"] * loss_kld
            # + search_params["beta"] * mmd_regularizer
        )
        loss_first_term = jnp.mean(loss_ordinal + CE_categorical)
        jax_log_debug("mean(ord+cat):{mean} (ord:{o} + cat:{c})",
                      mean=loss_first_term, o=jnp.mean(loss_ordinal), c=CE_categorical)
        jax_log_debug("weight_penalty:{wp} + alpha*loss_kld: {a}*{kld}",
                      wp=weight_penalty, a=search_params["alpha"], kld=loss_kld)
        return loss

    loss_value = loss_fn(state.params)
    grads = jax.grad(loss_fn)(state.params)
    # jax_log_debug("grads: {g}", g=grads)
    return state.apply_gradients(grads=grads), loss_value
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # , allow_int=True)
    # _, grad = grad_fn(optimizer.target)
    # optimizer = optimizer.apply_gradient(grad)
    # return optimizer


@partial(jax.jit, static_argnums=(0,))
def eval(hashed_architecture, model, eval_ds, search_params):
    module_logger.info("eval start")
    architecture = json.loads(hashed_architecture)

    y_batch = None
    if architecture["y_shape"] != [0]:
        y_batch = jnp.split(
            eval_ds,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(eval_ds.ndim - 1),
        )[1]

        eval_ds = jnp.split(
            eval_ds,
            np.cumsum(architecture["x_shape"] + architecture["y_shape"]),
            axis=(eval_ds.ndim - 1),
        )[0]
    else:
        eval_ds = eval_ds

    VAE = TabularVAE(
        encoder_widths=architecture["layers_size"],
        decoder_widths=architecture["layers_size"][::-1],
        x_shape=architecture["x_shape"],
        y_shape=architecture["y_shape"],
        ordinal_feature_sizes=architecture["ordinal_feature_sizes"],
        categorical_feature_sizes=architecture["categorical_feature_sizes"],
        search_params=search_params,
    )
    recon_xs, mean, logvar = VAE.apply({"params": model}, eval_ds, y_batch)

    return compute_metrics(architecture, recon_xs, eval_ds, mean, logvar, search_params)
