from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, random as jrnd, lax, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
import optax
import equinox as eqx
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from . import vi


def training_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TextColumn("•"),
        "Elapsed",
        TimeElapsedColumn(),
        TextColumn("•"),
        "Remainning",
        TimeRemainingColumn(),
        TextColumn("•"),
        "Loss",
        TextColumn("{task.fields[loss]:.3f}"),
        "MA Loss",
        TextColumn("{task.fields[mean]:.3f}"),
    )


@dataclass
class Opt:
    min_iter: int = 50
    max_iter: int = 50
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1
    weight_decay: float = 1e-3
    beta: float = 0.95
    seed: int = 0
    noise_eta: float = 0.5
    noise_gamma: float = 0.8
    valid_ratio: float = 0.2
    validation_size: int = 80


# def make_optimizer(model, opt: Opt):
#     optimizer = optax.chain(
#         optax.clip_by_global_norm(opt.clip_norm),
#         optax.add_noise(opt.noise_eta, opt.noise_gamma, opt.seed),
#         optax.scale_by_adam(),
#         optax.add_decayed_weights(opt.weight_decay),
#         optax.scale_by_learning_rate(opt.learning_rate),
#     )
#     opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

#     return optimizer, opt_state


# def train(
#     model: smoother.XFADS,
#     data,
#     *,
#     seed: int,
#     opt: Opt,
# ) -> smoother.XFADS:
#     chex.assert_equal_shape(data, dims=(0, 1))

#     def batch_elbo(model, key, ts, moment_s, moment_p, ys) -> Array:
#         _elbo = jax.vmap(
#             jax.vmap(
#                 partial(
#                     vi.elbo,
#                     eloglik=model.likelihood.eloglik,
#                     approx=model.hyperparam.approx,
#                     mc_size=model.hyperparam.mc_size,
#                 )
#             )
#         )  # (batch, seq)

#         keys = jrnd.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)

#         return _elbo(keys, ts, moment_s, moment_p, ys)

#     def batch_loss(model, key, tb, yb, ub, cb) -> Scalar:
#         key, subkey = jrnd.split(key)
#         chex.assert_equal_shape((tb, yb, ub, cb), dims=(0, 1))

#         key, subkey = jrnd.split(key)
#         _, moment, moment_p = model(tb, yb, ub, cb, key=subkey)

#         key, subkey = jrnd.split(key)
#         free_energy = -batch_elbo(model, subkey, tb, moment, moment_p, yb)

#         loss = (
#             jnp.mean(free_energy)
#             + model.hyperparam.noise_penalty * model.forward.loss()
#             # + hyperparam.noise_penalty * model.backward.loss()
#         )

#         return loss

#     def optimize(model, seed, batch_loss_func, *data):
#         rng = np.random.default_rng(seed)
#         key = jrnd.key(seed)

#         n_devices = len(jax.devices())

#         n: int = np.size(data[0], 0)
#         perm = rng.permutation(n)
#         n_samples_per_device = int(n * opt.valid_ratio / n_devices)
#         n_valid = n_samples_per_device * n_devices  # n_devices-dividible for sharding

#         indices_valid, indices_training = perm[:n_valid], perm[n_valid:]
#         valid_set = tuple(jnp.asarray(arr[indices_valid]) for arr in data)
#         training_set = tuple(jnp.asarray(arr[indices_training]) for arr in data)

#         mesh = jax.make_mesh((n_devices,), ("batch",))

#         optimizer, opt_state = make_optimizer(model, opt)

#         @eqx.filter_jit
#         def evaluate(model, key, data):
#             model = eqx.nn.inference_mode(model)
#             return shard_map(
#                 lambda *d: batch_loss(model, key, *d),
#                 mesh,
#                 in_specs=(
#                     P("batch", None),
#                     P("batch", None, None),
#                     P("batch", None, None),
#                     P("batch", None, None),
#                 ),
#                 out_specs=P(),
#                 check_rep=False,
#             )(*data)

#         key, eval_key = jrnd.split(key)
#         batch_size = opt.batch_size
#         min_iter = opt.min_iter
#         max_iter = opt.max_iter
#         valid_loss = evaluate(model, eval_key, valid_set)

#         with training_progress() as pbar:
#             task_id = pbar.add_task("Fitting", total=opt.max_iter, loss=valid_loss)

#             # lax.scan cannot handle equinox Module. see https://docs.kidger.site/equinox/faq/
#             params, static = eqx.partition(model, eqx.is_inexact_array)

#             def should_continue(carry):
#                 i, converged, mean_loss, params, opt_state, key = carry
#                 return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

#             def do_epoch(carry):
#                 i, converged, mean_loss, params, opt_state, key = carry
#                 key, subkey = jrnd.split(key)
#                 N = jnp.size(training_set[0], 0)
#                 perm = jax.random.permutation(subkey, N)

#                 def batch_step(carry, batch_idx):
#                     params, opt_state, key = carry
#                     model = eqx.combine(params, static)
#                     batch_indices = jax.lax.dynamic_slice_in_dim(
#                         perm, batch_idx * batch_size, batch_size
#                     )
#                     batch = tuple(arr[batch_indices] for arr in training_set)

#                     # compute gradients
#                     key, subkey = jrnd.split(key)
#                     loss, grads = eqx.filter_value_and_grad(batch_loss_func)(
#                         model, subkey, *batch
#                     )

#                     # update parameters
#                     updates, opt_state = optimizer.update(grads, opt_state, model)
#                     model = eqx.apply_updates(model, updates)
#                     params, _ = eqx.partition(model, eqx.is_inexact_array)

#                     return (params, opt_state, key), None

#                 (params, opt_state, key), _ = jax.lax.scan(
#                     batch_step, (params, opt_state, key), jnp.arange(N // batch_size)
#                 )
#                 model = eqx.combine(params, static)

#                 key, subkey = jrnd.split(key)
#                 valid_loss = evaluate(model, subkey, valid_set)
#                 jax.debug.callback(
#                     lambda x: pbar.update(task_id, advance=1, loss=x), valid_loss
#                 )

#                 converged = jnp.logical_and(
#                     i > min_iter, jnp.isclose(valid_loss, mean_loss)
#                 )
#                 mean_loss = ((mean_loss * i + 1) + valid_loss) / (i + 2)

#                 return i + 1, converged, mean_loss, params, opt_state, key

#             i, converged, mean_loss, params, opt_state, key = lax.while_loop(
#                 should_continue,
#                 do_epoch,
#                 (0, False, valid_loss, params, opt_state, key),
#             )

#             model = eqx.combine(params, static)

#         return model

#     model = optimize(model, seed, batch_loss, *data)

#     return model


# >>> Full JAX
def train_test_split(arrays, *, rng, test_ratio=None, test_size=None, train_size=None):
    data_size = arrays[0].shape[0]
    if test_size is None:
        test_size = int(test_ratio * data_size)
    if train_size is None:
        train_size = data_size - test_size
    perm = rng.permutation(data_size)

    return tuple(
        array[perm[test_size : train_size + test_size]] for array in arrays
    ), tuple(array[perm[:test_size]] for array in arrays)


def to_shard(arrays, sharding=None):
    return tuple(jax.device_put(arr, sharding) for arr in arrays)


def batch_elbo(model, key, ts, moment_s, moment_p, ys) -> Array:
    _elbo = jax.vmap(
        jax.vmap(
            partial(
                vi.elbo,
                eloglik=model.likelihood.eloglik,
                approx=model.hyperparam.approx,
                mc_size=model.hyperparam.mc_size,
            )
        )
    )  # (batch, seq)

    keys = jrnd.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)

    return _elbo(keys, ts, moment_s, moment_p, ys)


def train_fast(model, data, *, conf):
    key = jrnd.key(conf.seed)
    rng = np.random.default_rng(conf.seed)

    # >>> Prepare data
    n_devices = len(jax.devices())
    mesh = jax.make_mesh((n_devices,), ("batch",))
    sharding = NamedSharding(mesh, P("batch"))

    # batch size is required to be multiple of the number of devices
    # validation size is required to be multile of batch_size

    data_size = len(data[0])
    batch_size = int(conf.batch_size / n_devices) * n_devices
    valid_size = int(conf.validation_size / batch_size) * batch_size
    train_size = int((data_size - valid_size) / batch_size) * batch_size

    train_set, valid_set = train_test_split(
        data, rng=rng, test_size=valid_size, train_size=train_size
    )

    train_set = to_shard(train_set, sharding)
    valid_set = to_shard(valid_set, sharding)
    # <<<

    # >>> Prepare optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(conf.clip_norm),
        optax.add_noise(conf.noise_eta, conf.noise_gamma, conf.seed),
        optax.scale_by_adam(),
        optax.add_decayed_weights(conf.weight_decay),
        optax.scale_by_learning_rate(conf.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    # <<<

    # Define loss and grad
    @eqx.filter_jit
    def batch_loss(model, batch, key):
        tb, yb, ub, cb = batch

        key, ky = jrnd.split(key)
        _, moment, moment_p = model(tb, yb, ub, cb, key=ky)

        key, ky = jrnd.split(key)
        free_energy = -batch_elbo(model, ky, tb, moment, moment_p, yb)

        loss = (
            jnp.mean(free_energy)
            + model.hyperparam.noise_penalty * model.forward.loss()
            # + hyperparam.noise_penalty * model.backward.loss()
        )

        return loss

    @eqx.filter_jit
    def batch_grad_step(model, opt_state, batch, key):
        lss, grads = eqx.filter_value_and_grad(batch_loss)(
            model, batch, key
        )  # The gradient will be computed with respect to all floating-point JAX/NumPy arrays in the first argument
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return lss, model, opt_state

    # Main loop
    key, cey = jrnd.split(key)
    perm = jrnd.permutation(cey, train_size)
    min_iter = conf.min_iter
    max_iter = conf.max_iter
    beta = conf.beta

    with training_progress() as pbar:
        key, cey = jrnd.split(key)
        valid_loss = lax.stop_gradient(
            batch_loss(eqx.nn.inference_mode(model), valid_set, cey)
        )
        task_id = pbar.add_task(
            "Training", total=max_iter, loss=valid_loss, mean=valid_loss
        )

        params, static = eqx.partition(model, eqx.is_inexact_array)

        def train_cond(carry):
            params, opt_state, i, converged, *_ = carry
            return jnp.logical_and(
                i < max_iter, jnp.logical_or(jnp.logical_not(converged), i < min_iter)
            )

        def train_step(carry):
            params, opt_state, i, converged, mean_loss, key, idx, perm = carry

            def new_perm(key, perm, idx):
                perm = jrnd.permutation(key, train_size)
                return perm, 0

            def old_perm(key, perm, idx):
                return perm, idx

            key, cey = jrnd.split(key)
            perm, idx = lax.cond(
                idx + batch_size >= train_size, new_perm, old_perm, cey, perm, idx
            )

            model = eqx.combine(params, static)
            batch_idx = lax.dynamic_slice_in_dim(perm, idx, batch_size)
            batch = tuple(arr[batch_idx] for arr in train_set)
            key, cey = jrnd.split(key)
            _, model, opt_state = batch_grad_step(model, opt_state, batch, cey)

            key, cey = jrnd.split(key)
            valid_loss = lax.stop_gradient(
                batch_loss(eqx.nn.inference_mode(model), valid_set, cey)
            )
            jax.debug.callback(
                lambda vl, ml: pbar.update(task_id, advance=1, loss=vl, mean=ml),
                valid_loss,
                mean_loss,
            )

            params, _ = eqx.partition(model, eqx.is_inexact_array)
            converged = jnp.logical_and(
                jnp.isclose(mean_loss, valid_loss), valid_loss <= mean_loss
            )
            mean_loss = mean_loss * beta + valid_loss * (1 - beta)

            return (
                params,
                opt_state,
                i + 1,
                converged,
                mean_loss,
                key,
                idx + batch_size,
                perm,
            )

        key, cey = jrnd.split(key)
        params, *_ = lax.while_loop(
            train_cond,
            train_step,
            (params, opt_state, 0, False, valid_loss, cey, 0, perm),
        )
        model = eqx.combine(params, static)

    return model
