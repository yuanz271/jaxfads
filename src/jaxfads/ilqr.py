from functools import partial

import jax
from jax import lax, vmap
from jax import numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def rollout(x0, u, c, f):
    """
    Roll out a trajectory under given controls and dynamics.

    Parameters
    ----------
    x0 : Array, shape (D_x,)
        Initial state.
    u : Array, shape (T, D_u)
        Control sequence.
    c : Array, shape (T, D_c)
        Covariate sequence.
    f : callable
        Dynamics function mapping `(x_t, u_t, c_t) -> x_{t+1}`.

    Returns
    -------
    Array, shape (T, D_x)
        State trajectory excluding the final x_{T+1}.
    """
    def step(x_t, inputs):
        u_t, c_t = inputs
        x_tp1 = f(x_t, u_t, c_t)
        return x_tp1, x_t

    _, x = lax.scan(step, init=x0, xs=(u, c))

    return x


def cost_function(x, u, target, Q, R):
    """
    Quadratic trajectory cost relative to a target.

    Parameters
    ----------
    x : Array, shape (T, D_x)
        State trajectory.
    u : Array, shape (T, D_u)
        Control trajectory.
    target : Array, shape (T, D_x)
        Target state trajectory.
    Q : Array, shape (D_x, D_x)
        State cost matrix (PD).
    R : Array, shape (D_u, D_u)
        Control cost matrix (PD).

    Returns
    -------
    Array
        Scalar total cost 0.5 * sum_t(||x_t - target_t||_Q^2 + ||u_t||_R^2).
    """
    Qcho = jnp.linalg.cholesky(Q, upper=True)
    Rcho = jnp.linalg.cholesky(R, upper=True)

    def step_cost(x, u, target, Qcho, Rcho):
        vx = Qcho @ x - target
        vu = Rcho @ u
        return jnp.inner(vx, vx) + jnp.inner(vu, vu)

    costs = vmap(step_cost)(x, u, target, Qcho, Rcho)

    return jnp.sum(0.5 * costs)


def backward_pass(
    x,
    u,
    c,
    target,
    Q,
    R,
    Df,
    initial_damping=1e-3,
    damping_inc=2.0,
    damping_dec=0.5,
    min_damping=1e-6,
    max_damping=1e6,
):
    """
    Compute iLQR gains via a backward dynamic programming pass.

    Parameters
    ----------
    x : Array, shape (T, D_x)
        Current state rollout.
    u : Array, shape (T, D_u)
        Current control rollout.
    c : Array, shape (T, D_c)
        Covariates.
    target : Array, shape (T, D_x)
        Target states.
    Q : Array, shape (D_x, D_x)
        State cost matrix.
    R : Array, shape (D_u, D_u)
        Control cost matrix.
    Df : callable
        Linearization function returning `(F_x, F_u)` at `(x_t, u_t, c_t)`.
    initial_damping : float, default=1e-3
        Initial Levenberg–Marquardt damping.
    damping_inc : float, default=2.0
        Multiplicative increase factor when damping is insufficient.
    damping_dec : float, default=0.5
        Multiplicative decrease factor after a successful step.
    min_damping : float, default=1e-6
        Lower bound on damping.
    max_damping : float, default=1e6
        Upper bound on damping.

    Returns
    -------
    k : Array, shape (T, D_u)
        Feedforward control gains.
    K : Array, shape (T, D_u, D_x)
        Feedback control gains.
    """
    x_size = jnp.size(x, -1)  # state dimension
    u_size = jnp.size(u, -1)  # control dimension
    eye_u = jnp.eye(u_size)

    damping = initial_damping

    def step(carry, inputs):
        V_x, V_xx, damping = carry
        x_t, u_t, c_t, z_t, Q, R = inputs

        # Stage cost derivatives.
        C_x = Q @ (x_t - z_t)
        C_u = R @ u_t
        C_xx = Q
        C_uu = R
        C_ux = jnp.zeros((u_size, x_size))

        # Dynamics linearization (assuming you have a function that returns F_x and F_u).
        F_x, F_u = Df(x_t, u_t, c_t)  # User-defined for your system.

        # Compute Q-function derivatives.
        Q_x = C_x + F_x.T @ V_x
        Q_u = C_u + F_u.T @ V_x
        Q_xx = C_xx + F_x.T @ V_xx @ F_x
        Q_ux = C_ux + F_u.T @ V_xx @ F_x
        Q_uu = C_uu + F_u.T @ V_xx @ F_u

        def compute_gain(Q_uu, damping):
            Q_uu_reg = Q_uu + damping * eye_u
            Q_uu_cho = cho_factor(Q_uu_reg)

            # Compute control gains.
            def on_invalid_damping():
                return jnp.zeros_like(Q_u), jnp.zeros_like(Q_ux)

            def on_valid_damping():
                return -cho_solve(Q_uu_cho, Q_u), -cho_solve(Q_uu_cho, Q_ux)

            k_t, K_t = lax.cond(
                damping > max_damping, on_invalid_damping, on_valid_damping
            )

            return k_t, K_t

        def failed(vals):
            damping, k_t, K_t = vals
            return jnp.logical_or(jnp.any(jnp.isnan(k_t)), jnp.any(jnp.isnan(K_t)))

        def damp(vals):
            damping, k_t, K_t = vals
            damping = damping * damping_inc
            # jax.debug.print("damping = {damping}", damping=damping)
            k_t, K_t = compute_gain(Q_uu, damping)

            return damping, k_t, K_t

        k_t, K_t = compute_gain(Q_uu, damping)

        damping, k_t, K_t = lax.while_loop(failed, damp, (damping, k_t, K_t))

        # Update the value function approximations.
        V_x = Q_x + K_t.T @ Q_u
        V_xx = Q_xx + K_t.T @ Q_ux
        # Ensure symmetry for numerical stability.
        V_xx = 0.5 * (V_xx + V_xx.T)

        # Optionally, decrease damping if step looks promising.
        damping = jnp.maximum(damping * damping_dec, min_damping)

        return (V_x, V_xx, damping), (k_t, K_t)

    # Terminal cost derivatives.
    V_x = Q[-1] @ (x[-1] - target[-1])  # Gradient at final step.
    V_xx = Q[-1]  # Hessian at final step.

    _, (k, K) = lax.scan(
        step, init=(V_x, V_xx, damping), xs=(x, u, c, target, Q, R), reverse=True
    )

    return k, K


def line_search(alpha, k, K, x_optimal, u_optimal, c, target, Q, R, f):
    """
    Perform line search over step size to update controls.

    Parameters
    ----------
    alpha : float
        Step-size scalar in [0, 1].
    k : Array, shape (T, D_u)
        Feedforward gains.
    K : Array, shape (T, D_u, D_x)
        Feedback gains.
    x_optimal : Array, shape (T, D_x)
        Current best trajectory.
    u_optimal : Array, shape (T, D_u)
        Current best controls.
    c : Array, shape (T, D_c)
        Covariates.
    target : Array, shape (T, D_x)
        Target trajectory.
    Q : Array, shape (D_x, D_x)
        State cost matrix.
    R : Array, shape (D_u, D_u)
        Control cost matrix.
    f : callable
        Dynamics function.

    Returns
    -------
    new_cost : Array
        Scalar cost after applying the step.
    x_new : Array, shape (T, D_x)
        New state trajectory.
    u_new : Array, shape (T, D_u)
        New control trajectory.
    """
    def step(carry, inputs):
        x_t = carry
        k_t, K_t, x_opt, u_opt, c_t = inputs
        dx = x_t - x_opt
        u_t = u_opt + alpha * k_t + K_t @ dx
        x_tp1 = f(x_t, u_t, c_t)

        return x_tp1, (x_t, u_t)

    _, (x_new, u_new) = lax.scan(
        step, init=x_optimal[0], xs=(k, K, x_optimal, u_optimal, c)
    )

    new_cost = cost_function(x_new, u_new, target, Q, R)

    return new_cost, x_new, u_new


def arg_first_less_than(array, value):
    """
    Return index of first element less than a value or -1 if none.

    Parameters
    ----------
    array : Array, shape (N,)
        Input array.
    value : float
        Comparison threshold.

    Returns
    -------
    int
        Index of first element < value, or -1 if none.
    """
    arr = array < value
    return lax.cond(jnp.any(arr), jnp.argmax, lambda _: -1, arr)


def arg_smallest_less_than(array, value):
    """
    Return index of smallest element less than a value or -1 if none.

    Parameters
    ----------
    array : Array, shape (N,)
        Input array.
    value : float
        Comparison threshold.

    Returns
    -------
    int
        Index of smallest element < value, or -1 if none.
    """
    arr = array < value
    return lax.cond(jnp.any(arr), jnp.argmin, lambda _: -1, array)


def ilqr(x0, u, c, target, Q, R, f, Df, max_iter=100, tol=1e-6, verbose=False):
    """
    Iterative Linear-Quadratic Regulator (iLQR) optimization.

    Parameters
    ----------
    x0 : Array, shape (D_x,)
        Initial state.
    u : Array, shape (T, D_u)
        Initial control sequence.
    c : Array, shape (T, D_c)
        Covariate sequence.
    target : Array, shape (T, D_x)
        Target state trajectory.
    Q : Array, shape (D_x, D_x)
        State cost matrix (PD).
    R : Array, shape (D_u, D_u)
        Control cost matrix (PD).
    f : callable
        Dynamics function mapping `(x_t, u_t, c_t) -> x_{t+1}`.
    Df : callable
        Linearization function returning `(F_x, F_u)` at `(x_t, u_t, c_t)`.
    max_iter : int, default=100
        Maximum number of optimization iterations.
    tol : float, default=1e-6
        Convergence tolerance on cost improvement.
    verbose : bool, default=False
        Whether to print iteration logs via `jax.debug.print`.

    Returns
    -------
    u_optimal : Array, shape (T, D_u)
        Optimized control sequence.
    x_optimal : Array, shape (T, D_x)
        Corresponding state trajectory.
    success : bool
        Whether a successful line search step occurred in the last iteration.
    """
    # alphas = jnp.logspace(0, 15, num=50, base=0.5)
    alphas = 1 / jnp.geomspace(1, 1e5, num=50)

    u_optimal = u

    T = len(u)
    Q = jnp.broadcast_to(Q, (T,) + (Q.shape[-2:]))
    R = jnp.broadcast_to(R, (T,) + (R.shape[-2:]))

    x_optimal = rollout(x0, u_optimal, c, f)
    cost = cost_function(x_optimal, u_optimal, target, Q, R)

    def not_converged(carry):
        i, converge, success, cost, x_optimal, u_optimal = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converge))

    def do_iteration(carry):
        i, _, _, cost, x_optimal, u_optimal = carry
        k, K = backward_pass(x_optimal, u_optimal, c, target, Q, R, Df)
        new_costs, x_new, u_new = vmap(
            partial(
                line_search,
                k=k,
                K=K,
                x_optimal=x_optimal,
                u_optimal=u_optimal,
                c=c,
                target=target,
                Q=Q,
                R=R,
                f=f,
            )
        )(alphas)

        idx = arg_first_less_than(new_costs, cost)  # stable, prefer large alpha
        # idx = arg_smallest_less_than(new_costs, cost)  # unstable, prefer large improvement

        def on_failure(idx, alphas, cost, new_costs, x_new, u_new):
            return False, cost, jnp.nan, x_optimal, u_optimal

        def on_success(idx, alphas, cost, new_costs, x_new, u_new):
            return True, new_costs[idx], alphas[idx], x_new[idx], u_new[idx]

        success, new_cost, alpha, x_optimal, u_optimal = lax.cond(
            idx < 0, on_failure, on_success, idx, alphas, cost, new_costs, x_new, u_new
        )

        converge = abs(cost - new_cost) < tol

        if verbose:
            jax.debug.print(
                "Iteration {iteration}: cost = {cost:.4f}, new_cost = {new_cost:.4f}, alpha = {alpha:.4f}, line search = {success}, converge = {converge}",
                iteration=i,
                cost=cost,
                new_cost=new_cost,
                alpha=alpha,
                success=success,
                converge=converge,
            )

        return i + 1, converge, success, new_cost, x_optimal, u_optimal

    carry = lax.while_loop(
        not_converged, do_iteration, (0, False, False, cost, x_optimal, u_optimal)
    )

    i, converge, success, cost, x_optimal, u_optimal = carry

    return u_optimal, x_optimal, success
