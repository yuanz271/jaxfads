from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, vmap
# import matplotlib.pyplot as plt

from jaxfads.ilqr import ilqr, cost_function, backward_pass


def test_ilqr():
    def dynamics(x, u, c):
        return x + (0.5 * x * x + u) * 0.1

    dt = 0.1
    T = 50

    rng = np.random.default_rng(0)

    # Initial state
    x0 = rng.standard_normal(size=(5, 2)) * 0.1
    # Define a target trajectory (T+1 entries including terminal target)
    target = np.array(
        [np.array([np.sin(0.1 * k), np.cos(0.1 * k)]) for k in range(T + 1)]
    )
    # Initialize the control sequence; here control dimension is 1.
    u_init = np.zeros((5, T, 2))
    c = jnp.full((T, 1), fill_value=dt)
    # Define cost matrices:
    Q = np.eye(2)
    R = np.eye(2) * 0.01
    jac = jax.jacobian(dynamics, argnums=(0, 1))

    # Run iLQR:
    pilqr = jax.jit(
        partial(
            ilqr,
            c=c,
            target=target,
            Q=Q,
            R=R,
            f=dynamics,
            Df=jac,
            max_iter=10,
            verbose=True,
        )
    )
    vilqr = vmap(pilqr)

    # u_opt, x_opt = pilqr(x0[1], u_init[1])

    u_opt, x_opt, _ = vilqr(x0, u_init)

    # Plotting position tracking
    # time = np.linspace(0, T * dt, T + 1)
    # plt.figure(figsize=(10, 5))
    # plt.plot(*target.T, label="Target Trajectory", linewidth=1)
    # for x in x_opt:
    #     plt.plot(*x.T, label="Optimized Trajectory", linewidth=1)
    # # plt.plot(*x_opt.T, label="Optimized Trajectory", linewidth=1)
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    # plt.title("Trajectory Tracking (Higher-Dimensional System)")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("test_copilot.pdf")
    # plt.close()


def test_cost_function_matches_quadratic_form():
    """cost_function must compute stage + terminal quadratic costs.

    Stage: 0.5 * sum_{t=0}^{T-1} [(x_t-target_t)^T Q_t (x_t-target_t) + u_t^T R_t u_t]
    Terminal: 0.5 * (x_T - target_T)^T Q_T (x_T - target_T)
    """
    T, D_x, D_u = 4, 2, 1
    # T+1 states, T controls
    x = jnp.array([[1.0, 2.0], [0.5, 1.5], [0.0, 1.0], [-0.5, 0.5], [-1.0, 0.0]])
    u = jnp.ones((T, D_u))
    target = jnp.zeros((T + 1, D_x))
    Q = jnp.broadcast_to(jnp.diag(jnp.array([2.0, 3.0])), (T + 1, D_x, D_x))
    R = jnp.broadcast_to(jnp.eye(D_u) * 0.5, (T, D_u, D_u))

    cost = cost_function(x, u, target, Q, R)

    # Manual computation: stage costs + terminal cost
    expected = 0.0
    for t in range(T):
        dx = x[t] - target[t]
        expected += 0.5 * (dx @ Q[t] @ dx + u[t] @ R[t] @ u[t])
    # Terminal cost (state only)
    dx_T = x[T] - target[T]
    expected += 0.5 * (dx_T @ Q[T] @ dx_T)
    expected = jnp.array(expected)

    np.testing.assert_allclose(float(cost), float(expected), rtol=1e-5)


def test_backward_pass_analytical_1d():
    """Verify feedback and feedforward gains against hand-derived Riccati solution.

    System: x_{t+1} = x_t + u_t  (1D integrator, a=1, b=1, no covariates)
    T = 3 control steps, T+1 = 4 states.
    q = 2, r = 1.

    Uses a dynamically-consistent non-zero trajectory so that x[T] != x[T-1],
    ensuring that the terminal value function is initialised from the correct
    state (x_T, not x_{T-1}).

    Riccati recursion for P (value Hessian):
        P_T = q = 2                              (terminal)
        P_t = q + P_{t+1} * r / (r + P_{t+1})

        P_2 = 2 + 2/3          = 8/3
        P_1 = 2 + (8/3)/(11/3) = 30/11
        P_0 = 2 + (30/11)/(41/11) = 112/41

    Feedback gains K_t = -P_{t+1} / (r + P_{t+1}):
        K_0 = -30/41,  K_1 = -8/11,  K_2 = -2/3

    Feedforward gains depend on the trajectory; hand-derived for:
        x = [0, 0.5, 1.0, 2.0],  u = [0.5, 0.5, 1.0],  target = 0
        k_0 = -1/2,  k_1 = -19/22,  k_2 = -5/3
    """
    T = 3
    q, r = 2.0, 1.0

    def dynamics(x, u, c):
        return x + u

    Df = jax.jacobian(dynamics, argnums=(0, 1))

    # Non-trivial trajectory consistent with dynamics (x_{t+1} = x_t + u_t):
    #   x_0=0 + u_0=0.5 -> x_1=0.5 + u_1=0.5 -> x_2=1.0 + u_2=1.0 -> x_3=2.0
    x = jnp.array([[0.0], [0.5], [1.0], [2.0]])  # T+1 states
    u = jnp.array([[0.5], [0.5], [1.0]])  # T controls
    c = jnp.zeros((T, 1))
    target = jnp.zeros((T + 1, 1))  # regulate to zero
    Q = jnp.broadcast_to(jnp.array([[q]]), (T + 1, 1, 1))
    R = jnp.broadcast_to(jnp.array([[r]]), (T, 1, 1))

    k, K = backward_pass(x, u, c, target, Q, R, Df, initial_damping=1e-12)

    # Expected feedback gains (independent of trajectory)
    K_expected = np.array([[[-30 / 41]], [[-8 / 11]], [[-2 / 3]]])
    # Expected feedforward gains (trajectory-dependent, hand-derived)
    k_expected = np.array([[-1 / 2], [-19 / 22], [-5 / 3]])

    np.testing.assert_allclose(np.array(K), K_expected, atol=1e-5)
    np.testing.assert_allclose(np.array(k), k_expected, atol=1e-5)
