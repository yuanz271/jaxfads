from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, vmap
# import matplotlib.pyplot as plt

from jaxfads.ilqr import ilqr, cost_function


def test_ilqr():
    def dynamics(x, u, c):
        return x + (0.5 * x * x + u) * 0.1

    dt = 0.1
    T = 50

    rng = np.random.default_rng(0)

    # Initial state
    x0 = rng.standard_normal(size=(5, 2)) * 0.1
    # Define a target trajectory (for example, tracking a sine wave in position and a smooth velocity profile)
    target = np.array([np.array([np.sin(0.1 * k), np.cos(0.1 * k)]) for k in range(T)])
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
    """cost_function must compute 0.5 * sum_t (x-target)^T Q (x-target) + u^T R u."""
    T, D_x, D_u = 4, 2, 1
    x = jnp.array([[1.0, 2.0], [0.5, 1.5], [0.0, 1.0], [-0.5, 0.5]])
    u = jnp.ones((T, D_u))
    target = jnp.zeros((T, D_x))
    Q = jnp.broadcast_to(jnp.diag(jnp.array([2.0, 3.0])), (T, D_x, D_x))
    R = jnp.broadcast_to(jnp.eye(D_u) * 0.5, (T, D_u, D_u))

    cost = cost_function(x, u, target, Q, R)

    # Manual computation
    expected = 0.0
    for t in range(T):
        dx = x[t] - target[t]
        expected += 0.5 * (dx @ Q[t] @ dx + u[t] @ R[t] @ u[t])
    expected = jnp.array(expected)

    np.testing.assert_allclose(float(cost), float(expected), rtol=1e-5)
