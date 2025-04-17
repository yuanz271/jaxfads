# stochastic LQR with additive homogenenous noise has the same solution as determinitic dynamics does.
# instead of doing adjoint to optimize dynamics, we can use XFADS trajectory and iLQR.
# finite horizon, discrete time
import jax
from jax import lax, numpy as jnp, vmap
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array


def lqr_backward(z, Fz, Fu, Czz, Cuu):
    # Backward in Algo. 1
    # we can set the cross cost to be 0 (equiv. independent joint distribution of z and u)
    # the state cost is posterior of z
    # the control cost is prior of u
    
    # v[T] initial cost
    # V[T] initial cost
    # Check cz, cu, Czz, Cuz, Cuu in S15, the indices indicate the shape

    def step(carry: tuple[Array, Array], xs: tuple):
        # z[t+1] = Az[t] + Bu[t]
        S_tp1, s_tp1 = carry
        z_t, A, B, Czz, Cuu = xs
        
        M = jnp.linalg.multi_dot((B.T, S_tp1, B))
        L = cho_factor(M + Cuu)
        M = cho_solve(L, B.T)
        K_t = jnp.linalg.multi_dot((M, S_tp1, A))
        D = A - B @ K_t
        S_t = jnp.linalg.multi_dot((A.T, S_tp1, D)) + Czz
        s_t = D.T @ s_tp1 + Czz @ z_t
        k_t = M @ s_tp1
        u_t = - K_t @ z_t + k_t
        
        return (S_t, s_t), u_t
    
    S_T = Czz[-1]
    s_t = Czz[-1] @ z[-1]
    _, u = lax.scan(step, init=(S_T, s_t), xs=(z[:-1], Fz[:-1], Fu[:-1], Czz[:-1], Cuu[:-1]), reverse=True)

    return u


def lqr_forward(Fz, Fu, u, z0):
    def step(z_t, xs):
        Fz, Fu, u_t = xs
        z_tp1 = Fz @ z_t + Fu @ u_t
        return z_tp1, z_tp1
    
    _, zhat = lax.scan(step, init=z0, xs=(Fz[:-1], Fu[:-1], u))
    zhat = jnp.vstack((z0, zhat))
    return zhat

# <<< LQR

# >>> iLQR
# See Tassa 2014
# Notation:
# x: state
# u: control
# z: target
# f: dynamics, x[t+1] = f(x[t], u[t])
# Fx, Fu: Jacobian
# l: cost function, l(x, u), l(x[T])
# cx, cu, Cxx, Cux, Cuu: derivatives


def eval_cost(x, u, z, Q, R):
    UQ = jnp.linalg.cholesky(Q, upper=True)
    Udx = UQ @ (x - z)
    
    UR = jnp.linalg.cholesky(R, upper=True)
    Uu = UR @ u

    return jnp.inner(Udx, Udx) + jnp.inner(Uu, Uu)


def cost_g_and_H(x, u, z, Q, R):
    """Return cost vectors and matrices"""
    # (x - z)'Q(x - z)
    cx = -2 * Q @ z
    cu = jnp.zeros_like(u)
    Cxx = Q
    Cux = jnp.zeros((jnp.size(u), jnp.size(x)))
    Cuu = R

    return cx, cu, Cxx, Cux, Cuu


def rollout(x0, u, z, f, J, Q, R):
    def step(carry, inputs):
        x_t = carry
        u_t= inputs
        
        x_tp1= f(x_t, u_t)
        
        return x_tp1, x_t
    
    _, x = lax.scan(step, init=x0, xs=u)

    c = vmap(J)(x, u, z, Q, R)

    return x, u, c


def ilqr_forward(k, K, x0, x_ref, u_ref, z, f, J, Q, R, alpha):
    
    def step(carry, inputs):
        x_t = carry
        x_ref, u_ref, k, K = inputs

        u_t = u_ref + k * alpha
        dx = x_t - x_ref
        u_t = u_t + K @ dx
        
        x_tp1= f(x_t, u_t)
        
        return x_tp1, (x_t, u_t)
    
    x_T, (x, u) = lax.scan(step, init=x0, xs=(x_ref[:-1], u_ref[:-1], k, K))

    x = jnp.vstack((x, x_T))
    u = jnp.vstack((u, jnp.zeros_like(u_ref[-1])))  # u[T] is not used but counted in cost

    # print(f"{x.shape=}, {u.shape=}, {z.shape=}, {Q.shape=}, {R.shape=}")
    c = vmap(J)(x, u, z, Q, R)

    return x, u, c


def ilqr_backward(Fx, Fu, cx, cu, Cxx, Cux, Cuu, lam, reg):
    v = cx[-1]
    V = Cxx[-1]
    dV = jnp.zeros(2)

    def step(carry, inputs):
        v, V, dV = carry

        Fx, Fu, cx, cu, Cxx, Cux, Cuu = inputs

        qx = cx + Fx.T @ v
        qu = cu + Fu.T @ v

        Qxx = Cxx + jnp.linalg.multi_dot((Fx.T, V, Fx))
        Quu = Cuu + jnp.linalg.multi_dot((Fu.T, V, Fu))
        Qux = Cux + jnp.linalg.multi_dot((Fu.T, V, Fx))

        V_reg = V + lam * reg * jnp.eye(V.shape[0])
        Qux_reg = Cux + jnp.linalg.multi_dot((Fu.T, V_reg, Fx))

        QuuF = Cuu + jnp.linalg.multi_dot((Fu.T, V_reg, Fu)) + lam * (1 - reg) * jnp.eye(Cuu.shape[0])

        Luu = cho_factor(QuuF)

        k = -cho_solve(Luu, qu)
        K = -cho_solve(Luu, Qux_reg)

        Uuu = jnp.linalg.cholesky(Quu, upper=True)
        Uk = Uuu @ k
        UK = Uuu @ K

        dV = dV + jnp.array((jnp.inner(k, qu), 0.5* jnp.inner(Uk, Uk)))
        v = qx + K.T @ qu + Qux.T @ k + UK.T @ Uk
        V = Qxx + K.T @ Qux + Qux.T @ K  + UK.T @ UK
        V = 0.5 * (V + V.T)

        return (v, V, dV), (k, K)

    (_, _, dV), (k, K) = lax.scan(step, init=(v, V, dV), xs=(Fx[:-1], Fu[:-1], cx[:-1], cu[:-1], Cxx[:-1], Cux[:-1], Cuu[:-1]), reverse=True)

    return k, K, dV



def ilqr(f, z, u, Q, R, *, alphas=None, ftol=1e-7, gtol=1e-4, xtol=1e8, ltol=1e-5, maxiter=500, lam=1., dlam=1., flam=1.6, maxlam=1e10, minlam=1e-6, reg=0, minrr=0, verbose=False):
    """
    :param alphas: backtracking coefficients
    :param ftol: reduction exit criterion
    :param gtol: gradient exit criterion
    :param maxiter: maximum iterations
    :param lam: initial value for regularization lambda
    :param dlam: initial value for dlam
    :param flam: lambda scaling factor
    :param maxlam: maximum lambda
    :param minlam: minimum lambda
    :param reg: regularization type 1: Q_uu + lambda*I; 2: V_xx + lambda * I
    :param minz: minimal accepted reduction ratio
    """
    # hyperparameters from Tassa 2014
    if alphas is None:
        alphas = jnp.logspace(0, -3, 11)

    F = jax.jacobian(f, argnums=(0, 1))

    x0 = z[0]
    # >>> Check initial control sequence
    for alpha in alphas:
        x, u, cost = rollout(x0, alpha*u, z, f, eval_cost, Q, R)
        if jnp.all(jnp.abs(x) < xtol):
            break
    else:    
        raise RuntimeError("Initial control sequence caused divergence.")
    
    recompute = True

    for i in range(maxiter):
        if verbose:
            print(f"Iteration - {i}")
        # >>> STEP 1: differentiate on new trajectory
        if recompute:
            # differentiate dynamics
            Fs = vmap(F)(x, u)
            Cs = vmap(cost_g_and_H)(x, u, z, Q, R)
            recompute = False
        # <<< STEP 1

        # >>> STEP 2: backward pass
        bwd_succeeded = False
        while not bwd_succeeded:
            k, K, dV = ilqr_backward(*Fs, *Cs, lam, reg)
            
            if jnp.any(jnp.isnan(K)):
                lam, dlam = inc_lam(lam, dlam, flam, minlam)
                if lam > maxlam:
                    break
                continue
            bwd_succeeded = True
        
        gnorm = jnp.mean(jnp.max(jnp.abs(k) / (jnp.abs(u[:-1]) + 1), axis=1))  # check for termination due to small gradient
        
        if gnorm < gtol and lam < ltol:
            lam, dlam = dec_lam(lam, dlam, flam, minlam)
            break
        # <<< STEP 2

        # >>> STEP 3: line-search new control sequence, trajectory, cost
        fwd_succeeded = False
        if bwd_succeeded:
            # print(f"{k.shape=}, {K.shape=}, {x0.shape=}, {x.shape=}, {u.shape=}, {z.shape=}, {Q.shape=}, {R.shape=}, {alphas.shape=}")
            x_new, u_new, cost_new = vmap(lambda a: ilqr_forward(k, K, x0, x, u, z, f, eval_cost, Q, R, a))(alphas)
            # print(f"{cost=}")
            Dcost = jnp.sum(cost) - jnp.sum(cost_new, axis=1)  # sum over time, improvements
            w = jnp.argmax(Dcost)
            dcost = Dcost[w]
            alpha = alphas[w]
            expected = -alpha*(dV[0] + alpha*dV[1])
            
            if verbose:
                print(f"{dcost=}, {expected=}")

            if expected > 0:
                rr = dcost / expected
            else:
                rr = jnp.sign(dcost)
                # should not occur
            
            if verbose:
                print(f"{rr=}")
            fwd_succeeded = rr >= minrr  # TODO: strict >?
            
            if fwd_succeeded:
                cost_star = cost_new[w]
                x_star = x_new[w]
                u_star = u_new[w]
        # <<< STEP 3
        
        # >>> STEP 4: accept or discard new control sequence
        if fwd_succeeded:
            # decrease lambda
            lam, dlam = dec_lam(lam, dlam, flam, minlam)

            # accept
            u = u_star
            x = x_star
            cost = cost_star
            recompute = True

            if dcost < ftol:
                # success
                if verbose:
                    print("SUCCESS: cost change < tol")
                break
        else:
            # increase lambda
            lam, dlam = inc_lam(lam, dlam, flam, minlam)
            
            if lam > maxlam:
                if verbose:
                    print("EXIT: lambda > maximum lambda")
                break
        # <<< STEP 4
    else:
        if verbose:
            print("Maxiter reached.")

    return u, x

# >>> Regularization schedule
# [Tassa, 2012]

def inc_lam(lam, dlam, flam, minlam):
    dlam = max(dlam * flam, flam)
    lam = max(lam * dlam, minlam)
    return lam, dlam


def dec_lam(lam, dlam, flam, minlam):
    dlam = min(dlam / flam, 1/flam)
    lam = lam * dlam
    if lam < minlam:
        lam = 0 
    return lam, dlam
# <<< Regularization schedule