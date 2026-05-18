from __future__ import annotations

import numpy as np
from typing import Any, Optional

Array = np.ndarray


def _as_float_array(x: Array) -> Array:
    """Convert input to a finite float64 NumPy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Input array must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input array must contain only finite values.")
    return arr


def normalize_histogram(x: Array, eta: float = 1e-8) -> Array:
    """Return a strictly positive normalized histogram.

    Parameters
    ----------
    x : Array
        Nonnegative array or vector. The output keeps the same shape.
    eta : float, default=1e-8
        Small offset added before normalization. This is useful when some
        entries are exactly zero, e.g. for sparse grayscale images.

    Returns
    -------
    Array
        Array of the same shape as x with nonnegative entries summing to 1.
    """
    if eta < 0:
        raise ValueError("eta must be nonnegative.")

    arr = _as_float_array(x)
    if np.any(arr < 0):
        raise ValueError("Histogram entries must be nonnegative.")

    hist = arr + eta
    total_mass = float(hist.sum())
    if total_mass <= 0:
        raise ValueError("Histogram mass must be strictly positive after regularization.")

    return hist / total_mass


def make_grid_1d(n: int, normalize: bool = True) -> Array:
    """Create a 1D grid of support points.

    Parameters
    ----------
    n : int
        Number of support points.
    normalize : bool, default=True
        If True, return coordinates in [0, 1]. Otherwise return coordinates
        0, 1, ..., n-1.

    Returns
    -------
    Array
        Array of shape (n, 1).
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    if normalize:
        if n == 1:
            grid = np.array([0.0], dtype=np.float64)
        else:
            grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
    else:
        grid = np.arange(n, dtype=np.float64)

    return grid.reshape(n, 1)


def make_grid_2d(h: int, w: int, normalize: bool = True) -> Array:
    """Create a 2D grid of support points of shape (h*w, 2).

    Parameters
    ----------
    h : int
        Grid height.
    w : int
        Grid width.
    normalize : bool, default=True
        If True, coordinates are normalized independently to [0, 1] along each
        axis. Otherwise coordinates are integer pixel indices.

    Returns
    -------
    Array
        Array of shape (h*w, 2) in row-major order.
    """
    if not isinstance(h, int) or h <= 0:
        raise ValueError("h must be a positive integer.")
    if not isinstance(w, int) or w <= 0:
        raise ValueError("w must be a positive integer.")

    if normalize:
        ys = np.zeros(h, dtype=np.float64) if h == 1 else np.linspace(0.0, 1.0, h, dtype=np.float64)
        xs = np.zeros(w, dtype=np.float64) if w == 1 else np.linspace(0.0, 1.0, w, dtype=np.float64)
    else:
        ys = np.arange(h, dtype=np.float64)
        xs = np.arange(w, dtype=np.float64)

    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    grid = np.column_stack([yy.ravel(order="C"), xx.ravel(order="C")])
    return grid


def squared_euclidean_cost(X: Array, Y: Optional[Array] = None) -> Array:
    """Return the squared Euclidean cost matrix between two supports.

    Parameters
    ----------
    X : Array
        Support of shape (n, d) or a 1D array of length n.
    Y : Array, optional
        Support of shape (m, d) or a 1D array of length m. If omitted,
        the self-cost matrix on X is returned.

    Returns
    -------
    Array
        Cost matrix of shape (n, m) with entries
        C[i, j] = ||X[i] - Y[j]||_2^2.
    """
    X_arr = _as_float_array(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    elif X_arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")

    if Y is None:
        Y_arr = X_arr
    else:
        Y_arr = _as_float_array(Y)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        elif Y_arr.ndim != 2:
            raise ValueError("Y must be a 1D or 2D array.")

    if X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError("X and Y must have the same ambient dimension.")

    x_norm = np.sum(X_arr * X_arr, axis=1, keepdims=True)
    y_norm = np.sum(Y_arr * Y_arr, axis=1, keepdims=True).T
    C = x_norm + y_norm - 2.0 * (X_arr @ Y_arr.T)
    # Numerical roundoff can create tiny negative values; clip them away.
    np.maximum(C, 0.0, out=C)
    return C


def _validate_sinkhorn_inputs(
    a: Array,
    b: Array,
    C: Array,
    eps: float,
    max_iter: int,
    tol: float,
) -> tuple[Array, Array, Array]:
    """Validate inputs shared by Sinkhorn solvers."""
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if tol <= 0:
        raise ValueError("tol must be strictly positive.")

    a_arr = _as_float_array(a).reshape(-1)
    b_arr = _as_float_array(b).reshape(-1)
    C_arr = _as_float_array(C)

    if np.any(a_arr < 0) or np.any(b_arr < 0):
        raise ValueError("Histograms a and b must be entrywise nonnegative.")
    if C_arr.ndim != 2:
        raise ValueError("C must be a 2D cost matrix.")

    n, m = C_arr.shape
    if a_arr.size != n:
        raise ValueError("Length of a must match the number of rows of C.")
    if b_arr.size != m:
        raise ValueError("Length of b must match the number of columns of C.")

    mass_a = float(a_arr.sum())
    mass_b = float(b_arr.sum())
    if mass_a <= 0 or mass_b <= 0:
        raise ValueError("a and b must have strictly positive total mass.")
    if not np.isclose(mass_a, mass_b, atol=1e-12, rtol=1e-12):
        raise ValueError("a and b must have the same total mass.")

    return a_arr, b_arr, C_arr


def _safe_log_nonnegative(x: Array) -> Array:
    """Elementwise log on nonnegative arrays, with log(0) = -inf."""
    x_arr = _as_float_array(x)
    out = np.full_like(x_arr, -np.inf, dtype=np.float64)
    mask = x_arr > 0
    out[mask] = np.log(x_arr[mask])
    return out


def _safe_negentropy_term(P: Array) -> float:
    """Return sum_{ij} P_ij (log P_ij - 1) with the convention 0 log 0 = 0."""
    P_arr = _as_float_array(P)
    out = np.zeros_like(P_arr, dtype=np.float64)
    mask = P_arr > 0
    out[mask] = P_arr[mask] * (np.log(P_arr[mask]) - 1.0)
    return float(out.sum())


def _logsumexp(M: Array, axis: int) -> Array:
    """Stable log-sum-exp reduction along a given axis.

    This version allows entries equal to -inf, which naturally arise in
    log-domain Sinkhorn when some masses are exactly zero.
    """
    M_arr = np.asarray(M, dtype=np.float64)

    if M_arr.size == 0:
        raise ValueError("Input array must be non-empty.")
    if np.any(np.isnan(M_arr)):
        raise ValueError("Input array must not contain NaNs.")
    if np.any(np.isposinf(M_arr)):
        raise ValueError("Input array must not contain +inf.")

    max_M = np.max(M_arr, axis=axis, keepdims=True)

    # If an entire slice is -inf, then logsumexp should return -inf there.
    all_neg_inf = np.isneginf(max_M)

    # Avoid invalid operations like (-inf) - (-inf)
    safe_max = np.where(all_neg_inf, 0.0, max_M)
    stabilized = M_arr - safe_max
    sum_exp = np.sum(np.exp(stabilized), axis=axis, keepdims=True)

    out = np.where(all_neg_inf, -np.inf, np.log(sum_exp) + safe_max)
    return np.squeeze(out, axis=axis)


def _build_sinkhorn_output(
    P: Array,
    C: Array,
    eps: float,
    n_iter: int,
    converged: bool,
    row_hist: list[float],
    col_hist: list[float],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble a standardized output dictionary for Sinkhorn solvers."""
    cost = float(np.sum(P * C))
    out: dict[str, Any] = {
        "plan": P,
        "cost": cost,
        "reg_objective": cost + float(eps) * _safe_negentropy_term(P),
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "row_residual_l1": float(row_hist[-1]),
        "col_residual_l1": float(col_hist[-1]),
        "row_residual_history": [float(x) for x in row_hist],
        "col_residual_history": [float(x) for x in col_hist],
        "eps": float(eps),
    }
    if extra is not None:
        out.update(extra)
    return out


def sinkhorn(
    a: Array,
    b: Array,
    C: Array,
    eps: float,
    max_iter: int = 5000,
    tol: float = 1e-6,
    return_log: bool = False,
) -> Any:
    """Run standard Sinkhorn scaling on a fixed cost matrix.

    Parameters
    ----------
    a, b : Array
        Nonnegative histograms with the same total mass.
    C : Array
        Cost matrix of shape (n, m).
    eps : float
        Entropic regularization strength.
    max_iter : int, default=5000
        Maximum number of Sinkhorn iterations.
    tol : float, default=1e-6
        L1 tolerance on the marginal residuals.
    return_log : bool, default=False
        If True, also return log-scalings.

    Returns
    -------
    dict
        Dictionary containing the transport plan, cost, residual history,
        scaling vectors, and convergence metadata.

    Notes
    -----
    This is the standard scaling-domain implementation. It is efficient for
    moderate values of eps but may become numerically unstable when eps is too
    small. In that regime, ``sinkhorn_log`` might be prefered.
    """
    a_arr, b_arr, C_arr = _validate_sinkhorn_inputs(a, b, C, eps, max_iter, tol)

    K = np.exp(-C_arr / eps)
    u = np.ones_like(a_arr, dtype=np.float64)
    v = np.ones_like(b_arr, dtype=np.float64)

    check_every = 10
    tiny = np.finfo(np.float64).tiny

    row_hist: list[float] = []
    col_hist: list[float] = []
    P: Optional[Array] = None
    converged = False

    for it in range(1, max_iter + 1):
        Kv = K @ v
        if np.any(Kv <= tiny):
            raise FloatingPointError(
                "Standard Sinkhorn became numerically unstable because K @ v "
                "underflowed. Try sinkhorn_log for smaller eps."
            )
        u = a_arr / Kv

        KTu = K.T @ u
        if np.any(KTu <= tiny):
            raise FloatingPointError(
                "Standard Sinkhorn became numerically unstable because K.T @ u "
                "underflowed. Try sinkhorn_log for smaller eps."
            )
        v = b_arr / KTu

        if it == 1 or it % check_every == 0 or it == max_iter:
            P = (u[:, None] * K) * v[None, :]
            row_res = float(np.linalg.norm(P.sum(axis=1) - a_arr, ord=1))
            col_res = float(np.linalg.norm(P.sum(axis=0) - b_arr, ord=1))
            row_hist.append(row_res)
            col_hist.append(col_res)

            if max(row_res, col_res) <= tol:
                converged = True
                break

    if P is None:
        P = (u[:, None] * K) * v[None, :]
        row_hist.append(float(np.linalg.norm(P.sum(axis=1) - a_arr, ord=1)))
        col_hist.append(float(np.linalg.norm(P.sum(axis=0) - b_arr, ord=1)))

    extra: dict[str, Any] = {
        "u": u,
        "v": v,
        "K": K,
    }
    if return_log:
        extra["log_u"] = _safe_log_nonnegative(u)
        extra["log_v"] = _safe_log_nonnegative(v)

    return _build_sinkhorn_output(
        P=P,
        C=C_arr,
        eps=eps,
        n_iter=it,
        converged=converged,
        row_hist=row_hist,
        col_hist=col_hist,
        extra=extra,
    )


def sinkhorn_log(
    a: Array,
    b: Array,
    C: Array,
    eps: float,
    max_iter: int = 5000,
    tol: float = 1e-6,
    return_log: bool = False,
) -> Any:
    """Run a log-domain stabilized Sinkhorn solver.

    Parameters
    ----------
    a, b : Array
        Nonnegative histograms with the same total mass.
    C : Array
        Cost matrix of shape (n, m).
    eps : float
        Entropic regularization strength.
    max_iter : int, default=5000
        Maximum number of Sinkhorn iterations.
    tol : float, default=1e-6
        L1 tolerance on the marginal residuals.
    return_log : bool, default=False
        If True, also return the log-plan.

    Returns
    -------
    dict
        Dictionary containing the transport plan, cost, residual history,
        dual/log-domain variables, and convergence metadata.

    Notes
    -----
    This implementation is more stable than the standard scaling-domain solver
    for small values of eps.
    """
    a_arr, b_arr, C_arr = _validate_sinkhorn_inputs(a, b, C, eps, max_iter, tol)

    log_a = _safe_log_nonnegative(a_arr)
    log_b = _safe_log_nonnegative(b_arr)

    f = np.zeros_like(a_arr, dtype=np.float64)
    g = np.zeros_like(b_arr, dtype=np.float64)

    check_every = 10

    row_hist: list[float] = []
    col_hist: list[float] = []
    P: Optional[Array] = None
    converged = False

    for it in range(1, max_iter + 1):
        f = eps * (log_a - _logsumexp((g[None, :] - C_arr) / eps, axis=1))
        g = eps * (log_b - _logsumexp((f[:, None] - C_arr) / eps, axis=0))

        if it == 1 or it % check_every == 0 or it == max_iter:
            log_P = (f[:, None] + g[None, :] - C_arr) / eps
            P = np.exp(log_P)

            row_res = float(np.linalg.norm(P.sum(axis=1) - a_arr, ord=1))
            col_res = float(np.linalg.norm(P.sum(axis=0) - b_arr, ord=1))
            row_hist.append(row_res)
            col_hist.append(col_res)

            if max(row_res, col_res) <= tol:
                converged = True
                break

    if P is None:
        log_P = (f[:, None] + g[None, :] - C_arr) / eps
        P = np.exp(log_P)
        row_hist.append(float(np.linalg.norm(P.sum(axis=1) - a_arr, ord=1)))
        col_hist.append(float(np.linalg.norm(P.sum(axis=0) - b_arr, ord=1)))

    extra: dict[str, Any] = {
        "f": f,
        "g": g,
        "u": np.exp(f / eps),
        "v": np.exp(g / eps),
    }
    if return_log:
        extra["log_plan"] = (f[:, None] + g[None, :] - C_arr) / eps

    return _build_sinkhorn_output(
        P=P,
        C=C_arr,
        eps=eps,
        n_iter=it,
        converged=converged,
        row_hist=row_hist,
        col_hist=col_hist,
        extra=extra,
    )
