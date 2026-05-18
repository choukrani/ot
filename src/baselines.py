from __future__ import annotations

import numpy as np
from typing import Any, Dict

Array = np.ndarray


def _as_float_array(x: Array) -> Array:
    """Convert input to a finite float64 NumPy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Input array must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input array must contain only finite values.")
    return arr


def _validate_histogram(a: Array, name: str) -> Array:
    """Validate a nonnegative 1D histogram."""
    arr = _as_float_array(a).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be entrywise nonnegative.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError(f"{name} must have strictly positive total mass.")
    return arr


def exact_ot_1d_sorting(a: Array, b: Array, x: Array, y: Array, p: int = 2) -> Dict[str, Any]:
    if p <= 0:
        raise ValueError("p must be positive.")

    a_arr = _validate_histogram(a, "a")
    b_arr = _validate_histogram(b, "b")
    x_arr = _as_float_array(x).reshape(-1)
    y_arr = _as_float_array(y).reshape(-1)

    if a_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("a and x must have the same length.")
    if b_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("b and y must have the same length.")

    mass_a = float(a_arr.sum())
    mass_b = float(b_arr.sum())
    if not np.isclose(mass_a, mass_b, atol=1e-12, rtol=1e-12):
        raise ValueError("a and b must have the same total mass.")

    x_perm = np.argsort(x_arr, kind="stable")
    y_perm = np.argsort(y_arr, kind="stable")

    x_sorted = x_arr[x_perm]
    y_sorted = y_arr[y_perm]
    a_sorted = a_arr[x_perm].copy()
    b_sorted = b_arr[y_perm].copy()

    n = a_sorted.shape[0]
    m = b_sorted.shape[0]
    plan_sorted = np.zeros((n, m), dtype=np.float64)

    i, j = 0, 0
    cost = 0.0
    tol = 1e-12

    # Skip initial zero-mass entries.
    while i < n and a_sorted[i] <= tol:
        a_sorted[i] = 0.0
        i += 1
    while j < m and b_sorted[j] <= tol:
        b_sorted[j] = 0.0
        j += 1

    while i < n and j < m:
        mass = min(a_sorted[i], b_sorted[j])

        if mass > 0.0:
            plan_sorted[i, j] += mass
            cost += mass * abs(x_sorted[i] - y_sorted[j]) ** p

        a_sorted[i] -= mass
        b_sorted[j] -= mass

        # Advance past exhausted or numerically negligible entries.
        while i < n and a_sorted[i] <= tol:
            a_sorted[i] = 0.0
            i += 1
        while j < m and b_sorted[j] <= tol:
            b_sorted[j] = 0.0
            j += 1

    # It is valid to have only zero-mass entries left.
    while i < n and a_sorted[i] <= tol:
        a_sorted[i] = 0.0
        i += 1
    while j < m and b_sorted[j] <= tol:
        b_sorted[j] = 0.0
        j += 1

    if i != n or j != m:
        raise RuntimeError(
            "Mass sweep did not terminate correctly: nonzero residual mass remains."
        )

    # Map the sorted coupling back to the original indexing.
    inv_x_perm = np.empty_like(x_perm)
    inv_y_perm = np.empty_like(y_perm)
    inv_x_perm[x_perm] = np.arange(n)
    inv_y_perm[y_perm] = np.arange(m)
    plan = plan_sorted[inv_x_perm][:, inv_y_perm]

    return {
        "cost": float(cost),
        "plan": plan,
        "plan_sorted": plan_sorted,
        "x_sorted": x_sorted,
        "y_sorted": y_sorted,
        "a_sorted": a_arr[x_perm],
        "b_sorted": b_arr[y_perm],
        "x_perm": x_perm,
        "y_perm": y_perm,
    }


def exact_ot_lp(a: Array, b: Array, C: Array) -> Dict[str, Any]:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise ImportError(
            "exact_ot_lp requires SciPy. Please install scipy before using this baseline."
        ) from exc

    a_arr = _validate_histogram(a, "a")
    b_arr = _validate_histogram(b, "b")
    C_arr = _as_float_array(C)

    if C_arr.ndim != 2:
        raise ValueError("C must be a 2D cost matrix.")

    n, m = C_arr.shape
    if a_arr.shape[0] != n:
        raise ValueError("Length of a must match the number of rows of C.")
    if b_arr.shape[0] != m:
        raise ValueError("Length of b must match the number of columns of C.")

    mass_a = float(a_arr.sum())
    mass_b = float(b_arr.sum())
    if not np.isclose(mass_a, mass_b, atol=1e-12, rtol=1e-12):
        raise ValueError("a and b must have the same total mass.")

    c = C_arr.reshape(-1)

    A_eq = np.zeros((n + m, n * m), dtype=np.float64)
    b_eq = np.concatenate([a_arr, b_arr])

    # Row constraints: sum_j P_{ij} = a_i
    for i in range(n):
        A_eq[i, i * m : (i + 1) * m] = 1.0

    # Column constraints: sum_i P_{ij} = b_j
    for j in range(m):
        A_eq[n + j, j :: m] = 1.0

    bounds = [(0.0, None)] * (n * m)

    result = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return {
            "cost": np.inf,
            "plan": None,
            "success": False,
            "status": int(result.status),
            "message": result.message,
            "row_residual_l1": np.inf,
            "col_residual_l1": np.inf,
            "result": result,
        }

    plan = result.x.reshape(n, m)
    row_residual_l1 = float(np.linalg.norm(plan.sum(axis=1) - a_arr, ord=1))
    col_residual_l1 = float(np.linalg.norm(plan.sum(axis=0) - b_arr, ord=1))

    return {
        "cost": float(result.fun),
        "plan": plan,
        "success": True,
        "status": int(result.status),
        "message": result.message,
        "row_residual_l1": row_residual_l1,
        "col_residual_l1": col_residual_l1,
        "result": result,
    }
