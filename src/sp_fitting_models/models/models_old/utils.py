"""
Utility functions for model calculations.
"""

import numpy as np
import numpy.typing as npt
import numba as nb

# Gas constant in J/(mol·K)
R = 8.314


@nb.jit(nopython=True)
def solve_cubic_vectorized(a, b, c, d, x_low, x_high, max_iter=50) -> npt.NDArray[np.float64]:
    """
    Solve a x^3 + b x^2 + c x + d = 0 on the interval [x_low, x_high]
    using fully vectorized bisection.

    Args:
        a, b, c, d: Coefficients of the cubic equation.
        x_low, x_high: Bounds for the root search.
        max_iter: Maximum number of iterations for bisection.

    Returns:
        npt.NDArray[np.float64]: Approximated roots within the specified bounds.

    Raises:
        ValueError: If the root is not bracketed for some elements.
        RuntimeError: If the bisection does not converge within the maximum number of iterations.
    """

    # Shape unification
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)

    # Initial bounds
    xl: npt.NDArray[np.float64] = np.asarray(x_low, dtype=np.float64)
    xr: npt.NDArray[np.float64] = np.asarray(x_high, dtype=np.float64)

    # Evaluate cubic
    def f(x):
        return a * x**3 + b * x**2 + c * x + d

    fl = f(xl)
    fr = f(xr)
    mask_valid = fl * fr <= 0
    if not np.all(mask_valid):
        raise ValueError("Root not bracketed for some elements.")

    # Bisection method loop
    for _ in range(max_iter):
        xm = 0.5 * (xl + xr)
        fm = f(xm)

        left_mask = fl * fm <= 0
        # Update bounds based on which side contains the root
        xr = np.where(left_mask, xm, xr)
        xl = np.where(left_mask, xl, xm)
        fl = np.where(left_mask, fl, fm)
        fr = np.where(left_mask, fm, fr)

    if not np.all(np.abs(xr - xl) < 1e-10):
        raise RuntimeError("Bisection did not converge within the maximum number of iterations.")
    return 0.5 * (xl + xr)
