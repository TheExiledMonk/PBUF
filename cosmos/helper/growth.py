import numpy as np
from scipy.integrate import solve_ivp

from cosmos.optim.parameter_defaults import SIGMA8_PLANCK


def growth_factor(z, model, sigma8_0=SIGMA8_PLANCK, z_max=20.0, n_points=800):
    """
    Compute the linear growth factor D(z) by numerically integrating
    the full growth ODE for arbitrary cosmological background (LCDM or PBUF).

    Equation:
        d²D/d(ln a)² + [2 + dlnH/dln a] * dD/dln a - (3/2)*Ω_m(a)*D = 0

    Normalization:
        D(0) = 1  (growth factor today)

    Parameters
    ----------
    z : float or array
        Redshift(s) to evaluate D(z)
    model : LCDM or PBUF instance
        Must implement H(z) and density_parameters_at_z(z)
    sigma8_0 : float
        Normalization σ₈(0)
    z_max : float
        Max redshift for integration start (default: 20)
    n_points : int
        Number of points in log(a) grid

    Returns
    -------
    D(z) : float or np.ndarray
        Normalized linear growth factor
    """

    # Scale factor grid (integrate forward from high z to today)
    a_start = 1.0 / (1.0 + z_max)
    a_end = 1.0
    ln_a_span = (np.log(a_start), np.log(a_end))

    # Convert H(z) -> H(a)
    def H_of_a(a):
        z = 1.0 / a - 1.0
        return model.H(z)

    def Omega_m_of_a(a):
        z = 1.0 / a - 1.0
        omega = model.density_parameters_at_z(z)
        return omega.get("omega_m", 0.3)

    # Differential equation in ln(a)
    def dD_dlnA(ln_a, y):
        """
        y = [D, dD/dln a]
        """
        a = np.exp(ln_a)
        H = H_of_a(a)
        dlnH_dlnA = _dlnH_dlnA(a, H_of_a)
        Omega_m = Omega_m_of_a(a)
        D, Dp = y
        Dpp = -(2.0 + dlnH_dlnA) * Dp + 1.5 * Omega_m * D
        return [Dp, Dpp]

    # Initial conditions deep in matter domination
    y0 = [a_start, a_start]  # D ∝ a → D' = D initially

    # Integrate ODE
    sol = solve_ivp(
        fun=dD_dlnA,
        t_span=ln_a_span,
        y0=y0,
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )

    # Normalize so that D(a=1) = 1
    D_today = sol.y[0, -1]
    norm = 1.0 / D_today

    # Evaluate at requested z values
    z = np.atleast_1d(np.array(z, dtype=float))
    a_vals = 1.0 / (1.0 + z)
    D_vals = norm * sol.sol(np.log(a_vals))[0]

    # Return same shape as input
    if D_vals.size == 1:
        return float(D_vals)
    return D_vals


def _dlnH_dlnA(a, H_of_a, delta=1e-5):
    """
    Compute dlnH/dln a numerically, guarding against the domain boundaries.
    """
    a = float(a)
    a_minus = a * (1 - delta)
    a_plus = a * (1 + delta)

    if a_plus > 1.0:
        # Use a backward difference very close to a=1
        a_plus = min(1.0, a)
        a_minus = a / (1 + delta)
        if a_minus <= 0.0:
            a_minus = max(a * 0.5, 1e-12)
        H1 = H_of_a(a_minus)
        H2 = H_of_a(a)
        return (np.log(H2) - np.log(H1)) / (np.log(a) - np.log(a_minus))

    if a_minus <= 0.0:
        # Use a forward difference if we approach a→0
        a_minus = max(a * (1 - 0.5 * delta), 1e-12)
        a_plus = a * (1 + delta)
        H1 = H_of_a(a)
        H2 = H_of_a(a_plus)
        return (np.log(H2) - np.log(H1)) / (np.log(a_plus) - np.log(a))

    H1 = H_of_a(a_minus)
    H2 = H_of_a(a_plus)
    return (np.log(H2) - np.log(H1)) / (np.log(a_plus) - np.log(a_minus))


def growth_rate(z, model):
    """
    Compute growth rate f(z) = d ln D / d ln a numerically.
    """
    z = np.atleast_1d(np.array(z, dtype=float))
    D_vals = growth_factor(z, model)
    a_vals = 1.0 / (1.0 + z)
    ln_a = np.log(a_vals)
    ln_D = np.log(D_vals + 1e-30)

    if ln_a.size < 2:
        # Fall back to a small backward difference in ln a.
        delta = 1e-4
        a_hi = min(a_vals[0] * (1 + delta), 1.0)
        a_lo = max(a_vals[0] / (1 + delta), 1e-8)
        D_hi = growth_factor(1.0 / a_hi - 1.0, model)
        D_lo = growth_factor(1.0 / a_lo - 1.0, model)
        dlnD = np.log(D_hi + 1e-30) - np.log(D_lo + 1e-30)
        dlnA = np.log(a_hi) - np.log(a_lo)
        return float(dlnD / dlnA)

    order = np.argsort(ln_a)
    ln_a_sorted = ln_a[order]
    ln_D_sorted = ln_D[order]

    # Collapse duplicate ln(a) samples by averaging ln_D values.
    unique_ln_a, inverse = np.unique(ln_a_sorted, return_inverse=True)
    if unique_ln_a.size < ln_a_sorted.size:
        ln_D_unique = np.zeros(unique_ln_a.shape, dtype=float)
        counts = np.zeros(unique_ln_a.shape, dtype=float)
        for idx, bucket in enumerate(inverse):
            ln_D_unique[bucket] += ln_D_sorted[idx]
            counts[bucket] += 1
        ln_D_unique /= np.where(counts == 0, 1.0, counts)
    else:
        ln_D_unique = ln_D_sorted

    if unique_ln_a.size < 2:
        # Should not happen (handled earlier), but guard anyway.
        edge = 1
        grad_scalar = np.gradient(ln_D_unique, unique_ln_a, edge_order=edge)
        return float(grad_scalar[0])

    edge_order = 2 if unique_ln_a.size > 2 else 1
    grad_unique = np.gradient(ln_D_unique, unique_ln_a, edge_order=edge_order)

    # Map gradients back to original ordering
    dlnD_sorted = grad_unique[inverse]
    dlnD_dlnA = np.empty_like(ln_a_sorted)
    dlnD_dlnA[:] = dlnD_sorted

    # Restore original input order
    restored = np.empty_like(dlnD_dlnA)
    restored[order] = dlnD_dlnA
    return restored if restored.size > 1 else float(restored)


def fsigma8(z, model, sigma8_0=SIGMA8_PLANCK):
    """
    Compute the redshift-space distortion observable fσ8(z)
    using the full numerical growth factor.
    """
    D_z = growth_factor(z, model, sigma8_0)
    f_z = growth_rate(z, model)
    return f_z * sigma8_0 * D_z
