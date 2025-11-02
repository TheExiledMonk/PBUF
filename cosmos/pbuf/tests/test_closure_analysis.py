"""
Check PBUF closure and elastic contributions for different parameter values.
"""

from cosmos.pbuf.model import PBUF

def test_closure_and_elastic():
    """Test closure and elastic contributions for different parameter values."""

    base = dict(
        omega_m=0.3153,
        h=0.6736,
        omega_k=0.0,
        omega_r=9.2e-5,
        omega_b=0.02237/(0.6736**2),
        T_cmb=2.7255,
    )

    test_cases = [
        (5.0e-4, 1.0e8, 1.0, 0.7),
        (2.0e-3, 5.0e8, 1.5, 0.9),
        (1.0e-2, 1.0e9, 0.6, 1.2),
    ]

    print("PBUF Closure and Elastic Analysis...")
    print("=" * 70)
    print(f"{'α':<10} {'Rmax':<10} {'k_sat':<8} {'ε₀':<8} {'Ω_total':<10} {'Ω_σ(a=1)':<12}")
    print("-" * 70)

    for alpha, Rmax, k_sat, eps0 in test_cases:
        model = PBUF(**base, alpha=alpha, Rmax=Rmax, k_sat=k_sat, eps0=eps0)

        # Check closure
        omega_total = model.closure_today()
        omega_sigma_today = model.omega_sigma(1.0)
        print(f"{alpha:.2e} {Rmax:.2e} {k_sat:.2f}   {eps0:.2f}   {omega_total:.6f}   {omega_sigma_today:.6f}")

        # Check elastic contributions at different scale factors
        print("  Scale factor dependence:")
        for a in [0.1, 0.5, 1.0]:
            omega_sigma_a = model.omega_sigma(a)
            print(f"    a={a:.1f}: Ω_σ(a)={omega_sigma_a:.6f}")
        print()

if __name__ == "__main__":
    test_closure_and_elastic()
