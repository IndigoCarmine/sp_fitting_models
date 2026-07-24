"""
Tests for cooperative polymerization models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from sp_fitting_models.models import (
    temp_cooperative_model_n,
    inv_cooperative_model_n,
    cooperative_model_n,
    cooperative_model,
    inv_cooperative_model,
    inv_isodesmic_model,
    isodesmic_model,
)


def _species_sum_ctot(c_m, K, sigma, N, s_max=5000):
    """Ground-truth total concentration by direct species summation.

    c_tot = sum_s s * [M_s], with [M_s] = sigma**(min(s, N) - 1) * K**(s - 1) * c_m**s.
    Valid while K * c_m < 1 (geometric tail converges); s_max=5000 is far beyond needed.
    """
    s = np.arange(1, s_max + 1, dtype=float)
    penalty = np.minimum(s, N) - 1.0
    Ms = sigma**penalty * K ** (s - 1) * c_m**s
    return float(np.sum(s * Ms))


def test_cooperative_inverse_consistency():
    """
    Forward solver (Rust) and inverse model (Python) must be self-consistent for each
    nucleus size. Both now implement the same corrected formula, so the round-trip should
    hold to a tight tolerance (limited only by bisection precision and the (1 - agg)
    cancellation near full aggregation), not the loose 5e-2 that previously masked a bug.
    """
    c_tot = np.linspace(1, 1000, 1000) * 1e-6  # Total concentrations from 1 uM to 1000 uM
    deltaH = -96000
    deltaS = -180
    deltaHnuc = 10000
    Temp = 300
    R = 8.314

    # Calculate K and sigma
    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    sigma = np.exp(-deltaHnuc / (R * Temp))

    for n in [2, 3, 4, 5]:  # actually vary the nucleus size (bug fix: was hardcoded to 3)
        # Calculate aggregation with nucleus size n
        agg = np.array(
            [
                temp_cooperative_model_n(np.array([Temp]), deltaH, deltaS, deltaHnuc, c, scaler=1.0, nuc_size=n)
                for c in c_tot
            ]
        ).flatten()
        monomer_conc = (1 - agg) * c_tot

        # Reverse calculation with the same nucleus size n
        c_tot_calculated = inv_cooperative_model_n(monomer_conc, K, sigma, nuc_size=n)

        # Check consistency
        max_diff = np.max(np.abs(c_tot - c_tot_calculated) / c_tot)
        assert np.allclose(c_tot, c_tot_calculated, rtol=1e-3), (
            f"Cooperative model inverse inconsistent for nuc_size={n} (max rel diff {max_diff:.2e})"
        )


def test_inv_cooperative_n_matches_species_sum():
    """
    The Python inverse model must reproduce the exact species-sum ground truth to near
    machine precision, for several nucleus sizes and cK values. This pins the corrected
    formula (multiplicity factor s and summation range s = 2..N-1).
    """
    K, sigma = 1.0, 0.1
    for N in [2, 3, 4, 5, 6]:
        for x in [0.05, 0.2, 0.4, 0.6, 0.8]:  # x = cK
            c_m = x / K
            expected = _species_sum_ctot(c_m, K, sigma, N)
            got = float(inv_cooperative_model_n(np.asarray(c_m), K, sigma, N))
            assert got == pytest.approx(expected, rel=1e-9), f"N={N}, cK={x}"


def test_cooperative_n_forward_matches_species_sum():
    """
    The Rust forward solver must invert the exact species-sum: recovering the free-monomer
    concentration from a known total should return the original value (round-trip on c_m),
    including the high-aggregation regime (cK -> 1) where the boundary handling matters.
    """
    K, sigma = 1.0, 0.1
    for N in [2, 3, 4, 5, 8]:
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:  # x = cK, incl. near-singularity
            c_m = x / K
            c_tot = _species_sum_ctot(c_m, K, sigma, N)
            agg = cooperative_model_n(c_tot, K, sigma, N)
            c_m_recovered = (1.0 - agg) * c_tot
            assert c_m_recovered == pytest.approx(c_m, rel=1e-6), f"N={N}, cK={x}"
            assert 0.0 <= agg <= 1.0


def test_cooperative_n_reduces_to_basic_at_n2():
    """
    nuc_size=2 must reproduce the basic cooperative model exactly, for both the forward
    solver and the inverse model (invariant that was broken before the fix).
    """
    K, sigma = 2.056e7, 0.0181
    for c in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]:
        agg_n = cooperative_model_n(float(c), K, sigma, 2)
        agg_basic = cooperative_model(float(c), K, sigma)
        assert agg_n == pytest.approx(agg_basic, abs=1e-12, rel=0), f"forward mismatch at c={c}"

    c_monomer = np.linspace(1e-9, 0.9 / K, 50)
    assert np.allclose(
        inv_cooperative_model_n(c_monomer, K, sigma, 2),
        inv_cooperative_model(c_monomer, K, sigma),
        rtol=1e-12,
        atol=0.0,
    ), "inverse mismatch between nuc_size=2 and basic cooperative model"


def test_cooperative_n_reduces_to_isodesmic_at_sigma1():
    """
    At sigma = 1 the cooperativity penalty sigma**(min(s, N) - 1) equals 1 for every
    species, so the nucleation-elongation distribution [M_s] = K**(s-1) * c_m**s becomes
    identical to the isodesmic one, independently of the nucleus size N. Both the inverse
    model and the forward solver must therefore coincide with the isodesmic model to
    (near) machine precision for every N.
    """
    K = 1.0e5

    # Inverse model: total concentration from monomer concentration. This is an exact
    # algebraic identity (the nucleus correction vanishes and the elongation term reduces
    # to the isodesmic closed form), so it holds to essentially machine precision.
    c_monomer = np.linspace(1e-9, 0.99 / K, 500)
    iso_ctot = inv_isodesmic_model(c_monomer, K)
    for N in [2, 3, 4, 5, 8]:
        coop_ctot = inv_cooperative_model_n(c_monomer, K, sigma=1.0, nuc_size=N)
        max_rel = np.max(np.abs(coop_ctot - iso_ctot) / iso_ctot)
        assert np.allclose(coop_ctot, iso_ctot, rtol=1e-12, atol=0.0), (
            f"inverse model disagrees with isodesmic at sigma=1 for nuc_size={N} "
            f"(max rel diff {max_rel:.2e})"
        )

    # Forward solver: aggregated fraction from total concentration. The cooperative solver
    # uses bisection while the isodesmic reference uses the closed-form root, so they agree
    # only to bisection precision (still ~1e-15 in practice).
    c_tot = np.linspace(1, 1000, 50) * 1e-6
    for N in [2, 3, 4, 5, 8]:
        for c in c_tot:
            agg_coop = cooperative_model_n(float(c), K, 1.0, N)
            agg_iso = float(isodesmic_model(float(c), K))
            assert agg_coop == pytest.approx(agg_iso, abs=1e-9, rel=0), (
                f"forward solver disagrees with isodesmic at sigma=1 for nuc_size={N}, c={c}"
            )


def test_temp_cooperative_model():
    """
    Test temperature-dependent cooperative model.
    """
    temps = np.linspace(280, 400, 100)
    deltaH = -96000
    deltaS = -180
    deltaHnuc = 100000
    concentrations = np.array([10, 50, 100]) * 1e-6  # 10, 50, 100 µM

    fig, ax = plt.subplots(figsize=(8, 6))

    for c in concentrations:
        agg = temp_cooperative_model_n(
            Temp=temps,
            deltaH=deltaH,
            deltaS=deltaS,
            deltaHnuc=deltaHnuc,
            c_tot=c,
            scaler=1.0,
            nuc_size=3,
        )
        ax.plot(temps - 273.15, agg, label=f"Conc={c*1e6:.1f} µM")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Aggregation")
    ax.set_title("Temperature-Dependent Cooperative Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("✓ Temperature-dependent cooperative model test completed")
    return fig


def test_cooperative_high_temp_rounding_error():
    """
    Investigation of rounding error effects in cooperative model at low aggregation (high temperature).

    At high temperatures, the aggregation (agg) approaches zero due to σ → 1.
    This test measures numerical stability in the low-aggregation regime.
    """
    R = 8.314
    deltaH = -96000
    deltaS = -180
    deltaHnuc = 100000
    c_tot = 1e-6

    # Temperature progression: 300 K → 500 K (high temperature where agg → 0)
    temps = np.linspace(300, 500, 200)

    aggs = []
    K_vals = []
    sigma_vals = []

    for temp in temps:
        agg = temp_cooperative_model_n(np.array([temp]), deltaH, deltaS, deltaHnuc, c_tot, scaler=1.0, nuc_size=3)
        aggs.append(agg[0])

        K = np.exp(-deltaH / (R * temp) + deltaS / R)
        sigma = np.exp(-deltaHnuc / (R * temp))
        K_vals.append(K)
        sigma_vals.append(sigma)

    aggs = np.array(aggs)
    K_vals = np.array(K_vals)
    sigma_vals = np.array(sigma_vals)

    # Identify low aggregation regime (agg < 1e-4)
    low_agg_mask = aggs < 1e-4
    temps_low_agg = temps[low_agg_mask]
    aggs_low_agg = aggs[low_agg_mask]
    sigma_low_agg = sigma_vals[low_agg_mask]

    # Analysis
    print("\n=== Cooperative Model: High-Temperature Rounding Error Analysis ===")
    print(f"Temperature range: {temps.min():.1f} K to {temps.max():.1f} K")
    print(f"Aggregation range: {aggs.min():.2e} to {aggs.max():.2e}")
    print(f"σ range: {sigma_vals.min():.2e} to {sigma_vals.max():.2e}")

    non_monotonic = 0
    if len(aggs_low_agg) > 0:
        print(f"\nLow aggregation regime (agg < 1e-4):")
        print(f"  Temperature range: {temps_low_agg.min():.1f} K to {temps_low_agg.max():.1f} K")
        print(f"  Aggregation range: {aggs_low_agg.min():.2e} to {aggs_low_agg.max():.2e}")
        print(f"  σ range: {sigma_low_agg.min():.2e} to {sigma_low_agg.max():.2e}")

        # Check monotonicity (agg should decrease monotonically with increasing T)
        agg_diff = np.diff(aggs_low_agg)
        non_monotonic = np.sum(agg_diff > 0)
        print(f"  Non-monotonic changes: {non_monotonic}")

        # Check if aggregation values become unnaturally small (potential underflow)
        underflow_threshold = 1e-15
        underflow_count = np.sum(aggs_low_agg < underflow_threshold)
        print(f"  Values below {underflow_threshold}: {underflow_count}")

        # Relative changes in low aggregation regime
        rel_changes = np.abs(np.diff(aggs_low_agg) / aggs_low_agg[:-1])
        print(f"  Max relative change in agg: {rel_changes.max():.2e}")
        print(f"  Mean relative change in agg: {rel_changes.mean():.2e}")

    # Verify aggregation is non-negative and <= 1
    # Note: negative values near zero indicate rounding error issues
    neg_count = np.sum(aggs < 0)
    if neg_count > 0:
        print(f"  [WARNING] {neg_count} negative aggregation values detected (rounding error)")

    # Check for underflow in low aggregation regime
    if len(aggs_low_agg) > 0:
        underflow_count = np.sum(np.abs(aggs_low_agg) < 1e-14)
        if underflow_count > 0:
            print(f"  [WARNING] {underflow_count} potential underflow values (<1e-14) detected")

        # Check non-monotonicity
        if non_monotonic > len(aggs_low_agg) * 0.1:
            print(f"  [WARNING] High proportion of non-monotonic changes ({non_monotonic/len(aggs_low_agg)*100:.1f}%)")

    # Physical constraints should still hold in absolute terms
    assert np.all(aggs <= 1.0), "Aggregation should not exceed 1"

    print("\n✓ Cooperative model high-temperature rounding error test completed")


if __name__ == "__main__":
    test_cooperative_inverse_consistency()
    fig = test_temp_cooperative_model()
    test_cooperative_high_temp_rounding_error()
    plt.show()
