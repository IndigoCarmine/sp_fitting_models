"""
Tests for utility functions in models.utils module.
"""

import numpy as np
import pytest

from sp_fitting_models.models.utils import solve_cubic_vectorized


class TestSolveCubicVectorized:
    """Test suite for solve_cubic_vectorized function."""

    def test_cubic_normal_case(self):
        """Test normal 3rd-order polynomial with roots at boundaries."""
        # Test: x^3 - 2*x^2 + x = x(x-1)^2 = 0
        # Roots: x = 0, 1, 1 (analytical) - roots are AT the boundaries!
        a = np.array([1.0])
        b = np.array([-2.0])
        c = np.array([1.0])
        d = np.array([0.0])
        x_low = np.array([0.0])
        x_high = np.array([1.0])

        # Since f(0)=0 and f(1)=0, bisection will find one of these roots
        result = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
        assert isinstance(result, np.ndarray)
        assert 0.0 <= float(result[0]) <= 1.0
        # Verify result is close to a root
        x_val = float(result[0])
        f_val = a[0] * x_val**3 + b[0] * x_val**2 + c[0] * x_val + d[0]
        assert np.isclose(f_val, 0.0, atol=1e-9)

    def test_quadratic_case_a_zero(self):
        """
        Test when a=0 (quadratic equation) with roots at boundaries.

        For 2*x^2 - 2*x = 2*x(x-1), roots are x=0 and x=1 (the boundaries).
        Bisection will converge to one of these roots.
        """
        # Test: 2*x^2 - 2*x = 0 (roots: x=0, 1)
        a = np.array([0.0])
        b = np.array([2.0])
        c = np.array([-2.0])
        d = np.array([0.0])
        x_low = np.array([0.0])
        x_high = np.array([1.0])

        # Roots at boundaries are legitimate
        result = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
        x_val = float(result[0])
        f_result = b[0] * x_val**2 + c[0] * x_val + d[0]
        assert np.isclose(f_result, 0.0, atol=1e-9), f"Result should be close to a root, but f({x_val}) = {f_result}"

    def test_linear_case_a_b_zero(self):
        """
        Test when a=0 and b=0 (linear equation).

        Problem: c*x + d = 0 reduces to a linear equation.
        If the function is monotonic and bounded by the interval,
        it should still work. But if both endpoints have same sign,
        root bracketing fails.
        """
        # Test: 2*x - 1 = 0 (root: x=0.5)
        a = np.array([0.0])
        b = np.array([0.0])
        c = np.array([2.0])
        d = np.array([-1.0])
        x_low = np.array([0.0])
        x_high = np.array([1.0])

        try:
            result = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
            # Should find x ≈ 0.5
            assert np.isclose(float(result[0]), 0.5, atol=1e-9)
        except ValueError as e:
            pytest.fail(f"Linear case should work, but got error: {e}")

    def test_all_zero_coefficients(self):
        """
        Test degenerate case: a=b=c=d=0, giving f(x)=0 everywhere.

        When boundaries have same sign (both zero in this case), the function
        should still be solvable in principle since any x is a root.
        However, bisection may not converge properly or detect the issue.
        This test documents the current behavior.
        """
        a = np.array([0.0])
        b = np.array([0.0])
        c = np.array([0.0])
        d = np.array([0.0])
        x_low = np.array([0.0])
        x_high = np.array([1.0])

        # f(x) = 0 everywhere is an edge case
        # Current implementation returns a value (typically near left boundary)
        # without raising an error, since f(0)*f(1) = 0*0 <= 0 is satisfied
        result = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
        # Any point is technically a valid root for f(x)=0
        assert isinstance(result, np.ndarray)
        assert 0.0 <= float(result[0]) <= 1.0

    def test_quadratic_no_sign_change(self):
        """Test quadratic with no sign change in bracket interval."""
        # Test: x^2 + 1 = 0 (no real roots)
        a = np.array([0.0])
        b = np.array([1.0])
        c = np.array([0.0])
        d = np.array([1.0])
        x_low = np.array([0.0])
        x_high = np.array([1.0])

        # f(x) = x^2 + 1 > 0 for all x
        # No sign change, should raise ValueError
        with pytest.raises(ValueError):
            solve_cubic_vectorized(a, b, c, d, x_low, x_high)

    def test_vectorized_with_mixed_coefficients(self):
        """Test vectorized operation with array inputs including a=0 case."""
        # Multiple problems: some cubic (a!=0), some quadratic (a=0)
        a = np.array([1.0, 0.0, 1.0])
        b = np.array([-2.0, 2.0, 0.0])
        c = np.array([1.0, -2.0, 1.0])
        d = np.array([0.0, 0.0, 0.0])
        x_low = np.array([0.0, 0.0, 0.0])
        x_high = np.array([1.0, 1.0, 1.0])

        # This is a mixed case; all have roots at boundaries
        # Expected: should succeed as these are valid bracketed roots
        result = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
        assert len(result) == 3
        # All should be valid roots
        for i in range(3):
            x_val = float(result[i])
            f_val = a[i] * x_val**3 + b[i] * x_val**2 + c[i] * x_val + d[i]
            assert np.isclose(f_val, 0.0, atol=1e-9), f"Element {i}: f({x_val}) = {f_val} should be ~0"

    def test_stability_near_zero_a(self):
        """Test numerical stability when a is very small but non-zero."""
        # Nearly quadratic: a = 1e-10
        a = np.array([1e-10])
        b = np.array([-2.0])
        c = np.array([1.0])
        d = np.array([0.0])
        x_low = np.array([0.0])
        x_high = np.array([1.0])

        result = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
        # Should find a root in [0, 1]
        assert 0.0 <= float(result[0]) <= 1.0

        # Verify it's actually a valid root
        x_val = float(result[0])
        f_val = a[0] * x_val**3 + b[0] * x_val**2 + c[0] * x_val + d[0]
        assert np.isclose(f_val, 0.0, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
