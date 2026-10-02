"""
Robustness tests: the models must stay finite, within [0, 1] (times scaler) and smooth
for extreme parameters, because the fitting optimiser can propose such values.
"""

import itertools

import lmfit as lm
import numpy as np
import pytest

from sp_fitting_models import _core
from sp_fitting_models.data import TempVsAggData
from sp_fitting_models.fitting import objective_temp_cooperative
from sp_fitting_models.models import (
    temp_cooperative_model,
    temp_cooperative_model_n,
    temp_coop_iso_model,
    temp_isodesmic_model,
    temp_isodesmic_model_direct,
)

TEMPS = np.linspace(250, 420, 40)
DELTA_H = [-1e7, -1e6, -2e5, -96000, 0.0, 1e5, 1e6, 1e7]
DELTA_S = [-2000.0, -180.0, 0.0, 500.0]
DELTA_H_NUC = [-1e7, -1e5, 0.0, 1e4, 1e5, 1e7]
C_TOT = [1e-12, 1e-5, 1.0]


def _assert_valid(agg: np.ndarray) -> None:
    agg = np.asarray(agg)
    assert np.all(np.isfinite(agg)), agg
    assert np.all(agg >= 0.0) and np.all(agg <= 1.0), agg


@pytest.mark.parametrize("c_tot", C_TOT)
def test_isodesmic_extreme(c_tot):
    for dh, ds in itertools.product(DELTA_H, DELTA_S):
        _assert_valid(temp_isodesmic_model(TEMPS, dh, ds, c_tot))
        _assert_valid(temp_isodesmic_model_direct(TEMPS, dh, ds, c_tot))


@pytest.mark.parametrize("c_tot", C_TOT)
def test_cooperative_extreme(c_tot):
    for dh, ds, dhn in itertools.product(DELTA_H, DELTA_S, DELTA_H_NUC):
        _assert_valid(temp_cooperative_model(TEMPS, dh, ds, dhn, c_tot))
        _assert_valid(temp_cooperative_model_n(TEMPS, dh, ds, dhn, c_tot, nuc_size=4))


@pytest.mark.parametrize("c_tot", C_TOT)
def test_mixed_extreme(c_tot):
    for dh_i, dh_c, dhn in itertools.product(DELTA_H, DELTA_H, DELTA_H_NUC):
        _assert_valid(temp_coop_iso_model(TEMPS, dh_i, -180, dh_c, -180, dhn, c_tot))


def test_isodesmic_direct_matches_bisection():
    """The rewritten closed form must agree with the bisection over many decades of K*c."""
    for b in np.logspace(-15, 15, 61):
        direct = _core.isodesmic_model_direct(1e-5, b / 1e-5)
        bisect = _core.isodesmic_model(1e-5, b / 1e-5, 100)
        assert direct == pytest.approx(bisect, rel=1e-9, abs=1e-14)


def test_small_aggregation_is_accurate():
    """Small K: aggregation ~ 2 K c (isodesmic). Previously this returned huge negative values."""
    for kc in [1e-5, 1e-8, 1e-12]:
        agg = _core.isodesmic_model(1e-5, kc / 1e-5, 100)
        assert agg == pytest.approx(2 * kc, rel=1e-4, abs=1e-15)
        agg_direct = _core.isodesmic_model_direct(1e-5, kc / 1e-5)
        assert agg_direct == pytest.approx(2 * kc, rel=1e-4)


def test_monotonic_in_parameter_sweep():
    """No oscillation: aggregation must be monotone in K across many decades."""
    ks = np.logspace(-20, 30, 500)
    for sigma in [1e-8, 1e-3, 1.0]:
        agg = np.array([_core.cooperative_model(1e-5, k, sigma, 100) for k in ks])
        assert np.all(np.diff(agg) >= -1e-12)
        agg_n = np.array([_core.cooperative_model_n(1e-5, k, sigma, 4, 100) for k in ks])
        assert np.all(np.diff(agg_n) >= -1e-12)
    agg_iso = np.array([_core.isodesmic_model(1e-5, k, 100) for k in ks])
    assert np.all(np.diff(agg_iso) >= -1e-12)


@pytest.mark.parametrize("conc", [0.0, -1e-5, np.nan, np.inf])
def test_obvious_bad_input_raises(conc):
    """Data/config errors (never proposed by the optimiser) must raise, not hide as numbers."""
    with pytest.raises(ValueError):
        _core.isodesmic_model(conc, 1e5, 100)
    with pytest.raises(ValueError):
        _core.isodesmic_model_direct(conc, 1e5)
    with pytest.raises(ValueError):
        _core.cooperative_model(conc, 1e5, 1e-3, 100)
    with pytest.raises(ValueError):
        _core.cooperative_model_n(conc, 1e5, 1e-3, 3, 100)
    with pytest.raises(ValueError):
        _core.coop_iso_model(conc, 1e3, 1e5, 1e-3, 100)
    with pytest.raises(ValueError):
        temp_cooperative_model(TEMPS, -96000, -180, 20000, conc)
    with pytest.raises(ValueError):
        temp_isodesmic_model_direct(TEMPS, -96000, -180, conc)


@pytest.mark.parametrize("bad_t", [0.0, -10.0, np.nan])
def test_bad_temperature_raises(bad_t):
    temps = np.array([300.0, bad_t, 350.0])
    with pytest.raises(ValueError):
        temp_isodesmic_model(temps, -96000, -180, 1e-5)
    with pytest.raises(ValueError):
        temp_isodesmic_model_direct(temps, -96000, -180, 1e-5)
    with pytest.raises(ValueError):
        temp_cooperative_model(temps, -96000, -180, 20000, 1e-5)
    with pytest.raises(ValueError):
        temp_cooperative_model_n(temps, -96000, -180, 20000, 1e-5, nuc_size=3)
    with pytest.raises(ValueError):
        temp_coop_iso_model(temps, -96000, -180, -96000, -180, 20000, 1e-5)


def test_invalid_fit_params_return_nan():
    """Invalid parameter values (fit-related) give NaN rather than an exception."""
    c = 1e-5
    for k in [-1.0, np.nan]:
        assert np.isnan(_core.isodesmic_model(c, k, 100))
        assert np.isnan(_core.isodesmic_model_direct(c, k))
        assert np.isnan(_core.cooperative_model(c, k, 1e-3, 100))
        assert np.isnan(_core.cooperative_model_n(c, k, 1e-3, 3, 100))
        assert np.isnan(_core.coop_iso_model(c, k, 1e5, 1e-3, 100))
        assert np.isnan(_core.coop_iso_model(c, 1e3, k, 1e-3, 100))
    for sigma in [-1.0, np.nan]:
        assert np.isnan(_core.cooperative_model(c, 1e5, sigma, 100))
        assert np.isnan(_core.cooperative_model_n(c, 1e5, sigma, 3, 100))
        assert np.isnan(_core.coop_iso_model(c, 1e3, 1e5, sigma, 100))
    assert np.all(np.isnan(temp_cooperative_model(TEMPS, np.nan, -180, 20000, c)))
    assert np.all(np.isnan(temp_cooperative_model(TEMPS, -96000, -180, np.nan, c)))
    assert np.all(np.isnan(temp_isodesmic_model_direct(TEMPS, np.nan, -180, c)))


def test_physical_limits():
    """K = 0 / inf and sigma = 0 / inf are physical limits with finite answers."""
    c = 1e-5
    assert _core.isodesmic_model(c, 0.0, 100) == pytest.approx(0.0, abs=1e-12)
    assert _core.isodesmic_model_direct(c, 0.0) == 0.0
    assert _core.isodesmic_model(c, np.inf, 100) == 1.0
    assert _core.isodesmic_model_direct(c, np.inf) == 1.0
    for sigma in [0.0, 1e-3, np.inf]:
        assert _core.cooperative_model(c, 0.0, sigma, 100) == pytest.approx(0.0, abs=1e-12)
        assert _core.cooperative_model(c, np.inf, sigma, 100) == 1.0
        assert _core.cooperative_model_n(c, np.inf, sigma, 3, 100) == 1.0
    for sigma in [0.0, np.inf]:
        assert 0.0 <= _core.cooperative_model(c, 1e5, sigma, 100) <= 1.0
        assert 0.0 <= _core.cooperative_model_n(c, 1e5, sigma, 3, 100) <= 1.0


def test_fit_with_extreme_initial_guess():
    """A fit started far from the truth must stay finite and still converge."""
    temps = np.linspace(280, 400, 100)
    true = dict(deltaH=-96000, deltaS=-180, deltaHnuc=20000)
    data = [
        TempVsAggData(temp=temps, agg=np.asarray(temp_cooperative_model(temps, c_tot=c, **true)), concentration=c)
        for c in (50e-6, 100e-6)
    ]

    def make_params(dh, ds, dhn):
        params = lm.Parameters()
        params.add("deltaH", value=dh, min=-1e6, max=0)
        params.add("deltaS", value=ds, min=-2000, max=0)
        params.add("deltaHnuc", value=dhn, min=-1e5, max=2e5)
        params.add("scaler", value=1.0, vary=False)
        return params

    # Extremely far start: the optimiser may end in a local minimum, but every model
    # evaluation along the way must remain finite.
    params = make_params(-5e5, -1500, 1e5)
    assert np.all(np.isfinite(objective_temp_cooperative(params, data)))
    result = lm.minimize(objective_temp_cooperative, params, args=(data,), method="least_squares")
    assert np.all(np.isfinite(result.residual))

    # Far (but not degenerate) start: must recover the true parameters.
    result = lm.minimize(
        objective_temp_cooperative, make_params(-1.5e5, -300, 1.5e5), args=(data,), method="least_squares"
    )
    assert np.sum(result.residual**2) < 1e-10
    assert result.params["deltaH"].value == pytest.approx(true["deltaH"], rel=1e-4)
    assert result.params["deltaHnuc"].value == pytest.approx(true["deltaHnuc"], rel=1e-3)
