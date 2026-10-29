# -*- coding: utf-8 -*-
"""Unit tests .
Separate algebraic invariants, numerical consistency and model-only
physical assumptions; passing a test does not validate an experimental theory.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from npl_kinetics import (
    Parameters, build_generator, compute_rates, diagnostics, eigen_linear,
    exact_linear, find_reliable_peaks, initial_state, linear_rhs,
    nonlinear_jacobian, nonlinear_rhs, solve_stiff, stationary_linear,
    stationary_nonlinear,
)


@pytest.fixture(scope="module")
def base():
    """Shared reproducible input for all tests."""
    p = Parameters()  #  Default reference configuration
    rates = compute_rates(p)  #  Derive barriers and rate constants
    K = build_generator(rates)  # Linear generator
    x0 = initial_state(len(rates.ms))  #  All monomers initially free
    times = np.r_[0.0, np.geomspace(1e-3, 1e9, 400)]  
    return p, rates, K, x0, times


def test_reference_rates_match_previous_example(base):
    """ Reference numerical values from supplied model."""
    _, rates, _, _, _ = base
    expected_on = np.array([0.70515, 0.003813, 1.933e-5, 9.407e-8])
    expected_off = np.array([0.47235, 0.001632, 6.325e-6, 2.573e-8])
    assert np.allclose(rates.k_on, expected_on, rtol=2e-4)
    assert np.allclose(rates.k_off, expected_off, rtol=2e-4)


def test_2ml_threshold_is_parameter_dependent():
   
    p = Parameters(thickness_ml=(2, 3, 4, 5, 6))
    rates = compute_rates(p)
    assert rates.formation_effective_eV[0] > 0
    assert np.all(rates.formation_effective_eV[1:] < 0)
    # Changing reference bulk energy changes the sign.
    q = Parameters(volume_eV_A3=-2.2e-3, thickness_ml=(2, 3))
    assert compute_rates(q).formation_effective_eV[0] < 0


def test_thickness_gibbs_thomson_step(base):
    """Thinner groups have higher GT proxy."""
    _, rates, _, _, _ = base
    assert np.all(np.diff(rates.gt_thickness_eV) < 0)
    assert np.all(np.diff(rates.coexistence_factor) < 0)
    assert np.all(np.diff(rates.formation_effective_eV) < 0)


def test_lateral_correction_is_optional_and_monotone():
    """Smaller L adds a larger chemical potential."""
    raw = compute_rates(Parameters())
    #  Same 10-nm size for every group.
    with_l = compute_rates(Parameters(lateral_nm=(10., 10., 10., 10.)))
    assert np.all(with_l.gt_lateral_eV > 0)
    assert np.all(with_l.k_off > raw.k_off)
    # 3 ML: compare short and long lateral sizes 5nm VS 50nm
    short = compute_rates(Parameters(lateral_nm=(5., 20., 20., 20.)))
    long = compute_rates(Parameters(lateral_nm=(50., 20., 20., 20.)))
    assert short.coexistence_factor[0] > long.coexistence_factor[0]
    assert np.all(raw.gt_lateral_eV == 0)


def test_invalid_parameter_inputs_are_rejected():
    """Invalid shapes and nonphysical inputs fail fast."""
    with pytest.raises(ValueError):
        Parameters(temperature_K=-10)
    with pytest.raises(ValueError):
        Parameters(thickness_ml=(3, 3, 4))
    with pytest.raises(ValueError):
        Parameters(lateral_nm=(5., 10.))


def test_generator_has_zero_column_sums_and_nonnegative_offdiagonals(base):
    """ Zero column sums encode mass conservation."""
    _, _, K, _, _ = base
    assert K.shape == (5, 5)
    assert np.allclose(K.sum(axis=0), 0.0, rtol=0.0, atol=1e-14)
    offdiag = K.copy()
    np.fill_diagonal(offdiag, 0)
    assert offdiag.min() >= 0


def test_rhs_conserves_mass_at_arbitrary_positive_state(base):
    """ Both model right-hand sides have zero total derivative."""
    _, rates, K, _, _ = base
    x = np.array([.22, .33, .11, .22, .12])
    assert abs(linear_rhs(0., x, K).sum()) < 1e-15
    assert abs(nonlinear_rhs(0., x, rates).sum()) < 1e-15
    assert abs(nonlinear_rhs(0., x, rates, sqrt_epsilon=1e-12).sum()) < 1e-15


def test_linear_eigen_decomposition_agrees_with_expm(base):
    """ Eigen decomposition vs matrix exponential."""
    _, _, K, x0, _ = base
    times = np.array([0., 1., 1e3, 1e6, 1e9])
    eigen, eigvals, cond = eigen_linear(K, x0, times)
    exact = exact_linear(K, x0, times)
    assert np.max(np.abs(eigen - exact)) < 3e-8
    assert np.min(np.abs(eigvals)) == 0
    assert cond < 10
    #  Long-time expm has roundoff error.
    assert np.max(np.abs(exact.sum(axis=0) - 1.)) < 1e-7


def test_linear_stiff_solver_agrees_with_expm(base):
    """LSODA should match known linear reference."""
    _, rates, K, x0, _ = base
    times = np.array([0., 1e-3, 1., 1e3, 1e6, 1e9])
    sol = solve_stiff("linear", rates, K, x0, times, method="LSODA", rtol=1e-9, atol=1e-12)
    expected = exact_linear(K, x0, times)
    assert np.max(np.abs(sol.y - expected)) < 6e-8
    assert diagnostics(sol, x0)["mass_error_max"] < 1e-7


@pytest.fixture(scope="module")
def nonlinear_case(base):
    """ Cache a single long nonlinear solve for tests."""
    _, rates, K, x0, times = base
    sol = solve_stiff("nonlinear", rates, K, x0, times, method="LSODA",
                      rtol=1e-9, atol=1e-12)
    return sol


def test_nonlinear_mass_conservation(base, nonlinear_case):
    """Nonlinear flux exchange still conserves monomers."""
    _, _, _, x0, _ = base
    assert diagnostics(nonlinear_case, x0)["mass_error_max"] < 1e-11


def test_nonnegative_populations(base, nonlinear_case):
    """ No material negative inventory."""
    _, _, K, x0, times = base
    linear = exact_linear(K, x0, times[::50])
    assert linear.min() > -1e-10
    assert nonlinear_case.y.min() > -1e-10


def test_analytic_equilibrium_satisfies_balance(base):
    """Equilibria satisfy zero flux."""
    _, rates, K, _, _ = base
    xlin = stationary_linear(rates)
    xnon = stationary_nonlinear(rates)
    assert abs(xlin.sum() - 1.) < 1e-13
    assert abs(xnon.sum() - 1.) < 1e-13
    assert np.max(np.abs(linear_rhs(1., xlin, K))) < 1e-14
    assert np.max(np.abs(nonlinear_rhs(1., xnon, rates))) < 1e-14
    assert np.allclose(xlin[1:] / xlin[0], rates.k_on / rates.k_off, rtol=1e-12)
    assert np.allclose(np.sqrt(xnon[1:]) / xnon[0]**2,
                       rates.k_on / rates.k_off, rtol=1e-12)


def test_nonlinear_long_time_agrees_with_equilibrium(base, nonlinear_case):
    """Long-time state reaches nonlinear equilibrium."""
    _, rates, _, _, _ = base
    eq = stationary_nonlinear(rates)
    assert np.max(np.abs(nonlinear_case.y[:, -1] - eq)) < 2e-7


def test_nonlinear_peak_detection_rejects_plateau(base, nonlinear_case):
    """ Reject numerical plateau artifacts for 6 ML."""
    _, rates, K, _, _ = base
    peaks = find_reliable_peaks(nonlinear_case, rates, stationary_nonlinear(rates),
                                "nonlinear", K, prominence_min=1e-4)
    assert len(peaks) == 4
    assert all(p.t_seconds is not None for p in peaks[:3])
    assert peaks[0].t_seconds < peaks[1].t_seconds < peaks[2].t_seconds
    assert peaks[3].t_seconds is None


def test_sqrt_regularization_preserves_conservation_and_changes_model(base):
    """ Epsilon changes kinetics but preserves mass."""
    _, rates, _, _, _ = base
    eps = 1e-10
    x = np.array([.5, .125, .125, .125, .125])
    unreg = nonlinear_rhs(0., x, rates)
    reg = nonlinear_rhs(0., x, rates, sqrt_epsilon=eps)
    assert not np.array_equal(unreg, reg)
    assert abs(reg.sum()) < 1e-15
    eq = stationary_nonlinear(rates, sqrt_epsilon=eps)
    assert abs(eq.sum() - 1.) < 1e-12
    assert np.max(np.abs(nonlinear_rhs(0., eq, rates, sqrt_epsilon=eps))) < 1e-12
    J = nonlinear_jacobian(0., x, rates, sqrt_epsilon=eps)
    assert J.shape == (5, 5)
    assert np.allclose(J.sum(axis=0), 0., atol=1e-15)


def test_multiple_stiff_integrators_are_consistent_at_short_horizon(base):
    """Cross-check LSODA and Radau over a shorter horizon."""
    _, rates, K, x0, _ = base
    times = np.r_[0., np.geomspace(1e-3, 1e5, 60)]
    s_lsoda = solve_stiff("nonlinear", rates, K, x0, times,
                          method="LSODA", rtol=2e-8, atol=2e-11)
    s_radau = solve_stiff("nonlinear", rates, K, x0, times,
                          method="Radau", rtol=2e-8, atol=2e-11)
    assert np.max(np.abs(s_lsoda.y - s_radau.y)) < 2e-6


def test_linear_mass_conservation_after_lateral_gt_correction():
    """ Geometry changes rates, not conservation."""
    r = compute_rates(Parameters(lateral_nm=(10., 20., 30., 40.)))
    K = build_generator(r)
    assert np.allclose(K.sum(axis=0), 0, atol=1e-14)
    assert np.max(np.abs(K @ stationary_linear(r))) < 1e-14


def test_regularized_jacobian_matches_finite_differences(base):
    """ Check the regularized Jacobian by finite differences."""
    _, rates, _, _, _ = base
    eps = 1e-9
    x = np.array([.31, .12, .15, .20, .22])  # Interior positive state
    J = nonlinear_jacobian(0., x, rates, sqrt_epsilon=eps)
    step = 1e-6
    numerical = np.zeros_like(J)
    for column in range(len(x)):
        shift = np.zeros_like(x)
        shift[column] = step
        numerical[:, column] = (
            nonlinear_rhs(0., x + shift, rates, sqrt_epsilon=eps)
            - nonlinear_rhs(0., x - shift, rates, sqrt_epsilon=eps)
        ) / (2 * step)
    assert np.allclose(J, numerical, rtol=2e-8, atol=1e-9)


def test_zero_bound_population_has_no_detachment(base):
    """ Zero inventory cannot generate detachment flux."""
    _, rates, _, x0, _ = base
    raw = nonlinear_rhs(0., x0, rates, sqrt_epsilon=0.)
    reg = nonlinear_rhs(0., x0, rates, sqrt_epsilon=1e-9)
    assert np.allclose(raw[1:], rates.k_on)
    assert np.allclose(reg[1:], rates.k_on)
    assert raw[0] == -raw[1:].sum()
