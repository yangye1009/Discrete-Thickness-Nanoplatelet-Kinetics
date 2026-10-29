# -*- coding: utf-8 -*-
""" Discrete-thickness NPL kinetics.
"""

from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import eig, expm
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]
ModelKind = Literal["linear", "nonlinear"]


@dataclass(frozen=True)#Frozen Dataclass
class Parameters:
    """Input parameters (reference set).
     Energies are in eV, lengths in Å, temperature in K, time in s.
    """
    temperature_K: float = 473.15  #  Absolute temperature (K)
    kB_eV_K: float = 8.617333262e-5  #  Boltzmann constant (eV/K)
    layer_height_A: float = 3.10  # One-layer height (Å)
    surface_eV_A2: float = 5.7e-3  # Facet energy density (eV/Å²)
    edge_eV_A: float = 37.1e-3  # Step-edge line energy (eV/Å)
    volume_eV_A3: float = -1.5e-3  # Bulk formation energy density (eV/Å³)
    prefactor_s1: float = 2.8e11  #  Effective attempt prefactor (s⁻¹)

    thickness_ml: tuple[int, ...] = (3, 4, 5, 6)  #  Thickness groups (ML)
    lateral_nm: tuple[float, ...] | None = None  # Optional fixed lateral sizes(L)

    def __post_init__(self) -> None:
        """ Reject nonphysical or incompatible inputs."""
        if self.temperature_K <= 0 or self.kB_eV_K <= 0 or self.layer_height_A <= 0:
            raise ValueError("Temperature, kB, and layer height must be positive")
        if self.surface_eV_A2 < 0 or self.edge_eV_A < 0 or self.prefactor_s1 <= 0:
            raise ValueError("Surface/edge energies must be nonnegative and C0 positive")
        if not self.thickness_ml or any(m <= 0 for m in self.thickness_ml):
            raise ValueError("Thicknesses must be positive integers")
        if any(b <= a for a, b in zip(self.thickness_ml, self.thickness_ml[1:])):
            raise ValueError("Thicknesses must be strictly increasing")
        if self.lateral_nm is not None:
            if len(self.lateral_nm) != len(self.thickness_ml):
                raise ValueError("Provide one lateral size per thickness")
            if any(not np.isfinite(z) or z <= 0 for z in self.lateral_nm):
                raise ValueError("Lateral sizes must be positive finite nm values")

@dataclass(frozen=True) #k_on、k_off
class Rates:
    """ Derived arrays; all entries ordered by thickness_ml."""

    ms: FloatArray
    barrier_eV: FloatArray
    formation_bulk_eV: FloatArray
    gt_thickness_eV: FloatArray
    gt_lateral_eV: FloatArray
    formation_effective_eV: FloatArray
    coexistence_factor: FloatArray
    k_on: FloatArray
    k_off: FloatArray

# ============================================================================
#  TRACEABLE ORIGINAL-PAPER SOURCES
# ============================================================================
#  Values below are transcribed from the 2017 Riedinger paper/SI.

PAPER_RIEDINGER = (
    "Riedinger et al., How semiconductor nanoplatelets form, "
    "Nature Materials (2017); arXiv:1605.06553v2 with Supplementary Information"
)
PAPER_KNUESEL = (
    "Knuesel et al., Experimental Evidence for Two-Dimensional Ostwald Ripening "
    "in Semiconductor Nanoplatelets, Chemistry of Materials 32 (2020), "
    "DOI:10.1021/acs.chemmater.0c01238"
)

# Classify parameter provenance.
ORIGINAL_PAPER_VALUES = {
    "layer_height_A": {
        "value": 3.10, "unit": "Å", "method": "lattice geometry ",
        "source": "Riedinger SI  p.26, S4.a, L1=a0/2"},
    "surface_eV_A2": {
        "value": 5.7e-3, "unit": "eV/Å²", "method": "DFT (Se-rich, acetate)",
        "source": "Riedinger SI  pp.26,32, equations S1/S25-S28"},
    "edge_eV_A": {
        "value": 37.1e-3, "unit": "eV/Å", "method": "DFT mean of step geometries ",
        "source": "Riedinger SI  pp.26,32, S4.c; edge range 24–47 meV/Å"},
    "volume_eV_A3": {
        "value": -1.5e-3, "unit": "eV/Å³", "method": "experimental constraint ",
        "source": "Riedinger SI  p.33, S29-S30; 2 ML absent, 3 ML observed"},
    "prefactor_s1": {
        "value": 2.8e11, "unit": "s⁻¹", "method": "optical kinetic fit ",
        "source": "Riedinger SI  p.30, S19-S21; normalized absorbance"},
    "temperature_K": {
        "value": 473.15, "unit": "K", "method": "model/example condition ",
        "source": "Riedinger main text  pp.10-11, ~200°C"},
}


# Formula strings are documentary reference.
ORIGINAL_PAPER_EQUATIONS = {
    "S1": "ΔE = E_V ΔV + E_A ΔA + E_L ΔL",
    "S2": "ΔE(a,P_island,L_step)=L1 E_V a + L1 E_A P_island + E_L (L_step+4L1)",
    "S3": "minimum energy at fixed island area a (shape optimization)",
    "S4-S5": "E_wide(a)=L1 E_V a+(4L1 E_A+2E_L) sqrt(a)+4L1 E_L",
    "S6-S7": "E_narrow,m(a)=(L1 E_V+2E_A/m)a+2m L1² E_A+(m+4)L1 E_L",
    "S8": "N_tot = sum_{m in {0,3,4,5,6}} N_m(t)",
    "S9": "dN_m/dt=k_m N_0-k_{-m} N_m",
    "S10": "dN_0/dt=-N_0 sum_m k_m+sum_m k_{-m} N_m",
    "S11-S12": "dx/dt=Kx, K column sums=0",
    "S13-S18": "x(t)=U exp(Λt) U^{-1} x(0) (if K diagonalizable)",
    "S19": "k_m=C0 m exp(-ΔE‡_m/(k_B T))",
    "S20": "k_{-m}=C0 m exp(-ΔE‡_m/(k_B T)) A_m",
    "S21": "A_m=k_{-m}/k_m=exp(ΔH_f,m/(k_B T)); ΔH_f,m=2L1³ E_V+4L1² E_A/m",
    "S22": "dN_m/dt=k_m N_0²-k_{-m} sqrt(N_m)",
    "S23": "K_m=exp(-ΔH_f,m/(k_B T))=sqrt(N_m,eq)/(N_0,eq²)",
    "S24": "N_i,eq/N_j,eq=(K_i/K_j)²",
    "S29": "dE_narrow,m/da=L1 E_V+2E_A/m<0 for favorable lateral growth",
    "S30": "-E_A/L1 < E_V < -2E_A/(3L1) (2 ML disfavored, 3 ML favored)",
}

# 2020 CdS observations: validation context, not a transfer of CdSe parameters.
KNUSEL_2020_OBSERVATIONS = {
    "material": "CdS, NOT CdSe ",
    "ripening_temperature_C": [160, 170, 180, 190],
    "ripening_hold_h": 12 + 50/60,
    "baby_2ml_absorption_nm": 326,
    "population_peaks_nm": {"3_ML": 374, "4_ML": 407, "5_ML": 429},
    "interpretation": "2-ML disappears below 160°C; thicker groups appear with increased temperature",
    "source": "Knuesel et al. 2020  pp.4-5, Methods and Figure 2; "
              "2-ML/3-ML assignment via electron/light-hole transitions",
}


def paper_wide_island_energy(area_A2: float | FloatArray, p: Parameters) -> FloatArray:
    """ Paper S4-S5, wide-facet island energy (eV)."""
    a = np.asarray(area_A2, dtype=float)
    if np.any(a < 0):
        raise ValueError("Area a must be nonnegative")
    L1, EA, EL, EV = p.layer_height_A, p.surface_eV_A2, p.edge_eV_A, p.volume_eV_A3
    return (L1*EV)*a + (4*L1*EA+2*EL)*np.sqrt(a) + 4*L1*EL


def paper_wide_barrier(p: Parameters) -> tuple[float, float]:
    """ Return (critical area Å², peak energy eV) from S5."""
    A = p.layer_height_A*p.volume_eV_A3
    if A >= 0:
        raise ValueError("S5 classical maximum requires E_V < 0 ")
    B = 4*p.layer_height_A*p.surface_eV_A2 + 2*p.edge_eV_A
    area = (B/(2*abs(A)))**2
    energy = B**2/(4*abs(A)) + 4*p.layer_height_A*p.edge_eV_A
    return float(area), float(energy)


def paper_narrow_island_energy(area_A2: float | FloatArray, m: int, p: Parameters) -> FloatArray:
    """S6-S7 spanning-island branch."""
    a = np.asarray(area_A2, dtype=float)
    if m <= 0 or np.any(a < 0):
        raise ValueError("Positive thickness and nonnegative area required")
    L1, EA, EL, EV = p.layer_height_A, p.surface_eV_A2, p.edge_eV_A, p.volume_eV_A3
    return (L1*EV+2*EA/m)*a + 2*m*L1**2*EA + (m+4)*L1*EL


def paper_growth_threshold(p: Parameters) -> tuple[float, tuple[float, float]]:
    """S29-S30 Return critical m and allowable E_V interval in eV/Å³.
       S30 follows conditional 2-ML/3-ML observations, not a universal limit.
    """
    if p.volume_eV_A3 >= 0:
        raise ValueError("Growth threshold requires negative volume driving energy")
    critical_m = 2*p.surface_eV_A2/(p.layer_height_A*abs(p.volume_eV_A3))
    ev_low = -p.surface_eV_A2/p.layer_height_A
    ev_high = -2*p.surface_eV_A2/(3*p.layer_height_A)
    return float(critical_m), (float(ev_low), float(ev_high))


def paper_reference_audit(p: Parameters, r: Rates) -> dict:
    """ Machine-readable original-source audit. Track equation provenance. 
        Lateral GT and sqrt regularization are extensions.
    """
    comparisons = {}
    for key, record in ORIGINAL_PAPER_VALUES.items():
        actual = float(getattr(p, key))
        comparisons[key] = {**record, "used_value": actual,
                             "same_as_original": bool(np.isclose(actual, record["value"], rtol=1e-12, atol=0.))}
    area, wide = paper_wide_barrier(p)
    mcrit, ev_bounds = paper_growth_threshold(p)
    return {
        "reference_original_CdSe": PAPER_RIEDINGER,
        "reference_independent_CdS_experiment": PAPER_KNUESEL,
        "original_values": comparisons,
        "equations": ORIGINAL_PAPER_EQUATIONS,
        "CdS_independent_observations": KNUSEL_2020_OBSERVATIONS,
        "computed_from_selected_parameters": {
            "critical_island_area_A2_S5": area,
            "wide_peak_energy_eV_S5": wide,
            "narrow_intercept_eV_S7": {str(int(m)): float(e) for m,e in zip(r.ms, r.barrier_eV)},
            "formation_energy_eV_S21_or_extension": {str(int(m)): float(e) for m,e in zip(r.ms, r.formation_effective_eV)},
            "k_on_s1_S19": {str(int(m)): float(e) for m,e in zip(r.ms, r.k_on)},
            "k_off_s1_S20": {str(int(m)): float(e) for m,e in zip(r.ms, r.k_off)},
            "critical_thickness_S29": mcrit,
            "EV_bounds_eV_A3_S30": list(ev_bounds),
        },
        "original_equations_unmodified": bool(p.lateral_nm is None),  # S21 Only no lateral extension
        "barrier_model": "S7 narrow-branch intercept approximation; not the full minimum-energy island path",
        "lateral_gt_extension_enabled": bool(p.lateral_nm is not None),
        "note": "Rates are model-derived predictions, NOT directly measured constants; "
                "nonlinear rate-law units require normalized dimensionless inventory.",
    }


def export_paper_audit(path: Path, p: Parameters, r: Rates) -> None:
    """Export original-paper parameter provenance and
        model-derived data to a JSON report."""

    with path.open("w", encoding="utf-8") as f:
        json.dump(paper_reference_audit(p, r), f, ensure_ascii=False, indent=2)


def compute_rates(p: Parameters) -> Rates:
    """Compute Gibbs–Thomson proxy and rates.
    The barrier is a narrow-facet intercept approximation; 
    the thickness contribution is 2γΩ/h with Ω=2L1³. Optional lateral correction
    is 2γΩ/L for a square platelet, equal facet energies and fixed L.
    """
    ms = np.asarray(p.thickness_ml, dtype=float)  #  Vectorize ML labels
    L1, gamma = p.layer_height_A, p.surface_eV_A2
    omega = 2.0 * L1**3  #  Formula-unit volume proxy (Å³)
    kT = p.kB_eV_K * p.temperature_K  #  Thermal energy scale (eV)

    barrier = 2.0 * ms * L1**2 * gamma + (ms + 4.0) * L1 * p.edge_eV_A
    bulk = np.full_like(ms, omega * p.volume_eV_A3)  #  Bulk contribution
    thickness = 2.0 * gamma * omega / (ms * L1)  #  Thickness GT term
    lateral = np.zeros_like(ms)  # Disabled by default
    if p.lateral_nm is not None:
        lateral_A = 10.0 * np.asarray(p.lateral_nm, dtype=float)  # Convert nm to Å
        lateral = 2.0 * gamma * omega / lateral_A  # Lateral GT term

    dH_eff = bulk + thickness + lateral  #  Effective formation-energy proxy
    A = np.exp(dH_eff / kT)  # Coexistence factor k_off/k_on
    k_on = p.prefactor_s1 * ms * np.exp(-barrier / kT)  #  Activated attachment
    k_off = k_on * A  # Reverse effective rate via selected ratio
    return Rates(ms, barrier, bulk, thickness, lateral, dH_eff, A, k_on, k_off)


def build_generator(r: Rates) -> FloatArray:
    """ Construct the mass-conserving linear generator.
     State is (free pool, discrete populations);
     zero column sums
     conserve total inventory, with no direct m→m+1 transitions.
    """
    n = len(r.ms)  #  Number of thickness populations
    K = np.zeros((n + 1, n + 1), dtype=float)  #  Include free pool
    K[0, 0] = -float(np.sum(r.k_on))  #  Total free-pool outflux rate
    K[0, 1:] = r.k_off  #  Detachment back to free pool
    K[1:, 0] = r.k_on  #  Attachment into each population
    K[1:, 1:] = -np.diag(r.k_off)  #  Population outfluxes
    return K


def initial_state(n: int, total: float = 1.0) -> FloatArray:
    """ Initialize all inventory in the free pool."""
    if n < 1 or total <= 0:
        raise ValueError("Need at least one population and positive total inventory")
    x0 = np.zeros(n + 1, dtype=float)  # No initial bound monomers
    x0[0] = total  #  Free pool initially owns all inventory 1.0
    return x0

#1
def linear_rhs(t: float, x: FloatArray, K: FloatArray) -> FloatArray:
    """Linear ODE right-hand side; autonomous in t."""
    return K @ x 

#2
def nonlinear_rhs(t: float, x: FloatArray, r: Rates, sqrt_epsilon: float = 0.0) -> FloatArray:
    """ Nonlinear effective S22 model.
    (1)J_on,m=k_on,m*N0²;J_off,m=k_off,m*sqrt(Nm)。
    sqrt_epsilon>0 → sqrt(N) = sqrt(N+eps)-sqrt(eps),
    Optional epsilon regularization has a finite boundary slope,
    preserves zero detachment at zero, but CHANGES the original kinetics.
    
    (2)max(x,0) 
    Clipping here only protects roundoff/intermediate solver probes. 
    Large negative physical populations should be reported as integration failures.
    """
    if sqrt_epsilon < 0:
        raise ValueError("sqrt_epsilon must be >= 0")
    free = max(float(x[0]), 0.0)  #  Guard the free pool
    bound = np.maximum(x[1:], 0.0)  #  Guard square-root evaluations
    detach = np.sqrt(bound + sqrt_epsilon) - np.sqrt(sqrt_epsilon)
    flux = r.k_on * free**2 - r.k_off * detach  #  Net population flux
    return np.r_[-flux.sum(), flux]  #  Exact algebraic conservation

#Jacobian 
def nonlinear_jacobian(t: float, x: FloatArray, r: Rates, sqrt_epsilon: float) -> FloatArray:
    """ Jacobian of the REGULARIZED model only.
    The raw square-root law has an infinite derivative at N=0;
    only use this Jacobian when epsilon is strictly positive.
    """
    if sqrt_epsilon <= 0:
        raise ValueError("Analytical Jacobian requires sqrt_epsilon > 0")
    free = max(float(x[0]), 0.0)
    bound = np.maximum(x[1:], 0.0)
    on_slope = 2.0 * r.k_on * free  # ∂J_on/∂N0 / Free-pool derivative
    off_slope = r.k_off / (2.0 * np.sqrt(bound + sqrt_epsilon))
    n = len(r.ms)
    J = np.zeros((n + 1, n + 1), dtype=float)
    J[0, 0] = -on_slope.sum()
    J[0, 1:] = off_slope
    J[1:, 0] = on_slope
    J[1:, 1:] = -np.diag(off_slope)
    return J


def stationary_linear(r: Rates, total: float = 1.0) -> FloatArray:
    """Solve k_on*N0=k_off*Nm with mass balance."""
    ratio = r.k_on / r.k_off  # Nm/N0 / Population to free-pool ratio
    n0 = total / (1.0 + ratio.sum())
    return np.r_[n0, ratio * n0]


def stationary_nonlinear(r: Rates, total: float = 1.0,
                         sqrt_epsilon: float = 0.0) -> FloatArray:
    """ Positive nonlinear stationary inventory.
    For eps=0, Nm=r_m²*N0⁴. With eps>0, algebraically invert the
    regularized detachment term, then solve the scalar mass-balance equation.
    """
     #  Validate the total mass and regularization
    if total <= 0 or sqrt_epsilon < 0:
        raise ValueError("Invalid total or sqrt epsilon")
    ratio = r.k_on / r.k_off
    root_eps = np.sqrt(sqrt_epsilon)

     # Evaluate equilibrium populations from N0
    def bound_from_free(n0: float) -> FloatArray:
        """ Bound inventory as a function of free inventory."""
        z = ratio * n0**2
        return z**2 + 2.0 * root_eps * z  # (z+sqrt(eps))²-eps
        # Compute stationary inventories of NPL groups

    n0 = brentq(lambda z: z + bound_from_free(z).sum() - total, 0.0, total,
                xtol=1e-14, rtol=1e-14)  #  Positive conservation root
    return np.r_[n0, bound_from_free(n0)]


def exact_linear(K: FloatArray, x0: FloatArray, times: FloatArray) -> FloatArray:
    """ Reference solution via expm(K*t)@x0.
    For this 5-state model, direct matrix exponential evaluation
    avoids possible ill-conditioned eigenvector and spurious-zero-mode issues.
    """
    return np.column_stack([expm(K * float(t)) @ x0 for t in times])

#Eigenmode decomposition
def eigen_linear(K: FloatArray, x0: FloatArray, times: FloatArray,
                 zero_tolerance: float = 1e-12) -> tuple[FloatArray, FloatArray, float]:
    """ Eigenmode decomposition for educational comparison.
    
    Snap the known conservation mode to zero, and reject a
    significant imaginary remainder instead of blindly taking .real.
    """
    lam, U = eig(K)
    condition = float(np.linalg.cond(U))  # Eigenvector condition number
    if not np.isfinite(condition) or condition > 1e9:
        raise ArithmeticError("Eigenbasis is ill-conditioned; prefer expm")
    idx0 = int(np.argmin(np.abs(lam)))
    if abs(lam[idx0]) > zero_tolerance * max(1.0, np.max(np.abs(lam))):
        raise ArithmeticError("No sufficiently small conserved eigenvalue")
    lam[idx0] = 0.0
    coef = np.linalg.solve(U, x0)  #  Solve rather than invert U
    X_complex = U @ (np.exp(np.outer(lam, times)) * coef[:, None])
    if np.max(np.abs(X_complex.imag)) > 1e-9:
        raise ArithmeticError("Unexpected significant imaginary component")
    return X_complex.real, lam.real, condition

#solve_stiff linear/nonlinear
def solve_stiff(kind: ModelKind, r: Rates, K: FloatArray, x0: FloatArray,
                times: FloatArray, method: str = "LSODA", rtol: float = 1e-9,
                atol: float = 1e-12, sqrt_epsilon: float = 0.0):
    """ Stiff integration for linear or nonlinear kinetics.
    t_eval is an OUTPUT grid, not the internal adaptive step grid.
    """
    if method not in {"LSODA", "BDF", "Radau"}:
        raise ValueError("Choose LSODA, BDF, or Radau")
    if times.ndim != 1 or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be a strictly increasing nonnegative vector")
    if rtol <= 0 or atol <= 0:
        raise ValueError("Both tolerances must be positive")

    if kind == "linear":
        fun = lambda t, x: linear_rhs(t, x, K)
        jac = (lambda t, x: K) if method == "LSODA" else K  # Callable for LSODA
    elif kind == "nonlinear":
        fun = lambda t, x: nonlinear_rhs(t, x, r, sqrt_epsilon)
        jac = (lambda t, x: nonlinear_jacobian(t, x, r, sqrt_epsilon)) if sqrt_epsilon > 0 else None
    else:
        raise ValueError("kind must be linear or nonlinear")

    options = dict(method=method, t_eval=times, dense_output=True, rtol=rtol, atol=atol)
    if jac is not None:
        options["jac"] = jac  
    sol = solve_ivp(fun, (float(times[0]), float(times[-1])), x0, **options)
    if not sol.success:
        raise RuntimeError(f"{kind} {method} failed: {sol.message}")
    return sol


@dataclass(frozen=True)
class Peak:
    """ Peak classification"""
    thickness_ml: int
    t_seconds: float | None
    value: float | None
    reason: str

#peaks
def find_reliable_peaks(sol, r: Rates, equilibrium: FloatArray,
                        kind: ModelKind, K: FloatArray,
                        sqrt_epsilon: float = 0.0,
                        prominence_min: float = 1e-4) -> list[Peak]:
    """Peak detection by derivative crossing AND prominence.
    Argmax on output points can invent late-time plateau peaks.
    Require a +→− derivative crossing and material excess over equilibrium.
    """
    if sol.sol is None:
        raise ValueError("dense_output=True is required for peak refinement")
    if prominence_min < 0:
        raise ValueError("prominence_min must be nonnegative")
    rhs: Callable[[float, FloatArray], FloatArray]
    if kind == "linear":
        rhs = lambda t, x: linear_rhs(t, x, K)
    else:
        rhs = lambda t, x: nonlinear_rhs(t, x, r, sqrt_epsilon)

    results: list[Peak] = []
    times = sol.t
    for i, m in enumerate(r.ms):
        index = i + 1  # Column zero is the free pool
        yi = sol.y[index]
        derivatives = np.array([rhs(float(t), sol.sol(float(t)))[index] for t in times])
        #Bracket only an actual positive-to-negative sign crossing
        candidates = np.where((derivatives[:-1] > 0) & (derivatives[1:] < 0))[0]
        qualifying: list[tuple[float, float]] = []
        for j in candidates:
            root = brentq(lambda t: rhs(float(t), sol.sol(float(t)))[index],
                          float(times[j]), float(times[j + 1]))
            peak = float(sol.sol(root)[index])
            if peak - equilibrium[index] > prominence_min:
                qualifying.append((root, peak))
        if not qualifying:
            results.append(Peak(int(m), None, None,
                                "no robust finite maximum; plateau or no detectable peak"))
        else:
            #Earliest significant local maximum
            t_peak, peak = min(qualifying, key=lambda it: it[0])
            results.append(Peak(int(m), t_peak, peak, "finite peak verified"))
    return results


def diagnostics(sol, x0: FloatArray) -> dict[str, float]:
    """Numerical/physical diagnostics after integration."""
    return {
        "mass_error_max": float(np.max(np.abs(np.sum(sol.y, axis=0) - x0.sum()))),
        "minimum_population": float(np.min(sol.y)),
        "nfev": float(sol.nfev),
        "njev": float(sol.njev),
    }


def export_csv(path: Path, times: FloatArray, names: list[str],
               linear: FloatArray, nonlinear: FloatArray) -> None:
    """Export both time series to CSV."""
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["time_s"] + [f"linear_{name}" for name in names]
                        + [f"nonlinear_{name}" for name in names])
        for j, t in enumerate(times):
            writer.writerow([f"{t:.12g}"]
                            + [f"{y:.12g}" for y in linear[:, j]]
                            + [f"{y:.12g}" for y in nonlinear[:, j]])


def export_plot(path: Path, times: FloatArray, labels: list[str],
                linear: FloatArray, nonlinear: FloatArray) -> None:
    """ Write a side-by-side linear/nonlinear comparison plot."""

    import matplotlib.pyplot as plt 

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, values, title in zip(axes, (linear, nonlinear), ("Linear model", "Nonlinear model")):
        for idx, label in enumerate(labels):
            ax.semilogx(times[1:], values[idx, 1:], label=label)  #   Omit zero (t=0) on log axis
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Normalized monomer inventory")
    axes[1].legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    """End-to-end command-line demonstration."""
    parser = argparse.ArgumentParser(description="NPL discrete-population kinetics")
    parser.add_argument("--temperature", type=float, default=473.15, help="Temperature (K)")
    parser.add_argument("--t-end", type=float, default=1e9, help="Final simulation time (s) ")
    parser.add_argument("--points", type=int, default=400, help="Log time output points ")
    parser.add_argument("--method", choices=["LSODA", "BDF", "Radau"], default="LSODA")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--sqrt-epsilon", type=float, default=0.0,
                        help="Optional changed-model sqrt regularization ")
    parser.add_argument("--lateral-nm", nargs=4, type=float, metavar=("L3", "L4", "L5", "L6"),
                        help="Optional representative lateral sizes, nm ")
    parser.add_argument("--output", type=Path, default=Path("npl_outputs"),
                        help="Output directory ")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot ")
    args = parser.parse_args()
    if args.t_end <= 1e-3 or args.points < 4:
        parser.error("--t-end must exceed 1e-3 s and --points >= 4")

    p = Parameters(temperature_K=args.temperature,
                   lateral_nm=None if args.lateral_nm is None else tuple(args.lateral_nm))
    r = compute_rates(p)
    K = build_generator(r)
    x0 = initial_state(len(r.ms))
    times = np.r_[0.0, np.geomspace(1e-3, args.t_end, args.points)]

    x_exact = exact_linear(K, x0, times)  # Linear reference
    x_eig, eigvals, condU = eigen_linear(K, x0, times)  #  Verify eigenmodes
    lin = solve_stiff("linear", r, K, x0, times, method=args.method,
                      rtol=args.rtol, atol=args.atol)
    nonlin = solve_stiff("nonlinear", r, K, x0, times, method=args.method,
                         rtol=args.rtol, atol=args.atol, sqrt_epsilon=args.sqrt_epsilon)
    eq_linear = stationary_linear(r)
    eq_nonlinear = stationary_nonlinear(r, sqrt_epsilon=args.sqrt_epsilon)
    peaks = find_reliable_peaks(nonlin, r, eq_nonlinear, "nonlinear", K,
                                sqrt_epsilon=args.sqrt_epsilon)

    print("\n=== NPL Parameter table ===")
    print("ML   barrier_eV    effective_dH_eV      k_on_s^-1      k_off_s^-1")
    for i, m in enumerate(r.ms):
        print(f"{int(m):2d}   {r.barrier_eV[i]:.6f}      {r.formation_effective_eV[i]:+.6f}"
              f"        {r.k_on[i]:.6e}     {r.k_off[i]:.6e}")
    print("\n  Eigenvalues:", np.sort(eigvals))
    print("Eigenvector condition:", f"{condU:.3g}")
    print("eigen vs expm max difference:", f"{np.max(np.abs(x_eig-x_exact)):.3e}")
    print("stiff linear vs expm max difference:", f"{np.max(np.abs(lin.y-x_exact)):.3e}")
    for kind, sol in (("linear", lin), ("nonlinear", nonlin)):
        print(f"{kind} diagnostics :", diagnostics(sol, x0))
    print("linear equilibrium:", eq_linear)
    print("nonlinear equilibrium :", eq_nonlinear)
    print("nonlinear peak times  (h):")
    for item in peaks:
        when = "none " if item.t_seconds is None else f"{item.t_seconds/3600:.6g}"
        print(f"  {item.thickness_ml} ML: {when} ({item.reason})")

    args.output.mkdir(parents=True, exist_ok=True)
    names = ["N0"] + [f"N{int(m)}" for m in r.ms]
    export_csv(args.output / "npl_results.csv", times, names, x_exact, nonlin.y)
    export_paper_audit(args.output / "original_paper_reference.json", p, r)
    print("source audit:", args.output / "original_paper_reference.json")
    if not args.no_plot:
        export_plot(args.output / "npl_comparison.png", times, names, x_exact, nonlin.y)
    print(" Files written to:", args.output.resolve())


if __name__ == "__main__":
    main()
