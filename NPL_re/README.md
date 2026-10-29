# Discrete-Thickness Nanoplatelet Kinetics — Technical Guide and Validation Manual

> **Project type:** 2026 literature-based model reimplementation, numerical analysis and reproducibility exercise.  
> **Code:** `npl_kinetics.py` · **Tests:** `test_npl_kinetics.py` (19 pytest test functions).  
> **Physical system:** reduced CdSe nanoplatelet kinetics; normalized free-monomer and discrete 3–6-monolayer populations.  
> **Important distinction:** the code adapts equations and parameters reported by Riedinger *et al.* (2017). It does **not** reproduce the full original kinetic Monte Carlo model, perform DFT, or fit the CdS experiments reported by Knüsel *et al.* (2020).

## 1. Research aim and provenance

The purpose of this implementation is to examine how **thickness-dependent attachment and detachment kinetics** can generate sequential transient dominance among nanoplatelet populations that exchange material through a common monomer reservoir. It also serves as a numerical study of stiff mass-conserving ODEs, steady states, parameter sensitivity and software validation.

### 1.1 Which parts come from which sources?

| Source | Role in this repository | Not claimed |
|---|---|---|
| **[R1] Riedinger *et al.*, Nature Materials (2017)** | Island-energy formulas, rate-law structure, effective material parameters and thickness-instability criteria, as identified by the supplementary-equation references below | A full reproduction of the paper's kMC calculations, experimental images, or an independently developed theory |
| **[R2] Knüsel *et al.*, Chemistry of Materials (2020)** | Independent context for observed nanoplatelet ripening and thickness-resolved absorption features in **CdS** | Calibration of this **CdSe** kinetic model, or a prediction of CdS absorption spectra |
| **[R3] Ott *et al.*, Nano Letters (2017)** | Background on semiconductor nanoplatelet ripening | Numerical validation of this implementation |
| **This Python implementation** | A reduced 3–6 ML population model; SciPy stiff solvers; explicit mass-balance and stationary-state checks; source/parameter audit; optional lateral-size and regularization variants | New experimental evidence, a novel atomistic mechanism, or original DFT/kMC calculations |

Source pointers into [R1] supplementary information: **S1–S7** for energetics, **S8–S24** for rate equations, equilibria and rates, and **S29–S30** for the parameter-dependent 2-ML versus 3-ML growth criterion. Equation labels below are **literature references**, not numbering newly introduced by this code.

### 1.2 What is and is not reproduced?

**Implemented:** thickness-resolved rates; linear and nonlinear effective population exchange; a shared monomer reservoir; analytic linear reference solution; stiff ODE simulations; stationary-state calculations; conservation/equilibrium/solver checks; tests for false peak detection.

**Not implemented:** direct transitions $m\to m+1$ between platelets; individual platelet number and area distributions; stochastic kMC events; spatial diffusion fields; dynamic lateral-size growth; first-principles DFT; absorption wavelengths or spectra; parameter fits to new experimental datasets.

## 2. Environment, structure and quick start

Requires **Python ≥ 3.10** and `numpy`, `scipy`, `matplotlib`, `pytest`. Dependency versions are not pinned. `matplotlib` is only required for plots.

```text
NPL_re/
├── npl_kinetics.py          # Parameters, rate laws, ODEs, solvers, exports
├── test_npl_kinetics.py     # 19 test_* functions
└── README.md                # This technical guide
```

From the repository root:

```bash
cd NPL_re
python -m pip install numpy scipy matplotlib pytest
python npl_kinetics.py --help
python npl_kinetics.py
python -m pytest -q test_npl_kinetics.py
```

For tabular output without plotting:

```bash
python npl_kinetics.py --no-plot --output results_no_plot
```

The default time grid contains **$t=0$ plus 400 logarithmically spaced positive times**, starting at $10^{-3}$ s and extending to $10^9$ s. The integrator uses **adaptive internal steps**; these 401 points are a reporting grid, not the step sequence. The initial condition places all normalized inventory in the free pool.

## 3. Model definitions and equations

### 3.1 Conserved populations

With default thicknesses $m\in\{3,4,5,6\}$:

$$
\mathbf{x}(t)=[N_0(t),N_3(t),N_4(t),N_5(t),N_6(t)]^\mathsf{T},
\qquad N_{\mathrm{tot}}=N_0+\sum_m N_m=1.
$$

- $N_0$ is the **normalized inventory of free monomers**.
- $N_m$ is the **normalized number of monomer units in the thickness-$m$ population**, *not* the number of individual nanoplatelets.
- All groups exchange matter through the shared free pool. There is **no explicit direct $m\to m+1$ conversion** in the right-hand side.
- Total inventory is normalized to one by default; rates have effective units of s$^{-1}$ within this reduced normalization.

### 3.2 Facet energetics (source: [R1], SI S1, S5 and S7)

The general energy change is

$$
\Delta E=E_V\Delta V+E_A\Delta A+E_L\Delta L.\tag{S1}
$$

For a wide-facet island with area $a$:

$$
E_{\mathrm{wide}}(a)=L_1E_Va+(4L_1E_A+2E_L)\sqrt a+4L_1E_L.\tag{S5}
$$

For a spanning island on a facet of thickness $m$:

$$
E_{\mathrm{narrow},m}(a)=
\left(L_1E_V+\frac{2E_A}{m}\right)a
+2mL_1^2E_A+(m+4)L_1E_L.\tag{S7}
$$

**Implementation choice:** `compute_rates()` uses the **intercept** of the S7 narrow-facet energy branch as an **approximate** nucleation barrier,

$$
\Delta E_m^\ddagger\approx 2mL_1^2E_A+(m+4)L_1E_L.
$$

It does not numerically reconstruct the full nucleation free-energy landscape or reproduce individual kMC island events.

### 3.3 Thickness thermodynamics and optional lateral correction

The default [R1] S21-type formation-energy expression is

$$
\Delta H_{f,m}=2L_1^3E_V+\frac{4L_1^2E_A}{m},\qquad
A_m=\exp\!\left(\frac{\Delta H_{f,m}}{k_BT}\right)=\frac{k_{-m}}{k_m}.
$$

The implementation also uses $\Omega=2L_1^3$, $h=mL_1$ and $\gamma=E_A$ to express an effective formation energy:

$$
\Delta H_{f,m}^{\mathrm{eff}}=
\underbrace{\Omega E_V}_{\text{bulk}}+
\underbrace{\frac{2\gamma\Omega}{h}}_{\text{thickness}}+
\underbrace{\frac{2\gamma\Omega}{L_m}}_{\text{optional lateral extension}}.
$$

- With `lateral_nm=None` (**default**), the lateral contribution is zero and the S21-type thickness baseline is recovered.
- With `lateral_nm=(L3,L4,L5,L6)`, each population receives one **fixed** effective lateral length in nm (internally converted to Å). This simplified square-platelet/equal-effective-facet-energy **extension is not a dynamic model of lateral-size distributions**.
- A smaller fixed lateral length increases the effective formation-energy contribution and hence changes rates. This is a **model extension**, not a claim to have reproduced measured lateral growth.

### 3.4 Attachment and detachment (source: [R1], SI S19–S21)

$$
k_m=C_0m\exp\!\left(-\frac{\Delta E_m^\ddagger}{k_BT}\right),\qquad
k_{-m}=k_mA_m.
$$

`Rates.k_on` and `Rates.k_off` are aligned with `Rates.ms`. The rates are **calculated from effective model parameters**; they are not independent experimental measurements.

### 3.5 Linear and nonlinear mass-balance laws

**Linear exchange** (SI S8–S12):

$$
\dot N_m=k_mN_0-k_{-m}N_m,\qquad
\dot N_0=-\sum_m\dot N_m,\qquad
\dot{\mathbf{x}}=K\mathbf{x}.
$$

`build_generator()` constructs $K$ with **zero column sums**. `exact_linear()` evaluates $\exp(Kt)\mathbf{x}(0)$ and `eigen_linear()` supplies an independent eigenmode reconstruction (SI S13–S18).

**Nonlinear exchange** (SI S22):

$$
\dot N_m=k_mN_0^2-k_{-m}\sqrt{N_m},\qquad
\dot N_0=-\sum_m\dot N_m.
$$

`nonlinear_rhs()` protects evaluations near slightly negative numerical trial states using clipping. **This does not impose mathematical nonnegative constraints** on the solver; materially negative solutions should be treated as a warning.

For `sqrt_epsilon>0`, the detachment factor is deliberately modified:

$$
g_{\varepsilon}(N_m)=\sqrt{\max(N_m,0)+\varepsilon}-\sqrt{\varepsilon}.
$$

This regularization **changes the model** and is **disabled by default** (`sqrt_epsilon=0`). An analytic Jacobian is provided only when $\varepsilon>0$, and corresponds to positive interior states; clipping remains nonsmooth at the boundary.

### 3.6 Stationary states

Linear equilibrium satisfies

$$
\frac{N_m}{N_0}=\frac{k_m}{k_{-m}}.
$$

Unregularized nonlinear equilibrium satisfies

$$
\frac{\sqrt{N_m}}{N_0^2}=\frac{k_m}{k_{-m}},\qquad
N_m=\left(\frac{k_m}{k_{-m}}\right)^2N_0^4.
$$

For the regularized model, with $r_m=k_m/k_{-m}$:

$$
N_m=r_m^2N_0^4+2r_m\sqrt\varepsilon\,N_0^2.
$$

`stationary_linear()` evaluates the linear stationary state. `stationary_nonlinear()` uses `scipy.optimize.brentq` on a **scalar total-inventory equation**, rather than inferring equilibrium only by long-time integration.

## 4. Parameter values and provenance

The default physical configuration is `Parameters()`. The CLI varies temperature and optional lateral sizes; other material parameters can be changed through the Python API.

| Field | Default | Units | Meaning / provenance |
|---|---:|---|---|
| `temperature_K` | 473.15 | K | Example model temperature, 200 °C |
| `kB_eV_K` | `8.617333262e-5` | eV/K | Boltzmann constant |
| `layer_height_A` | 3.10 | Å | Monolayer height, [R1] SI S4.a |
| `surface_eV_A2` | `5.7e-3` | eV/Å² | DFT-based acetate-passivated CdSe(001) surface value from source |
| `edge_eV_A` | `37.1e-3` | eV/Å | Representative DFT-based step-edge value from source |
| `volume_eV_A3` | `-1.5e-3` | eV/Å³ | Source-constrained using 2-/3-ML observations; **not directly calculated by DFT here** |
| `prefactor_s1` | `2.8e11` | s$^{-1}$ | Effective fitted source prefactor |
| `thickness_ml` | `(3,4,5,6)` | ML | Default thickness categories |
| `lateral_nm` | `None` | nm | Optional lateral extension disabled by default |

At default values, the derived quantities (rounded) are:

| ML | Approx. narrow barrier (eV) | Formation energy (eV) | $k_m$ (s$^{-1}$) | $k_{-m}$ (s$^{-1}$) |
|---:|---:|---:|---:|---:|
| 3 | 1.133732 | −0.016337 | 7.0515e−1 | 4.7235e−1 |
| 4 | 1.358296 | −0.034596 | 3.8130e−3 | 1.6322e−3 |
| 5 | 1.582860 | −0.045551 | 1.9330e−5 | 6.3246e−6 |
| 6 | 1.807424 | −0.052855 | 9.4070e−8 | 2.5732e−8 |

Additional reference calculations using S5 give a wide-facet critical island area of approximately **242.69 Å²** and energy **1.58855 eV**; the approximate narrow-growth threshold is $m_c\approx2.4516$. These depend on the **chosen model parameters**.

## 5. Command-line usage

```bash
python npl_kinetics.py [OPTIONS]
```

| Flag | Default | Function / restrictions |
|---|---|---|
| `--temperature` | `473.15` | Positive temperature, K |
| `--t-end` | `1e9` | Final time in seconds, greater than `1e-3` |
| `--points` | `400` | Number of positive log-spaced output times (minimum 4); $t=0$ added |
| `--method` | `LSODA` | `LSODA`, `BDF`, `Radau` |
| `--rtol` | `1e-9` | Positive relative tolerance |
| `--atol` | `1e-12` | Positive absolute tolerance |
| `--sqrt-epsilon` | `0` | Zero = original unregularized law; positive = **modified kinetics** |
| `--lateral-nm` | omitted | **Exactly four positive lengths**: `L3 L4 L5 L6`, nm |
| `--output` | `npl_outputs` | Output directory, relative to working directory unless absolute |
| `--no-plot` | off | Skip PNG; CSV and JSON are still generated |
| `-h`, `--help` | — | Show argparse help |

**Not available as CLI flags:** changing the thickness list, the initial state, or choosing only one of the models. The supplied `main()` runs both models with a normalized total inventory of one. Use the Python API for nondefault populations or initial conditions.

```bash
# Default case
python npl_kinetics.py

# CSV + JSON only
python npl_kinetics.py --no-plot --output out_data

# 24 h simulation; 250 positive output times plus t=0
python npl_kinetics.py --t-end 86400 --points 250 --output out_24h

# Alternative stiff methods
python npl_kinetics.py --method Radau --rtol 1e-9 --atol 1e-12 --output out_radau
python npl_kinetics.py --temperature 500 --method BDF --output out_500K

# Optional fixed lateral dimensions (nm), 3 to 6 ML
python npl_kinetics.py --lateral-nm 10 15 20 25 --output out_lateral

# Optional MODEL-CHANGING regularization
python npl_kinetics.py --sqrt-epsilon 1e-9 --output out_eps
```

### Python API example

```python
import numpy as np
from npl_kinetics import (
    Parameters, compute_rates, build_generator, initial_state,
    exact_linear, solve_stiff, stationary_nonlinear, diagnostics,
)

p = Parameters(thickness_ml=(3, 4, 5, 6), lateral_nm=None)
r = compute_rates(p)
K = build_generator(r)
x0 = initial_state(len(r.ms), total=1.0)
times = np.r_[0.0, np.geomspace(1e-3, 1e6, 200)]

linear_reference = exact_linear(K, x0, times)
nonlinear_solution = solve_stiff(
    "nonlinear", r, K, x0, times,
    method="LSODA", rtol=1e-9, atol=1e-12,
    sqrt_epsilon=0.0,
)
print(diagnostics(nonlinear_solution, x0))
print(stationary_nonlinear(r))
```

## 6. Numerical methodology and validation

### 6.1 Why use stiff integrators?

At the reference parameters, nonzero eigenvalues of the *linear* generator are approximately $-1.18$, $-3.16\times10^{-3}$, $-1.03\times10^{-5}$ and $-3.76\times10^{-8}$ s$^{-1}$. These widely separated time scales can make long-horizon explicit integration inefficient or unstable. The code uses SciPy's **LSODA** by default and offers **BDF** and **Radau**.

`solve_stiff()` requests dense output. For the linear system, the known generator $K$ can be supplied as a Jacobian. For the nonlinear system, an analytic Jacobian is supplied **only when the modified regularized model has `sqrt_epsilon>0`**.

### 6.2 Independent numerical checks

| Check | What it establishes | What it does not establish |
|---|---|---|
| $\sum_i K_{ij}=0$ | Algebraic inventory conservation in the linear law | Validity of physical rate assumptions |
| $\sum_i\dot N_i=0$ | Algebraic inventory conservation for both right-hand sides | Strict numerical conservation at any tolerance |
| Matrix exponential vs eigenmodes vs stiff solve | Agreement of three linear solution routes | Uniqueness of the physical model |
| Nonlinear stationary root vs long-time trajectory | Equilibrium consistency | Experimental ripening equilibrium |
| LSODA vs Radau | Numerical robustness for tested settings | Global convergence under every parameter choice |
| Finite differences vs analytic regularized Jacobian | Local derivative implementation at positive interior states | Smoothness of the unregularized law at zero |
| Peak derivative and prominence check | Reduced susceptibility to flat-plateau numerical artifacts | Validation of experimentally observed peak times |

Always inspect mass drift, minimum population, solver status and sensitivity to tolerances. `rtol` and `atol` control **local** error estimates; they are not a guarantee of strict nonnegativity or a specified **global** solution error.

### 6.3 Illustrative default nonlinear behavior

| Thickness | Accepted finite-time maximum |
|---|---:|
| 3 ML | ~0.00127 h (~4.58 s) |
| 4 ML | ~0.527 h |
| 5 ML | ~159.5 h |
| 6 ML | **No verified finite-time maximum**; late-time plateau |

A peak is accepted only if there is a **positive-to-negative derivative crossing** and a sufficient excess above the stationary population (`prominence_min=1e-4` by default). A small numerical bump in an almost flat 6-ML curve must **not** be interpreted as a physical peak. Peak classifications can depend on parameters, solver tolerance, regularization, representative lateral sizes, and time horizon.

## 7. Outputs and reproducibility contract

The program generates these paths beneath `--output` (default `npl_outputs/` relative to the execution directory):

```text
npl_outputs/
├── npl_results.csv
├── original_paper_reference.json
└── npl_comparison.png            # absent with --no-plot
```

### `npl_results.csv`

The default case has **11 columns**: `time_s`, then five `linear_*` inventory fractions (`N0,N3,N4,N5,N6`), then five `nonlinear_*` fractions. The `linear_*` columns are calculated from the **matrix exponential** reference; numerical integration of the linear system serves as a verification route. The file uses UTF-8 with BOM and approximately 12 significant figures.

**These are normalized inventory fractions**, not dimensional particle concentrations, numbers of nanoplatelets, or optical intensities. The output grid includes $t=0$; the logarithmic comparison plot omits that first value because $\log(0)$ is undefined.

### `original_paper_reference.json`

`paper_reference_audit()` exports source/reference information, declared original values, current parameter values, source-equation identifiers, derived model rates/barriers, and flags recording whether the baseline equations or optional extensions were used. The data are **model calculations and provenance records**, not independently measured material constants.

### `npl_comparison.png` and console output

The PNG compares normalized linear and nonlinear population histories. Console diagnostics include rate tables, eigenvalues, eigenbasis conditioning, numerical comparisons, conservation error, stationary states and accepted peak times.

## 8. Automated test coverage

The file `test_npl_kinetics.py` contains **19 `test_*` functions**. Its fixtures cache the common reference configuration and a long nonlinear solution. Run:

```bash
python -m pytest -v test_npl_kinetics.py
python -m pytest -q test_npl_kinetics.py
python -m pytest -q test_npl_kinetics.py -k "jacobian"
python -m pytest -q test_npl_kinetics.py -k "mass or equilibrium"
```

| Category | Tests |
|---|---|
| Rate and parameter provenance | `test_reference_rates_match_previous_example`, `test_2ml_threshold_is_parameter_dependent`, `test_thickness_gibbs_thomson_step`, `test_lateral_correction_is_optional_and_monotone`, `test_invalid_parameter_inputs_are_rejected` |
| Algebraic identities | `test_generator_has_zero_column_sums_and_nonnegative_offdiagonals`, `test_rhs_conserves_mass_at_arbitrary_positive_state`, `test_linear_mass_conservation_after_lateral_gt_correction` |
| Independent linear solutions | `test_linear_eigen_decomposition_agrees_with_expm`, `test_linear_stiff_solver_agrees_with_expm` |
| Nonlinear integration | `test_nonlinear_mass_conservation`, `test_nonnegative_populations`, `test_nonlinear_long_time_agrees_with_equilibrium`, `test_multiple_stiff_integrators_are_consistent_at_short_horizon` |
| Equilibria and peaks | `test_analytic_equilibrium_satisfies_balance`, `test_nonlinear_peak_detection_rejects_plateau` |
| Optional regularization and derivatives | `test_sqrt_regularization_preserves_conservation_and_changes_model`, `test_regularized_jacobian_matches_finite_differences`, `test_zero_bound_population_has_no_detachment` |

**Interpretation:** these tests check internal mathematical identities and numerical agreement for selected settings. They do **not** constitute experimental confirmation of the underlying growth mechanism. This README describes the supplied tests; it does not itself certify that a particular checkout or software environment has passed them.

## 9. Limitations and interpretation boundaries

1. **Model reduction.** The states are thickness-class monomer inventories, not individually resolved platelets; there is no stochastic kMC simulation, concentration field, or dynamical lateral-size distribution.
2. **Source-model approximation.** The nucleation barrier uses an intercept-based approximation to [R1] S7, not the full free-energy landscape. A numerical match within this implementation is not equivalent to reproducing every result of [R1].
3. **Optional model changes.** A fixed lateral Gibbs–Thomson-like term and square-root regularization alter the model. Both are **off by default**; compare baseline and modified cases explicitly.
4. **Materials/observables.** The kinetic parameters are for a CdSe-inspired effective model. [R2] reports CdS experiments at 160–190 °C; its 2-/3-/4-/5-ML optical absorption peaks are independent empirical context, **not outputs calculated by this code**.
5. **Numerical safeguards are not constraints.** Clipping negative trial-state values is not a positivity-preserving integrator. Long time scales can cause solver-conditioning and peak-detection difficulties.
6. **Validation scope.** Conservation, solver agreement, and equilibrium tests establish **internal consistency**, not that the selected kinetic equations uniquely explain experiments.
7. **Physical assumptions.** Geometric factors, rates and parameters are effective approximations; they have not been independently inferred from new measurements here.

For reference, [R2] reports absorption signals at approximately **326 nm (2 ML), 374 nm (3 ML), 407 nm (4 ML), and 429 nm (5 ML)** in its CdS system. The current script **does not calculate absorption spectra or wavelengths**; those values should never be plotted as model predictions.

## 10. References and attribution

- **[R1]** Riedinger, A., Ott, F. D., Mule, A., *et al.* “An intrinsic growth instability in isotropic materials leads to quasi-two-dimensional nanoplatelets.” *Nature Materials* **16**, 743–748 (2017). DOI: [10.1038/nmat4889](https://doi.org/10.1038/nmat4889).
- **[R2]** Knüsel, P. N., Riedinger, A., Rossinelli, A. A., Ott, F. D., Mule, A. S., Norris, D. J. “Experimental Evidence for Two-Dimensional Ostwald Ripening in Semiconductor Nanoplatelets.” *Chemistry of Materials* **32**(7), 3312–3319 (2020). DOI: [10.1021/acs.chemmater.0c01238](https://doi.org/10.1021/acs.chemmater.0c01238).
- **[R3]** Ott, F. D., *et al.* “Ripening of Semiconductor Nanoplatelets.” *Nano Letters* **17**, 6870–6877 (2017). DOI: [10.1021/acs.nanolett.7b03191](https://doi.org/10.1021/acs.nanolett.7b03191).

**Attribution statement:** This repository is a **2026 reproduction and numerical learning project** based on published theory and literature parameters. AI assistance was used in the software-development workflow. The scientific credit for the referenced models and experiments remains with the original authors. The numerical tests and documented approximations are intended to make this particular implementation inspectable rather than to claim a new physical discovery.
