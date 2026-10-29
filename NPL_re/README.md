# Discrete-Thickness Nanoplatelet Kinetics: User Guide and Validation Manual 

> **Applies to:** `npl_kinetics.py` (model, solver, diagnostics, exports) and `test_npl_kinetics.py` (pytest checks).
> **Scope:** A reduced CdSe nanoplatelet (NPL) kinetic model with a free-monomer reservoir, discrete 3–6 monolayer (ML) populations, first-order and nonlinear exchange kinetics, an effective Gibbs–Thomson (GT) term, and stiff ODE integration in SciPy.
> **Scientific scope:** This is a **reduced effective-kinetics model** for numerical study and reproducibility. It is **not** a complete kinetic Monte Carlo reconstruction of the 2017 work or a direct calibration to the 2020 CdS experiment.

## 1. Files and quick start

Put the two Python files in the **same directory**:

```text
npl_project/
├── npl_kinetics.py         # Parameters, rate laws, ODEs, solvers, exports
├── test_npl_kinetics.py    # 19 pytest test functions
└── README.md               # This guide
```

Use **Python 3.10 or newer**; the implementation uses union annotations (`X | None`) and parameterized built-in containers (`tuple[...]`). Required for the simulation: **NumPy and SciPy**. **Matplotlib** is required when producing figures; **pytest** is required to run tests. The provided scripts do not pin dependency versions.

Install the packages:

```bash
python -m pip install numpy scipy matplotlib pytest
```

Check the interpreter and inspect all options:

```bash
python --version
python npl_kinetics.py --help
```

Run the default simulation:

```bash
python npl_kinetics.py
```

Run without plotting (Matplotlib not needed):

```bash
python npl_kinetics.py --no-plot --output results_no_plot
```

Run the test suite:

```bash
python -m pytest -v test_npl_kinetics.py
# Compact summary
python -m pytest -q test_npl_kinetics.py
```

By default, the physical initial condition is applied at **t=0** and the horizon extends to **1e9 s** . The output grid contains **t=0 plus 400 logarithmically spaced positive times**, starting at `1e-3 s`—typically **401 data rows**.

## 2. Scientific references and provenance

### 2.1 Primary theoretical source: CdSe

**[R1]** Riedinger, A.; Ott, F. D.; Mule, A.; *et al.* **An intrinsic growth instability in isotropic materials leads to quasi-two-dimensional nanoplatelets.** *Nature Materials* **2017**, *16*, 743–748. DOI: [10.1038/nmat4889](https://doi.org/10.1038/nmat4889).

- Key supplementary equations: **S1–S7** (island energetics), **S8–S24** (mass balance, kinetics, equilibrium, attachment/detachment coefficients), and **S29–S30** (the parameter-dependent 2-ML/3-ML growth criterion).

### 2.2 Independent experimental context: CdS parameter fit

**[R2]** Knüsel, P. N.; Riedinger, A.; Rossinelli, A. A.; Ott, F. D.; Mule, A. S.; Norris, D. J. **Experimental Evidence for Two-Dimensional Ostwald Ripening in Semiconductor Nanoplatelets.** *Chemistry of Materials* **2020**, *32*(7), 3312–3319. DOI: [10.1021/acs.chemmater.0c01238](https://doi.org/10.1021/acs.chemmater.0c01238).

`Knuesel_etal_MS_accepted.pdf` describes CdS NPL ripening and observations supporting redistribution from small platelets to thicker populations.
- The experiment includes ripening temperatures **160, 170, 180, and 190 °C**. The main text (p. 5, Fig. 2) discusses absorption features assigned to **3 ML: 374 nm; 4 ML: 407 nm; 5 ML: 429 nm**, as well as the thin **2-ML: 326 nm** signal.
- These wavelengths are **experimental CdS reference observations**, **not optical output** from this code. The program does not calculate excitonic transition wavelengths or absorption spectra.

**[R3]** Ott, F. D.; *et al.* **Ripening of Semiconductor Nanoplatelets.** *Nano Letters* **2017**, *17*, 6870–6877. DOI: [10.1021/acs.nanolett.7b03191](https://doi.org/10.1021/acs.nanolett.7b03191).

## 3. States, equations, and model assumptions

### 3.1 Discrete ML populations and free-monomer pool

The default thickness set is `thickness_ml=(3,4,5,6)` with state vector

\[
\mathbf x(t)=[N_0(t),N_3(t),N_4(t),N_5(t),N_6(t)]^{\mathsf T},\qquad
N_{\mathrm{tot}}=N_0+\sum_m N_m=1.
\]

Here, \(N_m\) is the **normalized number of monomer units stored in the thickness-m population**, **not** a count of individual platelets. The thickness populations exchange material through one shared free-monomer pool. **No direct `m → m+1` transition term exists** in the ODEs; sequential dominance emerges from pool-mediated exchange.

### 3.2 Nucleation energetics: supplementary Eqs. S1, S5, S7

Total energy change:

\[
\Delta E=E_V\Delta V+E_A\Delta A+E_L\Delta L.\tag{S1}
\]

Wide-facet island:

\[
E_{\mathrm{wide}}(a)=L_1E_Va+(4L_1E_A+2E_L)\sqrt a+4L_1E_L.\tag{S5}
\]

Spanning-island branch on a facet of thickness \(m\):

\[
E_{\mathrm{narrow},m}(a)=\left(L_1E_V+\frac{2E_A}{m}\right)a
+2mL_1^2E_A+(m+4)L_1E_L.\tag{S7}
\]

The actual `compute_rates()` implementation uses the **intercept of the narrow-facet S7 branch as an approximate nucleation barrier**:

\[
\Delta E_m^\ddagger\approx 2mL_1^2E_A+(m+4)L_1E_L.
\]

### 3.3 Baseline thermodynamics and optional Gibbs–Thomson correction

The default S21-type formation-energy expression is:

\[
\Delta H_{f,m}=2L_1^3E_V+\frac{4L_1^2E_A}{m},\qquad
A_m=\exp\!\left(\frac{\Delta H_{f,m}}{k_BT}\right)=\frac{k_{-m}}{k_m}.
\]

In `compute_rates()`, define \(\Omega=2L_1^3\), \(h=mL_1\), and \(\gamma=E_A\). The effective formation-energy proxy is implemented as

\[
\Delta H_{f,m}^{\mathrm{eff}}=
\underbrace{\Omega E_V}_{\text{bulk}}
+\underbrace{\frac{2\gamma\Omega}{h}}_{\text{thickness}}
+\underbrace{\frac{2\gamma\Omega}{L_m}}_{\text{optional lateral correction}}.
\]

- **Default (`lateral_nm=None`)**: no lateral term; the S21 baseline is recovered.
- **If `lateral_nm` is provided**: one **fixed representative** lateral dimension per thickness group, specified in nm and converted internally to Å. This is an **extension with simplified square-platelet/equal-effective-facet-energy geometry**

### 3.4 Attachment/detachment rates: supplementary Eqs. S19–S21

\[
k_m=C_0m\exp\!\left(-\frac{\Delta E_m^\ddagger}{k_BT}\right),\qquad
k_{-m}=k_mA_m.
\]

The arrays `Rates.k_on` and `Rates.k_off` follow the order in `Rates.ms`. Their reported units are `s^-1` for this **normalized effective model**. 

### 3.5 Linear and nonlinear ODE systems

First-order model (SI S8–S12):

\[
\dot N_m=k_mN_0-k_{-m}N_m,\qquad
\dot N_0=-\sum_m\dot N_m,\qquad
\dot{\mathbf x}=K\mathbf x.
\]

`build_generator()` creates a matrix with zero column sums. `exact_linear()` evaluates \(\exp(Kt)\mathbf x(0)\), and `eigen_linear()` checks the eigenmode decomposition (SI S13–S18).

Nonlinear model (SI S22):

\[
\dot N_m=k_mN_0^2-k_{-m}\sqrt{N_m},\qquad
\dot N_0=-\sum_m\dot N_m.
\]

With the **optional, model-changing** `sqrt_epsilon>0`, the detachment term becomes

\[
g_\varepsilon(N_m)=\sqrt{\max(N_m,0)+\varepsilon}-\sqrt\varepsilon.
\]

The `nonlinear_rhs()` implementation also protects the free-pool and bound-population inputs with `max`/`np.maximum`. This safeguards evaluations near negative numerical trial states, but it **does not impose strict nonnegative state constraints**. Large negative populations indicate a model or integration problem. The analytic Jacobian is provided **only when `sqrt_epsilon>0`** and is interpreted on positive interior states; clipping still introduces a nonsmooth boundary at zero.

### 3.6 Stationary distributions

Linear equilibrium: \(N_m/N_0=k_m/k_{-m}\). Original nonlinear equilibrium:

\[
\frac{\sqrt{N_m}}{N_0^2}=\frac{k_m}{k_{-m}},\qquad
N_m=\left(\frac{k_m}{k_{-m}}\right)^2N_0^4.
\]

For a regularized law and \(r_m=k_m/k_{-m}\):

\[
N_m=r_m^2N_0^4+2r_m\sqrt\varepsilon\,N_0^2.
\]

`stationary_linear()` evaluates the first-order distribution analytically. `stationary_nonlinear()` uses `scipy.optimize.brentq` to solve a **single scalar equation** enforcing total inventory, rather than integrating until an arbitrary long time.

## 4. Defaults, units, and source classification

`Parameters()` is a frozen dataclass. The CLI overrides **temperature** and **fixed lateral sizes**, whereas the other physical constants may be changed through the Python API, not the existing CLI.

| `Parameters` field | Default | Unit | Provenance / meaning |
|---|---:|---|---|
| `temperature_K` | `473.15` | K | 200 °C model/example temperature |
| `kB_eV_K` | `8.617333262e-5` | eV/K | Boltzmann constant |
| `layer_height_A` | `3.10` | Å | Geometrical monolayer height, Riedinger SI S4.a |
| `surface_eV_A2` | `5.7e-3` | eV/Å² | DFT, acetate-passivated CdSe(001) |
| `edge_eV_A` | `37.1e-3` | eV/Å | DFT-based representative step-edge value |
| `volume_eV_A3` | `-1.5e-3` | eV/Å³ | Constrained by 2/3-ML observations; **not a directly computed DFT value** |
| `prefactor_s1` | `2.8e11` | s⁻¹ | Effective fit to time-resolved 3/4-ML absorption behavior |
| `thickness_ml` | `(3,4,5,6)` | ML | Default discrete groups; customizable via API |
| `lateral_nm` | `None` | nm | Lateral GT extension disabled by default |

Model-derived rates with the reference inputs (not independently measured rate constants):

| m / ML | Approx. narrow barrier / eV | Effective formation energy / eV | `k_on` / s⁻¹ | `k_off` / s⁻¹ |
|---:|---:|---:|---:|---:|
| 3 | 1.133732 | −0.016337 | 7.0515e−1 | 4.7235e−1 |
| 4 | 1.358296 | −0.034596 | 3.8130e−3 | 1.6322e−3 |
| 5 | 1.582860 | −0.045551 | 1.9330e−5 | 6.3246e−6 |
| 6 | 1.807424 | −0.052855 | 9.4070e−8 | 2.5732e−8 |

As additional checks, the S5 wide-facet calculation gives a critical area of approximately **242.69 Å²** and an energy of **1.58855 eV**; the effective narrow-growth threshold is \(m_c\approx2.4516\). These values are **parameter-dependent model predictions**.

## 5. Command-line interface: every option and default

```bash
python npl_kinetics.py [OPTIONS]
```

| Flag | Type / accepted values | Default | Purpose |
|---|---|---|---|
| `--temperature` | `float` in K | `473.15` | Changes thermal energy and rate/equilibrium factors; must be positive |
| `--t-end` | `float` in seconds | `1e9` | End time; must be strictly greater than `1e-3` |
| `--points` | `int` | `400` | Number of positive logarithmic output times; minimum 4, plus t=0 |
| `--method` | `LSODA`, `BDF`, `Radau` | `LSODA` | Stiff SciPy ODE integrator |
| `--rtol` | `float` | `1e-9` | Relative local error tolerance; must be positive |
| `--atol` | `float` | `1e-12` | Absolute local error tolerance; must be positive |
| `--sqrt-epsilon` | `float` | `0.0` | 0 = original nonlinear square-root term; positive = **modified law** |
| `--lateral-nm` | **exactly four** floats | omitted | `L3 L4 L5 L6` in nm; positive values |
| `--output` | directory path | `npl_outputs` | Directory for files, created if needed |
| `--no-plot` | Boolean flag | off | Suppresses PNG generation; Matplotlib optional |
| `--help` / `-h` | flag | — | Prints argparse-generated help |

**Not implemented as CLI flags:** `--thickness-ml`, `--model`, `--initial-state`, and `--total`. The supplied `main()` runs **both models**, starts with normalized total inventory one **entirely in the free pool**, and fixes 3–6 ML by default. To change those assumptions, use the Python API and adjust the affected arguments.

**Command examples:**

```bash
# Default: both kinetic models, 200 °C, four populations
python npl_kinetics.py

# Save only CSV and JSON; no PNG dependency
python npl_kinetics.py --no-plot --output out_data

# End at 24 hours; 250 positive output samples (+ t=0)
python npl_kinetics.py --t-end 86400 --points 250 --output out_24h

# Cross-check using a different stiff integrator
python npl_kinetics.py --method Radau --rtol 1e-9 --atol 1e-12 --output out_radau

# Enable the optional geometry extension (L3 to L6)
python npl_kinetics.py --lateral-nm 10 15 20 25 --output out_lateral

# Enable a MODEL-CHANGING regularized square-root law
python npl_kinetics.py --sqrt-epsilon 1e-9 --output out_eps

# Vary temperature (other material parameters remain fixed)
python npl_kinetics.py --temperature 500 --method BDF --output out_500K
```

## 6. Python API and input/output contract

The program can also be imported:

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

Principal functions and classes:

| Function / class | Inputs | Return / side effect |
|---|---|---|
| `Parameters(...)` | Temperature, materials, thicknesses, optional lateral sizes | Validated input dataclass |
| `Rates` | Aligned arrays | Barrier, formation energies, coexistence factors, attachment/detachment rates |
| `compute_rates(p)` | `Parameters` | `Rates` |
| `build_generator(r)` | `Rates` | `(n+1) × (n+1)` linear generator `K` |
| `initial_state(n,total=1)` | Number of thickness groups, normalized total | `[total,0,...,0]` |
| `linear_rhs(t,x,K)` | State and linear generator | `K @ x` |
| `nonlinear_rhs(t,x,r,sqrt_epsilon=0)` | State, rate arrays, optional regularization | Nonlinear derivative vector |
| `nonlinear_jacobian(t,x,r,sqrt_epsilon)` | **positive** epsilon | Analytic Jacobian of the regularized RHS |
| `exact_linear(K,x0,times)` | Generator, initial state, time vector | Matrix-exponential solution, shape `(n+1,len(times))` |
| `eigen_linear(K,x0,times)` | Same | `(solution,eigenvalues,eigenvector_condition_number)` |
| `solve_stiff(kind,r,K,x0,times,...)` | `linear`/`nonlinear`, solver settings | SciPy `OdeResult` (`t`, `y`, `sol`, `success`, `nfev`, `njev`) |
| `stationary_linear(r,total=1)` | Rates, total | Linear stationary state |
| `stationary_nonlinear(r,total=1,sqrt_epsilon=0)` | Rates, total, regularization | Nonlinear stationary state |
| `find_reliable_peaks(sol,r,eq,kind,K,...)` | Dense numerical solution, equilibrium, threshold | One `Peak` per population; `t_seconds=None` if not verified |
| `diagnostics(sol,x0)` | Solver result and initial state | Conservation error, minimum inventory, evaluation counts |
| `paper_reference_audit(p,r)` | Parameters and rates | JSON-serializable source and computed-data dictionary |
| `export_csv` / `export_paper_audit` / `export_plot` | Output path and arrays | Writes files; returns `None` |

**Immutability caveat:** `@dataclass(frozen=True)` prevents attribute reassignment but does **not** make contained NumPy arrays read-only.

## 7. Stiff integration, numerical accuracy, and validation

### 7.1 Why stiff methods?

The default linear generator has widely separated nonzero decay eigenvalues, approximately \(-1.18,-3.16\times10^{-3},-1.03\times10^{-5},-3.76\times10^{-8}\) s⁻¹. Such multiple time scales can restrict the stability step of explicit solvers over a long horizon. The CLI therefore offers `LSODA` (automatic stiff/nonstiff switching), `BDF`, and `Radau`.

`solve_stiff()` requests `dense_output=True`. The `times` vector is an **output grid**, not the integrator's internal adaptive-step sequence. For the linear model, it supplies `K` as the Jacobian (a callable for LSODA). For the nonlinear model, it supplies the analytic Jacobian **only for `sqrt_epsilon>0`**; otherwise the integrator handles derivatives using its own methods.

### 7.2 Independent solution/verification routes

1. **`exact_linear`** evaluates `scipy.linalg.expm(K*t) @ x0`; this is a mathematically exact representation but is still evaluated in floating-point arithmetic.
2. **`eigen_linear`** evaluates an eigenmode decomposition, adjusts the known conserved zero eigenvalue, and checks eigenbasis conditioning and unwanted imaginary components.
3. **`solve_stiff`** computes linear or nonlinear solutions via LSODA, BDF, or Radau and compares them against analytic references, stationary states, and conservation invariants.

### 7.3 What to check

- **Algebraic conservation:** `K.sum(axis=0) ≈ 0`, and the sum of either RHS output is approximately zero.
- **Trajectory conservation:** \(\max_t|\sum_iN_i(t)-N_{\rm tot}|\) remains within chosen numerical tolerances.
- **Nonnegativity:** `minimum_population` should have no materially negative value; clipping in `nonlinear_rhs` is a safeguard, not a constrained solver.
- **Linear agreement:** eigenmode reconstruction, matrix exponential, and stiff linear integration match to suitable tolerances.
- **Stationarity:** zero net flux and conservation hold at the stationary state; long-horizon nonlinear results approach that state.
- **Jacobian:** finite-difference checks of the regularized Jacobian on positive interior states and its column-sum identity.
- **Reliable peaks:** a positive-to-negative derivative crossing **and** excess above the stationary population exceeding `prominence_min` (default `1e-4`). A late-time `np.argmax` on a flat 6-ML trajectory is not, by itself, evidence of a physical peak.

`rtol` and `atol` control **local numerical error estimates**. They do **not** guarantee a particular global conservation error or strict state positivity.

### 7.4 Indicative reference behavior

Using the default normalized setup, the nonlinear model typically yields:

| Population | Accepted finite-time peak |
|---|---:|
| 3 ML | approximately **0.00127 h** (~4.58 s) |
| 4 ML | approximately **0.527 h** |
| 5 ML | approximately **159.5 h** |
| 6 ML | **No verified finite peak**; approaches a late-time plateau |

These are results of the **reduced code model and its detection criteria**, not separately measured experimental peak times. The peak classification may change with temperature, geometry, regularization, solver tolerance, output sampling, or simulation horizon.

## 8. Files written: names, schemas, and caveats

By default, outputs are written under `npl_outputs/`; set `--output` to change the directory.

```text
npl_outputs/
├── npl_results.csv                # Normalized time courses for both kinetic laws
├── original_paper_reference.json # Original-source metadata and derived-data audit
└── npl_comparison.png            # Linear/nonlinear comparison; omitted with --no-plot
```

### 8.1 `npl_results.csv`

The first column is `time_s`. With four thickness populations there are **11 columns**: time, five linear state columns, and five nonlinear state columns:

```text
time_s,linear_N0,linear_N3,linear_N4,linear_N5,linear_N6,nonlinear_N0,nonlinear_N3,nonlinear_N4,nonlinear_N5,nonlinear_N6
0,...
0.001,...
...
```

**Important:** `linear_*` values are written from the **`exact_linear()` matrix-exponential solution**; `solve_stiff("linear",...)` is used for verification and console diagnostics, not for these CSV columns. CSV encoding is `utf-8-sig`, and floating-point values are formatted to roughly 12 significant digits. All state columns represent **normalized monomer inventory fractions**, not platelet counts, dimensional concentrations, or absorption amplitudes.

### 8.2 `original_paper_reference.json`

Top-level keys produced by `paper_reference_audit()`:

```json
{
  "reference_original_CdSe": "...",
  "reference_independent_CdS_experiment": "...",
  "original_values": {},
  "equations": {},
  "CdS_independent_observations": {},
  "computed_from_selected_parameters": {},
  "original_equations_unmodified": true,
  "barrier_model": "...",
  "lateral_gt_extension_enabled": false,
  "note": "..."
}
```

`original_values` compares reference parameter values to the current `Parameters` values, including origin and a `same_as_original` flag. `computed_from_selected_parameters` records the wide-facet critical area and peak energy, approximate narrow barriers, effective formation energies, thickness-specific rates, and the critical growth conditions.

### 8.3 `npl_comparison.png` and console output

The optional PNG plots the **linear** and **nonlinear** time courses side by side using a logarithmic time axis and normalized monomer inventories. The plotting function omits the `t=0` sample because `log(0)` is undefined. The console prints a parameter/rate table, eigenvalues, eigenbasis condition number, comparisons with the matrix exponential, integration diagnostics, stationary distributions, and classified nonlinear peak times.

## 9. Testing: `test_npl_kinetics.py`

The **uploaded test file contains 19 `test_*` functions**, using two module-scoped fixtures: `base` and `nonlinear_case`. Numerical thresholds in the tests are **acceptance criteria for this implementation**, not empirical validation of the physical theory.

| Test function | What it checks |
|---|---|
| `test_reference_rates_match_previous_example` | Computed default rates agree with the reference example |
| `test_2ml_threshold_is_parameter_dependent` | 2-ML viability depends on selected bulk energy |
| `test_thickness_gibbs_thomson_step` | Monotonic thickness GT and coexistence-factor trends |
| `test_lateral_correction_is_optional_and_monotone` | Optional geometry correction; smaller sizes increase the dissolution proxy |
| `test_invalid_parameter_inputs_are_rejected` | Selected invalid input cases are rejected |
| `test_generator_has_zero_column_sums_and_nonnegative_offdiagonals` | Linear generator's conservation and sign structure |
| `test_rhs_conserves_mass_at_arbitrary_positive_state` | Algebraic mass conservation for both RHS forms |
| `test_linear_eigen_decomposition_agrees_with_expm` | Eigenmode solution agrees with matrix exponential |
| `test_linear_stiff_solver_agrees_with_expm` | LSODA linear trajectory agrees with matrix exponential |
| `test_nonlinear_mass_conservation` | Integrated nonlinear trajectory conserves total inventory |
| `test_nonnegative_populations` | Output contains no significant negative inventory |
| `test_analytic_equilibrium_satisfies_balance` | Stationary states satisfy balance and zero net flux |
| `test_nonlinear_long_time_agrees_with_equilibrium` | Long-time numerical state approaches nonlinear equilibrium |
| `test_nonlinear_peak_detection_rejects_plateau` | Sequential 3→4→5 peaks; suppresses 6-ML plateau artifact |
| `test_sqrt_regularization_preserves_conservation_and_changes_model` | Regularization changes kinetics but conserves mass; Jacobian structure |
| `test_multiple_stiff_integrators_are_consistent_at_short_horizon` | LSODA/Radau agreement on a shorter interval |
| `test_linear_mass_conservation_after_lateral_gt_correction` | Lateral correction preserves the linear conservation identity |
| `test_regularized_jacobian_matches_finite_differences` | Analytic regularized Jacobian agrees with finite differences |
| `test_zero_bound_population_has_no_detachment` | Empty populations have zero detachment flux |

Run selected tests:

```bash
python -m pytest -q test_npl_kinetics.py -k "jacobian"
python -m pytest -q test_npl_kinetics.py -k "mass or equilibrium"
python -m pytest -v test_npl_kinetics.py::test_nonlinear_peak_detection_rejects_plateau
```

**Interpretation:** Passing tests verifies mathematical and software consistency under the tested configurations and tolerances. 

## 10. Optional versus default behavior

| Feature | Default | Optional configuration | Important limitation |
|---|---|---|---|
| Thickness classes | 3, 4, 5, 6 ML | `Parameters(thickness_ml=...)` via Python API | CLI `--lateral-nm` still hard-codes four inputs |
| Lateral GT | Off (`None`) | `--lateral-nm L3 L4 L5 L6` | Fixed representative dimensions, not a dynamic size distribution |
| Square-root regularization | Off (`epsilon=0`) | `--sqrt-epsilon >0` | Modifies the S22 nonlinear kinetics and stationary state |
| Stiff integrator | LSODA | BDF or Radau | Check conservation and convergence |
| Time samples | t=0 + 400 log times | `--points`, `--t-end` | Output samples are not internal steps; sparse sampling may miss roots |
| Plot | On | `--no-plot` | CSV/JSON remain available |
| Linear reference | `expm` | Additional eigenmode check | Floating-point errors increase for extreme times/conditioning |
