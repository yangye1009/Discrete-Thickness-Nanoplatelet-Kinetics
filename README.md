# Discrete-Thickness Nanoplatelet Kinetics

**A literature-based Python reimplementation and numerical validation study of reduced semiconductor nanoplatelet growth and ripening kinetics (2026).**

This project examines how a common monomer reservoir redistributes material among discrete **3–6-monolayer (ML) CdSe nanoplatelet populations**. It implements first-order and nonlinear population-balance equations informed by the supplementary theory in Riedinger *et al.* (2017), evaluates thickness-dependent energetics and attachment/detachment rates, and checks numerical trajectories against conservation laws, equilibrium conditions, and independent solution routes.

> **Research provenance.** This is an **educational/research reproduction and adaptation of published theory**, not a new mechanism, a complete reconstruction of the original kinetic Monte Carlo (kMC) calculations, or a fit to the CdS experiments of Knüsel *et al.* (2020). Numerical agreement within this reduced model is not experimental validation.

## At a glance

| Component | What is implemented | Qualification |
|---|---|---|
| Growth energetics | Wide- and narrow-facet island-energy expressions; thickness-dependent nucleation barriers | Approximate barrier extracted from a published narrow-facet expression |
| Population kinetics | Shared monomer pool and 3–6 ML populations; linear and nonlinear ODEs | Effective population model; not individual-platelet-resolved kMC |
| Thermodynamics | Thickness-dependent formation energy and attachment/detachment equilibrium ratio | CdSe reference values; optional lateral Gibbs–Thomson-like correction is an extension |
| Solvers | SciPy `LSODA`, `BDF`, `Radau`; matrix exponential and eigenmode references for the linear system | Stiff ODE integration with configurable tolerances |
| Verification | 19 pytest test functions; mass balance, positivity diagnostics, equilibria, solver comparisons, Jacobian checks, peak detection | Software/numerical checks, not empirical validation |
| Outputs | Time-dependent CSV, parameter/provenance JSON, and comparison PNG | Populations are normalized monomer inventories, **not** platelet counts or absorption spectra |

## Scientific question

Can a simplified, mass-conserving exchange model reproduce the *qualitative sequence* in which thin nanoplatelet populations become transiently prominent before thicker populations dominate, and how sensitive is that behavior to thickness-dependent rates and numerical treatment?

The default state vector is

$$
\mathbf{x}(t)=[N_0,N_3,N_4,N_5,N_6]^\mathsf{T},\qquad
N_0+\sum_{m=3}^{6}N_m=1.
$$

Here $N_0$ is the free-monomer inventory fraction and $N_m$ is the fraction of monomer units in the thickness-$m$ population; $N_m$ does **not** count individual nanoplatelets. All populations communicate through $N_0$, **without direct $m\to m+1$ conversions**.

Two kinetic forms are compared:

$$
\begin{aligned}
\text{Linear:}\quad&\dot N_m=k_m N_0-k_{-m}N_m,\\
\text{Nonlinear:}\quad&\dot N_m=k_m N_0^2-k_{-m}\sqrt{N_m},\\
&\dot N_0=-\sum_m\dot N_m.
\end{aligned}
$$

The model uses a thickness-dependent approximate nucleation barrier and an effective formation energy to compute the rates. See the [technical guide](NPL_re/README.md) for the energy expressions, source-equation mapping, parameter provenance, and solver details.

## Repository layout

```text
Discrete-Thickness-Nanoplatelet-Kinetics/
├── README.md                 # Place this overview at the repository root
├── NPL_re/
│   ├── npl_kinetics.py       # Model, rates, solvers, diagnostics, exports
│   ├── test_npl_kinetics.py  # 19 pytest test functions
│   └── README.md             # Full technical/validation guide
└── npl_outputs/              # Example/generated outputs (if retained)
```

## Quick start

Requires **Python 3.10+** and `numpy`, `scipy`, `matplotlib`, `pytest`.

```bash
# From the repository root
cd NPL_re
python -m pip install numpy scipy matplotlib pytest

# Simulate both the linear and nonlinear models
python npl_kinetics.py

# Run the verification suite
python -m pytest -q test_npl_kinetics.py

# Avoid plotting if only tabular data are needed
python npl_kinetics.py --no-plot --output npl_outputs

# Compare stiff integrators or change conditions
python npl_kinetics.py --method Radau --output out_radau
python npl_kinetics.py --temperature 500 --method BDF --output out_500K
```

The default settings use $T=473.15$ K (200 °C), a $10^9$ s simulation horizon, and 400 logarithmically spaced positive output times **plus $t=0$**. The default initial state places the entire normalized inventory in the free-monomer pool.

### Output interpretation

- `npl_results.csv`: linear and nonlinear normalized inventory histories for $N_0$ and the 3–6 ML populations.
- `original_paper_reference.json`: reference inputs, derived quantities, and source/classification metadata.
- `npl_comparison.png`: plotted population histories, unless `--no-plot` is supplied.

With the default reduced nonlinear model, the reported finite-time peak locations are approximately **4.58 s (3 ML)**, **0.527 h (4 ML)**, and **159.5 h (5 ML)**; **6 ML approaches a plateau without a verified finite-time peak**. These are model-specific numerical results, **not experimental ripening times**. See the technical guide for the peak-detection criterion and tolerance sensitivity.

## Verification strategy

The code checks several independent properties:

1. **Mass conservation:** zero column sums of the linear generator and small drift in total inventory during integration.
2. **Cross-solver agreement:** linear trajectories from the matrix exponential, eigenmode reconstruction, and stiff numerical integration.
3. **Equilibria:** stationary mass-balance and zero-flux conditions, including a scalar root solve for the nonlinear stationary state.
4. **Numerical sensitivity:** comparisons between stiff integrators, nonnegative-population diagnostics, and finite-difference checks of the *regularized* nonlinear Jacobian.
5. **Peak classification:** positive-to-negative derivative crossings with a prominence criterion, avoiding false peaks in nearly flat late-time trajectories.

**Passing tests establishes consistency under the tested cases and tolerances; it does not demonstrate that the simplified model uniquely explains experimental observations.**

## Published theory vs. implementation choices

| From the literature | Implementation / adaptation in this repository |
|---|---|
| Island-energy relations and growth-layer theory of Riedinger *et al.* (2017) | Uses the narrow-facet energy **intercept** as an approximate barrier in the rate calculation |
| Supplementary linear and nonlinear exchange forms | Restricts the state space to normalized 3–6 ML populations plus one free-monomer pool |
| Thickness-dependent thermodynamic contribution | Adds an **optional** lateral-size Gibbs–Thomson-like term using fixed representative lateral lengths; disabled by default |
| Original square-root nonlinear detachment law | Provides an **optional model-changing** regularization (`--sqrt-epsilon`), disabled by default |
| Experimental CdS ripening context (Knüsel *et al.*, 2020) | Used as background only: **no CdS fitting, optical transition calculation, or absorption-spectrum prediction** |

## Limits and next steps

- This is a **reduced effective kinetic model**, not an atomistic DFT calculation, full kMC simulator, or spatially resolved nucleation/growth model.
- The populations represent **monomer inventory fractions**, not size distributions or platelet number concentrations.
- Lateral dimensions are fixed when the optional correction is enabled; their evolution is not solved dynamically.
- Reference material parameters are associated primarily with **CdSe**; reported **CdS** optical peaks come from independent experimental literature and are not computed here.
- Stiff solver tolerances control local error estimates, not strict positivity or global error guarantees. Very long integrations and weak/flat peaks require extra scrutiny.
- Further validation would require systematic comparisons with published calculations and, for physical parameter inference, experimental measurements matched to the same material and observables.

## References and credit

**[R1]** Riedinger, A., Ott, F. D., Mule, A., *et al.* (2017). “An intrinsic growth instability in isotropic materials leads to quasi-two-dimensional nanoplatelets.” *Nature Materials* **16**, 743–748. https://doi.org/10.1038/nmat4889

**[R2]** Knüsel, P. N., Riedinger, A., Rossinelli, A. A., Ott, F. D., Mule, A. S., Norris, D. J. (2020). “Experimental Evidence for Two-Dimensional Ostwald Ripening in Semiconductor Nanoplatelets.” *Chemistry of Materials* **32**, 3312–3319. https://doi.org/10.1021/acs.chemmater.0c01238

**[R3]** Ott, F. D., *et al.* (2017). “Ripening of Semiconductor Nanoplatelets.” *Nano Letters* **17**, 6870–6877. https://doi.org/10.1021/acs.nanolett.7b03191

**Project status:** Independent study of the published models and a 2026 Python implementation with AI-assisted development. The underlying physical theory and cited experimental findings belong to the original authors; this repository does not claim an original experimental or theoretical discovery. For detailed assumptions and reproducibility procedures, read [`NPL_re/README.md`](NPL_re/README.md).
