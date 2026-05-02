# A-7A Corsair II Stability Analysis — Code Design

**Date:** 2026-05-02
**Course:** AERO 321 — Flight Dynamics
**Deliverable target:** `src/321_Final_Project/main.py` (single Python file) plus PNG plots in `src/321_Final_Project/figures/`

---

## 1. Goal

Produce a single-file Python program that performs every numerical task required by the AERO 321 final project for the A-7A Corsair II at the cruise trim condition (15,000 ft, M = 0.6). The program supports the LaTeX report at `src/321_Final_Project/321_final_project.tex` by:

- Computing all dimensional longitudinal and lateral-directional stability derivatives.
- Building the longitudinal and lateral-directional state-space matrices.
- Computing eigenvalues, modal parameters (ζ, ωn, ωd, T, t½, τ), and eigenvector contributions.
- Identifying each mode (short period, phugoid, dutch roll, roll subsidence, spiral).
- Plotting impulse responses to each control input.
- Classifying each mode against MIL-F-8785C flying-qualities requirements.

The student transcribes the printed numerical results into the LaTeX report by hand. The Python file itself is included in the report appendix.

## 2. Non-goals

- No CLI, no argparse, no config files. The program runs as `python main.py` and that is the only invocation.
- No `.tex` snippet generation. Numbers go to stdout for hand-transcription.
- No `python-control` or other "black-box" libraries. NumPy + SciPy + matplotlib only.
- No tests. This is a one-shot computation; correctness is verified by inspection against textbook reference values.
- No nonlinear simulation. Strictly linearized perturbation analysis at trim.

## 3. Success criteria

1. `python main.py` from `src/321_Final_Project/` runs to completion, prints all required tables, and writes 4 PNGs to `figures/`.
2. Numerical results are within ~1% of textbook A-7A values where available (e.g., Roskam Vol. 1 Appendix B).
3. Console output is unambiguously formatted so the user can transcribe values into LaTeX without misreading.
4. Code style is indistinguishable from the user's hand-written Python (matches `~/Downloads/Aero-306-Project-1-1/` and `~/PycharmProjects/Capstone-Systems-Engineering/scripts/` conventions; see memory file `feedback_python_style.md`).
5. PNGs use `matplotlib` `usetex=True` with serif fonts and render cleanly at 300 dpi when included in the report via `\includegraphics`.

## 4. Inputs (constants block at top of file)

### 4.1 Geometry, mass, inertias

```
S    = 375 ft^2          # wing area
b    = 38.7 ft           # wingspan
c    = 10.8 ft           # mean aerodynamic chord
W    = 21,889 lbf        # weight
Ixx  = 13,635  slug-ft^2
Iyy  = 58,966  slug-ft^2
Izz  = 67,560  slug-ft^2
Ixz  = 2,933   slug-ft^2
g    = 32.174 ft/s^2
m    = W/g
```

### 4.2 Trim condition (Cruise, 15,000 ft, M = 0.6)

```
alt  = 15,000 ft
M    = 0.6
rho  = 1.4962e-3 slug/ft^3       (1976 US Standard Atmosphere)
a    = 1057.4 ft/s
U1   = M * a    = 634.4 ft/s
qbar = 0.5*rho*U1^2 = 301.0 lbf/ft^2
alpha1   = 4.0 deg
delta_e1 = -3.87 deg
theta1   = alpha1   (level cruise; flight-path angle gamma1 = 0)
```

### 4.3 Longitudinal nondimensional derivatives

From the project handout. `CLM`, `CDM`, `CmM` Mach-derivatives are listed by the handout but their effects are folded into `CLU`, `CDU`, `CMU` for `u`-derivatives, so they are recorded as comments only.

```
CL1, CD1, CM1   = 0.19, 0.02, 0
CL0, CD0, CM0   = 0.149, 0.0205, -0.08
CLU, CDU, CMU   = -0.294, -0.0364, 0.032
CLA, CDA, CMA   = 4.42, 0.378, -0.437
CLAD, CMAD      = 0.0, -0.752
CLQ, CMQ        = 1.42, -3.94
CLDE, CDDE, CMDE = 0.59, -0.042, -0.912
```

### 4.4 Lateral-directional nondimensional derivatives

```
CYB,  CLB,  CNB  = -0.715, -0.087, 0.075
CYP,  CLP,  CNP  = 0.0,    -0.265, 0.0
CYR,  CLR,  CNR  = 0.0,     0.10, -0.30
CYDA, CLDA, CNDA = -0.025,  0.055, 0.00575
CYDR, CLDR, CNDR = 0.21,    0.020, -0.0925
```

### 4.5 Thrust derivative assumptions (handout omits thrust derivatives)

```
CTX1   = CD1            # steady cruise: thrust = drag
CTXU   = -2 * CD1       # constant-thrust jet (cancels +2*CD1 from qbar)
CMTU   = 0              # thrust line through cg
CMTA   = 0
```

These assumptions are documented at the top of the code and noted in the printed output.

## 5. File structure

A single file `src/321_Final_Project/main.py` organized top-to-bottom as:

1. Module docstring (one-paragraph summary).
2. `from __future__ import annotations`.
3. Imports: `numpy`, `matplotlib.pyplot`, `scipy.linalg.eig`, `scipy.linalg.expm`, `pathlib.Path`.
4. `# ============================================================================` banner — Aircraft data constants block (Sections 4.1–4.5 above).
5. Banner — Part 1: dimensional derivative computation functions.
6. Banner — Parts 2 & 3: state-space matrix builders.
7. Banner — Parts 4 & 5: modal analysis and mode identification.
8. Banner — Part 6: impulse response computation and plotting.
9. Banner — Part 7: flying-qualities classification per MIL-F-8785C.
10. Banner — Console printing helpers (`_print_header`, `_print_derivative_table`, `_print_matrix`, `_print_modal_table`, `_print_eigenvector_table`, `_print_flying_qualities`).
11. `if __name__ == "__main__":` block — runs the full pipeline and saves figures.

Style conforms to the conventions in `feedback_python_style.md` (Google-style docstrings, PEP 604 type hints, `# === Banner ===` major dividers, `# --- inline ---` minor dividers, `_private` helpers, `UPPER_SNAKE_CASE` constants, double-quoted strings, two blank lines between top-level functions).

## 6. Computational design (per part)

### Part 1 — Dimensional derivatives

Two functions: `compute_longitudinal_derivatives()` and `compute_lateral_derivatives()`. Each returns a `dict[str, float]` keyed by symbolic name. Formulas implemented exactly as boxed in `321_final_project.tex`. Examples:

```
Xu  = -(qbar*S/(m*U1)) * (CDU + 2*CD1)
Xa  = -(qbar*S/m) * (CDA - CL1)
Xde = -(qbar*S/m) * CDDE

Zu  = -(qbar*S/(m*U1)) * (CLU + 2*CL1)
Za  = -(qbar*S/m) * (CLA + CD1)
Zad = -(qbar*S*c/(2*m*U1)) * CLAD
Zq  = -(qbar*S*c/(2*m*U1)) * CLQ
Zde = -(qbar*S/m) * CLDE

Mu  =  (qbar*S*c/(Iyy*U1)) * (CMU + 2*CM1)
Ma  =  (qbar*S*c/Iyy) * CMA
Mad =  (qbar*S*c**2/(2*Iyy*U1)) * CMAD
Mq  =  (qbar*S*c**2/(2*Iyy*U1)) * CMQ
Mde =  (qbar*S*c/Iyy) * CMDE

XTu =  (qbar*S/(m*U1)) * (CTXU + 2*CTX1)
MTu =  (qbar*S*c/(Iyy*U1)) * (CMTU + 2*0.0)
MTa =  (qbar*S*c/Iyy) * CMTA
```

`w`-derivatives produced by `Zw = Za/U1`, `Mw = Ma/U1`, `Mwd = Mad/U1`, and printed alongside `α`-derivatives.

Lateral analogues with `b` replacing `c` and the appropriate inertia.

### Parts 2 & 3 — State-space matrices

`build_longitudinal_ss(d)` and `build_lateral_ss(d)` each return `(A, B)` numpy arrays.

Longitudinal state vector `[Δu, Δw, Δq, Δθ]ᵀ`, control vector `[Δδe, Δδт]ᵀ`:

```
A_long = [[Xu,                Xw,              0,             -g     ],
          [Zu,                Zw,              U1,             0     ],
          [Mu + Mwd*Zu,       Mw + Mwd*Zw,    Mq + Mwd*U1,    0     ],
          [0,                 0,               1,              0     ]]

B_long = [[Xde,               XdT             ],
          [Zde,               ZdT             ],
          [Mde + Mwd*Zde,     MdT + Mwd*ZdT   ],
          [0,                 0               ]]
```

`XdT`, `ZdT`, `MdT` thrust controls are zero in the absence of throttle perturbation data; the column is retained for completeness so the matrix matches the handout image.

Lateral state vector `[Δβ, Δp, Δr, Δφ]ᵀ`, control vector `[Δδa, Δδr]ᵀ`. Built per the handout image:

```
A_lat = [[Yb/U1,            Yp/U1,            -(1 - Yr/U1),       g*cos(theta1)/U1],
         [Lb_prime,         Lp_prime,         Lr_prime,           0               ],
         [Nb_prime,         Np_prime,         Nr_prime,           0               ],
         [0,                1,                0,                  0               ]]

B_lat = [[0,                Ydr/U1          ],
         [Lda_prime,        Ldr_prime       ],
         [Nda_prime,        Ndr_prime       ],
         [0,                0               ]]
```

Because `Ixz ≠ 0`, only the rolling- and yawing-moment derivatives (`L*` and `N*`) are converted to **primed** form; side-force `Y*` derivatives are unchanged because they have no inertial coupling:

```
denom = 1 - Ixz**2 / (Ixx * Izz)
L_prime(X) = (L_X + (Ixz/Ixx) * N_X) / denom    # for X in {b, p, r, da, dr}
N_prime(X) = (N_X + (Ixz/Izz) * L_X) / denom    # for X in {b, p, r, da, dr}
```

This is the standard Roskam treatment and is consistent with the handout image (which shows no explicit `Ixz` terms in the matrix).

### Parts 4 & 5 — Modal analysis

`modal_analysis(A, state_names)` returns a list of dicts, one per eigenvalue:

```
{
    "lambda":   complex,
    "wn":       float,         # |lambda|
    "zeta":     float,         # -Re(lambda) / wn
    "wd":       float,         # Im(lambda)
    "period":   float | None,  # 2*pi / wd  (None if real eigenvalue)
    "t_half":   float | None,  # ln(2) / |Re(lambda)| if Re < 0, else None
    "t_double": float | None,  # ln(2) /  Re(lambda)  if Re > 0, else None
    "tau":      float | None,  # -1 / Re(lambda)      for real eigenvalues
    "vector":   np.ndarray,    # right eigenvector, complex
}
```

`identify_longitudinal_modes(modes)` partitions the eigenvalues by inspection: the higher-frequency complex pair is the **short period**, the lower-frequency complex pair is the **phugoid**.

`identify_lateral_modes(modes)` partitions: the complex pair is the **dutch roll**, the fast real eigenvalue (most negative) is the **roll subsidence**, the slow real eigenvalue (closest to zero) is the **spiral**.

Eigenvector contributions are reported as a magnitude-and-phase table per mode per state. Magnitudes are normalized so the largest entry per mode = 1.0.

### Part 6 — Impulse response

`impulse_response(A, B, u_index, t)` returns `X` of shape `(len(t), n_states)`:

```
X[k, :] = expm(A * t[k]) @ B[:, u_index]
```

Two longitudinal figures (`long_impulse_de.png`, `long_impulse_dT.png`), each with 4 stacked subplots (Δu in ft/s, Δα = Δw/U₁ in rad, Δq in rad/s, Δθ in rad). Time grid: `t = np.linspace(0, 200, 1500)` to capture the phugoid period.

Two lateral figures (`lat_impulse_da.png`, `lat_impulse_dr.png`), each with 4 stacked subplots (Δβ, Δp, Δr, Δφ in rad or rad/s). Time grid: `t = np.linspace(0, 30, 1500)`.

All four figures saved at 300 dpi, `usetex=True`, serif font, single solid black line per subplot, light gridlines.

### Part 7 — Flying qualities (MIL-F-8785C, Class IV fighter, Cat B cruise)

Five classifier functions, each returning a string `"Level 1"`, `"Level 2"`, `"Level 3"`, or `"Worse than Level 3"`:

- `classify_short_period(zeta_sp, wn_sp, n_alpha)` — checks ζ_sp band (0.35–1.30 for Level 1) and CAP = ωn²/n_α band (0.085–3.6 for Level 1).
- `classify_phugoid(zeta_ph)` — Level 1: ζ ≥ 0.04; Level 2: ζ ≥ 0; Level 3: t_double ≥ 55 s.
- `classify_dutch_roll(zeta_dr, wn_dr)` — Level 1 requires ζ ≥ 0.08, ζ·ωn ≥ 0.15, and ωn ≥ 0.4.
- `classify_roll_mode(tau_r)` — Level 1: τ ≤ 1.4 s; Level 2: ≤ 3 s; Level 3: ≤ 10 s.
- `classify_spiral(t_double_spiral)` — Level 1: t_double ≥ 20 s; Level 2: ≥ 12 s; Level 3: ≥ 4 s. (Stable spiral = Level 1 automatically.)

`n_alpha = (U1 / g) * (-Za / U1) = -Za / g` is the load-factor sensitivity.

Output is a printed table:

```
Mode             Metric                 Value      Level    Comment
Short period     zeta                   0.42       1        --
Short period     CAP (1/s^2)            1.85       1        --
Phugoid          zeta                   0.07       1        --
Dutch roll       zeta                   0.12       1        --
Dutch roll       zeta * wn (1/s)        0.31       1        --
Dutch roll       wn (rad/s)             2.6        1        --
Roll subsidence  tau (s)                0.55       1        --
Spiral           t_double (s)           --         1        Stable
```

## 7. Output specification

### 7.1 Console output

Run order, each section preceded by a 78-char `=`-banner:

1. Assumptions block (atmosphere, thrust derivative assumptions, primed-derivative usage).
2. Trim summary table (`U1`, `qbar`, `m`, `theta1`, etc.).
3. Longitudinal dimensional derivatives — three columns: symbol, value, units.
4. Lateral-directional dimensional derivatives — same format.
5. `A_long` matrix (4×4) with row/column labels.
6. `B_long` matrix (4×2) with row/column labels.
7. `A_lat`  matrix (4×4) with row/column labels.
8. `B_lat`  matrix (4×2) with row/column labels.
9. Longitudinal modal table: mode | λ | ωn | ζ | T | t½ | τ.
10. Longitudinal eigenvector table: mode × state grid of `magnitude ∠ phase°`.
11. Lateral-directional modal table.
12. Lateral-directional eigenvector table.
13. Flying-qualities table (per Part 7 above).
14. List of saved figure paths.

All numerical values printed with 4 significant figures (`{:.4g}`). Matrix entries aligned in fixed-width columns.

### 7.2 Figure output

Saved to `src/321_Final_Project/figures/` (created if absent):

- `long_impulse_de.png` — 4-subplot column, response to elevator impulse.
- `long_impulse_dT.png` — 4-subplot column, response to throttle impulse (zeros if `XdT = ZdT = MdT = 0`, but file is still produced for completeness).
- `lat_impulse_da.png` — 4-subplot column, response to aileron impulse.
- `lat_impulse_dr.png` — 4-subplot column, response to rudder impulse.

Each figure: 6.5 in wide × 8 in tall, 300 dpi, serif font with `usetex=True`, axis labels in LaTeX math, light gray gridlines, single solid black line per subplot, time on x-axis in seconds.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `usetex=True` requires a working LaTeX install on PATH at `python main.py` runtime. | The user has confirmed they run inside PyCharm with a working LaTeX install, so this is acceptable. If it fails, falling back to `mathtext` is a one-line change to `plt.rcParams`. |
| Mode identification by inspection (frequency band) could mis-tag modes for unusual aircraft. | The A-7A is a conventional fighter with well-separated phugoid/short-period and well-separated roll/spiral; mode separation will be obvious. The identification function uses defensive sorting (highest-`wn` complex pair = short period) and prints both candidates so the user can sanity-check. |
| Thrust-derivative assumptions could mis-state `XTu`. | Assumptions are printed at the top of stdout and labeled in the code so the user can override if the grader expects different conventions. |
| Primed-derivative form for `Ixz ≠ 0` could be flipped vs. what the grader expects. | This is the Roskam convention; the matrix image in the project handout shows no explicit `Ixz` terms, which implies primed derivatives are intended. Comment in code explains the choice. |

## 9. Dependencies

```
numpy
scipy        (linalg.eig, linalg.expm)
matplotlib   (with text.usetex = True)
```

No `python-control`, no `slycot`, no other libraries. Standard library: `pathlib` only.

## 10. Out of scope (explicitly not included)

- Nonlinear simulation.
- Trim solver (trim is given by the handout).
- Comparison plots vs. published A-7A flight-test data.
- Sensitivity analysis or Monte Carlo over derivative uncertainty.
- Compilation of the LaTeX report (the user runs the LaTeX build themselves in PyCharm).
