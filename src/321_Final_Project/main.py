"""A-7A Corsair II -- longitudinal and lateral-directional stability analysis.

Computes dimensional stability derivatives, builds the linearized state-space
matrices, performs eigenvalue and modal analysis, plots impulse responses to
each control input, and classifies each mode against the MIL-F-8785C flying
qualities requirements (Class IV, Category B). Single-script entry point:

    python main.py

Output: numerical tables to stdout (for hand-transcription into the LaTeX
report) and four PNG figures written to ./figures/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, expm


# =============================================================================
# Aircraft data -- A-7A Corsair II, cruise trim @ 15,000 ft, M = 0.6
# =============================================================================
# -- geometry, mass, inertias --
S, B, CBAR = 375.0, 38.7, 10.8                              # ft^2, ft, ft
W = 21889.0                                                 # lbf
IXX, IYY, IZZ, IXZ = 13635.0, 58966.0, 67560.0, 2933.0      # slug-ft^2
G = 32.174                                                  # ft/s^2
M_MASS = W / G                                              # slug

# -- atmosphere (1976 US Standard at 15,000 ft) and trim --
ALT_FT = 15000.0
MACH = 0.6
RHO = 1.4962e-3                                             # slug/ft^3
A_SOUND = 1057.4                                            # ft/s
U1 = MACH * A_SOUND                                         # ft/s
QBAR = 0.5 * RHO * U1**2                                    # lbf/ft^2
ALPHA1 = np.deg2rad(4.0)                                    # rad
DELTA_E1 = np.deg2rad(-3.87)                                # rad
THETA1 = ALPHA1                                             # level cruise (gamma1 = 0)

# -- longitudinal nondimensional derivatives (per project handout) --
CL1, CD1, CM1 = 0.19, 0.02, 0.0
CL0, CD0, CM0 = 0.149, 0.0205, -0.08
CLU, CDU, CMU = -0.294, -0.0364, 0.032
CLA, CDA, CMA = 4.42, 0.378, -0.437
CLAD, CMAD = 0.0, -0.752
CLQ, CMQ = 1.42, -3.94
CLDE, CDDE, CMDE = 0.59, -0.042, -0.912
# Mach derivatives (CLM = 0.012, CDM = 0, CmM = -0.005) are folded into
# CLU/CDU/CMU per the handout convention; recorded here for reference only.

# -- lateral-directional nondimensional derivatives --
# Note: rolling-moment coefficients use lowercase "l" (Cl_*) to disambiguate
# from lift coefficients (CL*). C_L is lift, C_l is rolling moment.
CYB, Cl_B, CNB = -0.715, -0.087, 0.075
CYP, Cl_P, CNP = 0.0, -0.265, 0.0
CYR, Cl_R, CNR = 0.0, 0.10, -0.30
CYDA, Cl_DA, CNDA = -0.025, 0.055, 0.00575
CYDR, Cl_DR, CNDR = 0.21, 0.020, -0.0925

# -- thrust derivative assumptions (handout omits thrust derivatives) --
# Steady cruise: thrust balances drag, so C_TX1 = C_D1.
# Constant-thrust jet model: C_TXu = -2 * C_TX1 (cancels +2*C_D1 from qbar).
# Pitching-moment thrust derivatives assumed zero (thrust line through cg).
CTX1 = CD1
CTXU = -2.0 * CD1
CMTU = 0.0
CMTA = 0.0

# -- output paths --
FIG_DIR = Path(__file__).parent / "figures"


# =============================================================================
# Part 1a: longitudinal dimensional stability derivatives
# =============================================================================
def compute_longitudinal_derivatives() -> dict[str, float]:
    """Compute every dimensional longitudinal stability derivative at trim.

    Returns:
        dict[str, float]: keys are symbolic names (Xu, Xa, Xw, Xde, Zu, ...,
            Mq, Mwd, Mde, XTu, MTu, MTa) with values in SI-equivalent English
            engineering units (1/s, 1/(ft*s), ft/s^2, etc.).

    Notes:
        Formulas implemented per the boxed equations in 321_final_project.tex.
        The +2*C_(*)1 terms in u-derivatives come from differentiating qbar.
        Thrust derivatives use the steady-cruise + constant-thrust assumptions
        documented at the top of this file.
    """
    d: dict[str, float] = {}

    # -- X-force derivatives --
    d["Xu"]  = -(QBAR * S / (M_MASS * U1)) * (CDU + 2.0 * CD1)
    d["Xa"]  = -(QBAR * S / M_MASS) * (CDA - CL1)
    d["Xw"]  = d["Xa"] / U1
    d["Xde"] = -(QBAR * S / M_MASS) * CDDE
    d["XTu"] =  (QBAR * S / (M_MASS * U1)) * (CTXU + 2.0 * CTX1)

    # -- Z-force derivatives --
    d["Zu"]  = -(QBAR * S / (M_MASS * U1)) * (CLU + 2.0 * CL1)
    d["Za"]  = -(QBAR * S / M_MASS) * (CLA + CD1)
    d["Zw"]  = d["Za"] / U1
    d["Zad"] = -(QBAR * S * CBAR / (2.0 * M_MASS * U1)) * CLAD
    d["Zwd"] = d["Zad"] / U1
    d["Zq"]  = -(QBAR * S * CBAR / (2.0 * M_MASS * U1)) * CLQ
    d["Zde"] = -(QBAR * S / M_MASS) * CLDE

    # -- M pitching-moment derivatives --
    d["Mu"]  =  (QBAR * S * CBAR / (IYY * U1)) * (CMU + 2.0 * CM1)
    d["Ma"]  =  (QBAR * S * CBAR / IYY) * CMA
    d["Mw"]  =  d["Ma"] / U1
    d["Mad"] =  (QBAR * S * CBAR**2 / (2.0 * IYY * U1)) * CMAD
    d["Mwd"] =  d["Mad"] / U1
    d["Mq"]  =  (QBAR * S * CBAR**2 / (2.0 * IYY * U1)) * CMQ
    d["Mde"] =  (QBAR * S * CBAR / IYY) * CMDE
    d["MTu"] =  (QBAR * S * CBAR / (IYY * U1)) * CMTU
    d["MTa"] =  (QBAR * S * CBAR / IYY) * CMTA

    # -- thrust controls (no throttle-perturbation data in handout) --
    d["XdT"] = 0.0
    d["ZdT"] = 0.0
    d["MdT"] = 0.0

    return d


def _print_derivative_table(title: str, derivs: dict[str, float],
                            units: dict[str, str]) -> None:
    """Print a three-column table: symbol, value (4 sig figs), units."""
    print()
    print("-" * 78)
    print(f" {title}")
    print("-" * 78)
    print(f" {'Symbol':<8}  {'Value':>14}  {'Units':<14}")
    for name, value in derivs.items():
        print(f" {name:<8}  {value:>14.4g}  {units.get(name, ''):<14}")


_LONG_UNITS: dict[str, str] = {
    "Xu":  "1/s",        "Xa":  "ft/s^2",   "Xw":  "1/s",       "Xde": "ft/s^2",
    "XTu": "1/s",        "XdT": "ft/s^2",
    "Zu":  "1/s",        "Za":  "ft/s^2",   "Zw":  "1/s",       "Zad": "ft/s",
    "Zwd": "1",          "Zq":  "ft/s",     "Zde": "ft/s^2",    "ZdT": "ft/s^2",
    "Mu":  "1/(ft*s)",   "Ma":  "1/s^2",    "Mw":  "1/(ft*s)",  "Mad": "1/s",
    "Mwd": "1/ft",       "Mq":  "1/s",      "Mde": "1/s^2",     "MTu": "1/(ft*s)",
    "MTa": "1/s^2",      "MdT": "1/s^2",
}


# =============================================================================
# Part 1b: lateral-directional dimensional stability derivatives
# =============================================================================
def compute_lateral_derivatives() -> dict[str, float]:
    """Compute every dimensional lateral-directional derivative at trim.

    Returns:
        dict[str, float]: keys are symbolic names (Yb, Yp, Yr, Yda, Ydr,
            Lb, Lp, Lr, Lda, Ldr, Nb, Np, Nr, Nda, Ndr) with values in
            English engineering units. These are the "unprimed" derivatives;
            the primed forms (which absorb Ixz cross-coupling) are computed
            separately in build_lateral_ss().

    Notes:
        Formulas per the boxed equations in 321_final_project.tex. The lateral
        channel has no u-derivatives; symmetric trim gives Cy1 = Cl1 = Cn1 = 0
        so there is no +2*C_(*)1 kinematic term.
    """
    d: dict[str, float] = {}

    # -- Y side-force derivatives --
    d["Yb"]  = (QBAR * S / M_MASS) * CYB
    d["Yp"]  = (QBAR * S * B / (2.0 * M_MASS * U1)) * CYP
    d["Yr"]  = (QBAR * S * B / (2.0 * M_MASS * U1)) * CYR
    d["Yda"] = (QBAR * S / M_MASS) * CYDA
    d["Ydr"] = (QBAR * S / M_MASS) * CYDR

    # -- L rolling-moment derivatives --
    d["Lb"]  = (QBAR * S * B / IXX) * Cl_B
    d["Lp"]  = (QBAR * S * B**2 / (2.0 * IXX * U1)) * Cl_P
    d["Lr"]  = (QBAR * S * B**2 / (2.0 * IXX * U1)) * Cl_R
    d["Lda"] = (QBAR * S * B / IXX) * Cl_DA
    d["Ldr"] = (QBAR * S * B / IXX) * Cl_DR

    # -- N yawing-moment derivatives --
    d["Nb"]  = (QBAR * S * B / IZZ) * CNB
    d["Np"]  = (QBAR * S * B**2 / (2.0 * IZZ * U1)) * CNP
    d["Nr"]  = (QBAR * S * B**2 / (2.0 * IZZ * U1)) * CNR
    d["Nda"] = (QBAR * S * B / IZZ) * CNDA
    d["Ndr"] = (QBAR * S * B / IZZ) * CNDR

    return d


_LAT_UNITS: dict[str, str] = {
    "Yb":  "ft/s^2", "Yp":  "ft/s",   "Yr":  "ft/s",
    "Yda": "ft/s^2", "Ydr": "ft/s^2",
    "Lb":  "1/s^2",  "Lp":  "1/s",    "Lr":  "1/s",
    "Lda": "1/s^2",  "Ldr": "1/s^2",
    "Nb":  "1/s^2",  "Np":  "1/s",    "Nr":  "1/s",
    "Nda": "1/s^2",  "Ndr": "1/s^2",
}


# =============================================================================
# Parts 2 & 3a: longitudinal state-space matrices
# =============================================================================
# State vector:   x_long = [du, dw, dq, dtheta]^T
# Control vector: u_long = [d_de, d_dT]^T
def build_longitudinal_ss(d: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (A_long, B_long) per the matrix form in the project handout.

    Args:
        d: dictionary returned by compute_longitudinal_derivatives().

    Returns:
        (A_long, B_long): A is 4x4, B is 4x2. State ordering is
            [du, dw, dq, dtheta]; control ordering is [d_de, d_dT].

    Notes:
        Mwd entries appear in row 3 (the q-equation) because the Mwd*Zw term
        comes from substituting the w-equation into the q-equation to remove
        the alpha-dot dependency. This is the standard "small-disturbance"
        longitudinal A-matrix as it appears in the project handout image.
    """
    Xu, Xw, Xde       = d["Xu"], d["Xw"], d["Xde"]
    Zu, Zw, Zde       = d["Zu"], d["Zw"], d["Zde"]
    Mu, Mw, Mwd       = d["Mu"], d["Mw"], d["Mwd"]
    Mq, Mde           = d["Mq"], d["Mde"]
    XdT, ZdT, MdT     = d["XdT"], d["ZdT"], d["MdT"]

    A = np.array([
        [Xu,                 Xw,                 0.0,             -G          ],
        [Zu,                 Zw,                 U1,               0.0        ],
        [Mu + Mwd * Zu,      Mw + Mwd * Zw,      Mq + Mwd * U1,    0.0        ],
        [0.0,                0.0,                1.0,              0.0        ],
    ])

    B = np.array([
        [Xde,                XdT             ],
        [Zde,                ZdT             ],
        [Mde + Mwd * Zde,    MdT + Mwd * ZdT ],
        [0.0,                0.0             ],
    ])

    return A, B


def _print_matrix(name: str, M: np.ndarray,
                  row_labels: list[str], col_labels: list[str]) -> None:
    """Print a labeled matrix with 4 significant figures per entry."""
    print()
    print("-" * 78)
    print(f" {name}")
    print("-" * 78)
    col_width = 14
    header = " " * 10 + "".join(f"{c:>{col_width}}" for c in col_labels)
    print(header)
    for i, row in enumerate(M):
        row_str = "".join(f"{v:>{col_width}.4g}" for v in row)
        print(f" {row_labels[i]:<8}  {row_str}")


# =============================================================================
# Parts 2 & 3b: lateral-directional state-space matrices (primed derivatives)
# =============================================================================
# State vector:   x_lat = [dbeta, dp, dr, dphi]^T
# Control vector: u_lat = [d_da, d_dr]^T
def _prime_lateral(d: dict[str, float]) -> dict[str, float]:
    """Return the primed L and N derivatives that absorb Ixz cross-coupling.

    Args:
        d: dictionary from compute_lateral_derivatives().

    Returns:
        dict[str, float]: same keys as d, but L* and N* entries replaced by
            their primed forms. Y* entries pass through unchanged because the
            side-force equation has no inertial coupling.

    Notes:
        This is the standard Roskam treatment for Ixz != 0:
            L'_X = (L_X + (Ixz/Ixx) * N_X) / (1 - Ixz^2 / (Ixx*Izz))
            N'_X = (N_X + (Ixz/Izz) * L_X) / (1 - Ixz^2 / (Ixx*Izz))
        for X in {b, p, r, da, dr}. Without this, the matrix is wrong by a
        few percent because the rolling and yawing accelerations are coupled
        through the product of inertia.
    """
    denom = 1.0 - IXZ**2 / (IXX * IZZ)
    primed = dict(d)  # copy Y* entries unchanged
    for x in ("b", "p", "r", "da", "dr"):
        Lx, Nx = d[f"L{x}"], d[f"N{x}"]
        primed[f"L{x}"] = (Lx + (IXZ / IXX) * Nx) / denom
        primed[f"N{x}"] = (Nx + (IXZ / IZZ) * Lx) / denom
    return primed


def build_lateral_ss(d: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (A_lat, B_lat) using primed derivatives for L* and N*.

    Args:
        d: dictionary from compute_lateral_derivatives() (unprimed values).

    Returns:
        (A_lat, B_lat): A is 4x4, B is 4x2. State ordering is
            [dbeta, dp, dr, dphi]; control ordering is [d_da, d_dr].

    Notes:
        Yb/U1, Yp/U1, Yr/U1 appear in the beta-equation (row 1) because the
        sideslip rate is dbeta/dt = (1/U1) * (sum of side accelerations).
        The (1,3) entry is -(1 - Yr/U1) rather than -1 to keep the form
        general; for the A-7A Yr = 0 so it collapses to -1.
    """
    p = _prime_lateral(d)

    A = np.array([
        [d["Yb"] / U1,    d["Yp"] / U1,    -(1.0 - d["Yr"] / U1),    G * np.cos(THETA1) / U1],
        [p["Lb"],         p["Lp"],         p["Lr"],                  0.0],
        [p["Nb"],         p["Np"],         p["Nr"],                  0.0],
        [0.0,             1.0,             0.0,                      0.0],
    ])

    B = np.array([
        [0.0,             d["Ydr"] / U1   ],
        [p["Lda"],        p["Ldr"]        ],
        [p["Nda"],        p["Ndr"]        ],
        [0.0,             0.0             ],
    ])

    return A, B


# =============================================================================
# Parts 4 & 5: eigenvalue and modal analysis
# =============================================================================
def modal_analysis(A: np.ndarray) -> list[dict]:
    """Compute eigenvalues, modal parameters, and right eigenvectors.

    Args:
        A: square state matrix (n x n).

    Returns:
        list[dict]: one entry per eigenvalue, each containing:
            "lambda":    complex eigenvalue,
            "wn":        natural frequency |lambda|,
            "zeta":      damping ratio -Re(lambda)/wn,
            "wd":        damped frequency Im(lambda) (>= 0 by convention),
            "period":    2*pi/wd if wd > 0 else None,
            "t_half":    ln(2)/|Re(lambda)| if Re < 0 else None (time to half),
            "t_double":  ln(2)/Re(lambda) if Re > 0 else None (time to double),
            "tau":       -1/Re(lambda) for real eigenvalues, else None,
            "vector":    right eigenvector (length-n complex array).

    Notes:
        Returns one dict per eigenvalue (so a complex conjugate pair appears
        as two entries). Downstream identification functions partition these
        into named modes.
    """
    eigvals, eigvecs = eig(A)
    out: list[dict] = []
    for i, lam in enumerate(eigvals):
        sigma = float(lam.real)
        omega = float(lam.imag)
        wn = float(np.abs(lam))
        is_real = abs(omega) < 1e-9

        entry: dict = {
            "lambda":   lam,
            "wn":       wn,
            "zeta":     (-sigma / wn) if wn > 1e-12 else 0.0,
            "wd":       abs(omega),
            "period":   (2.0 * np.pi / abs(omega)) if not is_real else None,
            "t_half":   (np.log(2.0) / abs(sigma)) if sigma < 0.0 else None,
            "t_double": (np.log(2.0) / sigma) if sigma > 0.0 else None,
            "tau":      (-1.0 / sigma) if (is_real and abs(sigma) > 1e-12) else None,
            "vector":   eigvecs[:, i],
        }
        out.append(entry)
    return out


def _print_modal_table(title: str, modes: list[dict], mode_names: list[str]) -> None:
    """Print a per-mode table: name, lambda, wn, zeta, period, t_half/t_double, tau."""
    print()
    print("-" * 78)
    print(f" {title}")
    print("-" * 78)
    print(f" {'Mode':<18} {'lambda':>22} {'wn':>8} {'zeta':>8} {'T(s)':>8} "
          f"{'t1/2 or t2(s)':>14} {'tau(s)':>8}")
    for name, m in zip(mode_names, modes):
        lam_str = f"{m['lambda'].real:+.4g}{m['lambda'].imag:+.4g}j"
        T   = f"{m['period']:.4g}" if m["period"]   is not None else "--"
        t12 = (f"{m['t_half']:.4g}" if m["t_half"]  is not None
               else (f"-{m['t_double']:.4g}" if m["t_double"] is not None else "--"))
        tau = f"{m['tau']:.4g}" if m["tau"] is not None else "--"
        print(f" {name:<18} {lam_str:>22} {m['wn']:>8.4g} {m['zeta']:>8.4g} "
              f"{T:>8} {t12:>14} {tau:>8}")


# =============================================================================
# Parts 4 & 5 (cont.): mode identification and eigenvector display
# =============================================================================
def identify_longitudinal_modes(modes: list[dict]) -> dict[str, dict]:
    """Tag the four longitudinal eigenvalues as short period / phugoid pairs.

    Args:
        modes: list of mode dicts from modal_analysis().

    Returns:
        dict with keys "short_period" and "phugoid"; each value is the mode
        dict for ONE of the eigenvalues in that conjugate pair (the one with
        positive imaginary part by convention, for plotting).

    Notes:
        Identification is by natural frequency: the higher-wn complex pair is
        the short period, the lower-wn complex pair is the phugoid. For the
        A-7A both pairs are well-separated (~3 rad/s vs. ~0.07 rad/s) so this
        is unambiguous.
    """
    complex_modes = [m for m in modes if m["wd"] > 1e-9 and m["lambda"].imag > 0]
    complex_modes.sort(key=lambda m: m["wn"], reverse=True)
    if len(complex_modes) < 2:
        raise ValueError(
            f"Expected 2 complex pairs in longitudinal modes; got {len(complex_modes)}"
        )
    return {"short_period": complex_modes[0], "phugoid": complex_modes[1]}


def identify_lateral_modes(modes: list[dict]) -> dict[str, dict]:
    """Tag the four lateral eigenvalues as dutch roll / roll / spiral.

    Args:
        modes: list of mode dicts from modal_analysis().

    Returns:
        dict with keys "dutch_roll", "roll", "spiral".

    Notes:
        Dutch roll = the unique complex pair (positive-imaginary representative).
        Roll subsidence = the more-negative real eigenvalue (fast, well damped).
        Spiral = the real eigenvalue closest to zero (slow; can be slightly
        positive for an aircraft with weak spiral stability).
    """
    complex_modes = [m for m in modes if m["wd"] > 1e-9 and m["lambda"].imag > 0]
    real_modes    = [m for m in modes if m["wd"] <= 1e-9]
    if len(complex_modes) != 1 or len(real_modes) != 2:
        raise ValueError(
            f"Expected 1 complex pair + 2 real eigenvalues; got "
            f"{len(complex_modes)} complex, {len(real_modes)} real"
        )
    real_modes.sort(key=lambda m: m["lambda"].real)  # most negative first
    return {
        "dutch_roll": complex_modes[0],
        "roll":       real_modes[0],
        "spiral":     real_modes[1],
    }


def _print_eigenvector_table(title: str, modes_named: dict[str, dict],
                             state_names: list[str]) -> None:
    """Print a magnitude-and-phase table of right eigenvectors per mode.

    Magnitudes are normalized so the largest entry per mode equals 1.0;
    phases are reported in degrees relative to that largest entry.
    """
    print()
    print("-" * 78)
    print(f" {title} (magnitude / phase deg, normalized to largest entry)")
    print("-" * 78)
    header = f" {'State':<10}"
    for name in modes_named:
        header += f" {name:>22}"
    print(header)

    for i, state in enumerate(state_names):
        row = f" {state:<10}"
        for m in modes_named.values():
            v = m["vector"]
            j = int(np.argmax(np.abs(v)))
            v_norm = v / v[j]
            entry = v_norm[i]
            row += f"  {abs(entry):>9.4g} / {np.rad2deg(np.angle(entry)):>+7.2f}"
        print(row)


# =============================================================================
# Part 6: impulse response computation and plotting
# =============================================================================
def impulse_response(A: np.ndarray, B: np.ndarray, u_index: int,
                     t: np.ndarray) -> np.ndarray:
    """Compute the closed-form impulse response of an LTI system.

    Args:
        A: state matrix (n x n).
        B: input matrix (n x m).
        u_index: which column of B (which control input) is being impulsed.
        t: time grid (length-N, monotonically increasing from 0).

    Returns:
        np.ndarray: (N x n) array; row k is x(t[k]) = expm(A * t[k]) @ B[:, u_index].

    Notes:
        For an impulse delta(t) on input u_index, the response is
        x(t) = e^{At} * B[:, u_index] for t >= 0. We avoid the convolution
        integral by using this closed form; expm is called per time step,
        which is fine for the modest N used here.
    """
    b = B[:, u_index]
    X = np.empty((len(t), A.shape[0]))
    for k, tk in enumerate(t):
        X[k, :] = expm(A * tk) @ b
    return X


def _plot_impulse(t: np.ndarray, X: np.ndarray, state_labels: list[str],
                  ylabels: list[str], title: str, out_path: Path) -> None:
    """Plot a stacked column of impulse responses, one subplot per state."""
    n = X.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 8.0), sharex=True)
    for i in range(n):
        axes[i].plot(t, X[:, i], color="black", linewidth=1.0)
        axes[i].set_ylabel(ylabels[i])
        axes[i].grid(True, color="0.85", linewidth=0.5)
    axes[-1].set_xlabel(r"Time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_long_impulse(A: np.ndarray, B: np.ndarray, out_dir: Path) -> list[Path]:
    """Generate the two longitudinal impulse-response figures (de, dT)."""
    t = np.linspace(0.0, 200.0, 1500)
    ylabels = [r"$\Delta u$ (ft/s)", r"$\Delta w$ (ft/s)",
               r"$\Delta q$ (rad/s)", r"$\Delta \theta$ (rad)"]
    state_labels = ["du", "dw", "dq", "dtheta"]
    paths: list[Path] = []
    for u_idx, name, title in [
        (0, "long_impulse_de.png",
         r"Longitudinal impulse response to $\delta_e$"),
        (1, "long_impulse_dT.png",
         r"Longitudinal impulse response to $\delta_T$"),
    ]:
        X = impulse_response(A, B, u_idx, t)
        out = out_dir / name
        _plot_impulse(t, X, state_labels, ylabels, title, out)
        paths.append(out)
    return paths


def plot_lat_impulse(A: np.ndarray, B: np.ndarray, out_dir: Path) -> list[Path]:
    """Generate the two lateral-directional impulse-response figures (da, dr)."""
    t = np.linspace(0.0, 30.0, 1500)
    ylabels = [r"$\Delta \beta$ (rad)", r"$\Delta p$ (rad/s)",
               r"$\Delta r$ (rad/s)", r"$\Delta \phi$ (rad)"]
    state_labels = ["dbeta", "dp", "dr", "dphi"]
    paths: list[Path] = []
    for u_idx, name, title in [
        (0, "lat_impulse_da.png",
         r"Lateral-directional impulse response to $\delta_a$"),
        (1, "lat_impulse_dr.png",
         r"Lateral-directional impulse response to $\delta_r$"),
    ]:
        X = impulse_response(A, B, u_idx, t)
        out = out_dir / name
        _plot_impulse(t, X, state_labels, ylabels, title, out)
        paths.append(out)
    return paths


# =============================================================================
# Part 7: flying qualities (MIL-F-8785C, Class IV fighter, Category B - cruise)
# =============================================================================
def classify_short_period(zeta_sp: float, wn_sp: float,
                          n_alpha: float) -> dict[str, float | str]:
    """Classify short-period damping and CAP per MIL-F-8785C, Class IV, Cat B.

    Returns:
        dict with keys "zeta" (level string), "cap" (level string), and
        "cap_value" (float, the computed CAP = wn^2 / n_alpha in 1/s^2).
        Level strings are "Level 1", "Level 2", "Level 3", or "Worse than Level 3".
    """
    # zeta_sp band (Cat B): L1 [0.30, 2.00], L2 [0.20, 2.00], L3 [0.15, ...]
    if 0.30 <= zeta_sp <= 2.00:
        zeta_level = "Level 1"
    elif 0.20 <= zeta_sp <= 2.00:
        zeta_level = "Level 2"
    elif zeta_sp >= 0.15:
        zeta_level = "Level 3"
    else:
        zeta_level = "Worse than Level 3"

    # CAP band (Cat B): L1 [0.085, 3.6], L2 [0.038, 10.0], L3 [0.038, ...]
    cap = wn_sp**2 / n_alpha
    if 0.085 <= cap <= 3.6:
        cap_level = "Level 1"
    elif 0.038 <= cap <= 10.0:
        cap_level = "Level 2"
    elif cap >= 0.038:
        cap_level = "Level 3"
    else:
        cap_level = "Worse than Level 3"

    return {"zeta": zeta_level, "cap": cap_level, "cap_value": cap}


def classify_phugoid(zeta_ph: float, t_double_ph: float | None) -> str:
    """Phugoid: L1 zeta >= 0.04; L2 zeta >= 0; L3 t_double >= 55 s."""
    if zeta_ph >= 0.04:
        return "Level 1"
    if zeta_ph >= 0.0:
        return "Level 2"
    if t_double_ph is not None and t_double_ph >= 55.0:
        return "Level 3"
    return "Worse than Level 3"


def classify_dutch_roll(zeta_dr: float, wn_dr: float) -> str:
    """Dutch roll: L1 zeta>=0.08, zeta*wn>=0.15, wn>=0.4 (rad/s)."""
    if zeta_dr >= 0.08 and zeta_dr * wn_dr >= 0.15 and wn_dr >= 0.4:
        return "Level 1"
    if zeta_dr >= 0.02 and zeta_dr * wn_dr >= 0.05 and wn_dr >= 0.4:
        return "Level 2"
    if zeta_dr >= 0.0 and wn_dr >= 0.4:
        return "Level 3"
    return "Worse than Level 3"


def classify_roll(tau_r: float) -> str:
    """Roll subsidence: L1 tau<=1.4 s; L2 tau<=3.0 s; L3 tau<=10.0 s."""
    if tau_r <= 1.4:
        return "Level 1"
    if tau_r <= 3.0:
        return "Level 2"
    if tau_r <= 10.0:
        return "Level 3"
    return "Worse than Level 3"


def classify_spiral(t_double_spiral: float | None) -> str:
    """Spiral: L1 t_double>=20 s (Cat B); L2 >=12 s; L3 >=4 s. Stable = L1."""
    if t_double_spiral is None:
        return "Level 1"  # spiral is stable (real root <= 0)
    if t_double_spiral >= 20.0:
        return "Level 1"
    if t_double_spiral >= 12.0:
        return "Level 2"
    if t_double_spiral >= 4.0:
        return "Level 3"
    return "Worse than Level 3"


def _print_flying_qualities(long_modes: dict[str, dict],
                            lat_modes: dict[str, dict],
                            n_alpha: float) -> None:
    """Run all five classifiers and print a single consolidated table."""
    sp = long_modes["short_period"]
    ph = long_modes["phugoid"]
    dr = lat_modes["dutch_roll"]
    rl = lat_modes["roll"]
    sp_res = classify_short_period(sp["zeta"], sp["wn"], n_alpha)

    print()
    print("-" * 78)
    print(" Flying qualities (MIL-F-8785C, Class IV, Category B - cruise)")
    print("-" * 78)
    print(f" {'Mode':<18} {'Metric':<22} {'Value':>10}  {'Level':<20}")

    print(f" {'Short period':<18} {'zeta':<22} {sp['zeta']:>10.4g}  {sp_res['zeta']:<20}")
    print(f" {'Short period':<18} {'CAP (1/s^2)':<22} {sp_res['cap_value']:>10.4g}  {sp_res['cap']:<20}")
    print(f" {'Phugoid':<18} {'zeta':<22} {ph['zeta']:>10.4g}  "
          f"{classify_phugoid(ph['zeta'], ph['t_double']):<20}")
    print(f" {'Dutch roll':<18} {'zeta':<22} {dr['zeta']:>10.4g}  "
          f"{classify_dutch_roll(dr['zeta'], dr['wn']):<20}")
    print(f" {'Dutch roll':<18} {'zeta * wn (1/s)':<22} {dr['zeta']*dr['wn']:>10.4g}  --")
    print(f" {'Dutch roll':<18} {'wn (rad/s)':<22} {dr['wn']:>10.4g}  --")
    print(f" {'Roll subsidence':<18} {'tau (s)':<22} {rl['tau']:>10.4g}  "
          f"{classify_roll(rl['tau']):<20}")
    sp_t2 = lat_modes["spiral"]["t_double"]
    spiral_str = f"{sp_t2:>10.4g}" if sp_t2 is not None else f"{'stable':>10}"
    print(f" {'Spiral':<18} {'t_double (s)':<22} {spiral_str}  "
          f"{classify_spiral(sp_t2):<20}")


# =============================================================================
# Main pipeline
# =============================================================================
def main() -> None:
    """Run the full A-7A stability analysis pipeline."""
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size":   11,
    })
    print("=" * 78)
    print(" A-7A Corsair II Stability Analysis -- Cruise (15,000 ft, M=0.6)")
    print("=" * 78)
    print(f" U1   = {U1:>9.3f} ft/s")
    print(f" qbar = {QBAR:>9.3f} lbf/ft^2")
    print(f" m    = {M_MASS:>9.3f} slug")
    print(f" rho  = {RHO:>9.4e} slug/ft^3")
    print(f" theta1 = {np.rad2deg(THETA1):>7.3f} deg")

    long_d = compute_longitudinal_derivatives()
    _print_derivative_table("Longitudinal dimensional derivatives",
                            long_d, _LONG_UNITS)

    lat_d = compute_lateral_derivatives()
    _print_derivative_table("Lateral-directional dimensional derivatives",
                            lat_d, _LAT_UNITS)

    A_long, B_long = build_longitudinal_ss(long_d)
    _print_matrix("A_long", A_long,
                  ["du_dot", "dw_dot", "dq_dot", "dtheta_dot"],
                  ["du", "dw", "dq", "dtheta"])
    _print_matrix("B_long", B_long,
                  ["du_dot", "dw_dot", "dq_dot", "dtheta_dot"],
                  ["d_de", "d_dT"])

    A_lat, B_lat = build_lateral_ss(lat_d)
    _print_matrix("A_lat", A_lat,
                  ["dbeta_dot", "dp_dot", "dr_dot", "dphi_dot"],
                  ["dbeta", "dp", "dr", "dphi"])
    _print_matrix("B_lat", B_lat,
                  ["dbeta_dot", "dp_dot", "dr_dot", "dphi_dot"],
                  ["d_da", "d_dr"])

    long_modes = identify_longitudinal_modes(modal_analysis(A_long))
    lat_modes  = identify_lateral_modes(modal_analysis(A_lat))

    _print_modal_table("Longitudinal modes", list(long_modes.values()),
                       list(long_modes.keys()))
    _print_modal_table("Lateral-directional modes", list(lat_modes.values()),
                       list(lat_modes.keys()))

    _print_eigenvector_table("Longitudinal eigenvectors", long_modes,
                             ["du", "dw", "dq", "dtheta"])
    _print_eigenvector_table("Lateral eigenvectors", lat_modes,
                             ["dbeta", "dp", "dr", "dphi"])

    n_alpha = -long_d["Za"] / G   # load-factor sensitivity to alpha (1/rad -> g/rad)
    _print_flying_qualities(long_modes, lat_modes, n_alpha)

    fig_paths  = plot_long_impulse(A_long, B_long, FIG_DIR)
    fig_paths += plot_lat_impulse(A_lat, B_lat, FIG_DIR)
    print()
    print("-" * 78)
    print(" Saved figures")
    print("-" * 78)
    for p in fig_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
