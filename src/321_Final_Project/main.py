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
# Main pipeline
# =============================================================================
def main() -> None:
    """Run the full A-7A stability analysis pipeline."""
    FIG_DIR.mkdir(exist_ok=True)
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


if __name__ == "__main__":
    main()
