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


if __name__ == "__main__":
    main()
