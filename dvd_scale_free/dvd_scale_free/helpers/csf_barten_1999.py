import numpy as np

# ============================================================
# Barten (1999) adult CSF (implemented in NumPy for the 1D sf grid)
# ============================================================
def _as_float_array(x):
    return np.asarray(x, dtype=float)

def optical_MTF_Barten1999(u, sigma=0.5):
    u = _as_float_array(u)
    sigma = _as_float_array(sigma)
    return np.exp(-2.0 * np.pi**2 * sigma**2 * u**2)

def sigma_Barten1999(sigma_0=0.5/60, C_ab=0.08/60, d=3.0):
    sigma_0 = _as_float_array(sigma_0)
    C_ab = _as_float_array(C_ab)
    d = _as_float_array(d)
    return np.sqrt(sigma_0**2 + (C_ab * d)**2)

def retinal_illuminance_Barten1999(L=20, d=3.0, stiles_crawford_correction=True):
    d = _as_float_array(d)
    L = _as_float_array(L)
    E = (np.pi * d**2) / 4.0 * L
    if stiles_crawford_correction:
        E *= (1.0 - (d / 9.7)**2 + (d / 12.4)**4)
    return E

def function_contrast_sensitivity_Barten1999(
    u,
    sigma=sigma_Barten1999(0.5/60, 0.08/60, 3.0),
    k=3.0,
    T=0.1,
    X_0=60,
    X_max=12,
    N_max=15,
    n=0.03,
    p=1.2e6,
    E=retinal_illuminance_Barten1999(20, 3.0),
    phi_0=3e-8,
    u_0=7.0
):
    """
    Adult spatial CSF S_adult(u) per Barten (1999).

    We clip u away from 0 to avoid the low-frequency term singularity:
      phi_0 / (1 - exp(-(u/u0)^2))
    """
    u = _as_float_array(u)
    u = np.clip(u, 0.5, 32.0)

    M_opt = optical_MTF_Barten1999(u, sigma)
    denom_1 = 2.0 / T * (1.0 / X_0**2 + 1.0 / X_max**2 + u**2 / N_max**2)
    denom_2 = 1.0 / (n * p * E) + phi_0 / (1.0 - np.exp(-(u / u_0)**2))
    return (M_opt / k) / np.sqrt(denom_1 * denom_2)

def csf_relative_gain(u, u_min=0.5, u_max=32.0, ngrid=400) -> np.ndarray:
    """
    Return normalized gain G(u)=S(u)/max(S) using a fixed reference grid for the peak.
    This avoids the peak depending on the image frequency sampling.
    """
    u = np.asarray(u, dtype=float)
    grid = np.logspace(np.log10(u_min), np.log10(u_max), ngrid)
    peak = float(np.max(function_contrast_sensitivity_Barten1999(grid)))
    S = function_contrast_sensitivity_Barten1999(np.clip(u, u_min, u_max))
    return (S / (peak + 1e-30)).astype(np.float32)
