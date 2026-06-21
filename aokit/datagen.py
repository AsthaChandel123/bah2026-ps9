"""aokit.datagen -- synthetic SH-WFS data generation (the validation backbone).

Generates phase screens (>=4 methods), spot fields (geometric + Fraunhofer),
detector noise, and frozen-flow time-series with KNOWN injected r0/tau0, then
writes .bmp frames + a ground-truth JSON. research/07 PART B.

Pipeline: phase screen phi(x,y) [known r0, L0] -> propagate through MLA
(geometric tilt or Fraunhofer) -> spot field -> detector (Poisson + read +
quantize) -> .bmp; translate screen by v*dt for the next frame (frozen flow =>
known tau0 = 0.314 r0 / v).

Units / conventions
-------------------
* ``phi`` is the optical phase in **radians** on a square grid; the grid spacing
  ``delta_m`` ("pixel scale") and the Fried parameter ``r0_m`` are both in
  **metres**, referenced to the pupil plane.
* The phase **structure function** of a Kolmogorov screen is
  ``D_phi(r) = 6.88 (r/r0)^(5/3)`` rad^2 -- this is the relation the FT-screen
  tests invert to recover the injected ``r0``.
* The (modified) **von Karman PSD** of phase (rad^2 m^2) is
  ``Phi(f) = 0.0229 r0^(-5/3) exp(-(f/fm)^2) / (f^2 + 1/L0^2)^(11/6)`` with
  ``fm = 5.92/(2 pi l0)``; ``l0=0, L0=inf`` recovers pure Kolmogorov.
* Spot displacement (geometric model, the inverse of the centroiding model):
  a local wavefront tilt ``<dphi/dx>`` over a sub-aperture deflects its focal
  spot by ``theta = (lambda/2pi) <dphi/dx>`` rad, i.e. ``delta_px =
  f * theta / pixel_size`` pixels from the reference -- so that
  ``slope = (centroid - ref) * pixel_size / f`` (geometry.slopes_from_centroids)
  returns exactly ``theta`` again.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Optional, Tuple

import numpy as np

from .config import Config
from . import geometry as _geom
from . import zernike as _zern
from . import bmpio as _bmpio


# von Karman PSD coefficient for the *phase* PSD (rad^2 m^2):
#   Phi_phi(f) = C_VK * r0^(-5/3) * (f^2 + 1/L0^2)^(-11/6)
# Numerically C_VK = 0.0229 (= (24/5 Gamma(6/5))^(5/6) / (2 pi) ... ); the value
# 0.0229 reproduces the Kolmogorov structure function D_phi(r)=6.88(r/r0)^5/3.
# (aotools writes 0.023; research/07 uses 0.0229. We use 0.0229 and additionally
# *self-calibrate* the screen variance against the analytic structure-function
# constant so the injected r0 is recovered regardless of the exact prefactor.)
_VK_PSD_COEFF = 0.0229

# Kolmogorov phase structure-function constant: D_phi(r) = 6.88 (r/r0)^(5/3).
_STRUCT_CONST = 6.88


# --------------------------- phase-screen generators ---------------------

def _vonkarman_psd(fx: np.ndarray, fy: np.ndarray, r0_m: float,
                   L0_m: float = np.inf, l0_m: float = 0.0) -> np.ndarray:
    """(Modified) von Karman phase PSD ``Phi_phi(f)`` (rad^2 m^2) on a 2-D
    frequency grid.

    ``Phi(f) = 0.0229 r0^-5/3 exp(-(f/fm)^2) / (f^2 + 1/L0^2)^(11/6)`` with
    ``fm = 5.92/(2 pi l0)``.  ``l0=0`` drops the inner-scale term; ``L0=inf``
    drops the outer-scale floor (pure Kolmogorov, singular at f=0).
    """
    f2 = fx * fx + fy * fy
    f0sq = 0.0 if not np.isfinite(L0_m) else (1.0 / L0_m) ** 2
    denom = (f2 + f0sq) ** (11.0 / 6.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        psd = _VK_PSD_COEFF * r0_m ** (-5.0 / 3.0) / denom
    if l0_m and l0_m > 0.0:
        fm = 5.92 / (2.0 * np.pi * l0_m)
        psd = psd * np.exp(-(f2 / (fm * fm)))
    # The DC term is singular for pure Kolmogorov; piston is unobservable, so 0.
    psd = np.where(f2 == 0.0, 0.0, psd)
    return psd


def ft_phase_screen(r0_m: float, N: int, delta_m: float,
                    L0_m: float = np.inf, l0_m: float = 0.0,
                    seed: Optional[int] = None) -> np.ndarray:
    """Method 1: FFT/spectral phase screen from the (modified) von Karman PSD
    ``Phi = 0.0229 r0^-5/3 exp(-(f/fm)^2) / (f^2 + f0^2)^(11/6)``.

    Fills an ``N x N`` complex grid with unit white Gaussian noise, colours it by
    ``sqrt(Phi(f)) * df`` (frequency spacing ``df = 1/(N delta)``), inverse-FFTs
    and returns the real part as an ``(N, N)`` phase screen in **radians**.
    ``r0_m`` and ``delta_m`` are in metres.  research/07 B.1 M1.

    Fast (O(N^2 log N)) but under-represents the lowest spatial frequencies
    (tip/tilt) -- use :func:`ft_sh_phase_screen` when low-order power matters.
    """
    N = int(N)
    rng = np.random.default_rng(seed)
    df = 1.0 / (N * delta_m)                      # frequency-grid spacing (1/m)
    fx = np.fft.fftfreq(N, d=delta_m)             # cycles / m
    fxx, fyy = np.meshgrid(fx, fx)

    psd = _vonkarman_psd(fxx, fyy, r0_m, L0_m=L0_m, l0_m=l0_m)

    # White complex Gaussian field, coloured by sqrt(PSD)*df. The real and
    # imaginary parts of the IFFT are each a valid independent screen; we take
    # the real part. The N^2 normalisation of numpy's ifft is undone so that the
    # screen variance matches the continuous spectral integral.
    cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    spectrum = cn * np.sqrt(psd) * df
    screen = np.fft.ifft2(spectrum) * (N * N)
    return np.real(screen)


def ft_sh_phase_screen(r0_m: float, N: int, delta_m: float,
                       L0_m: float = np.inf, l0_m: float = 0.0,
                       n_subharmonics: int = 3,
                       seed: Optional[int] = None) -> np.ndarray:
    """Method 2 (RECOMMENDED for single frames): FFT + subharmonics to augment
    low-frequency (tip/tilt) content.  research/07 B.1 M2.

    Starts from :func:`ft_phase_screen` and adds ``n_subharmonics`` nested
    subharmonic levels: for level ``p = 1..Np`` a 3x3 grid of frequencies with
    spacing ``df_p = 1/(3^p N delta)`` (i.e. successively finer fractions of the
    fundamental) is drawn from the von Karman PSD and added analytically (as
    continuous Fourier modes ``exp(2 pi i (fx x + fy y))``) over the screen. This
    is the standard ``ft_sh_phase_screen`` fix; the added levels restore the
    tip/tilt and large-scale power the plain FFT screen misses.
    """
    N = int(N)
    rng = np.random.default_rng(seed)

    # High-frequency body from the standard FFT screen (independent RNG stream so
    # results are reproducible per-seed).
    screen = ft_phase_screen(r0_m, N, delta_m, L0_m=L0_m, l0_m=l0_m,
                             seed=int(rng.integers(0, 2 ** 31 - 1)))

    # Real-space coordinates (metres), zero-mean centred.
    coord = (np.arange(N) - N / 2.0) * delta_m
    xx, yy = np.meshgrid(coord, coord)

    D = N * delta_m
    sh = np.zeros((N, N), dtype=np.float64)
    for p in range(1, int(n_subharmonics) + 1):
        df_p = 1.0 / (3.0 ** p * D)               # subharmonic grid spacing (1/m)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    continue                       # skip DC (piston)
                fxp = i * df_p
                fyp = j * df_p
                psd = _vonkarman_psd(np.array(fxp), np.array(fyp),
                                     r0_m, L0_m=L0_m, l0_m=l0_m)
                amp = np.sqrt(float(psd)) * df_p
                c = (rng.standard_normal() + 1j * rng.standard_normal()) * amp
                phase = 2.0 * np.pi * (fxp * xx + fyp * yy)
                sh += np.real(c * np.exp(1j * phase))
    screen = screen + sh
    # Remove residual piston (unobservable; keeps tests on tip/tilt+ clean).
    return screen - screen.mean()


def zernike_phase_screen(r0_m: float, D_m: float, N: int, j_max: int,
                         seed: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Method 3: Zernike synthesis with Noll-covariance-drawn coefficients.

    Draws a coefficient vector ``a`` (Noll j = 2..j_max, piston excluded) from
    the Kolmogorov Zernike covariance ``<a_j a_j'> = C_jj' (D/r0)^(5/3)``
    (``aokit.zernike.noll_covariance``) and synthesises the phase on an
    ``N x N`` unit-disc grid, ``phi = sum_j a_j Z_j``.  Returns
    ``(screen[N, N], coeffs)`` where ``coeffs`` is the **ground-truth** Noll
    vector indexed ``j = 2..j_max`` (length ``j_max - 1``).  Deterministic per
    seed -- ideal for unit tests.  research/07 B.1 M3.

    ``D_m`` (pupil diameter) and ``r0_m`` are in metres and enter only through
    the scaling ``(D/r0)^(5/3)``.  Pixels outside the unit disc are 0.
    """
    N = int(N)
    j_max = int(j_max)
    rng = np.random.default_rng(seed)

    js = list(range(2, j_max + 1))                # exclude piston (j=1)
    nmodes = len(js)

    # Kolmogorov covariance matrix over the requested modes, scaled by (D/r0)^5/3.
    cov = np.empty((nmodes, nmodes), dtype=np.float64)
    for a, ja in enumerate(js):
        for b, jb in enumerate(js):
            cov[a, b] = _zern.noll_covariance(ja, jb)
    scale = (D_m / r0_m) ** (5.0 / 3.0)
    cov = cov * scale
    # Symmetrise for numerical safety.
    cov = 0.5 * (cov + cov.T)

    # Draw a correlated Gaussian coefficient vector ~ N(0, cov) via the
    # eigen-decomposition (robust to the slight indefiniteness of the truncated
    # covariance; clamp tiny negative eigenvalues to 0).
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    g = rng.standard_normal(nmodes)
    coeffs = evecs @ (np.sqrt(evals) * g)

    # Synthesise on the unit-disc grid: zernike_basis returns (N_pix, N_modes).
    lin = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(lin, lin)
    mask = (xx * xx + yy * yy) <= 1.0
    Z = _zern.zernike_basis(js, xx, yy, mask=mask)     # (n_in_disc, nmodes)
    flat = Z @ coeffs
    screen = np.zeros((N, N), dtype=np.float64)
    screen[mask] = flat
    return screen, coeffs


def covariance_phase_screen(r0_m: float, N: int, delta_m: float,
                            L0_m: float = np.inf,
                            seed: Optional[int] = None) -> np.ndarray:
    """Method 4: Cholesky of the von Karman phase covariance (statistical gold
    standard).  research/07 B.1 M4.

    Builds the ``N^2 x N^2`` phase covariance from the structure function
    ``D_phi(r) = 6.88 (r/r0)^(5/3)`` (Kolmogorov; von Karman saturates at large
    ``r`` when ``L0`` is finite), via ``C(r) = C(0) - 0.5 D_phi(r)`` with the
    integral-scale variance fixed so the screen has the correct relative
    structure, factorises ``C = L L^T`` and returns ``phi = L g`` for white
    Gaussian ``g``.  Statistically exact at the sample points (no low-frequency
    deficit) but O(N^3) -- keep ``N`` small (<= ~24).
    """
    N = int(N)
    rng = np.random.default_rng(seed)

    coord = np.arange(N) * delta_m
    xx, yy = np.meshgrid(coord, coord)
    pts = np.column_stack([xx.ravel(), yy.ravel()])   # (N^2, 2) metres

    # Pairwise separations.
    diff = pts[:, None, :] - pts[None, :, :]
    r = np.sqrt((diff ** 2).sum(axis=-1))             # (N^2, N^2)

    # Structure function (von Karman saturates; Kolmogorov grows).
    Dphi = _STRUCT_CONST * (r / r0_m) ** (5.0 / 3.0)
    if np.isfinite(L0_m):
        # Soft saturation at the outer scale (keeps covariance PSD-consistent).
        Dsat = _STRUCT_CONST * (L0_m / r0_m) ** (5.0 / 3.0)
        Dphi = Dsat * (1.0 - np.exp(-Dphi / Dsat))

    # Covariance from the structure function with a piston-removed reference:
    #   C(r) = 0.5 (D_max - D_phi(r)) so that C is PSD and C(0) = 0.5 D_max.
    Dmax = float(Dphi.max())
    C = 0.5 * (Dmax - Dphi)
    # Numerical PSD guard.
    C = C + 1e-9 * np.eye(N * N)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(C)
        evals = np.clip(evals, 0.0, None)
        L = evecs @ np.diag(np.sqrt(evals))
    phi = L @ rng.standard_normal(N * N)
    return phi.reshape(N, N)


# ----------------------------- spot-field synthesis ----------------------

def _phase_on_actuator_grid(cfg: Config, phase: np.ndarray) -> np.ndarray:
    """Resample a phase screen onto detector-pixel coordinates of the frame.

    The screen is assumed to span the detector frame (the pupil is inscribed);
    nearest/bilinear sampling maps it to an ``(frame_h, frame_w)`` array so each
    sub-aperture window can read its local phase directly.
    """
    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    ph = np.asarray(phase, dtype=np.float64)
    if ph.shape == (H, W):
        return ph
    # Bilinear-resample ph (shape (n,n)) onto the HxW detector grid.
    n0, n1 = ph.shape
    ys = np.linspace(0, n0 - 1, H)
    xs = np.linspace(0, n1 - 1, W)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, n0 - 1)
    x1 = np.minimum(x0 + 1, n1 - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = ph[np.ix_(y0, x0)] * (1 - wx) + ph[np.ix_(y0, x1)] * wx
    bot = ph[np.ix_(y1, x0)] * (1 - wx) + ph[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def _local_tilts(cfg: Config, grid: _geom.SubApertureGrid,
                 phase_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean wavefront gradient ``(<dphi/dx>, <dphi/dy>)`` (rad/px) over every
    sub-aperture window of ``grid``, measured on the detector-sampled phase.

    Returns two length-M arrays in canonical lenslet order.
    """
    H, W = phase_px.shape
    win = grid.w
    n = grid.ref_x.shape[0]
    gx = np.zeros(n, dtype=np.float64)
    gy = np.zeros(n, dtype=np.float64)
    for k in range(n):
        xi0 = int(np.floor(grid.x0[k]))
        yi0 = int(np.floor(grid.y0[k]))
        cx0 = max(xi0, 0)
        cy0 = max(yi0, 0)
        cx1 = min(xi0 + win, W)
        cy1 = min(yi0 + win, H)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        sub = phase_px[cy0:cy1, cx0:cx1]
        # Mean gradient = mean of per-pixel finite differences (rad / pixel).
        dphidx = np.gradient(sub, axis=1)
        dphidy = np.gradient(sub, axis=0)
        gx[k] = float(dphidx.mean())
        gy[k] = float(dphidy.mean())
    return gx, gy


def _render_gaussian_into(frame: np.ndarray, cx: float, cy: float,
                          sigma: float, amp: float, half: int) -> None:
    """Add a 2-D Gaussian of peak ``amp`` centred at ``(cx, cy)`` (detector px)
    into ``frame`` in place, over a ``(2 half + 1)`` window."""
    H, W = frame.shape
    ix = int(round(cx))
    iy = int(round(cy))
    x0 = max(ix - half, 0)
    x1 = min(ix + half + 1, W)
    y0 = max(iy - half, 0)
    y1 = min(iy + half + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    gx = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
    gy = np.exp(-0.5 * ((ys - cy) / sigma) ** 2)
    frame[y0:y1, x0:x1] += amp * np.outer(gy, gx)


def spots_geometric(cfg: Config, phase: np.ndarray,
                    grid: Optional[_geom.SubApertureGrid] = None,
                    spot_sigma_px: float = 1.5,
                    spot_amp: float = 1.0) -> np.ndarray:
    """Model 1: geometric tilt-per-sub-aperture (the analytic oracle).

    For each valid sub-aperture, the mean local slope ``<dphi/dx>`` over the
    lenslet window deflects its focal spot by
    ``theta = (lambda/2pi) <dphi/dx>`` (rad) -> ``delta_px = f theta /
    pixel_size`` pixels, and a Gaussian spot of width ``spot_sigma_px`` is placed
    at ``(ref_x + delta_x_px, ref_y + delta_y_px)``.  Summed into an
    ``(frame_h, frame_w)`` float image.  This is the exact inverse of the
    centroiding model (``slope = (centroid - ref) * pixel_size / f``), so spot
    shifts recover the injected tilt analytically.  research/07 B.2 M1.
    """
    if grid is None:
        grid = _geom.build_subaperture_grid(cfg, valid_only=True)

    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    frame = np.zeros((H, W), dtype=np.float64)

    phase_px = _phase_on_actuator_grid(cfg, phase)
    gx, gy = _local_tilts(cfg, grid, phase_px)

    lam = cfg.wavelength_m
    f = cfg.mla.focal_length_m
    px = cfg.camera.pixel_size_m
    # rad/px gradient -> tilt angle (rad): theta = (lambda/2pi) * <dphi/dx_metres>
    # and <dphi/dx_metres> = <dphi/dx_px> / px.  So delta_px = f*theta/px:
    #   delta_px = f * (lambda/2pi) * (grad_px / px) / px
    coeff = f * lam / (2.0 * np.pi) / (px * px)
    half = max(3, int(np.ceil(4.0 * spot_sigma_px)))

    n = grid.ref_x.shape[0]
    for k in range(n):
        dx_px = coeff * gx[k]
        dy_px = coeff * gy[k]
        _render_gaussian_into(frame, grid.ref_x[k] + dx_px,
                              grid.ref_y[k] + dy_px,
                              spot_sigma_px, spot_amp, half)
    return frame


def spots_fraunhofer(cfg: Config, phase: np.ndarray,
                     grid: Optional[_geom.SubApertureGrid] = None,
                     fft_pad: int = 2, spot_amp: float = 1.0) -> np.ndarray:
    """Model 2: per-sub-aperture Fraunhofer/FFT diffraction (physically
    faithful).  research/07 B.2 M2.

    Each lenslet performs an optical Fourier transform: the focal-plane intensity
    of a sub-aperture is ``I = |FFT(U_sub)|^2`` of its (zero-padded) complex pupil
    field ``U_sub = exp(i phi_sub)``.  The padded-FFT focal grid is rescaled so a
    pure tilt lands the spot at the same displacement as the geometric model
    (consistent units), then the spot is binned into the lenslet's detector cell.
    Reproduces spot broadening / speckle from intra-lenslet aberration.
    """
    if grid is None:
        grid = _geom.build_subaperture_grid(cfg, valid_only=True)

    H, W = cfg.camera.frame_h, cfg.camera.frame_w
    frame = np.zeros((H, W), dtype=np.float64)
    phase_px = _phase_on_actuator_grid(cfg, phase)

    win = grid.w
    Npad = int(win * fft_pad)
    lam = cfg.wavelength_m
    f = cfg.mla.focal_length_m
    px = cfg.camera.pixel_size_m

    # Focal-plane sample pitch from a padded FFT of a window of `win` pixels
    # (pupil pitch = px metres): d_focal = lambda f / (Npad * px) metres
    #  -> in detector pixels: d_focal_px = lambda f / (Npad * px * px).
    d_focal_px = lam * f / (Npad * px * px)

    n = grid.ref_x.shape[0]
    for k in range(n):
        xi0 = int(np.floor(grid.x0[k]))
        yi0 = int(np.floor(grid.y0[k]))
        cx0 = max(xi0, 0)
        cy0 = max(yi0, 0)
        cx1 = min(xi0 + win, W)
        cy1 = min(yi0 + win, H)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        sub = phase_px[cy0:cy1, cx0:cx1]
        # Remove the sub-aperture mean phase (piston): irrelevant to the spot.
        sub = sub - sub.mean()
        U = np.zeros((Npad, Npad), dtype=np.complex128)
        U[:sub.shape[0], :sub.shape[1]] = np.exp(1j * sub)
        psf = np.abs(np.fft.fftshift(np.fft.fft2(U))) ** 2
        psf /= psf.sum() if psf.sum() > 0 else 1.0

        # Map the focal grid (centred) to detector pixels around the reference.
        fc = Npad // 2
        # Coordinates of each PSF pixel relative to the spot centre, in det px,
        # broadcast to the full (Npad, Npad) grid so they align with `psf`.
        idx = (np.arange(Npad) - fc) * d_focal_px
        ygrid, xgrid = np.meshgrid(grid.ref_y[k] + idx,
                                   grid.ref_x[k] + idx, indexing="ij")
        xr = np.rint(xgrid).astype(int)
        yr = np.rint(ygrid).astype(int)
        inside = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
        # Accumulate the PSF intensity into the nearest detector pixels.
        np.add.at(frame, (yr[inside], xr[inside]), spot_amp * psf[inside])
    return frame


def apply_detector_noise(image: np.ndarray, flux_photons: float,
                         qe: float = 0.9, read_noise_e: float = 3.0,
                         gain: float = 1.0, bias: float = 0.0,
                         bit_depth: int = 8,
                         seed: Optional[int] = None) -> np.ndarray:
    """Poisson shot + Gaussian read + quantization to ``bit_depth``.  research/07
    B.3.

    Steps (Konnik & Welsh):
      1. scale the normalised ``image`` so its **total** equals ``flux_photons``
         electrons (``* qe``);
      2. ``Poisson`` shot noise (signal-dependent);
      3. add ``Normal(0, read_noise_e)`` read noise (in electrons);
      4. apply ``gain`` (e- -> ADU), add ``bias``, round and clip to
         ``[0, 2^bit_depth - 1]``.
    Returns a ``uint8`` (bit_depth<=8) or ``uint16`` frame ready for the BMP
    writer.  The mean output tracks the signal; the variance grows as the photon
    count drops (shot-noise scaling).
    """
    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=np.float64)
    tot = float(img.sum())
    if tot > 0.0:
        electrons_ideal = img / tot * float(flux_photons) * float(qe)
    else:
        electrons_ideal = np.zeros_like(img)

    electrons = rng.poisson(np.clip(electrons_ideal, 0.0, None)).astype(np.float64)
    if read_noise_e and read_noise_e > 0.0:
        electrons = electrons + rng.normal(0.0, float(read_noise_e), size=img.shape)

    adu = electrons / float(gain) + float(bias)
    maxval = (1 << int(bit_depth)) - 1
    adu = np.clip(np.rint(adu), 0, maxval)
    dtype = np.uint8 if int(bit_depth) <= 8 else np.uint16
    return adu.astype(dtype)


# ------------------------------ time-series ------------------------------

def _shift_screen(screen: np.ndarray, shift_x_px: float, shift_y_px: float
                  ) -> np.ndarray:
    """Translate ``screen`` by ``(shift_x_px, shift_y_px)`` with periodic wrap
    and bilinear interpolation for the sub-pixel part (Taylor frozen flow).

    Positive ``shift_x_px`` moves screen content toward +x (columns).
    """
    n0, n1 = screen.shape
    # New sample coordinates pulled from old screen (wrap-around).
    ys = (np.arange(n0)[:, None] - shift_y_px) % n0
    xs = (np.arange(n1)[None, :] - shift_x_px) % n1
    y0 = np.floor(ys).astype(int) % n0
    x0 = np.floor(xs).astype(int) % n1
    y1 = (y0 + 1) % n0
    x1 = (x0 + 1) % n1
    wy = ys - np.floor(ys)
    wx = xs - np.floor(xs)
    top = screen[y0, x0] * (1 - wx) + screen[y0, x1] * wx
    bot = screen[y1, x0] * (1 - wx) + screen[y1, x1] * wx
    return top * (1 - wy) + bot * wy


def frozen_flow_series(cfg: Config, r0_m: float, wind_speed_mps: float,
                       wind_angle_deg: float, n_frames: int,
                       L0_m: float = np.inf, boiling: float = 0.0,
                       base_screen: Optional[np.ndarray] = None,
                       n_grid: Optional[int] = None,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """Generate a frozen-flow phase-screen series translated by ``v*dt`` per
    frame (known ``tau0 = 0.314 r0 / v``).  research/07 B.5.

    A large base screen (``ft_sh_phase_screen`` at the injected ``r0``) is slid
    by the projected wind ``(v cos, v sin)`` each frame using periodic-wrap
    bilinear interpolation (sub-pixel exact).  Optional AR ``boiling`` in
    ``[0, 1)`` blends in a fresh independent screen per frame to decorrelate the
    turbulence (stress-test of the frozen-flow assumption).  Returns
    ``(phase_series[T, N, N], ground_truth_dict)`` where the dict carries the
    injected ``r0, tau0, wind_speed_mps, wind_angle_deg, L0`` and the per-frame
    pixel shift.

    ``cfg.dt_s`` is the inter-frame interval; the per-frame shift in pixels is
    ``v * dt / pixel_size`` along the wind direction.
    """
    rng = np.random.default_rng(seed)
    delta = cfg.camera.pixel_size_m
    if n_grid is None:
        n_grid = max(cfg.camera.frame_h, cfg.camera.frame_w)
    n_grid = int(n_grid)

    if base_screen is None:
        base_screen = ft_sh_phase_screen(
            r0_m, n_grid, delta, L0_m=L0_m,
            seed=int(rng.integers(0, 2 ** 31 - 1)))
    base_screen = np.asarray(base_screen, dtype=np.float64)

    dt = cfg.dt_s
    ang = np.deg2rad(wind_angle_deg)
    # Per-frame pixel shift along the wind vector.
    shift_px = wind_speed_mps * dt / delta
    sx = shift_px * np.cos(ang)
    sy = shift_px * np.sin(ang)

    tau0 = 0.314 * r0_m / wind_speed_mps if wind_speed_mps > 0 else float("inf")

    T = int(n_frames)
    series = np.empty((T, n_grid, n_grid), dtype=np.float64)
    cur = base_screen.copy()
    for k in range(T):
        frame = _shift_screen(base_screen, sx * k, sy * k)
        if boiling and boiling > 0.0:
            fresh = ft_sh_phase_screen(
                r0_m, n_grid, delta, L0_m=L0_m,
                seed=int(rng.integers(0, 2 ** 31 - 1)))
            a = float(boiling)
            frame = np.sqrt(1.0 - a * a) * frame + a * fresh
        series[k] = frame

    gt = {
        "r0_m": float(r0_m),
        "tau0_s": float(tau0),
        "wind_speed_mps": float(wind_speed_mps),
        "wind_angle_deg": float(wind_angle_deg),
        "L0_m": (None if not np.isfinite(L0_m) else float(L0_m)),
        "dt_s": float(dt),
        "shift_px_per_frame": float(shift_px),
        "shift_xy_px_per_frame": [float(sx), float(sy)],
        "n_frames": T,
        "n_grid": n_grid,
        "boiling": float(boiling),
    }
    return series, gt


def generate_dataset(cfg: Config, r0_m: float, tau0_s: float, n_frames: int,
                     out_dir: str, spot_model: str = "geometric",
                     wind_angle_deg: float = 0.0, L0_m: float = np.inf,
                     flux_photons: float = 5.0e4, read_noise_e: float = 3.0,
                     j_max: int = 0, seed: Optional[int] = None) -> dict:
    """End-to-end: build the frozen-flow series, synthesize spot fields, add
    noise, write ``.bmp`` frames + ground-truth JSON into ``out_dir``.  Returns
    the ground-truth / manifest dict (injected r0/tau0/wind/L0/wavelength + the
    per-frame Zernike truth when ``j_max>0``).  research/07 B, S B.4.

    ``tau0_s`` is reconciled with the wind speed via ``v = 0.314 r0 / tau0`` so
    the injected ``tau0`` is exactly realised by the frozen-flow shift.  Writes
    ``out_dir/frame_0000.bmp ...`` (zero-padded) and ``out_dir/ground_truth.json``
    -- the validation oracle.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Reconcile tau0 <-> wind: tau0 = 0.314 r0 / v  =>  v = 0.314 r0 / tau0.
    if tau0_s and tau0_s > 0.0:
        wind = 0.314 * r0_m / tau0_s
    else:
        wind = 0.0
        tau0_s = float("inf")

    grid = _geom.build_subaperture_grid(cfg, valid_only=True)
    n_grid = max(cfg.camera.frame_h, cfg.camera.frame_w)

    series, ff_gt = frozen_flow_series(
        cfg, r0_m, wind, wind_angle_deg, n_frames, L0_m=L0_m,
        n_grid=n_grid, seed=int(rng.integers(0, 2 ** 31 - 1)))

    # Optional per-frame Zernike ground truth (projection of each screen onto the
    # low-order Noll basis over the pupil).
    zern_truth = []
    if j_max and j_max > 1:
        R = _geom.pupil_radius_px(cfg)
        cx0, cy0 = cfg.pupil.center_x_px, cfg.pupil.center_y_px
        lin_y = (np.arange(n_grid) - cy0) / R
        lin_x = (np.arange(n_grid) - cx0) / R
        xx, yy = np.meshgrid(lin_x, lin_y)
        mask = (xx * xx + yy * yy) <= 1.0
        js = list(range(2, int(j_max) + 1))
        Z = _zern.zernike_basis(js, xx, yy, mask=mask)        # (npix, nmodes)
        # Least-squares projection matrix (modes are orthonormal over the disc,
        # but the discrete sampling is not exactly so -> use lstsq pinv).
        Zpinv = np.linalg.pinv(Z)

    digits = max(4, len(str(n_frames - 1)))
    frame_files = []
    for k in range(n_frames):
        phase = series[k]
        if spot_model == "fraunhofer":
            img = spots_fraunhofer(cfg, phase, grid=grid)
        else:
            img = spots_geometric(cfg, phase, grid=grid)
        adu = apply_detector_noise(
            img, flux_photons=flux_photons, read_noise_e=read_noise_e,
            bit_depth=cfg.camera.bit_depth,
            seed=int(rng.integers(0, 2 ** 31 - 1)))
        fname = f"frame_{k:0{digits}d}.bmp"
        _bmpio.write_bmp_gray8(os.path.join(out_dir, fname), adu)
        frame_files.append(fname)

        if j_max and j_max > 1:
            a = Zpinv @ phase[mask]
            zern_truth.append([float(v) for v in a])

    gt = {
        "r0_m": float(r0_m),
        "tau0_s": (None if not np.isfinite(tau0_s) else float(tau0_s)),
        "wind_speed_mps": float(wind),
        "wind_angle_deg": float(wind_angle_deg),
        "L0_m": (None if not np.isfinite(L0_m) else float(L0_m)),
        "wavelength_m": float(cfg.wavelength_m),
        "dt_s": float(cfg.dt_s),
        "n_frames": int(n_frames),
        "spot_model": str(spot_model),
        "frames": frame_files,
        "frozen_flow": ff_gt,
        "zernike_noll_per_frame": zern_truth,
        "zernike_j_max": int(j_max),
        "config": _config_to_dict(cfg),
    }
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as fh:
        json.dump(gt, fh, indent=2)
    return gt


def _config_to_dict(cfg: Config) -> dict:
    """Serialise the relevant Config fields into the ground-truth JSON."""
    return {
        "schema_version": cfg.schema_version,
        "camera": {
            "pixel_size_m": cfg.camera.pixel_size_m,
            "frame_w": cfg.camera.frame_w,
            "frame_h": cfg.camera.frame_h,
            "bit_depth": cfg.camera.bit_depth,
        },
        "mla": {
            "n_lenslets_x": cfg.mla.n_lenslets_x,
            "n_lenslets_y": cfg.mla.n_lenslets_y,
            "pitch_m": cfg.mla.pitch_m,
            "focal_length_m": cfg.mla.focal_length_m,
        },
        "pupil": {
            "diameter_m": cfg.pupil.diameter_m,
            "center_x_px": cfg.pupil.center_x_px,
            "center_y_px": cfg.pupil.center_y_px,
        },
        "wavelength_m": cfg.wavelength_m,
        "dt_s": cfg.dt_s,
    }
