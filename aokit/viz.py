"""aokit.viz -- matplotlib visualizations (offline, headless-safe).

Plots for the PS9 SH-WFS pipeline (research/06 S13 -- offline visualization):

* :func:`plot_spot_field`        -- SH-WFS detector frame + sub-aperture boxes /
                                    reference-spot markers.
* :func:`plot_slope_quiver`      -- slope vectors as a quiver over the grid.
* :func:`plot_phase_map`         -- reconstructed wavefront phase map + colorbar.
* :func:`plot_zernike_spectrum`  -- Zernike coefficient amplitudes vs Noll index
                                    (optional Kolmogorov overlay).
* :func:`plot_r0_tau0_timeseries`-- r0 / tau0 trends vs time/frame (twin axes).
* :func:`plot_turbulence_trends` -- r0 / tau0 trends from a fixed cadence dt.
* :func:`plot_residual`          -- true / recon / residual side-by-side maps.
* :func:`plot_actuator_map`      -- 2-D DM actuator-stroke map on the Fried grid.
* :func:`plot_method_comparison` -- bar chart cross-validating r0 / tau0
                                    estimators (with error bars).
* :func:`summary_figure`         -- small multi-panel dashboard helper.

HEADLESS: this module forces the non-interactive ``Agg`` backend at import time
(before ``pyplot`` is imported), so every function works without a display.
Each plotting function follows the same contract:

    f(<data...>, *, savepath=None, ax=None, dpi=150, ...)

* If ``ax`` is given the artist is drawn into it and its parent ``Figure`` is
  returned (the figure is NOT closed -- the caller owns it).
* If ``ax`` is omitted a new ``Figure``/``Axes`` is created.  When ``savepath``
  is given the figure is saved as a PNG at ``dpi`` and then **closed**, and the
  (closed) ``Figure`` is still returned for reference; otherwise the live
  ``Figure`` is returned.

Imported optionally by ``aokit.__init__`` (matplotlib is an optional
dependency).
"""
from __future__ import annotations

# --- Headless backend: MUST be selected before pyplot is imported. ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from typing import Optional, Sequence, Union  # noqa: E402
import numpy as np                              # noqa: E402

__all__ = [
    "plot_spot_field",
    "plot_slope_quiver",
    "plot_phase_map",
    "plot_zernike_spectrum",
    "plot_r0_tau0_timeseries",
    "plot_turbulence_trends",
    "plot_residual",
    "plot_actuator_map",
    "plot_method_comparison",
    "summary_figure",
]


# ============================================================================
# Internal helpers
# ============================================================================

def _new_ax(ax, figsize=(6.0, 5.0)):
    """Return ``(fig, ax, created)`` -- reuse ``ax`` if given, else make one.

    ``created`` is True when this call owns the figure (used to decide whether a
    standalone save/close is appropriate).
    """
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def _finish(fig, savepath, dpi, created):
    """Save (if requested) and close a figure we created; return the figure.

    Only figures created *inside* this module (``created`` True) are auto-saved
    and closed, so passing your own ``ax`` never closes your figure unexpectedly.
    """
    if savepath is not None and created:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def _as_2d(arr):
    """Coerce input to a 2-D float array (squeeze trailing singleton axes)."""
    a = np.asarray(arr, dtype=float)
    a = np.squeeze(a)
    if a.ndim == 1:
        # Best-effort: lay a 1-D vector out as a near-square image.
        n = a.size
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.nan, dtype=float)
        padded[:n] = a
        a = padded.reshape(side, side)
    elif a.ndim != 2:
        raise ValueError(f"expected a 2-D phase map, got shape {a.shape}")
    return a


def _apply_mask(img, mask):
    """Return a float copy of ``img`` with pixels outside ``mask`` set to NaN."""
    out = np.array(img, dtype=float, copy=True)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == out.shape:
            out[~m] = np.nan
    return out


def _xy_from_grid(grid_or_positions):
    """Extract sub-aperture / actuator (x, y) reference positions from many
    accepted container shapes.

    Accepts: an ``aokit.geometry`` object exposing ``subaps``/``acts`` or
    ``ref_x``/``ref_y`` (or ``x``/``y``); a tuple/list ``(x, y)``; or an
    ``(N, 2)`` array of positions.  Returns ``(x, y)`` 1-D float arrays.
    """
    g = grid_or_positions
    # Geometry bundle -> use its valid sub-apertures.
    if hasattr(g, "subaps"):
        sub = g.subaps
        return (np.asarray(sub.ref_x, dtype=float),
                np.asarray(sub.ref_y, dtype=float))
    # SubApertureGrid-like.
    if hasattr(g, "ref_x") and hasattr(g, "ref_y"):
        return (np.asarray(g.ref_x, dtype=float),
                np.asarray(g.ref_y, dtype=float))
    # ActuatorGrid-like.
    if hasattr(g, "x") and hasattr(g, "y"):
        return (np.asarray(g.x, dtype=float),
                np.asarray(g.y, dtype=float))
    # (x, y) tuple / list of two arrays.
    if isinstance(g, (tuple, list)) and len(g) == 2:
        x = np.asarray(g[0], dtype=float).ravel()
        y = np.asarray(g[1], dtype=float).ravel()
        return x, y
    # (N, 2) array of positions.
    arr = np.asarray(g, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].copy(), arr[:, 1].copy()
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[0].copy(), arr[1].copy()
    raise ValueError(
        "could not interpret grid_or_positions; pass a Geometry, a "
        "SubApertureGrid, an (x, y) tuple, or an (N, 2) array"
    )


def _split_slopes(sx, sy, n):
    """Return length-``n`` x/y slope arrays.

    ``sy`` may be ``None`` if ``sx`` is the full 2M block vector
    ``[sx_1..sx_M, sy_1..sy_M]`` (the canonical layout); otherwise ``sx``/``sy``
    are taken as separate per-sub-aperture arrays.
    """
    sx = np.asarray(sx, dtype=float).ravel()
    if sy is None:
        if sx.size == 2 * n:
            return sx[:n], sx[n:]
        if sx.size == n:
            return sx, np.zeros(n, dtype=float)
        raise ValueError(
            f"slope vector length {sx.size} matches neither n={n} nor 2n={2*n}"
        )
    sy = np.asarray(sy, dtype=float).ravel()
    return sx, sy


# ============================================================================
# 1. Spot field
# ============================================================================

def plot_spot_field(image: np.ndarray, subaps=None, centroids=None,
                    title: str = "SH-WFS spot field",
                    save: Optional[str] = None, *,
                    savepath: Optional[str] = None, ax=None, dpi: int = 150,
                    grid=None, cmap: str = "gray", box_color: str = "tab:cyan",
                    ref_color: str = "tab:red"):
    """Show the SH-WFS detector frame (grayscale) with optional sub-aperture
    boxes and reference-spot / measured-centroid markers.

    Parameters
    ----------
    image : 2-D array
        The detector frame (intensity).  Displayed with origin at the top-left
        (detector convention: x = column, y = row).
    subaps, grid : SubApertureGrid | Geometry, optional
        If given, draws each sub-aperture window as a box and its reference
        centroid as a marker.  ``grid`` is an alias accepted for convenience; a
        :class:`~aokit.geometry.Geometry` bundle is unwrapped to its ``subaps``.
    centroids : (M, 2) array | (x, y) tuple, optional
        Measured spot centroids to overlay (e.g. the current frame's spots).
    save, savepath : str, optional
        PNG output path (``savepath`` preferred; ``save`` kept for back-compat).
    """
    savepath = savepath if savepath is not None else save
    sub = grid if subaps is None else subaps
    if sub is not None and hasattr(sub, "subaps"):
        sub = sub.subaps

    img = np.asarray(image, dtype=float)
    if img.ndim == 3:                      # collapse RGB -> luminance
        img = img.mean(axis=2)

    fig, ax, created = _new_ax(ax, figsize=(6.0, 6.0))
    im = ax.imshow(img, cmap=cmap, origin="upper", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="intensity")

    if sub is not None:
        x0 = np.asarray(sub.x0, dtype=float)
        y0 = np.asarray(sub.y0, dtype=float)
        w = float(sub.w)
        h = float(sub.h)
        valid = getattr(sub, "valid", None)
        for k in range(x0.size):
            edge = box_color
            if valid is not None and not bool(np.ravel(valid)[k]):
                edge = "0.4"
            ax.add_patch(Rectangle((x0[k], y0[k]), w, h, fill=False,
                                   edgecolor=edge, linewidth=0.6, alpha=0.7))
        if hasattr(sub, "ref_x") and hasattr(sub, "ref_y"):
            ax.scatter(np.asarray(sub.ref_x, dtype=float),
                       np.asarray(sub.ref_y, dtype=float),
                       s=10, marker="+", c=ref_color, linewidths=0.8,
                       label="reference")

    if centroids is not None:
        cx, cy = _xy_from_grid(centroids)
        ax.scatter(cx, cy, s=12, facecolors="none", edgecolors="yellow",
                   linewidths=0.8, label="centroid")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)   # top-left origin
    return _finish(fig, savepath, dpi, created)


# ============================================================================
# 2. Slope quiver
# ============================================================================

def plot_slope_quiver(grid_or_positions, sx, sy=None, *,
                      title: str = "SH-WFS slopes",
                      savepath: Optional[str] = None, ax=None, dpi: int = 150,
                      scale: Optional[float] = None, color_by_mag: bool = True):
    """Quiver plot of slope vectors over the sub-aperture grid.

    Parameters
    ----------
    grid_or_positions :
        Anything :func:`_xy_from_grid` understands -- a Geometry, a
        SubApertureGrid, an ``(x, y)`` tuple, or an ``(N, 2)`` positions array.
    sx, sy : array
        Slope components per sub-aperture.  ``sy`` may be ``None`` if ``sx`` is
        the full ``2M`` block vector ``[sx_1..sx_M, sy_1..sy_M]``.
    scale : float, optional
        Passed to ``quiver(..., scale=...)``; ``None`` lets matplotlib autoscale.
    color_by_mag : bool
        Colour the arrows by slope magnitude (adds a colorbar).
    """
    x, y = _xy_from_grid(grid_or_positions)
    n = x.size
    u, v = _split_slopes(sx, sy, n)

    fig, ax, created = _new_ax(ax, figsize=(6.0, 6.0))
    mag = np.hypot(u, v)
    if color_by_mag and np.any(mag > 0):
        q = ax.quiver(x, y, u, v, mag, angles="xy",
                      scale=scale, scale_units="xy" if scale else None,
                      cmap="viridis", width=0.004)
        fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="|slope| (rad)")
    else:
        ax.quiver(x, y, u, v, angles="xy",
                  scale=scale, scale_units="xy" if scale else None,
                  width=0.004, color="tab:blue")

    ax.scatter(x, y, s=4, c="0.5", zorder=1)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()                       # detector top-left origin
    return _finish(fig, savepath, dpi, created)


# ============================================================================
# 3. Phase map
# ============================================================================

def plot_phase_map(W: np.ndarray, mask: Optional[np.ndarray] = None,
                   title: str = "Reconstructed wavefront",
                   save: Optional[str] = None, *,
                   savepath: Optional[str] = None, ax=None, dpi: int = 150,
                   cmap: str = "RdBu_r", units: str = "rad",
                   symmetric: bool = True):
    """Display a reconstructed phase map (rad) with a colorbar.

    Pixels outside ``mask`` (if given) are set to NaN so they render
    transparent.  By default the colour scale is symmetric about zero (a
    diverging map), appropriate for a piston-removed wavefront.
    """
    savepath = savepath if savepath is not None else save
    img = _apply_mask(_as_2d(W), mask)

    fig, ax, created = _new_ax(ax, figsize=(6.0, 5.0))
    vmin = vmax = None
    if symmetric and np.any(np.isfinite(img)):
        amax = float(np.nanmax(np.abs(img)))
        if amax > 0:
            vmin, vmax = -amax, amax
    im = ax.imshow(img, cmap=cmap, origin="lower", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=f"phase ({units})")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return _finish(fig, savepath, dpi, created)


# ============================================================================
# 4. Zernike spectrum
# ============================================================================

def plot_zernike_spectrum(coeffs: np.ndarray,
                          title: str = "Zernike spectrum",
                          save: Optional[str] = None, *,
                          savepath: Optional[str] = None, ax=None,
                          dpi: int = 150, j_start: int = 2,
                          kolmogorov=None, D_over_r0: Optional[float] = None,
                          amplitude: bool = True):
    """Bar plot of Zernike coefficients vs Noll index.

    Parameters
    ----------
    coeffs : 1-D array
        Modal coefficients, Noll-ordered.  By convention the first entry maps to
        Noll ``j = j_start`` (default 2, piston excluded) -- consistent with
        ``zernike.zernike_basis``.
    amplitude : bool
        If True (default) plot ``|a_j|``; otherwise the signed coefficient.
    kolmogorov : array | callable, optional
        Expected per-mode amplitude/variance to overlay (e.g. the Kolmogorov
        ``sqrt(c_j) (D/r0)^(5/6)`` RMS).  An array is used directly (aligned to
        ``coeffs``); a callable is invoked as ``kolmogorov(j)`` per Noll index.
    D_over_r0 : float, optional
        Convenience: if given (and ``kolmogorov`` is None), overlay the analytic
        Kolmogorov RMS expectation ``sqrt(noll_variance(j)) * (D/r0)^(5/6)``
        using :mod:`aokit.zernike` (skipped silently if unavailable).
    """
    savepath = savepath if savepath is not None else save
    a = np.asarray(coeffs, dtype=float).ravel()
    j = np.arange(j_start, j_start + a.size)
    height = np.abs(a) if amplitude else a

    fig, ax, created = _new_ax(ax, figsize=(7.0, 4.0))
    ax.bar(j, height, width=0.8, color="tab:blue", alpha=0.85,
           label="measured")

    # Optional Kolmogorov / expectation overlay.
    expect = None
    if kolmogorov is not None:
        if callable(kolmogorov):
            expect = np.array([float(kolmogorov(int(jj))) for jj in j])
        else:
            expect = np.asarray(kolmogorov, dtype=float).ravel()
            if expect.size != a.size:
                m = min(expect.size, a.size)
                expect = expect[:m]
                j_e = j[:m]
            else:
                j_e = j
    elif D_over_r0 is not None:
        try:
            from . import zernike as _zern
            scale = float(D_over_r0) ** (5.0 / 6.0)
            expect = np.array(
                [np.sqrt(max(_zern.noll_variance(int(jj)), 0.0)) * scale
                 for jj in j])
        except Exception:               # pragma: no cover - optional overlay
            expect = None

    if expect is not None:
        j_e = j if expect.size == a.size else j[:expect.size]
        ax.plot(j_e, np.abs(expect), "o--", color="tab:red", markersize=4,
                label="Kolmogorov expectation")

    ax.set_title(title)
    ax.set_xlabel("Noll index $j$")
    ax.set_ylabel("|coefficient| (rad)" if amplitude else "coefficient (rad)")
    ax.grid(True, axis="y", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    return _finish(fig, savepath, dpi, created)


# ============================================================================
# 5. r0 / tau0 time series
# ============================================================================

def plot_r0_tau0_timeseries(times, r0_series=None, tau0_series=None, *,
                            title: str = "Turbulence trends",
                            savepath: Optional[str] = None, ax=None,
                            dpi: int = 150, xlabel: str = "time (s)"):
    """Line plot of estimated r0 and/or tau0 vs time/frame (twin y-axes).

    ``times`` may be a 1-D array of timestamps/frame indices.  Either series may
    be omitted.  r0 is drawn on the left axis (m), tau0 on a twin right axis (ms).
    """
    t = np.asarray(times, dtype=float).ravel()

    fig, ax, created = _new_ax(ax, figsize=(7.5, 4.0))
    handles = []
    if r0_series is not None:
        r0 = np.asarray(r0_series, dtype=float).ravel()
        m = min(t.size, r0.size)
        h1, = ax.plot(t[:m], r0[:m], "-", color="tab:blue", label=r"$r_0$")
        handles.append(h1)
        ax.set_ylabel(r"$r_0$ (m)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

    if tau0_series is not None:
        tau0 = np.asarray(tau0_series, dtype=float).ravel()
        ax2 = ax.twinx()
        m = min(t.size, tau0.size)
        h2, = ax2.plot(t[:m], tau0[:m] * 1e3, "-", color="tab:red",
                       label=r"$\tau_0$")
        handles.append(h2)
        ax2.set_ylabel(r"$\tau_0$ (ms)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    if handles:
        ax.legend(handles, [h.get_label() for h in handles],
                  loc="best", fontsize=8)
    return _finish(fig, savepath, dpi, created)


def plot_turbulence_trends(r0_ts: Sequence[float], tau0_ts: Sequence[float],
                           dt_s: float, save: Optional[str] = None, *,
                           savepath: Optional[str] = None, ax=None,
                           dpi: int = 150):
    """Time trends of r0 and tau0 over a sequence at fixed cadence ``dt_s``.

    Thin wrapper over :func:`plot_r0_tau0_timeseries` that builds the time axis
    from the frame index and ``dt_s``.  Kept for the stub contract.
    """
    savepath = savepath if savepath is not None else save
    r0 = np.asarray(r0_ts, dtype=float).ravel()
    tau0 = np.asarray(tau0_ts, dtype=float).ravel()
    n = max(r0.size, tau0.size)
    times = np.arange(n) * float(dt_s)
    return plot_r0_tau0_timeseries(
        times, r0_series=r0 if r0.size else None,
        tau0_series=tau0 if tau0.size else None,
        savepath=savepath, ax=ax, dpi=dpi,
    )


# ============================================================================
# 6. Residual (true / recon / residual)
# ============================================================================

def plot_residual(W_before: np.ndarray, W_after: np.ndarray,
                  mask: Optional[np.ndarray] = None, save: Optional[str] = None,
                  *, savepath: Optional[str] = None, dpi: int = 150,
                  cmap: str = "RdBu_r",
                  labels=("true", "reconstructed", "residual"),
                  units: str = "rad"):
    """Side-by-side true / recon / residual maps sharing one colour scale.

    ``residual = W_before - W_after``.  The three panels share a symmetric
    diverging colour scale (about zero) and a single colorbar, so the residual
    is directly comparable to the inputs.  Returns the :class:`Figure`.

    (The stub names the first two args ``W_before``/``W_after``; here they are
    the *true* and *reconstructed* maps and the residual is their difference --
    equivalent to the before/after-correction interpretation.)
    """
    savepath = savepath if savepath is not None else save
    a = _apply_mask(_as_2d(W_before), mask)
    b = _apply_mask(_as_2d(W_after), mask)
    # Align shapes defensively (reconstruction may differ by a row/col).
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    res = a - b

    panels = [a, b, res]
    finite = np.concatenate([p[np.isfinite(p)].ravel() for p in panels
                             if np.any(np.isfinite(p))]) \
        if any(np.any(np.isfinite(p)) for p in panels) else np.array([0.0])
    amax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if amax <= 0:
        amax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    im = None
    rms = float(np.sqrt(np.nanmean(res ** 2))) if np.any(np.isfinite(res)) \
        else float("nan")
    titles = list(labels)
    titles[2] = f"{labels[2]} (RMS={rms:.3g} {units})"
    for axk, data, ttl in zip(axes, panels, titles):
        im = axk.imshow(data, cmap=cmap, origin="lower", vmin=-amax, vmax=amax,
                        interpolation="nearest")
        axk.set_title(ttl)
        axk.set_xticks([])
        axk.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label=f"phase ({units})")

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


# ============================================================================
# 7. Actuator map
# ============================================================================

def plot_actuator_map(commands: np.ndarray, n_act_x: int, n_act_y: int,
                      title: str = "DM actuator map (stroke units)",
                      save: Optional[str] = None, *,
                      savepath: Optional[str] = None, ax=None, dpi: int = 150,
                      cmap: str = "RdBu_r", units: str = "stroke"):
    """2-D image of the actuator-stroke map on the Fried ``n_act_y x n_act_x``
    grid (row-major flattening, matching ``aokit.geometry``).
    """
    savepath = savepath if savepath is not None else save
    a = np.asarray(commands, dtype=float).ravel()
    n = int(n_act_x) * int(n_act_y)
    if a.size < n:
        a = np.concatenate([a, np.full(n - a.size, np.nan)])
    grid = a[:n].reshape(int(n_act_y), int(n_act_x))

    fig, ax, created = _new_ax(ax, figsize=(5.5, 5.0))
    amax = float(np.nanmax(np.abs(grid))) if np.any(np.isfinite(grid)) else 1.0
    if amax <= 0:
        amax = 1.0
    im = ax.imshow(grid, cmap=cmap, origin="lower", interpolation="nearest",
                   vmin=-amax, vmax=amax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=units)
    ax.set_title(title)
    ax.set_xlabel("actuator column")
    ax.set_ylabel("actuator row")
    return _finish(fig, savepath, dpi, created)


# ============================================================================
# 8. Method comparison (cross-validation bar chart)
# ============================================================================

def plot_method_comparison(labels: Sequence[str], values: Sequence[float],
                           errors: Optional[Sequence[float]] = None, *,
                           title: str = "Estimator comparison",
                           ylabel: str = "estimate",
                           reference: Optional[float] = None,
                           savepath: Optional[str] = None, ax=None,
                           dpi: int = 150):
    """Bar chart comparing multiple r0 (or tau0) estimators -- the
    cross-validation visual (research/04 S5: independent estimators must agree).

    Parameters
    ----------
    labels : sequence of str
        Estimator names (e.g. ``["R1_zernike", "R2_slope", ...]``).
    values : sequence of float
        The estimate from each method (aligned with ``labels``).
    errors : sequence of float, optional
        Symmetric error bars per estimator.
    reference : float, optional
        A horizontal reference line (e.g. the injected ground-truth / median).
    """
    labels = list(labels)
    vals = np.asarray(values, dtype=float).ravel()
    n = len(labels)
    x = np.arange(n)
    yerr = None
    if errors is not None:
        yerr = np.asarray(errors, dtype=float).ravel()
        if yerr.size != n:
            yerr = None

    fig, ax, created = _new_ax(ax, figsize=(max(6.0, 0.8 * n + 2.0), 4.5))
    ax.bar(x, vals, yerr=yerr, width=0.7, color="tab:green", alpha=0.85,
           capsize=4, ecolor="0.3")

    if reference is not None:
        ax.axhline(float(reference), color="tab:red", linestyle="--",
                   linewidth=1.2, label=f"reference = {float(reference):.3g}")
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _finish(fig, savepath, dpi, created)


# ============================================================================
# 9. Dashboard helper (optional)
# ============================================================================

def summary_figure(phase=None, coeffs=None, slopes_grid=None,
                   r0_tau0=None, *, mask=None,
                   title: str = "PS9 wavefront summary",
                   savepath: Optional[str] = None, dpi: int = 150):
    """Compact multi-panel dashboard combining several views.

    Any panel whose data is omitted is left blank.  Panels:
      (1) reconstructed phase map,        (2) Zernike spectrum,
      (3) slope quiver,                   (4) r0/tau0 trend.

    Parameters
    ----------
    phase : 2-D array, optional
        Wavefront phase map -> :func:`plot_phase_map`.
    coeffs : 1-D array, optional
        Zernike coefficients -> :func:`plot_zernike_spectrum`.
    slopes_grid : tuple ``(grid_or_positions, sx, sy)``, optional
        Slope quiver inputs -> :func:`plot_slope_quiver`.
    r0_tau0 : tuple ``(times, r0_series, tau0_series)``, optional
        Turbulence trend inputs -> :func:`plot_r0_tau0_timeseries`.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    (ax_phase, ax_zern), (ax_quiv, ax_trend) = axes

    if phase is not None:
        plot_phase_map(phase, mask=mask, ax=ax_phase)
    else:
        ax_phase.set_axis_off()

    if coeffs is not None:
        plot_zernike_spectrum(coeffs, ax=ax_zern)
    else:
        ax_zern.set_axis_off()

    if slopes_grid is not None:
        g = slopes_grid
        sx = g[1] if len(g) > 1 else None
        sy = g[2] if len(g) > 2 else None
        plot_slope_quiver(g[0], sx, sy, ax=ax_quiv)
    else:
        ax_quiv.set_axis_off()

    if r0_tau0 is not None:
        g = r0_tau0
        plot_r0_tau0_timeseries(
            g[0],
            r0_series=g[1] if len(g) > 1 else None,
            tau0_series=g[2] if len(g) > 2 else None,
            ax=ax_trend,
        )
    else:
        ax_trend.set_axis_off()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
