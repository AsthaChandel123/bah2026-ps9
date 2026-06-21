# Spot Detection & Centroiding for the SH-WFS (PS9, Stage 1)

**Project:** ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 9
**Pipeline stage:** Stage 1 — convert each Shack-Hartmann (SH) lenslet spot-field frame (a `.bmp`) into per-sub-aperture x/y spot displacements (slopes).
**Hard constraint:** full reconstruction loop < ~10 ms (atmospheric coherence time τ0). Centroiding is run N_subap times per frame, so it must be **O(1) per sub-aperture** in the hot path and amenable to a tight C implementation.
**Scope of this document:** spot detection + centroiding only. Wavefront reconstruction, Zernike fitting, r0/τ0 estimation and DM actuator mapping are downstream stages covered elsewhere.

---

## 0. TL;DR / Executive Summary

- The whole field reduces to one estimator family: **center-of-gravity (CoG) and its conditioned variants** (thresholded, weighted, iteratively weighted, windowed). Everything else (quad-cell, parabola, Gaussian fit, correlation, matched-filter, maximum-likelihood, brightest-pixel, ML/CNN) is either a *special case*, a *peak refinement*, or a *heavier optimum* of the same problem: estimate sub-pixel spot position under photon (shot) + read noise.
- For **lab-simulated turbulence, real-time, point-source spots on a science-grade camera with a fixed Fried grid**, the winning recipe is **Thresholded + Windowed Weighted CoG (TWCoG)** inside a fixed per-lenslet window, with a one-time **flat-wavefront reference calibration**, a **noise-floor threshold (~3-5σ_read)**, and a **per-camera bias lookup table** to cancel the weighting-induced and pixel-locking biases. This is O(1)/sub-aperture, branch-light, and reaches near-Cramér-Rao accuracy at the SNR a lab rig delivers.
- Keep **plain CoG** as the always-on baseline and as the inner kernel; keep **correlation/matched-filter** only as a fallback if the supplied spots turn out extended/elongated; keep **CNN/ML out of the real-time loop** (use offline only, if at all).

A ranked recommendation and a full comparison table are at the end (Sections 11-12).

---

## 1. Geometry, notation, and the measurement equation

### 1.1 What a sub-aperture measures
Each lenslet of focal length `f_MLA` images the local wavefront tilt over its sub-aperture as a focused spot. If the average wavefront gradient (local tilt angle) across lenslet `k` is `θ_x = ∂W/∂x` (radians, W in length units), the spot moves on the detector by

```
Δx_spot = f_MLA · θ_x          (small-angle, spot displacement in length units)
```

So the **slope estimate** the reconstructor wants is

```
s_x = θ_x = Δx_spot / f_MLA  = (x_centroid − x_ref) · p_pix / f_MLA      [radians]
```

where:
- `x_centroid` = measured centroid (pixels),
- `x_ref` = reference centroid for that lenslet from a flat wavefront (pixels),
- `p_pix` = detector pixel pitch (length/pixel; given as "pixel size" in the dataset),
- `f_MLA` = MLA focal length (given).

This is the only place `p_pix` and `f_MLA` enter. Everything in this document is about estimating `x_centroid` accurately and fast; the conversion above is one multiply per axis per sub-aperture. (Operating principle and proportionality: integrated wavefront gradient across a lenslet ∝ centroid displacement — see Wikipedia SH-WFS and Thomas/MNRAS below.)

### 1.2 Sub-aperture image model and noise
A sub-aperture window is an `N×N` patch `I(i,j)` (typically N≈ spot-spacing in pixels; e.g. 8-32). The pixel value model:

```
I(i,j) = G · [ Σ photons ]  + dark(i,j) + bias + n_shot(i,j) + n_read(i,j)
```
- **Photon (shot) noise:** `n_shot ~ Poisson`, variance = signal in photo-electrons. Dominates at high flux.
- **Read noise:** `n_read ~ Gaussian(0, σ_read²)`, constant per pixel. Dominates at low flux and grows with window area.
- **Dark / bias / fixed-pattern:** removed by calibration (Section 9).

Two regimes drive every algorithm choice:
- **Photon-noise limited** (bright spots): error floor set by photon count; plain CoG is near-optimal if the window is tight.
- **Read-noise limited** (faint spots / many pixels): error grows with the number of summed pixels; thresholding/weighting/windowing are essential to suppress empty-pixel read noise.

A lab turbulence rig with a science-grade camera and a real laser/LED source usually sits in the **medium-to-high flux** regime — good news, it makes the simple, fast estimators viable.

---

## 2. The noise / error formulas you will actually budget against

These are the load-bearing equations for choosing parameters; sources: Thomas et al. *MNRAS* 371, 323 (2006); the *IntechOpen* "Measurement Error of Shack–Hartmann WFS" chapter; Cramér–Rao / maximum-likelihood treatment (Barrett et al., PMC2581470).

### 2.1 Plain CoG centroid variance
For a spot of equivalent Gaussian width `σ_s` (pixels) over an `N×N` window:

- **Photon-noise term** (N_ph total photo-electrons in window):
```
σ²_cog,photon ≈ σ_s² / N_ph                         (per axis, pixel²)
```
(Often quoted as `σ_photon = G_s / sqrt(V_s)` with `G_s` the Gaussian width, `V_s` total signal e-. This is the photon-noise floor; plain CoG reaches the Cramér-Rao bound for a critically/under-sampled Gaussian to within a small constant.)

- **Read-noise term** (σ_read per pixel, N×N window):
```
σ²_cog,read ≈ (σ_read² / N_ph²) · ( N²(N²−1) / 12 )   (per axis, pixel²)
```
Key insight: the read-noise error **grows as ~N⁴ with window side** because the sum of squared pixel-distances over an N×N window is `Σ d² ∝ N²(N²−1)/12`. This single fact is why you **never CoG the full sub-aperture** — you window and threshold. (IntechOpen: `σ²_cr = N_r²/V_r² · L₁L₂(L₁²−1)/12`.)

### 2.2 Propagation to wavefront error
Per-sub-aperture slope (angular) error from a centroid error `σ_c` (pixels):
```
σ_angle = σ_c · p_pix / f_MLA                         [rad]
```
For the same `σ_c`, **longer f_MLA → smaller slope error** (bigger lever arm), but **smaller dynamic range** (spot leaves the window sooner) — a fundamental f_MLA trade. The reconstructed-phase error scales with the reconstruction matrix `E`:
```
σ_φ ≈ σ_c · (D² / 2 λ f_MLA) · sqrt( Σ_j Σ_k (e_{j,2k-1}+e_{j,2k})² )
```
i.e. centroid error is the dominant, multiplicatively-propagated source of wavefront-reconstruction error in an SH-AO system. **Centroiding accuracy is the single biggest lever on final wavefront quality** — worth doing well.

### 2.3 Cramér–Rao lower bound (the target to beat)
With combined Poisson+Gaussian noise, the Fisher information for spot position from the per-pixel mean response `k̄_m(θ)`:
```
F_jk ≈ Σ_m [ R² / (σ² + R²·k̄_m(θ)) ] · ∂k̄_m/∂θ_j · ∂k̄_m/∂θ_k ,   Var{θ̂} ≥ [F⁻¹]_jj
```
Maximum-likelihood (Section 7) achieves this bound; the practical CoG variants get within a small factor of it at lab SNR for far less compute. Use the CRLB as the yardstick when validating your C implementation against simulated frames.

---

## 3. Method catalogue (≥12 distinct techniques)

For each: math → complexity → sub-pixel accuracy → noise behaviour → bias → real-time/C suitability.

### M1. Simple Center of Gravity (CoG / first moment / "centroid")
**Math.**
```
x̂ = Σ_ij i·I(i,j) / Σ_ij I(i,j) ,   ŷ = Σ_ij j·I(i,j) / Σ_ij I(i,j)
```
**Complexity.** Single pass over the window: `~3 N²` multiply-adds, 1-2 divides per sub-aperture → **O(1) per sub-aperture** (O(N²) in pixels, but N is small & fixed). No branches. Trivially SIMD/streamable.
**Sub-pixel accuracy.** Excellent in principle (continuous estimator); reaches photon-noise floor `σ_s/√N_ph` for a tight window.
**Noise.** Photon-optimal-ish; **read noise is the killer** — every empty pixel contributes `σ_read` weighted by its distance² (Section 2.1). Unbounded read-noise growth with window size.
**Bias.** Unbiased for a symmetric spot fully inside a generous window. **Two bias sources:** (a) *truncation/non-linearity* when the spot nears the window edge — response coefficient < 1, gain error grows as FOV shrinks; (b) negligible pixel-locking if well sampled. Must calibrate gain for finite FOV.
**C/real-time.** **Best-in-class.** The canonical kernel; running-sum accumulators, no transcendental functions. This is the kernel every other CoG variant wraps. aotools `centre_of_gravity()` is literally `(idx*img).sum()/img.sum()`.

### M2. Thresholded CoG (TCoG)
**Math.** Subtract a threshold `I_T` and clip to zero before CoG:
```
I'(i,j) = max( I(i,j) − I_T , 0 ),   then CoG on I'.
I_T = max( T·I_max , m·σ_read )   with T≈0.05–0.10, m≈3–5
```
**Complexity.** CoG + one compare/subtract per pixel → still **O(1)/sub-aperture**, one extra pass (or fused). One branch per pixel (clip), cheap.
**Sub-pixel accuracy.** Better than plain CoG under read noise because empty-pixel noise is zeroed.
**Noise.** Strongly suppresses read-noise term (removes the `N⁴` tail). At very low flux, threshold must track the noise floor (`m·σ_read`) not just `T·I_max`.
**Bias.** Introduces a **threshold-dependent bias**: subtracting `I_T` from kept pixels biases the centroid toward the window center and creates a mild non-linearity; choosing `T` optimally is fiddly and flux-dependent. Also creates pixel/peak-locking if the surviving footprint is tiny.
**C/real-time.** Excellent, near-CoG cost. The pragmatic default and the most-used method in fielded SH systems. (Thomas/MNRAS: 10% of max is a common default; SHARPEST uses 1.3–1.5σ_read plus a "≥3 pixels above threshold" validity gate.)

### M3. Weighted CoG (WCoG) — Gaussian-weighted / matched weighting
**Math.** Multiply by a fixed weighting function `W(i,j)` (usually a Gaussian centered on the reference, FWHM ≈ spot FWHM) before the moment:
```
x̂_WCoG = Σ i·I·W / Σ I·W ,  W(i,j)=exp(−((i−x_c)²+(j−y_c)²)/2σ_w²)
```
**Complexity.** CoG with a precomputed weight LUT → `~2×` CoG cost, still **O(1)/sub-aperture**. Weights are constant per lenslet ⇒ **precompute once** into a table (key O(1) speedup).
**Sub-pixel accuracy.** It is the **maximum-likelihood estimator of spot position under Gaussian noise when W matches the spot** (Thomas/MNRAS; Vyas "Gaussian Pattern Matching"). Optimal weight FWHM ≈ 4–4.5 px at Nyquist sampling. Best accuracy/noise trade of the cheap methods.
**Noise.** Photon-noise variance ≈ `0.795·(λ/πd_p)²·(1+σ_w²/σ_eq²)/N_ph`; read-noise ≈ `1.56·(N_r/N_ph)²·γ²`. Suppresses both far better than CoG; nearly removes FOV dependence.
**Bias.** **Has a structural bias:** because W is centered at a fixed point, the estimate is pulled toward the weight center proportional to (true offset)×(displacement) — needs a **gain factor γ** to linearize and/or an empirically-derived calibration curve. This bias is the central caveat (mitigation in Section 8).
**C/real-time.** Excellent with a precomputed weight LUT + precomputed gain γ; one extra elementwise multiply vs CoG. Strongly recommended core.

### M4. Iteratively Weighted CoG (IWCoG / IWC)
**Math.** WCoG where the weight's center and width are updated from the previous iteration's estimate:
```
init: (x_c,y_c) = window center, σ_w ≈ N/4
repeat: compute WCoG with W centered at (x_c,y_c) → new (x̂,ŷ); set (x_c,y_c)=(x̂,ŷ); (optionally re-estimate σ_w)
until convergence (2–4 iters typically)
```
**Complexity.** `n_iter ×` WCoG (n_iter≈2–5). **No longer strictly O(1)** — it's O(n_iter) per sub-aperture with data-dependent iteration count (bad for fixed-budget real-time). 
**Sub-pixel accuracy.** Best of the CoG family for **large spot excursions / large aberrations** and low SNR — weight follows the spot, killing the WCoG center bias. Can reach <0.01 px under sufficient SNR (Akondi/Roopashree). Convergence improved by seeding W at the brightest pixel on iter 1; pick the iteration with max W–spot correlation.
**Noise.** Lowest reconstructed-phase variance at low SNR among classical methods (Baker & Moallem, *Opt. Express* 15, 5147).
**Bias.** Largely removes the fixed-weight bias of WCoG; residual depends on stop criterion.
**C/real-time.** Marginal: variable iteration count breaks the hard 10 ms budget guarantee. **Use a fixed small iteration count (e.g. 2)** if adopted, or reserve for an offline/refinement path. For PS9's lab spots (modest excursions) the iteration usually isn't needed.

### M5. Windowed CoG / Stream-based floating-window CoG
**Math.** CoG computed in a window that **floats with the spot** (centered on a running estimate / brightest region) rather than fixed to the lenslet cell. In the stream/FPGA formulation the CoG accumulators slide with the incoming pixel raster so the effective window tracks the spot.
**Complexity.** **O(1)/sub-aperture, single raster pass**, accumulator-based — designed for line-streaming hardware; no frame buffer needed. (MDPI Electronics 12(7):1714; Opt. Express stream-processing, PMID 29047936.)
**Sub-pixel accuracy.** Decouples the noise/bias-vs-dynamic-range trade from window size: accuracy holds even as the spot crosses the nominal sub-aperture boundary → **increased dynamic range and reduced cross-talk sensitivity**.
**Noise.** Because the window tracks the spot and stays tight, read-noise stays bounded regardless of excursion.
**Bias.** Avoids edge-truncation bias (spot stays centered in its floating window). 
**C/real-time.** **Excellent and the most hardware-friendly** — exactly the pattern to use if you later push to FPGA/streaming. On CPU it maps to a per-lenslet window whose origin is set from a coarse peak find.

### M6. Quad-cell (QC) / 2×2
**Math.** Differential signal between halves:
```
x̂ = γ · (I_right − I_left)/(I_right + I_left) ,   ŷ = γ · (I_top − I_bottom)/(I_top + I_bottom)
```
**Complexity.** 4 sums + 2 divides → cheapest possible, **O(1)**. aotools `quadCell` returns `Σright−Σleft, Σbottom−Σtop`.
**Sub-pixel accuracy.** Fine **only in a tiny linear range** (|offset| ≲ spot radius). Needs the spot to straddle the 4 cells.
**Noise.** Photon `σ²∝1.57·κ/N_ph`; read `σ²∝4(N_r/N_ph)²·κ` — works to ~10 photons/subap at 3e- read noise (Thomas/MNRAS).
**Bias.** **Strong cubic non-linearity** `f_nl ∝ −x³/σ_s²`; gain `γ` depends on spot size/seeing and must be continuously calibrated (centroid-gain tracking). Atmospheric error penalty ~1.4·W² — worst of the methods for turbulence.
**C/real-time.** Fast but **poor fit for PS9**: a science-grade camera gives many pixels/spot, so collapsing to 2×2 throws away information, needs gain tracking, and has tiny dynamic range. Mention for completeness; do **not** use as primary.

### M7. Brightest-pixel selection centroiding (Basden)
**Math.** Keep the `n_b` brightest pixels in the window, subtract the `n_b`-th brightest value, clip negatives, then CoG (a data-adaptive thresholding):
```
thr = sort(I)[−n_b]; I' = clip(I − thr, 0, ∞); CoG(I')
```
aotools `brightest_pixel` does exactly this; `n_b ≈ round(threshold·W·H)`.
**Complexity.** Partial-sort/selection (O(N² log N²) or O(N²) with a selection algorithm) + CoG. **Slightly heavier than TCoG** but still per-sub-aperture O(1)-ish; the sort is on a tiny window.
**Sub-pixel accuracy / noise.** Robust to read noise and background level because it adapts to the spot's own brightest pixels; near-optimum `n_b≈40` over a wide range of flux, read-noise, spot size (Basden & Myers). Good for **elongated LGS spots**.
**Bias.** Similar threshold-style bias; depends on `n_b`.
**C/real-time.** Good; the selection is cheap on small windows. A solid alternative to a fixed threshold when spot brightness varies frame-to-frame.

### M8. Parabolic / quadratic 3-point interpolation (peak refinement)
**Math.** Fit a parabola to the peak pixel and its two neighbours per axis. With samples `α,β,γ` (left, peak, right):
```
p = ½ · (α − γ)/(α − 2β + γ) ∈ [−½, ½]      (sub-pixel offset from peak pixel)
peak value ≈ β − ¼(α−γ)p
```
**Complexity.** **O(1), a handful of flops** after a peak find. Per axis independent.
**Sub-pixel accuracy.** Crude (0.05–0.1 px typical) but extremely cheap; *exact* only for a Gaussian sampled on a dB/log scale (DSP result). Mostly used as the sub-pixel step **on a correlation surface**, not on raw spots.
**Noise.** Uses only 3 samples ⇒ throws away the rest of the spot ⇒ noisy; sensitive to the peak-pixel choice.
**Bias.** **Heavy pixel/peak-locking bias** — estimates pull toward integer pixel positions; sinusoidal anti-symmetric bias, period 1 px (peak-locking paper, arXiv:1801.06836). Needs the anti-symmetry correction.
**C/real-time.** Very fast; use **only** as the interpolation kernel for correlation/template methods, with peak-locking correction.

### M9. Gaussian fitting (least-squares PSF fit)
**Math.** Fit `I(i,j) ≈ A·exp(−((i−x₀)²+(j−y₀)²)/2σ²)+B` by nonlinear least squares (Levenberg–Marquardt) or via log-linearization (Caruana/iterated). Estimate `(x₀,y₀)`.
**Complexity.** **Iterative, O(iters·N²) with matrix solves** — by far the most expensive classical method per sub-aperture. Not O(1).
**Sub-pixel accuracy.** Highest among classical methods when the spot truly is Gaussian and SNR is decent; centroid shift grows more slowly than CoG as image quality degrades (star-tracker literature). Sub-0.05 px achievable.
**Noise.** Good (uses all pixels, model-based), but degrades and can diverge at low SNR / non-Gaussian spots; sensitive to initial guess.
**Bias.** Low if the model matches; **model mismatch** (diffraction rings, elongation, truncation) introduces bias. Some pixel-locking via the discrete grid (in the peak-locking comparison set).
**C/real-time.** **Poor** for a 10 ms loop with hundreds of sub-apertures (nonlinear solve per spot). Reserve for offline reference calibration / accuracy ground-truth, not the live loop.

### M10. Correlation / cross-correlation centroiding (and matched filter)
**Math.** Correlate the sub-image with a reference/template `R`, then locate the correlation peak sub-pixel:
```
C(u,v) = Σ_ij I(i,j)·R(i−u,j−v)       (or via FFT: C = IFFT( FFT(I)·conj(FFT(R)) ))
shift = subpixel_peak(C)   via TCoG / parabola / Gaussian fit on C
```
aotools `cross_correlate` (FFT-based) + `correlation_centroid` (thresholded CoG on the correlation surface).
**Complexity.** Direct: O(N²·M²); FFT: O(N² log N) per sub-aperture — **heavier than CoG**, but tractable for small N. Template-as-reference correlation reuses one FFT of R.
**Sub-pixel accuracy.** Excellent and robust for **extended / structured / elongated spots** (LGS, solar scene-based SH) where CoG fails; equivalent to matched filtering. Reference can be updated in real time (MNRAS 439, 968).
**Noise.** Strong noise rejection outside the spot core (correlation suppresses uncorrelated noise). Similarity-function choice matters: ADF, ADF², SDF, CCF compared in the literature.
**Bias.** The **sub-pixel peak step inherits peak-locking** (the same parabola/CoG biases) — needs anti-symmetry correction (≈7× bias reduction to <0.02 px, arXiv:1801.06836). Gradient cross-correlation reduces this for scene-based sensing (Opt. Express 26, 17549).
**C/real-time.** Moderate. **Overkill for compact point-source spots** (PS9's likely case). Keep as a **fallback** if the supplied spots are extended/elongated, or for scene-based robustness.

### M11. Maximum-likelihood (ML) spot-position estimation
**Math.** Maximize the data likelihood under the true noise model. Poisson:
```
ln Pr(g|θ) = Σ_m [ −ḡ_m(θ) + g_m ln ḡ_m(θ) − ln g_m! ]
```
With nuisance brightness `I₀` and background `b`:
```
ln Pr(g|τ,I₀,b) = −I₀ Σ f_m(τ) − M b + Σ g_m ln[ I₀ f_m(τ) + b ]
```
**Complexity.** Search/optimization over τ (multigrid/simplex), **iterative**; with precomputed **mean detector response functions (MDRFs)** stored as LUTs it can be made real-time-ish, but it's still the heaviest principled estimator.
**Sub-pixel accuracy.** **Hits the Cramér–Rao bound**; ~doubles dynamic range vs CoG for quad-cell, "nearly unbiased" where CoG is non-linear, up to ~4× residual-wavefront-error reduction vs tilt-then-reconstruct (Barrett et al., PMC2581470).
**Noise.** Optimal by construction (it *is* the noise model).
**Bias.** Minimal (unbiased asymptotically).
**C/real-time.** **Not for the live 10 ms loop** unless heavily LUT-accelerated. Excellent as the **gold-standard reference** to validate the fast estimator against in simulation.

### M12. Template matching (precomputed shifted templates / LUT correlation)
**Math.** Precompute a bank of spot templates at sub-pixel shifts `{R_δ}`; pick the `δ` maximizing match (cross-correlation or min SSD), optionally interpolate between best matches. A discretized ML/correlation hybrid.
**Complexity.** O(K·N²) for K templates (or O(N²) per template with early-out). **Fully precomputable templates** → table-driven, branch-predictable; tunable accuracy/speed via K.
**Sub-pixel accuracy.** Set by template grid density; can be very high with fine `δ` grid.
**Noise/bias.** Inherits matched-filter noise rejection; bias controlled by template fidelity and grid spacing.
**C/real-time.** Good if K is small and templates are cache-resident; a practical way to get correlation-like robustness with LUT speed. Niche for PS9 (compact spots) but a clean O(1)-with-precompute option.

### M13. Deep-learning / CNN centroiding (modern/ML)
**Math.** A CNN (e.g. ResNet variant, lightweight object-detector, or event-based EBWFNet) maps the raw sub-image (or whole frame) → centroid(s) / slopes / even directly to Zernike coefficients, learned from labelled/simulated data.
**Complexity.** Training offline; inference is a forward pass — fast on GPU, but a **dependency-heavy, hard-to-certify** component for an embedded C loop. Reported 300-400% reconstruction-time reduction and 315% dynamic-range increase vs traditional pipelines (but on GPU/python).
**Sub-pixel accuracy.** Strong in **extreme regimes**: very low SNR, overlapping/missing spots, large aberration, spot–subaperture mismatch (Opt. Express 26, 31675 ANN method; ScienceDirect lightweight detector; Nature Sci.Rep. 2024).
**Noise/bias.** Learns the noise statistics; risk of **domain shift** (lab vs training distribution) and unmodeled bias.
**C/real-time.** **Out of scope for the PS9 real-time C loop.** Possible offline use: generate ground-truth labels, or post-hoc robustness for pathological frames. Not recommended as the primary, given the speed/efficiency judging criterion and C-language steer.

### M14 (bonus). Global spot-matching / Hausdorff-PSO (very large dynamic range)
**Math.** Treat spot↔reference assignment as global optimization: parameterize the wavefront by ~15 Zernike coeffs, minimize the **Hausdorff distance** between detected and predicted spot sets via Particle Swarm Optimization, with a one-to-one penalty (Light: Adv. Manuf. 2024.007).
**Complexity.** ~8.8 ms/iteration, ~100 iters → **seconds/frame**. Not real-time, but solves extreme cross-talk (spots crossing many cells, 50% missing spots, ~14–24× dynamic-range gain).
**Use for PS9.** Only relevant if lab turbulence is strong enough to scramble spot↔lenslet correspondence. For Fried-grid lab data this is almost certainly unnecessary; noted as the heavy-artillery option for spot-wandering/cross-talk.

---

## 4. Spot detection & sub-aperture grid definition (before any centroiding)

1. **Grid model.** The MLA pitch + detector pixel pitch + focal length define a *nominal* lenslet grid. Compute the expected sub-aperture box centers from `MLA_pitch / p_pix` (pixels per lenslet) across the pupil. For a Fried geometry, lenslet centers sit on a regular square grid aligned to (and offset by half a pitch from) the DM actuator grid — keep this alignment explicit because the reconstructor needs it.
2. **Registration / calibration of the grid.** Acquire a **flat-wavefront (reference) frame** (collimated beam, no turbulence). Detect the bright spot in each cell, CoG it → these are the **reference centroids `x_ref,k`** and define the true cell origins. Build the per-lenslet window list (origin, size) from these. Registration corrects for MLA-to-detector rotation/offset/magnification.
3. **Active sub-aperture mask.** Rank cells by **flux = Σ pixel counts in cell**; keep only cells above a fraction (HCIPy uses ~50-75% of the unobscured-cell flux) → excludes under-illuminated edge/pupil-boundary lenslets that would inject garbage slopes. Store the mask once.
4. **Validity gate per frame.** Require ≥ ~3 pixels above the noise threshold in a cell before trusting its centroid (SHARPEST rule); otherwise flag/hold-last/interpolate. Cheap and prevents NaNs from empty cells.

---

## 5. Reference spot calibration (flat wavefront)

- The slope is **always** `(centroid − reference)`, never the raw centroid. Reference centroids come from the flat-wavefront frame (Section 4.2), averaged over many flat frames to beat down noise (HCIPy stores `slopes_ref`; SHARPEST CoGs the averaged background-subtracted flat).
- Calibrate with the **same centroiding algorithm and same window/threshold** you'll use live, so that algorithmic bias (WCoG pull, threshold bias, pixel-locking) is **common-mode and cancels** in the subtraction. This is the cheapest, most effective bias killer available — exploit it.
- Store `x_ref,k, y_ref,k` per lenslet in a table.

---

## 6. Thresholding strategies (decisive for read-noise suppression)

- **Fixed fraction of max:** `T·I_max`, T≈5-10%. Simple; flux-dependent; default in many tools.
- **Noise-floor (recommended):** `m·σ_read` with m≈3-5 (SHARPEST: 1.3-1.5σ plus the ≥3-pixel rule). Ties the threshold to the camera's measured read noise → stable across flux.
- **Adaptive / dynamic windowing:** adapt threshold per cell and float the window to the spot (Opt. Lett./Appl. Opt. 48, 6088 adaptive-thresholding + dynamic-windowing) — robust against uneven illumination, diffraction, source instability, and spot-to-cell-center deviation; increases dynamic range.
- **Brightest-pixel (data-adaptive):** keep top-`n_b` pixels (M7) — automatically tracks per-frame brightness.
- Always combine threshold with a **tight window** so the `N⁴` read-noise term (Section 2.1) never blows up.

---

## 7. Spot wandering, cross-talk, dynamic range

- **Lenslet-bound vs optical dynamic range.** The usual definition binds pixels permanently to a lenslet; a spot may not leave its cell. "Optical dynamic range" (avoid *image* overlap, not cell overlap) is larger by ~(#lenslets across pupil) (Opt. Express 29, 8417). Lab turbulence rarely needs the optical regime.
- **Cross-talk** = a spot drifting into a neighbour's cell, corrupting both centroids. Mitigations, cheapest first: (a) **tight per-cell window + threshold** (zeros the intruding tail); (b) **floating/windowed CoG** (M5) tracks the spot, tolerant to boundary crossing; (c) **image-segmentation + neighbouring-region search** / sorting methods (ScienceDirect S0030401819304602; Appl. Opt. 64, 10462) to re-assign spots; (d) **global Hausdorff-PSO matching** (M14) for the extreme case. For a Fried-grid lab dataset, (a)+(b) suffice.

---

## 8. Bias sources and their fixes (do not skip — this sets the accuracy floor)

| Bias source | Which methods | Fix |
|---|---|---|
| **Finite-FOV / truncation gain (<1)** | CoG, TCoG | Calibrate a per-cell gain; or window-floating (M5) |
| **Threshold bias / non-linearity** | TCoG, brightest-pixel | Same algo at calibration (common-mode cancel); pick T at noise floor |
| **Weight-center pull** | WCoG | Gain factor γ + empirical calibration curve; or IWCoG (M4) |
| **Pixel/peak-locking (period-1px, anti-symmetric)** | parabola, Gaussian fit, correlation-peak, CoG (mild) | Exploit anti-symmetry of bias vs sub-pixel position → ~7× reduction, <0.02 px (arXiv:1801.06836); or build a **bias LUT** indexed by intra-pixel position |
| **Non-uniform lenslet illumination** | all (intensity-gradient across cell) | Subtract intensity gradient / normalize per cell before centroiding (PMC7535117) |
| **Quad-cell cubic non-linearity & gain drift** | quad-cell | Centroid-gain tracking via slope discrepancy; avoid QC for PS9 |

**Two precomputed O(1) bias killers to implement:** (1) common-mode cancellation via same-algorithm reference calibration; (2) a small **intra-pixel bias correction LUT** (measured once from simulated/known shifts) added after centroiding. Both are table lookups — zero hot-loop cost.

---

## 9. Detector / `.bmp` data handling and preprocessing

1. **Read the BMP.** 8-bit (or 24-bit grayscale-encoded) bottom-up rows, 4-byte row padding, BGR order if color. Convert to a single intensity plane (`float` or `uint16` working buffer). Watch the row-order flip and padding when indexing pixels — an off-by-one here silently biases every centroid.
2. **Bias/dark subtraction.** Subtract an averaged dark frame (camera capped, same exposure) to remove bias + dark current + fixed-pattern. Without it, centroid and intensity are wrong (SHARPEST). 
3. **Flat-fielding.** Divide by a normalized flat to correct pixel-to-pixel gain / vignetting / lenslet transmission variations. Per-cell intensity normalization + intensity-gradient removal reduces illumination-induced centroid bias.
4. **Background subtraction.** Per-frame or rolling background (median of empty regions) removes residual stray light.
5. **Threshold + window** (Sections 4, 6) → centroid.
Pipeline order: **BMP→intensity → −dark → /flat → −background → mask/window/threshold → centroid → −reference → ×(p_pix/f_MLA) → slope.**

---

## 10. What enables O(1) / LUT precomputation (real-time C checklist)

- **Per-lenslet window geometry** (origin, size) — precomputed from calibration → array of structs.
- **Reference centroids** `x_ref,k,y_ref,k` — table.
- **Active sub-aperture mask** — bitfield.
- **Weighting function** `W(i,j)` per cell (or one shared if spots identical) — precomputed LUT; WCoG becomes CoG-with-weights, no exp() at runtime.
- **WCoG gain γ** and **slope scale** `p_pix/f_MLA` — scalars per axis.
- **Intra-pixel bias-correction LUT** — small 1-D/2-D table.
- **Running-sum / streaming accumulators** for CoG (M5) — single raster pass, no frame buffer, FPGA-portable.
- **Index tables** `i·W`, `j·W` precomputed so the moment is just `Σ (iW)·I` — turns each moment into a dot product (SIMD/`fma`).
With all of the above, the per-sub-aperture cost is a fixed handful of fused multiply-adds + 2 divides + 2 table lookups — comfortably inside the 10 ms budget for hundreds of sub-apertures, single-threaded, before any SIMD/threading.

---

## 11. Comparison table

Accuracy/noise/bias are relative ranks for **compact point-source spots at lab (medium-high) SNR**. "O(1)?" = constant work per sub-aperture (fixed, data-independent).

| # | Method | Per-subap complexity | O(1)? | Sub-pixel accuracy | Photon-noise | Read-noise | Bias | Real-time C fit | PS9 verdict |
|---|---|---|---|---|---|---|---|---|---|
| M1 | Plain CoG | ~3N² MAC, 1 div | ✅ | Good (tight win) | Near-floor | **Poor (N⁴)** | Truncation gain | ★★★★★ | Baseline kernel |
| M2 | Thresholded CoG | CoG + clip | ✅ | Good→V.Good | Good | **Good** | Threshold bias | ★★★★★ | **Core** |
| M3 | Weighted CoG | CoG + weight LUT | ✅ | **V.Good (ML-opt)** | **V.Good** | **V.Good** | Weight pull (calib.) | ★★★★☆ | **Core** |
| M4 | Iter. Weighted CoG | n_iter×WCoG | ⚠️ (n_iter) | **Excellent (lo-SNR/big shift)** | Excellent | Excellent | Low | ★★★☆☆ | Optional/offline |
| M5 | Windowed/stream CoG | 1 raster pass | ✅ | V.Good | Good | **V.Good** | Low (no edge) | ★★★★★ | **Core / FPGA path** |
| M6 | Quad-cell | 4 sums, 2 div | ✅ | Tiny lin. range | OK | OK | **Cubic + gain drift** | ★★★★★(speed) | Avoid (info loss) |
| M7 | Brightest-pixel | sort(N²)+CoG | ~✅ | V.Good | Good | **Good** | Threshold-like | ★★★★☆ | Good alt. |
| M8 | Parabola 3-pt | ~10 flops | ✅ | Crude | Poor (3 samp) | Poor | **Peak-locking** | ★★★★★ | Only for corr. peak |
| M9 | Gaussian fit | iter LSQ + solve | ❌ | **Highest (if Gaussian)** | V.Good | V.Good | Model mismatch | ★☆☆☆☆ | Offline ref only |
| M10 | Correlation/matched | O(N²logN) FFT | ❌ | Excellent (extended) | Excellent | Excellent | Peak-locking | ★★★☆☆ | Fallback (extended spots) |
| M11 | Maximum-likelihood | iter search (+LUT) | ❌ | **CRLB-optimal** | Optimal | Optimal | Minimal | ★★☆☆☆ | Gold-standard ref |
| M12 | Template matching | K·N² (precomp) | ~✅ | Tunable high | Excellent | Excellent | Template fidelity | ★★★☆☆ | Niche LUT option |
| M13 | CNN / deep learning | NN forward pass | ❌(GPU) | Excellent (extreme) | Learned | Learned | Domain shift | ★☆☆☆☆ | Offline only |
| M14 | Hausdorff-PSO global | ~sec/frame | ❌ | High (huge DR) | — | — | Low | ☆☆☆☆☆ | Extreme cross-talk only |

---

## 12. Recommended approach for PS9

**Context that drives the choice:** lab-simulated (repeatable, controllable) turbulence; a real point-source beacon on a science-grade camera (so spots are compact and reasonably bright → medium-high SNR, not the photon-starved LGS regime); a fixed Fried-geometry lenslet grid (so spot↔lenslet correspondence is stable, excursions modest); and a hard real-time + computational-efficiency judging criterion steering toward C and O(1) kernels.

### Primary algorithm: **Thresholded + Windowed Weighted CoG (TWCoG)** — O(1)/sub-aperture
Per sub-aperture, per frame:
1. Slice the **precomputed per-lenslet window** (from flat-wavefront calibration).
2. Apply **noise-floor threshold** `I_T = max(T·I_max, m·σ_read)`, m≈3-5; clip negatives. Enforce the **≥3-pixels-above-threshold validity gate**.
3. Compute **weighted first moments** using a **precomputed Gaussian weight LUT** centered on the reference (FWHM ≈ spot FWHM ≈ 4-4.5 px): `x̂ = Σ(i·W·I)/Σ(W·I)` via precomputed `i·W`, `j·W` index tables (each moment = one dot product, `fma`-friendly).
4. Apply the **WCoG gain γ** (precomputed) and the **intra-pixel bias-correction LUT**.
5. Subtract the **reference centroid** (same algorithm/threshold/window used in calibration → common-mode bias cancels).
6. Convert to slope: `s = (centroid − ref)·p_pix / f_MLA`.

This sits within a constant flop budget per sub-aperture, has no runtime transcendentals, no data-dependent branching beyond the clip, and reaches near-Cramér-Rao accuracy at lab SNR. It is the documented best accuracy/noise trade among the cheap estimators (Thomas/MNRAS; Vyas Gaussian-pattern-matching), with the WCoG bias neutralized by calibration + LUT.

### Keep alongside (cheap insurance):
- **Plain CoG** as the inner kernel and an always-available baseline/sanity estimator (it *is* TWCoG with W=1, T=0). Use it to validate TWCoG and to bootstrap the reference-spot detection.
- **Windowed/floating-window CoG (M5)** logic for any cells whose spot drifts toward the boundary — preserves dynamic range and kills cross-talk without extra cost; also the clean migration path to FPGA/streaming if you push for hardware speed.
- **Brightest-pixel (M7)** as a drop-in if per-frame brightness varies a lot (auto-tracks the threshold).

### Use only if the data demands it:
- **Correlation / matched-filter (M10)** + parabola/CoG sub-pixel **with peak-locking anti-symmetry correction** — switch to this *only* if the supplied spots turn out extended/elongated/structured (then CoG-family bias explodes and correlation wins).
- **IWCoG (M4)** with a **fixed 2-iteration cap** if large aberrations push spots far from references and you can afford 2× the WCoG cost.

### Offline / validation only (never in the 10 ms loop):
- **Maximum-likelihood (M11)** and **Gaussian fitting (M9)** as **ground-truth references** to benchmark TWCoG against on simulated frames with known injected slopes (verify you're within a small factor of the CRLB).
- **CNN/ML (M13)** — skip for the real-time loop; optional offline robustness experiment for pathological frames only.

### One-time calibration to implement (sets the accuracy floor):
Flat-wavefront reference centroids (averaged) · per-lenslet windows · active-subaperture flux mask · dark frame · flat field · WCoG weight LUT + gain γ · intra-pixel bias LUT · slope scale `p_pix/f_MLA`. All are tables/scalars → the live loop is pure O(1) table-driven arithmetic.

**Bottom line:** TWCoG (thresholded + windowed + Gaussian-weighted CoG) with full precomputation and reference-subtraction bias cancellation is the right primary estimator for PS9 — it is the maximum-likelihood spot-position estimator under Gaussian noise, runs O(1) per sub-aperture in tight C, and the entire methods landscape (quad-cell, parabola, Gaussian fit, correlation, ML, CNN) either reduces to it, refines it, or is too heavy for the loop. Centroiding accuracy dominates final wavefront error, so the calibration/bias-LUT work (cheap at runtime) is where the real accuracy is won.

---

## 13. Sources

**Core centroiding comparisons & noise theory**
- Thomas, Fusco, Tokovinin, et al., "Comparison of centroid computation algorithms in a Shack–Hartmann sensor," *MNRAS* 371, 323 (2006). https://academic.oup.com/mnras/article/371/1/323/980402
- Thomas et al., "Optimized centroid computing in a Shack-Hartmann sensor," NOIRLab archive PDF. https://noirlab.edu/science/sites/default/files/media/archives/pages/5490-123-1-en.pdf
- "Study of centroiding algorithms to optimize Shack-Hartmann…," AO4ELT (2010). https://ao4elt.edpsciences.org/articles/ao4elt/pdf/2010/01/ao4elt_05004.pdf
- "Performance of centroiding algorithms at low light level conditions in adaptive optics," arXiv:1001.1503. https://arxiv.org/pdf/1001.1503
- "Optimization of Existing Centroiding Algorithms for Shack Hartmann Sensor," arXiv:0908.4328. https://arxiv.org/pdf/0908.4328
- "Measurement Error of Shack-Hartmann Wavefront Sensor," IntechOpen (centroid error variance formulas, wavefront-error propagation). https://www.intechopen.com/chapters/26717

**Weighted / iteratively weighted / Gaussian pattern matching**
- Baker & Moallem, "Iteratively weighted centroiding for Shack-Hartmann wave-front sensors," *Opt. Express* 15, 5147 (2007). https://opg.optica.org/oe/abstract.cfm?uri=OE-15-8-5147 ; OSTI UCRL-JRNL-229735 https://www.osti.gov/biblio/908382
- Akondi, Roopashree, Prasad, "Improved iteratively weighted centroiding for accurate spot detection in LGS-based SH sensor," SPIE 7588 (2010). https://www.spiedigitallibrary.org/conference-proceedings-of-spie/7588/758806/
- Vyas, Roopashree, Prasad, "Centroid Detection by Gaussian Pattern Matching in Adaptive Optics," arXiv:0910.3386. https://arxiv.org/pdf/0910.3386
- "Laser guide stars for ELTs: efficient SH-WFS design using the weighted centre-of-gravity algorithm," *MNRAS* 396, 1513 (2009). https://academic.oup.com/mnras/article/396/3/1513/1746680 ; arXiv:0903.4165 https://arxiv.org/pdf/0903.4165

**Quad-cell, maximum-likelihood, brightest-pixel**
- Barrett, Myers, et al., "Maximum-likelihood methods in wavefront sensing: stochastic models and likelihood functions," PMC2581470. https://pmc.ncbi.nlm.nih.gov/articles/PMC2581470/
- "Measuring the centroid gain of a Shack–Hartmann quad-cell WFS by using slope discrepancy," ResearchGate 7626001. https://www.researchgate.net/publication/7626001
- Basden & Myers, "Wavefront sensing with a brightest pixel selection algorithm," ResearchGate 228405312. https://www.researchgate.net/publication/228405312
- Tokovinin, "AO tutorial 3: wave-front sensors." https://www.ctio.noirlab.edu/~atokovin/tutorial/part3/wfs.html

**Correlation / matched filter / peak-locking**
- Ellerbroek et al., "Peak-locking centroid bias in Shack-Hartmann wavefront sensing," *MNRAS* 476, 300 (2018); arXiv:1801.06836. https://arxiv.org/pdf/1801.06836
- "Gradient cross-correlation algorithm for scene-based Shack-Hartmann wavefront sensing," *Opt. Express* 26, 17549 (2018). https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-13-17549
- "Improvement of correlation-based centroiding methods for point source SH-WFS," *Optik* (2017). https://www.sciencedirect.com/science/article/abs/pii/S0030401817310775
- "Real-time correlation reference update for astronomical adaptive optics," *MNRAS* 439, 968 (2014). https://academic.oup.com/mnras/article/439/1/968/1749171
- Quadratic peak interpolation formula (Smith, *SASP*). https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html

**Windowing / dynamic range / cross-talk / spot matching**
- "Adaptive thresholding and dynamic windowing method for automatic centroid detection of digital SH-WFS," *Appl. Opt.* 48, 6088 (2009). https://opg.optica.org/ao/abstract.cfm?uri=ao-48-32-6088
- "Shack-Hartmann wavefront sensor optical dynamic range," *Opt. Express* 29, 8417 (2021). https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-6-8417
- "Large dynamic range SH-WFS based on adaptive spot matching" (Hausdorff + PSO), *Light: Adv. Manuf.* (2024). https://www.light-am.com/en/article/doi/10.37188/lam.2024.007
- "Large dynamic range SH wavefront measurement based on image segmentation and a neighbouring-region search algorithm," *Optik* (2019). https://www.sciencedirect.com/science/article/abs/pii/S0030401819304602

**FPGA / stream processing (O(1) hardware path)**
- "FPGA Implementation of SH Wavefront Sensing Using Stream-Based Center of Gravity," *Electronics* 12(7):1714 (2023). https://www.mdpi.com/2079-9292/12/7/1714
- "Centroid estimation for a Shack–Hartmann WFS based on stream processing," *Appl. Opt.* (2017), PMID 29047936. https://www.ncbi.nlm.nih.gov/pubmed/29047936

**Modern / ML**
- "Centroid computation for SH-WFS in extreme situations based on artificial neural networks," *Opt. Express* 26, 31675 (2018). https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-24-31675
- "A highly adaptive centroid positioning method for SH-WFS based on lightweight object detection network," *Optik* (2025). https://www.sciencedirect.com/science/article/abs/pii/S0030399225009053
- "Convolutional neural network for improved event-based SH wavefront reconstruction (EBWFNet)," *Appl. Opt.* (2024), PMID 38856590. https://pubmed.ncbi.nlm.nih.gov/38856590/
- "Experimental wavefront sensing techniques based on deep learning using a Hartmann-Shack sensor," *Sci. Rep.* (2024). https://www.nature.com/articles/s41598-024-80615-8

**Libraries / tooling**
- AOtools centroiders source (centre_of_gravity, brightest_pixel, cross_correlate, correlation_centroid, quadCell). https://aotools.readthedocs.io/en/v1.0/_modules/aotools/image_processing/centroiders.html ; image-processing docs https://aotools.readthedocs.io/en/latest/image_processing.html ; AOtools paper arXiv:1910.04414 https://arxiv.org/pdf/1910.04414
- HCIPy SH-WFS tutorial (estimator, reference slopes, subaperture flux mask). https://docs.hcipy.org/dev/tutorials/ShackHartmannWFS/ShackHartmannWFS.html
- photutils centroids (centroid_com, centroid_quadratic, centroid_2dg). https://photutils.readthedocs.io/en/stable/user_guide/centroids.html
- SH-WFS operating principle (spot displacement ∝ integrated wavefront gradient). https://en.wikipedia.org/wiki/Shack%E2%80%93Hartmann_wavefront_sensor

**Preprocessing / illumination bias / Fried geometry**
- "Centroid error due to non-uniform lenslet illumination in the SH-WFS," PMC7535117. https://pmc.ncbi.nlm.nih.gov/articles/PMC7535117/
- SHARPEST (dark/background subtraction, 5×5 window, 1.3-1.5σ threshold, ≥3-pixel rule), arXiv:2310.09564. https://arxiv.org/pdf/2310.09564
- "Wavefront Reconstruction of Shack-Hartmann with Under-Sampling of Sub-Apertures" (Fried-grid / under-sampling), *Photonics* 10(1):65 (2023). https://www.mdpi.com/2304-6732/10/1/65
