# Turbulence Characterization from SH-WFS Time-Series: Fried Parameter r0 and Coherence Time tau0

**Project:** ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 9 (Shack-Hartmann WFS reconstruction & turbulence characterization)
**Scope of this document:** Deriving the spatial turbulence strength (Fried parameter **r0**) and the temporal turbulence strength (coherence time **tau0**), plus the supporting parameters (outer scale L0, seeing, Greenwood/Tyler frequencies, isoplanatic angle, Strehl, scintillation) from a time-series of reconstructed wavefronts / Zernike coefficients / raw SH-WFS slopes.
**Core deliverable:** PS9 explicitly requires r0 and tau0 to be derived "from the same data." The robustness story for the hackathon is to compute **>=7 independent r0 estimators** and **>=5 independent tau0 estimators** and show they agree (cross-validation). This document gives the equations, data requirements, accuracy and cross-checks for each.

> Notation: `D` = pupil (beam) diameter, `d` = sub-aperture (lenslet) diameter projected on the pupil, `lambda` = wavelength, `k = 2*pi/lambda`, `Cn2(h)` = refractive-index structure constant profile, `phi` = wavefront phase (rad), `W` = optical path (length units), `s` = SH slope (angle of arrival), `a_i`/`b_i` = true/reconstructed Zernike coefficients, `Z_i` = Zernike mode (Noll index `i`), `v` = wind speed, `zeta` = zenith angle. In the lab (PS9) `zeta = 0`, `sec(zeta) = 1`, and the turbulence is a small number of discrete phase screens rather than a vertical Cn2(h) profile — so the **integrated/profile forms collapse to single-screen forms**, which is convenient.

---

## 0. Theoretical foundation (read this first)

### 0.1 Kolmogorov power spectrum of turbulence
The refractive-index fluctuations in the inertial subrange (between inner scale `l0` ~ mm and outer scale `L0` ~ tens of m) follow Kolmogorov statistics. The 3-D power spectrum of refractive index is

```
Phi_n(kappa) = 0.033 * Cn2 * kappa^(-11/3)          (Kolmogorov, l0 << 2*pi/kappa << L0)
```

The corresponding **2-D phase power spectrum** after propagation (the quantity that matters for a WFS) is

```
W_phi(f) = 0.023 * r0^(-5/3) * f^(-11/3)             (Kolmogorov, spatial freq f in cycles/m)
```

The leading constant is `0.0229` (= 0.023) in the cycles-per-metre convention used by aotools/Assémat & Wilson 2006; in radians-per-metre convention it appears as `0.49 r0^-5/3 kappa^-11/3`. (Source: aotools `ft_phase_screen`; Conan/Roddier.)

### 0.2 von Karman power spectrum (finite outer scale L0)
Real turbulence has a finite outer scale `L0`, which flattens the spectrum at low spatial frequency. Replace the pure power law by the **von Karman** form:

```
W_phi(f) = 0.0229 * r0^(-5/3) * ( f^2 + 1/L0^2 )^(-11/6)        (von Karman)
```

A "modified von Karman" spectrum additionally multiplies by `exp(-(f/f_inner)^2)` with `f_inner ~ 5.92/(2*pi*l0)` to model the inner-scale dissipation. Kolmogorov is the limit `L0 -> infinity`. The outer scale is critical for the **low-order modes (tip/tilt, focus)**: it reduces their variance relative to the Kolmogorov prediction and is therefore measurable from the modal variance curve. Typical on-sky `L0 ~ 20-25 m`; in a lab phase plate it is set by the plate's largest correlated structure.
(Sources: aotools turbulence docs; A&A 2014 multiwavelength SH outer-scale; Ziad GSM.)

### 0.3 Definition of the Fried parameter r0
`r0` is the diameter over which the RMS wavefront phase error is ~1 rad. Formally, from the Cn2 profile,

```
r0 = [ 0.423 * k^2 * sec(zeta) * INT Cn2(h) dh ]^(-3/5)      (0.423 = 16*pi^(5/3)*... / ...)
   = 0.185 * lambda^(6/5) * [ sec(zeta) * INT Cn2 dh ]^(-3/5)
```

Equivalent form used in aotools (`cn2_to_r0`): `r0 = (0.423 * (2*pi/lambda)^2 * Cn2)^(-3/5)`.
Key scaling: **`r0 ∝ lambda^(6/5)`** — r0 grows with wavelength, so seeing improves toward the IR. (Source: Roddier 1981; aotools `atmos_conversions.py`; Claire Max AO intro.)

### 0.4 Phase structure function (the spatial backbone of every r0 estimator)
The Kolmogorov phase structure function (variance of phase difference at separation `r`) is

```
D_phi(r) = <|phi(x+r) - phi(x)|^2> = 6.88 * (r / r0)^(5/3)        (Kolmogorov)
```

`6.88 = 2 * (24/5 * Gamma(6/5))^(5/6) = 6.883`. This single equation is the definition of r0 in real space: when `r = r0`, `D_phi = 6.88 rad^2`. The **von Karman** structure function is finite at large `r`:

```
D_phi,vK(r) = (1.0299/ ... ) * (L0/r0)^(5/3) * [ ... Bessel/Gamma terms ...]    (saturates at 2*sigma^2 for r >> L0)
```
aotools provides `structure_function_vk(separation, r0, L0)` and `structure_function_kolmogorov`. (Sources: Noll 1976; aotools.)

### 0.5 Total wavefront variance over the aperture
Integrating the Kolmogorov spectrum over a circular pupil of diameter D (piston removed) gives the **total residual phase variance** (Noll 1976):

```
sigma_phi^2 = 1.0299 * (D / r0)^(5/3)        [rad^2]   (piston removed)
```

This is the master normalization for the "modal-variance" family of r0 estimators. Removing more modes leaves smaller residuals (see Noll table below).

---

## 1. FRIED PARAMETER r0 — SEVEN INDEPENDENT ESTIMATORS

### Method R1 — Zernike coefficient variance vs Noll Kolmogorov coefficients (PRIMARY)
**Idea.** Noll (1976) computed, for Kolmogorov turbulence over a circular aperture, the variance that each Zernike mode `i` carries. Each modal variance scales as `(D/r0)^(5/3)`. Measure the variance of each reconstructed Zernike coefficient over the time-series and fit to Noll's coefficients to solve for `D/r0`.

**Equations.**
```
<a_i^2>  =  c_i * (D/r0)^(5/3)        [rad^2 at the sensing wavelength]
=>  (D/r0)^(5/3) = <a_i^2> / c_i   for each mode i, then average / fit.
```
Per-mode Noll Kolmogorov coefficients `c_i` (variance of the i-th Zernike, in `rad^2` per `(D/r0)^(5/3)`):

| Mode (Noll i) | Name | Coefficient c_i |
|---|---|---|
| 2,3 | tip, tilt | 0.448 each |
| 4 | defocus | 0.0232 |
| 5,6 | astigmatism | 0.0232 each |
| 7,8 | coma | 0.00619 each |
| 9,10 | trefoil | 0.00619 each |
| 11 | spherical | 0.00245 |

These follow Noll's general formula (his Eq. for the diagonal Zernike-Kolmogorov covariance). The **residual** after perfectly correcting the first `J` modes, `Delta_J`, obeys:

| Modes removed J | Residual Delta_J [ (D/r0)^(5/3) ] | Meaning |
|---|---|---|
| 1 (piston) | 1.0299 | total variance |
| 2 (tip) | 0.582 | |
| 3 (tip+tilt) | 0.134 | image-motion removed |
| 4 (+defocus) | 0.111 | |
| 5,6 (+astig) | 0.0880 / 0.0648 | |
| 7..10 (+coma,trefoil) | ~0.0587 ... 0.0379 | |
| 11 (+spherical) | 0.0306 | |
| large J | `0.2944 * J^(-sqrt(3)/2) * (D/r0)^(5/3)` | asymptotic |

(Source: Noll, R.J. 1976, "Zernike polynomials and atmospheric turbulence," JOSA 66, 207 — Table IV; reproduced widely.)

**Practical recipe (what NAOMI / MNRAS papers actually do):**
- Reconstruct Zernike coefficients `b_i(t)` for every frame; compute `Var(b_i)` across the time-series.
- **Exclude tip/tilt (i=2,3)** — they are contaminated by telescope/bench vibration and (in the lab) by mount/air-handling motion. Use **mid-order modes, e.g. i = 4..15** (NAOMI) or radial orders giving Noll **i = 7..19** (MNRAS).
- Fit `log Var(b_i) = log[ c_i_vK(r0,L0) + sigma_noise_i^2 ]` over the chosen modes to solve for `r0` (and L0).
- **Iteratively subtract the modal cross-coupling and aliasing bias** `sigma_cc,i^2` (the dominant bias — see §3.3); 2-3 iterations converge to sub-percent.

**Data needed:** time-series of reconstructed Zernike coefficients (needs the reconstruction step first). **Accuracy:** sub-percent on r0 once cross-coupling and noise are removed (MNRAS 2019; A&A NAOMI 2023). **Robustness:** very high — the workhorse; the only method that *also* yields L0. **Con:** depends on reconstruction quality and on correctly modelling the noise/cross-coupling bias.

### Method R2 — SH slope (gradient) variance and the slope structure function
**Idea.** Skip full reconstruction; use the raw centroids. The SH measures local wavefront *gradient* (angle of arrival) per sub-aperture. The variance of these slopes is directly tied to r0 through the structure function differentiated over the sub-aperture.

**Equation (slope-variance r0, the aotools / Saint-Jacques 1998 method).**
```
r0 = f( Var(slopes), lambda, d )      via  sigma_slope^2  proportional to (d/r0)^(5/3) / d^2
```
aotools implements `r0_from_slopes(slopes, wavelength, subapDiam)` and its inverse `slope_variance_from_r0(r0, wavelength, subapDiam)`, using Saint-Jacques (1998, PhD thesis, App. A). Physically the **G-tilt variance over one sub-aperture of diameter d** is

```
sigma_slope^2  ~  0.162 * (lambda / d)^2 * (d / r0)^(5/3) / lambda^2   (G-tilt; angle^2)
            i.e. <alpha^2> = 0.170 * lambda^2 * r0^(-5/3) * d^(-1/3)   (G-tilt one axis)
                 <alpha^2> = 0.184 * lambda^2 * r0^(-5/3) * d^(-1/3)   (Z-tilt one axis)
```
(The 0.170 vs 0.184 distinction is **G-tilt** [gradient/centroid tilt, what a SH actually measures] vs **Z-tilt** [Zernike/wavefront tilt]; Conan/Tyler.)

**Slope structure function** (spatial): build `D_s(r) = <|s(x+r) - s(x)|^2>` from pairs of sub-apertures separated by `r` and fit the Kolmogorov form to extract r0. This is the SLODAR/SHIMM-style approach and also enables Cn2 profiling.

**Data needed:** single frame gives an instantaneous estimate (average over sub-apertures); time-series reduces noise. **Accuracy:** good, but biased by centroid noise (subtract `sigma_noise^2`) and by the finite spot size. **Robustness:** high — independent of the reconstruction algorithm, so it is an excellent **cross-check on R1**. (Sources: aotools `turbulence/slopecovariance` & `r0_from_slopes`; Saint-Jacques 1998; SHIMM arXiv:2303.00153.)

### Method R3 — DIMM: differential tip/tilt variance between two sub-apertures
**Idea.** The DIMM measures the *differential* image motion between two sub-apertures separated by baseline `B`. Differential motion cancels common-mode bench vibration and tracking error (the key DIMM advantage), so it gives a **vibration-immune r0**. With a SH-WFS you simply pick pairs of lenslets as virtual DIMM apertures.

**Equations (Sarazin & Roddier 1990; Tokovinin 2002).** Differential variance of spot motion, longitudinal (parallel to baseline) and transverse:
```
sigma_l^2 = K_l * lambda^2 * r0^(-5/3) * D_sub^(-1/3)
sigma_t^2 = K_t * lambda^2 * r0^(-5/3) * D_sub^(-1/3)
```
with response coefficients (`b = B / D_sub` is the normalized separation):
```
K_l = 0.358 * (1 - 0.541 * b^(-1/3))         (longitudinal)
K_t = 0.358 * (1 - 0.811 * b^(-1/3))         (transverse)
Tokovinin (2002) refinement:
K_l = 0.340 * (1 - 0.570 b^(-1/3) - 0.040 b^(-7/3))
K_t = 0.364 * (1 - 0.798 b^(-1/3) - 0.018 b^(-7/3))
```
Solve for r0 from each:
```
r0 = [ K * lambda^2 * D_sub^(-1/3) / sigma^2 ]^(3/5)
```
Average the longitudinal and transverse estimates (they should agree — internal consistency check).

**Data needed:** time-series of two (or many) sub-aperture centroids. **Accuracy:** the field standard for site seeing; insensitive to static aberrations and tracking. **Robustness:** very high, and **immune to common-mode lab vibration** — extremely valuable as a cross-check in a noisy lab. **Con:** needs >=2 sub-apertures with a useful baseline; short-exposure regime corrections needed if spots are elongated. (Sources: Sarazin & Roddier 1990 A&A 227, 294; Tokovinin 2002 PASP, "From Differential Image Motion to Seeing," IOPscience 10.1086/342683; GDIMM arXiv:1811.07310.)

### Method R4 — Total wavefront phase variance (Noll normalization)
**Idea.** Compute the spatial variance of the reconstructed wavefront over the pupil (piston removed) for each frame, average over time, and invert Noll's master relation.

**Equation.**
```
sigma_phi^2 = 1.0299 * (D/r0)^(5/3)
=>  r0 = D * ( 1.0299 / sigma_phi^2 )^(3/5)
```
If tip/tilt are removed (recommended in the lab), use the tip/tilt-removed coefficient instead:
```
sigma_phi,TT-removed^2 = 0.134 * (D/r0)^(5/3)   =>  r0 = D * (0.134 / sigma^2)^(3/5)
```

**Data needed:** reconstructed phase maps (in rad at lambda). **Accuracy:** moderate — sensitive to reconstruction edge effects, unseen high-order content (fitting error) and noise propagation; tends to under/over-estimate if the pupil sampling is coarse. **Robustness:** simple, fast, good order-of-magnitude cross-check. **Con:** must convert reconstructed OPD to radians (`phi = 2*pi*W/lambda`) and remove piston/tip/tilt consistently. (Source: Noll 1976.)

### Method R5 — Kolmogorov phase structure function fit `D_phi(r) = 6.88 (r/r0)^(5/3)`
**Idea.** From the reconstructed phase maps, compute the empirical structure function `D_phi(r)` by averaging squared phase differences over all point pairs at separation `r`, then fit the 5/3 power law to extract r0 directly.

**Equation.**
```
D_phi(r) = < |phi(x+r) - phi(x)|^2 >   (empirical, average over x and over frames)
Fit:  D_phi(r) = 6.88 * (r/r0)^(5/3)   =>  r0 = r * (6.88 / D_phi(r))^(3/5)
```
Fit in the inertial range only (separations between ~`d` and ~`D/2`). The **slope of log D_phi vs log r should be 5/3 (=1.667)** — checking this slope is itself a turbulence-model validation (Kolmogorov vs non-Kolmogorov). Deviation/flattening at large `r` reveals the **outer scale L0** (use the von Karman structure function fit there).

**Data needed:** reconstructed phase maps; single frame works but time-averaging is far better. **Accuracy:** good and model-transparent; you literally see whether the data is Kolmogorov. **Robustness:** high; **the most direct visual proof** that the reconstruction "conforms to the turbulence characteristics" (an explicit PS9 evaluation criterion). **Con:** edge/aliasing effects at the largest separations. (Sources: Roddier; aotools `structure_function`; Max AO intro.)

### Method R6 — von Karman spectrum fit including OUTER SCALE L0
**Idea.** Generalize R1/R5: fit the measured **modal variance curve** (or the 2-D phase PSD, or the structure function) to the von Karman model with two free parameters `(r0, L0)`. The low-order modes (especially tip/tilt and focus) constrain L0 because a finite L0 suppresses their variance below the Kolmogorov value.

**Equations.** Use the von Karman phase PSD `W_phi(f) = 0.0229 r0^(-5/3) (f^2 + 1/L0^2)^(-11/6)`, or equivalently the von Karman Zernike variances `<a_i^2>_vK(r0, L0)` (Conan 2008; Takato & Yamaguchi 1995). Fit:
```
min over (r0, L0):  SUM_i [ log Var(b_i) - log( <a_i^2>_vK(r0,L0) + sigma_noise_i^2 + sigma_cc_i^2 ) ]^2
```

**Data needed:** time-series of modal coefficients (or slopes). **Accuracy:** r0 sub-percent; L0 reliable only if `D` is an appreciable fraction of `L0` (NAOMI's 1.8 m could *not* constrain L0; the MNRAS 14x14 / larger D could, for L0 = 4-32 m). In the lab L0 is whatever the phase plate imposes — likely measurable. **Robustness:** the canonical way to get L0; reduces to Kolmogorov as `L0 -> inf`. **Con:** L0–r0 degeneracy if the aperture is small; needs good low-order statistics. (Sources: MNRAS 483, 1192 (arXiv:1811.08396); A&A NAOMI 2023 aa46952-23; A&A 2014 aa24476-14.)

### Method R7 — Seeing FWHM relation `epsilon = 0.98 lambda/r0`
**Idea.** If a focal-plane (science camera) image or the SH spot long-exposure size is available, the seeing-limited FWHM gives r0 directly. Even without a science camera, the **long-exposure broadening of individual SH spots** measures the high-order (sub-aperture) seeing.

**Equation.**
```
epsilon_FWHM = 0.98 * lambda / r0            [rad]    (long-exposure seeing)
=>  r0 = 0.98 * lambda / epsilon_FWHM
seeing [arcsec] = (0.98 * lambda / r0) * (180*3600/pi)
```
For finite outer scale the effective FWHM is smaller (Tokovinin 2002 correction with L0). The SH "spot-size seeing" variant (active-optics SH; MNRAS 421, 3019) uses the **growth of spot width** vs the diffraction limit per lenslet.

**Data needed:** a focal-plane long-exposure image, OR statistics of SH spot sizes (single sensor, time-averaged). **Accuracy:** moderate; affected by static optics and detector sampling; the 0.98 constant assumes pure Kolmogorov + infinite L0. **Robustness:** an *independent optical-domain* check that does not use slopes/Zernikes at all — valuable orthogonality. **Con:** needs imaging data / careful PSF calibration. (Sources: aotools `r0_to_seeing`; Dhillon PHY217; Tokovinin 2002.)

---

## 2. COHERENCE TIME tau0 — FIVE+ INDEPENDENT METHODS

`tau0` is the timescale over which the turbulent phase changes by ~1 rad^2; it sets how fast the AO loop (and PS9 reconstruction) must run. The structure-function/AO definition is

```
tau0 = 0.314 * r0 / V_eff       with   V_eff = [ INT Cn2(h) V(h)^(5/3) dh / INT Cn2(h) dh ]^(3/5)
```
In the lab there is one (or few) moving screen(s), so `V_eff` is simply the screen translation speed. The various tau0 methods differ in *how* the timescale is extracted from the data.

### Method T1 — Temporal autocorrelation of Zernike / slope time-series (PRIMARY, direct)
**Idea.** For each mode (or slope), compute the temporal autocorrelation `C_i(tau)` over the frame sequence and read off the decorrelation time. This needs *no* wind model — it measures the timescale directly.

**Equations.**
```
C_i(tau) = < b_i(t) * b_i(t+tau) >_t = C_turb,i(tau) + sigma_noise_i^2 * delta(tau)
```
- The white-noise term sits **only at tau=0**; fitting `C_i(tau)` for small tau>0 with a smooth model and extrapolating to tau=0 **isolates the measurement noise** `sigma_noise_i^2` (this is also how you get the noise term needed in R1/R6 — a neat dual use; A&A NAOMI 2023).
- Define the modal **coherence time as the 1/e point** (or the e-fold of the normalized ACF): `C_i(tau_c)/C_i(0) = 1/e`.
- For an *atmospheric coherence time*, combine modes: the overall tau0 follows from the curvature of the autocorrelation of the full phase, or from the lowest well-measured high-order mode (tip/tilt decorrelate slowest; focus/astig give cleaner tau0 because they are less vibration-contaminated).

**Data needed:** time-series (the whole point of PS9's millisecond cadence). The **frame interval delta_t must be << tau0** (a few ms vs ~10 ms is adequate but marginal — sample as fast as possible). **Accuracy:** good if many independent realizations; the 1/e definition is convention-dependent so quote the definition. **Robustness:** high; the most assumption-free tau0. **Con:** needs long records for low-frequency (large-scale) modes; aliasing if undersampled. (Sources: A&A NAOMI 2023; Conan 1995.)

### Method T2 — Temporal power spectral density (PSD) of modes and Kolmogorov temporal slopes
**Idea.** Take the FFT of each modal (or slope) time-series to get its temporal PSD. Kolmogorov + frozen flow predicts characteristic **power-law slopes** and a **knee/cutoff frequency** proportional to `v/D` (or `v` x spatial frequency). The knee location gives the timescale; the slopes validate the model.

**Equations (Conan, Rousset & Madec 1995).** For a single layer of speed `v`, each Zernike's temporal PSD is flat at low f and breaks to a power law above a cutoff:
```
cutoff frequency:  f_c,i  ~  0.3 * (n_i + 1) * v / D      (increases with radial order n_i)
```
Asymptotic high-frequency slopes:
```
tip/tilt (n=1):     PSD ∝ f^(-11/3)        (high-f)
focus & higher:     PSD ∝ f^(-17/3)        (high-f)
low-frequency end:  PSD ∝ f^(-2/3)         (for tip/tilt, below cutoff)
```
The **Greenwood-related break of the full-phase PSD** scales as `f_G` (next method). Fitting the cutoff `f_c` and knowing `D` yields `v`, hence `tau0 = 0.314 r0/v`.

**Data needed:** evenly-sampled time-series, long enough for frequency resolution `df = 1/T`. **Accuracy:** good; the slopes are a strong turbulence-model check. **Robustness:** high; PSD separates turbulence (low-f, colored) from noise (high-f, white floor) cleanly — also gives a second noise estimate. **Con:** needs many frames; windowing/detrending care. (Sources: Conan et al. 1995 JOSA A 12, 1559; Glindemann AO notes; Roddier et al. 1993 for the 17/3 slope.)

### Method T3 — Greenwood frequency f_G and tau0 = 0.314 r0/v
**Idea.** The Greenwood frequency is the closed-loop bandwidth needed to track the turbulence; it is the natural frequency-domain partner of tau0.

**Equations.**
```
f_G = 2.31 * lambda^(-6/5) * [ sec(zeta) INT Cn2(h) v(h)^(5/3) dh ]^(3/5)
single-layer / lab form:        f_G = 0.426 * v / r0
relation to coherence time:     tau0 = 0.314 * r0 / v = 0.134 / f_G
```
So once you have r0 (from §1) and a wind speed `v` (from T2, T5, or known plate speed), you get f_G and tau0 consistently. Equivalently, the residual error of a first-order AO loop of bandwidth `f_3dB` is `sigma^2 = (f_G/f_3dB)^(5/3)`.

**Data needed:** r0 + v (v from any wind estimator). **Accuracy:** as good as r0 and v. **Robustness:** the standard bridge between spatial (r0) and temporal (tau0) domains; lets you cross-check tau0 from T1 against `0.134/f_G`. **Con:** requires a v estimate. (Sources: Greenwood 1977 JOSA 67, 390; Wikipedia "Greenwood frequency"; Hardy 1998.)

### Method T4 — Temporal structure function (time-domain, reaches 1 rad^2)
**Idea.** The time analogue of the spatial structure function. Build `D_phi(tau) = <|phi(t+tau) - phi(t)|^2>` (for the full phase, or per mode, or for differential piston between sub-apertures) and find where it reaches the defining variance.

**Equations.**
```
D_phi(tau) = < |phi(t+tau) - phi(t)|^2 >
Coherence time t0 :  D_phi(t0) = 1 rad^2   =>   t0 = 2^(-3/5) * tau0 = 0.66 * tau0
Short-lag (frozen flow):  D_phi(tau) ~ (tau / t1)^2 ,  t1 = 0.273 (r0/V2)(d/r0)^(1/6)
AA (angle-of-arrival) coherence time:  D_x|y(tau_a) = D_sat / e
```
By Taylor's frozen-flow hypothesis the **temporal** structure function at lag `tau` equals the **spatial** one at shift `r = v*tau`, i.e. `D_phi(tau) = 6.88 (v*tau/r0)^(5/3)` — so fitting `D_phi(tau)` to a 5/3 law in `tau` directly returns `v/r0`, hence tau0.

**Data needed:** time-series (slopes or modes or differential motion). **Accuracy:** good; consistent with T1 by construction but a different statistic, so a genuine cross-check. **Robustness:** high; the differential-piston version is vibration-immune (like DIMM in time). **Con:** definition (1 rad^2 vs 1/e) must be stated. (Sources: A&A 2007 aa5788-06 "Atmospheric coherence times in interferometry: definition and measurement"; GPI wind paper arXiv:2410.01193.)

### Method T5 — Taylor frozen-flow + wind-speed v from spatio-temporal cross-correlation
**Idea.** Under Taylor's frozen-flow hypothesis the phase screen translates rigidly at velocity `v`. Cross-correlate the slope/phase map at time `t` with the map at `t+delta_t`; the spatial offset of the correlation peak is `v*delta_t`, giving the wind vector directly. Then `tau0 = 0.314 r0/v`.

**Equations.**
```
C(r, tau) = < s(x, t) * s(x+r, t+tau) >    ;    peak at  r_peak = v * tau
=> v = r_peak / tau  (vector, per layer);   tau0 = 0.314 * r0 / |V_eff|
V_eff = [ SUM_layers Cn2_l v_l^(5/3) / SUM Cn2_l ]^(3/5)   (multi-layer weighting)
```
This is the SLODAR / S-SCIDAR / GPI-telemetry method and resolves **multiple layers** if present.

**Data needed:** time-series of slopes (or phase) with enough sub-apertures to see the spatial shift; cadence fast enough that the peak moves < pupil per frame. **Accuracy:** good for a dominant layer; degrades with few sub-apertures or a dominant ground layer (NAOMI 4x4 struggles; MNRAS 14x14 fine). In the lab with a single translating plate this is **clean and direct**. **Robustness:** high; gives `v` independently of T2/T3, closing the loop. **Con:** needs decent spatial sampling; frozen-flow can break down (boiling) — detectable as a decaying, not just shifting, correlation peak. (Sources: GPI wind arXiv:2410.01193; SLODAR; SHIMM profiling arXiv:2603.02817.)

### Method T6 (bonus) — Tyler tracking frequency (tip/tilt timescale)
**Idea.** The Tyler frequency is the bandwidth needed to correct *tip/tilt* specifically (image motion), as opposed to f_G for the full wavefront. Useful because tip/tilt dominate the variance and have their own (slower) timescale.

**Equation (Tyler 1994).**
```
f_T = 0.368 * v * r0^(-1/6) * D^(-5/6)    (one common form)
    = 0.331 * D^(-1/6) * lambda^(-1) * [ INT Cn2 v^2 dh ]^(1/2)   (profile form)
```
Gives a tip/tilt-specific timescale `~ 1/f_T` to compare against T1 restricted to modes 2,3. (Sources: Tyler 1994 JOSA A 11, 358, "Bandwidth considerations for tracking through turbulence"; Hardy 1998.)

---

## 3. SUPPORTING PARAMETERS AND BIASES

### 3.1 Isoplanatic angle theta0
```
theta0 = [ 2.914 * k^2 * sec(zeta)^(8/3) * INT Cn2(h) h^(5/3) dh ]^(-3/5)
       = 0.314 * r0 / h_eff          (single effective layer at height h_eff)
aotools: theta0 = 0.057 * lambda^(6/5) * ( SUM Cn2 * h^(5/3) )^(-3/5)
```
In a lab single-screen setup theta0 is essentially set by the screen-to-pupil geometry; less central for PS9 but cheap to report. (Sources: Roddier; aotools `isoplanaticAngle`.)

### 3.2 Strehl ratio from residual variance (Maréchal / extended Maréchal)
```
S = exp( - sigma_phi_residual^2 )          (Maréchal; valid sigma^2 < ~1-2 rad^2, S > ~0.1)
S = exp( - sigma_phi^2 - sigma_log-amp^2 ) (extended Maréchal, includes scintillation)
```
Lets you predict the Strehl that the derived r0 implies, and the **post-correction residual** `sigma^2 = Delta_J (D/r0)^(5/3)` (Noll table) for a DM correcting `J` modes — directly relevant to PS9's actuator-map deliverable. (Sources: eikonaloptics Maréchal tutorial; Noll 1976.)

### 3.3 The dominant biases you MUST remove (else r0 is wrong)
1. **Measurement (centroid) noise** — adds a white variance to every slope/mode. Remove via the autocorrelation-at-tau=0 trick (T1) or the high-f PSD floor (T2), then subtract `sigma_noise_i^2` before fitting r0. (A&A NAOMI 2023.)
2. **Modal cross-coupling / aliasing** — Zernike-derivative non-orthogonality leaks unmodelled high-order modes into the measured low-order variances; this is the **dominant bias** at radial orders 7-12. Remove iteratively with the cross-talk matrix `C = H+ H_perp` (MNRAS 2019); 2-3 iterations -> sub-percent.
3. **Tip/tilt contamination** — bench/air/mount vibration inflates modes 2,3. **Exclude tip/tilt from the r0 fit**; use DIMM (R3, differential) if you need a vibration-immune r0.
4. **Finite outer scale** — fitting Kolmogorov when L0 is finite biases r0 high; use von Karman (R6) or fit L0.

### 3.4 Scintillation index (amplitude, weak-turbulence)
```
sigma_I^2 = ( <I^2> - <I>^2 ) / <I>^2          (normalized intensity variance per sub-aperture)
weak turbulence:  sigma_I^2 = sigma_Rytov^2 = 1.23 * Cn2 * k^(7/6) * L^(11/6)
```
SH sub-aperture flux fluctuations give a cheap scintillation index; mainly a turbulence-strength sanity check and a flag for non-frozen propagation. (Sources: Andrews & Phillips; SHIMM.)

---

## 4. MASTER TABLE OF ESTIMATORS

| ID | Parameter | Core formula | Data needed | Single frame? | Pros | Cons |
|----|-----------|--------------|-------------|---------------|------|------|
| R1 | r0 (+L0) | `Var(a_i)=c_i (D/r0)^5/3`, fit modes 4-15 | reconstructed Zernikes, time-series | no | gold standard; gives L0; sub-% | needs reconstruction + bias removal |
| R2 | r0 | `r0_from_slopes`, `sigma_slope^2 ∝ (d/r0)^5/3/d^2` | raw centroids/slopes | yes (avg subaps) | reconstruction-free; fast | centroid-noise & spot-size bias |
| R3 | r0 | DIMM: `sigma_l,t^2=K_l,t lambda^2 r0^-5/3 D_sub^-1/3` | 2+ sub-aperture centroids, time-series | no | vibration-immune; field standard | needs baseline; short-exp corr. |
| R4 | r0 | `sigma_phi^2=1.03(D/r0)^5/3` | reconstructed phase maps | yes | trivial, fast | edge/fitting-error/noise sensitive |
| R5 | r0 (+L0) | `D_phi(r)=6.88(r/r0)^5/3` fit | reconstructed phase maps | yes (better w/ time) | model-transparent; validates Kolmogorov | large-r aliasing |
| R6 | r0+L0 | von Karman fit `(f^2+1/L0^2)^-11/6` | modes or slopes, time-series | no | only clean route to L0 | r0–L0 degeneracy if D<<L0 |
| R7 | r0 | `epsilon=0.98 lambda/r0` | focal-plane image / SH spot sizes | yes | orthogonal (image domain) | needs imaging; static-aberration sens. |
| T1 | tau0 | `C_i(tau)`, 1/e of ACF | modal/slope time-series | no | assumption-free; gives noise too | convention-dependent; long records |
| T2 | tau0 (v) | PSD slopes + cutoff `f_c~0.3(n+1)v/D` | time-series (FFT) | no | validates temporal Kolmogorov; noise floor | needs many frames |
| T3 | tau0/f_G | `tau0=0.314 r0/v=0.134/f_G`, `f_G=0.426 v/r0` | r0 + v | n/a | bridges spatial<->temporal | needs v |
| T4 | tau0 | `D_phi(tau)=1 rad^2 -> t0=0.66 tau0` | time-series | no | differential version vibration-immune | definition-dependent |
| T5 | v -> tau0 | cross-corr peak `r=v*tau`; `tau0=0.314 r0/v` | slope maps, time-series | no | direct wind; multi-layer | needs spatial sampling |
| T6 | tip/tilt time | Tyler `f_T=0.368 v r0^-1/6 D^-5/6` | r0 + v | n/a | tip/tilt-specific bandwidth | approximate |
| S1 | theta0 | `theta0=0.314 r0/h` | geometry + r0 | n/a | cheap | minor for lab |
| S2 | Strehl | `S=exp(-sigma^2)` | residual variance | yes | links to DM/actuator deliverable | small-aberration validity |
| S3 | scint. | `sigma_I^2=1.23 Cn2 k^7/6 L^11/6` | sub-aperture flux | yes | frozen-flow / strength flag | weak-turbulence only |

---

## 5. RECOMMENDED MULTI-METHOD APPROACH FOR PS9 (cross-validated robustness)

The grading rewards (a) phase maps that "conform to the turbulence characteristics," (b) statistical turbulence parameters, and (c) speed. The robustness story is **multiple independent estimators that agree**. Concrete pipeline:

### Step A — Pre-processing (shared by everything)
1. Centroid every SH spot, form slopes `s_x, s_y` per sub-aperture per frame (this also feeds reconstruction).
2. Reconstruct the wavefront (zonal Southwell/Fried-geometry **or** modal Zernike) -> `phi(x,y,t)` and `b_i(t)`.
3. **Estimate and store the measurement noise** `sigma_noise^2` from the tau=0 autocorrelation jump (T1) and from the high-frequency PSD floor (T2). Subtract it everywhere before fitting r0/tau0.

### Step B — r0 from >=7 estimators, then reconcile
Compute all of:
- **R1** Zernike-variance fit (modes 4-15, von Karman, iterate out cross-coupling) — **primary**, also yields L0.
- **R2** slope-variance r0 (reconstruction-independent) — checks the reconstruction.
- **R3** DIMM differential tip/tilt from sub-aperture pairs (both longitudinal & transverse) — **vibration-immune anchor**.
- **R4** total phase variance `1.03(D/r0)^5/3` (TT-removed: `0.134`).
- **R5** structure-function fit `6.88(r/r0)^5/3` — **and report the fitted log-log slope** (should be ~1.667) as turbulence-model proof.
- **R6** von Karman `(r0, L0)` joint fit — reports L0 and confirms r0.
- **R7** seeing `0.98 lambda/r0` from SH spot-size growth (or focal image if available).

Reconcile: tabulate the seven r0 values, take a robust central estimate (median), and **report the spread as the systematic uncertainty**. If R3 (vibration-immune) disagrees with R1/R4, suspect tip/tilt contamination; if R5's slope != 5/3 or R6 finds small L0, report non-Kolmogorov / finite-outer-scale and switch the other estimators to von Karman. **Agreement of >=7 estimators within their error bars is the headline robustness claim.**

### Step C — tau0 from >=5 estimators, then reconcile
- **T1** 1/e temporal autocorrelation of mid-order modes (focus/astig — vibration-clean) — **primary**.
- **T2** temporal PSD: fit cutoff `f_c ~ 0.3(n+1)v/D` and confirm the -11/3 (tilt) / -17/3 (higher) slopes -> v -> tau0.
- **T4** temporal structure function `D_phi(tau)`; frozen-flow fit `6.88(v tau/r0)^5/3` -> v/r0; report `t0` where `D=1 rad^2` (`tau0 = t0/0.66`).
- **T5** spatio-temporal cross-correlation peak tracking -> wind vector `v` (and detect boiling vs frozen flow) -> `tau0 = 0.314 r0/v`.
- **T3** consistency bridge: from the r0 of Step B and the v of T2/T5, compute `f_G = 0.426 v/r0` and `tau0 = 0.134/f_G`; it must match T1/T4.
- (**T6** Tyler `f_T` for the tip/tilt-only timescale, optional.)

Reconcile: the **wind speed v** is the common currency — T2, T4, T5 must give the same v; T1 and T4 must give the same tau0; T3 ties tau0 to r0 and v. Report median tau0 and spread. If the lab plate speed is known, it is the ground truth that all of T2/T4/T5 should recover — a powerful validation.

### Step D — Derived / sanity parameters
- L0 from R6 (and the flattening of R5/T-structure functions).
- f_G = 0.426 v/r0; tau0 = 0.134/f_G (T3) — internal closure.
- Strehl `S=exp(-sigma_res^2)` with `sigma_res^2 = Delta_J (D/r0)^5/3` for the DM's correctable J modes (feeds the actuator-map deliverable and lets you predict corrected performance).
- theta0, scintillation index — cheap extras that round out the characterization.

### Why this is robust (the narrative for judges)
- **Three orthogonal data domains** are used: raw slopes (R2, R3, T5), reconstructed phase (R1, R4, R5, R6, T1, T2, T4), and intensity/imaging (R7, S3). A bug in one domain cannot fake agreement across all three.
- **Vibration immunity is built in** via the differential estimators (R3 in space, T4-differential in time), which is essential in a lab with bench/air-handling motion.
- **Model validity is tested, not assumed**: R5's 5/3 slope and T2's -11/3 / -17/3 slopes prove (or disprove) Kolmogorov; R6 quantifies the outer scale; a decaying (not shifting) cross-correlation peak (T5) flags non-frozen boiling.
- **r0 and tau0 are linked through v and f_G** (T3), so the spatial and temporal results must be mutually consistent — a single closed, self-checking system, exactly what PS9 asks for from "the same data."

---

## 6. KEY EQUATIONS QUICK-REFERENCE CARD

```
Kolmogorov phase PSD:        W_phi(f) = 0.0229 r0^-5/3 f^-11/3
von Karman phase PSD:        W_phi(f) = 0.0229 r0^-5/3 (f^2 + 1/L0^2)^-11/6
Structure function (space):  D_phi(r) = 6.88 (r/r0)^5/3
Total variance (pupil):      sigma_phi^2 = 1.0299 (D/r0)^5/3        (TT-removed: 0.134)
Fried parameter:             r0 = [0.423 k^2 sec(zeta) INT Cn2 dh]^-3/5 ,  r0 ∝ lambda^6/5
Zernike modal variance:      Var(a_i) = c_i (D/r0)^5/3   (c_tilt=0.448, c_focus=c_astig=0.0232,...)
Slope variance (G-tilt):     <alpha^2> = 0.170 lambda^2 r0^-5/3 d^-1/3   (Z-tilt: 0.184)
DIMM differential variance:  sigma_l,t^2 = K_l,t lambda^2 r0^-5/3 D_sub^-1/3
                             K_l=0.358(1-0.541 b^-1/3),  K_t=0.358(1-0.811 b^-1/3)   [b=B/D_sub]
Seeing:                      epsilon = 0.98 lambda / r0
Coherence time:              tau0 = 0.314 r0 / V_eff = 0.134 / f_G ;  t0(1 rad^2)=0.66 tau0
Greenwood frequency:         f_G = 0.426 v / r0 = 2.31 lambda^-6/5 [INT Cn2 v^5/3 dh]^3/5
Tyler frequency:             f_T = 0.368 v r0^-1/6 D^-5/6
Isoplanatic angle:           theta0 = 0.314 r0 / h_eff
Temporal PSD slopes:         tilt f^-2/3 (low) -> f^-11/3 (high); higher modes -> f^-17/3
Temporal PSD cutoff:         f_c,i ~ 0.3 (n_i+1) v / D
Effective wind speed:        V_eff = [INT Cn2 V^5/3 dh / INT Cn2 dh]^3/5
Strehl (Marechal):           S = exp(-sigma_phi^2)
Scintillation (weak):        sigma_I^2 = 1.23 Cn2 k^7/6 L^11/6
```

---

## 7. SOURCES (with URLs)

**Foundational theory**
- Noll, R. J. 1976, "Zernike polynomials and atmospheric turbulence," *JOSA* 66, 207 — total variance 1.0299(D/r0)^5/3, residual Delta_J table, modal coefficients. PDF: https://e-l.unifi.it/pluginfile.php/1055871/mod_resource/content/1/Appunti_2020_Lezione%2014_3_NOLL1976.pdf
- Fried, D. L. 1965/1966 — definition of r0 and seeing; foundational.
- Roddier, F. (ed.) 1999, *Adaptive Optics in Astronomy*, Cambridge Univ. Press — r0, tau0, theta0 definitions (Roddier, Gilli & Lund 1982 forms).
- Hardy, J. W. 1998, *Adaptive Optics for Astronomical Telescopes*, Oxford Univ. Press — Greenwood/Tyler, bandwidth.
- Greenwood, D. P. 1977, "Bandwidth specification for adaptive optics systems," *JOSA* 67, 390. Summary: https://en.wikipedia.org/wiki/Greenwood_frequency
- Tyler, G. A. 1994, "Bandwidth considerations for tracking through turbulence," *JOSA A* 11, 358 — Tyler tracking frequency.
- Conan, Rousset & Madec 1995, "Wave-front temporal spectra in high-resolution imaging through turbulence," *JOSA A* 12, 1559 — temporal PSD slopes & cutoffs. https://www.osapublishing.org/abstract.cfm?uri=josaa-12-7-1559

**SH-WFS turbulence-parameter estimation (most directly applicable to PS9)**
- Andrade, P. P. et al. 2019, "Estimation of atmospheric turbulence parameters from Shack-Hartmann WFS measurements," *MNRAS* 483, 1192 — Zernike-variance r0/L0, modal cross-coupling iterative correction. https://academic.oup.com/mnras/article/483/1/1192/5203639 ; arXiv: https://arxiv.org/abs/1811.08396
- "Integrated turbulence parameters' estimation from NAOMI AO telemetry data," *A&A* 2023, aa46952-23 — modes 4-15 fit, noise from autocorrelation, seeing 0.976 lambda/r0, MASS-DIMM comparison. https://www.aanda.org/articles/aa/full_html/2023/10/aa46952-23/aa46952-23.html ; arXiv: https://arxiv.org/pdf/2307.15178
- SHIMM: "SHIMM: A Versatile Seeing Monitor for Astronomy," arXiv:2303.00153 — SH-based r0/seeing, scintillation. https://arxiv.org/pdf/2303.00153
- "Single-star optical turbulence profiling techniques for SHIMM and other Shack-Hartmann instruments," arXiv:2603.02817 — slope structure-function / profiling.
- "Fried Parameter Estimation from Single Wavefront Sensor Image with ANNs," arXiv:2504.17029 — ML alternative for single-frame r0. https://arxiv.org/abs/2504.17029

**DIMM / seeing monitors**
- Sarazin, M. & Roddier, F. 1990, "The ESO differential image motion monitor," *A&A* 227, 294 — DIMM K_l, K_t coefficients.
- Tokovinin, A. 2002, "From Differential Image Motion to Seeing," *PASP* 114, 1156 — refined K_l/K_t, finite-exposure & outer-scale corrections. https://iopscience.iop.org/article/10.1086/342683
- GDIMM: "Turbulence monitoring at the Plateau de Calern with the GDIMM instrument," arXiv:1811.07310 ; generalized DIMM arXiv:1811.07561, arXiv:1904.07093.

**Coherence time / wind**
- "Atmospheric coherence times in interferometry: definition and measurement," *A&A* 2007, aa5788-06 — tau0 = 0.314 r0/V_eff, V_eff weighting, t0 = 0.66 tau0. https://www.aanda.org/articles/aa/full/2007/02/aa5788-06/aa5788-06.right.html ; PDF https://www.aanda.org/articles/aa/pdf/2007/02/aa5788-06.pdf
- "Estimating Atmospheric Wind Speeds From Gemini Planet Imager AO Telemetry," arXiv:2410.01193 — spatio-temporal cross-correlation wind retrieval. https://arxiv.org/pdf/2410.01193

**Outer scale L0**
- "Multiwavelength active-optics Shack-Hartmann sensor for monitoring seeing and turbulence outer scale," *A&A* 2014, aa24476-14. https://www.aanda.org/articles/aa/full_html/2014/12/aa24476-14/aa24476-14.html ; arXiv:1408.4911
- Ziad et al., Generalized Seeing Monitor (GSM) — L0 ~ 20-25 m typical.

**Software / implementations (formulas you can copy)**
- AOtools turbulence docs: https://aotools.readthedocs.io/en/latest/turbulence.html — `r0_from_slopes`, `structure_function_vk`, `phasescreen` (Assémat & Wilson 2006).
- AOtools `atmos_conversions.py`: https://github.com/AOtools/aotools/blob/main/aotools/turbulence/atmos_conversions.py — `cn2_to_r0` (0.423), `r0_to_seeing` (0.98), `coherenceTime` (0.057 lambda^6/5 (SUM Cn2 v^5/3)^-3/5), `isoplanaticAngle` (0.057 lambda^6/5 (SUM Cn2 h^5/3)^-3/5).

**Teaching references (clear derivations & typical values)**
- Claire Max, "Adaptive Optics: An Introduction" (UCSC AY289): https://ucolick.org/~max/289/Assigned%20Readings/160114%20-%20Introduction/Introduction_Max.pdf
- V. Dhillon, PHY217 Adaptive Optics: https://vikdhillon.staff.shef.ac.uk/teaching/phy217/telescopes/phy217_tel_adaptive.html — r0 ~ 10 cm @ 500 nm, t0 ~ 10 ms, r0 ∝ lambda^6/5.
- A. Campbell (ROE), "Atmospheric Turbulence and its Influence on Adaptive Optics": https://www.roe.ac.uk/ifa/postgrad/pedagogy/2008_campbell.pdf
- A. Glindemann (ESO), "Adaptive Optics on Large Telescopes": https://www2.mpia.de/AO/INSTRUMENTS/FPRAKT/AOonLargeTelescopes.pdf

**Strehl / scintillation**
- Maréchal approximation tutorial (Eikonal Optics): https://eikonaloptics.com/blogs/tutorials/image-quality-metrics-the-strehl-ratio-and-the-marechal-approximation
- Andrews & Phillips, *Laser Beam Propagation through Random Media* — Rytov variance / scintillation index.

> **Numerical-constant caveat:** small differences in leading constants across the literature (e.g. von Karman PSD 0.0229 vs 0.023; G-tilt 0.170 vs Z-tilt 0.184; DIMM 0.358 vs Tokovinin's 0.340/0.364; seeing 0.98 vs 0.976) come from definition/normalization conventions (G-tilt vs Z-tilt, cycles/m vs rad/m, finite-exposure and outer-scale corrections). Pick one convention, state it, and keep it consistent across all estimators so the cross-checks are apples-to-apples. Several formula constants above were confirmed from the aotools source and the cited papers; a couple (e.g. the exact Saint-Jacques slope-variance coefficient and Tyler's exact prefactor) should be re-verified against the primary papers/aotools source before being hard-coded into the PS9 pipeline.
