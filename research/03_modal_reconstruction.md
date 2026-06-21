# Modal Wavefront Reconstruction for SH-WFS (ISRO BAH 2026 — PS9)

**Scope of this report.** Modal reconstruction of the Shack–Hartmann wavefront sensor (SH-WFS) signal: fitting the measured sub-aperture x/y slopes to an orthogonal polynomial basis (primarily **Zernike**, secondarily **Karhunen–Loève**) to recover (a) the wavefront phase map `W(x,y)` and (b) the modal coefficient vector `a` that feeds turbulence characterisation (`r0`, `τ0`) and the deformable-mirror (DM) actuator map. This is the companion of the *zonal* reconstruction report; here the emphasis is the polynomial basis, the slope→coefficient→wavefront chain, the Noll Zernike/Kolmogorov statistics, and the bridge to turbulence parameters.

Companion problem context (from `idea.md`): time series of `.bmp` SH-WFS frames at a few-ms cadence; MLA and DM are on a **Fried geometry**; outputs required are `W(xᵢ,yᵢ)` per frame, `r0`, `τ0`, and DM actuator maps with inter-actuator coupling. Real-time constraint: correction faster than the ~10 ms atmospheric coherence time. **Key consequence for modal methods: once the basis and geometry are fixed, reconstruction is a single precomputed matrix–vector product `a = R·s` (and optionally `W = Z·a`), i.e. O(N_modes × N_slopes) multiply–accumulates per frame — trivially real-time and ideal for a C/BLAS `gemv`.**

---

## 0. The full chain at a glance

```
 BMP frame ──► spot centroids ──► slopes s (2·N_sub vector, x & y)
                                         │
                                         │  s = M·a + noise        (forward model; M = modal interaction / "Γ" matrix)
                                         ▼
   modal coefficients  a = R·s = M⁺·s    (R precomputed: pseudo-inverse / SVD / MAP)
                                         │
            ┌────────────────────────────┼─────────────────────────────┐
            ▼                            ▼                             ▼
  wavefront  W(x,y)=Σ aⱼ Zⱼ(x,y)   turbulence: fit ⟨aⱼ²⟩→(D/r0)^(5/3)   DM map = conjugate(W) → actuator strokes
```

Three matrices are computed **once, offline**:
1. `M` — the **modal interaction matrix** (a.k.a. modal influence / "poke" matrix): column j is the SH-WFS slope response to a unit of Zernike mode j. Built either analytically (Noll derivative/Γ matrices) or numerically (calibration or sub-aperture integration of mode gradients).
2. `R = M⁺` — the **reconstructor** (Moore–Penrose pseudo-inverse, possibly SVD-truncated / Tikhonov / MAP-regularised).
3. `Z` — the **synthesis matrix**: column j is mode `Zⱼ` sampled on the output grid, so `W = Z·a`. (Often the wavefront map is not even formed explicitly for the DM; the DM command is computed directly from `a`.)

Per frame at run time only the cheap products `a = R·s` and (if needed) `W = Z·a` remain — both are `gemv` calls.

---

## 1. Zernike polynomials: definition, Noll indexing, normalisation

### 1.1 Definition

On the unit disc (ρ ∈ [0,1], θ azimuth) the Zernike polynomials separate into radial and azimuthal parts:

```
Z_even (n,m): Z = √(n+1)·√2 · R_n^m(ρ)·cos(mθ)      m > 0
Z_odd  (n,m): Z = √(n+1)·√2 · R_n^m(ρ)·sin(mθ)      m < 0
Z       (n,0): Z = √(n+1)        · R_n^0(ρ)              m = 0
```

with the **radial polynomial** (n−m even, else 0):

```
                (n−m)/2      (−1)^k (n−k)!
 R_n^m(ρ)  =     Σ      ───────────────────────────────────  ρ^(n−2k)
                 k=0    k! ((n+m)/2 − k)! ((n−m)/2 − k)!
```

The normalisation factor `N_n^m = √(n+1)` for m=0 and `√(2(n+1))` for m≠0 makes the modes **orthonormal** in the RMS sense over the unit disc:

```
 (1/π) ∫∫_disc  Zⱼ Zⱼ' dA  =  δ_{jj'}
```

This is **Noll normalisation** (RMS = 1 over the pupil): every coefficient `aⱼ` then *is* the RMS wavefront contributed by that mode (in the same length/phase units as `W`). This is the convention used by Noll 1976 and by `aotools` (`norm="noll"`). [Noll 1976; aotools]

### 1.2 Noll single-index `j` (the AO standard)

Noll (1976) introduced the linear ordering used throughout AO. Index `j` starts at **1 = piston**, orders by increasing radial order `n`, and within an `n` by increasing `|m|`; even `j` ↔ cos (`m>0`), odd `j` ↔ sin (`m<0`). The `aotools` implementation (verbatim logic) inverts `j → (n,m)`:

```python
def zernIndex(j):
    n = int((-1.0 + sqrt(8*(j-1)+1)) / 2.0)
    p = j - (n*(n+1))//2
    k = n % 2
    m = (int((p+k)//2)*2) - k
    if m != 0:
        m *= (+1 if j % 2 == 0 else -1)   # even j -> +m (cos), odd j -> -m (sin)
    return [n, m]
```
[AOtools `aotools/functions/zernike.py`, `zernIndex`]

**Why Noll, not OSA/ANSI, for PS9.** The OSA/ANSI single index `j = (n(n+2)+m)/2` (starts at 0) is common in ophthalmology. Noll is the de-facto standard in *atmospheric* AO because the **Noll residual-variance table (§3) and all Kolmogorov coefficient statistics are tabulated against the Noll j**. Use Noll to stay aligned with the turbulence literature and your `r0` bridge.

### 1.3 The Noll Zernike table (j = 1 … 21)

| Noll j | (n, m) | Name | Polynomial (un-normalised core × √-factor) |
|---|---|---|---|
| 1 | (0, 0) | Piston | 1 |
| 2 | (1, +1) | Tip (x-tilt) | √4 · ρcosθ = 2ρcosθ |
| 3 | (1, −1) | Tilt (y-tilt) | 2ρsinθ |
| 4 | (2, 0) | Defocus | √3 (2ρ²−1) |
| 5 | (2, −2) | Astigmatism (oblique, 45°) | √6 ρ²sin2θ |
| 6 | (2, +2) | Astigmatism (vertical, 0/90°) | √6 ρ²cos2θ |
| 7 | (3, −1) | Coma (y) | √8 (3ρ³−2ρ)sinθ |
| 8 | (3, +1) | Coma (x) | √8 (3ρ³−2ρ)cosθ |
| 9 | (3, −3) | Trefoil (oblique) | √8 ρ³sin3θ |
| 10 | (3, +3) | Trefoil (vertical) | √8 ρ³cos3θ |
| 11 | (4, 0) | Primary spherical | √5 (6ρ⁴−6ρ²+1) |
| 12 | (4, +2) | Secondary astigmatism (vert.) | √10 (4ρ⁴−3ρ²)cos2θ |
| 13 | (4, −2) | Secondary astigmatism (obl.) | √10 (4ρ⁴−3ρ²)sin2θ |
| 14 | (4, +4) | Quadrafoil (vert.) | √10 ρ⁴cos4θ |
| 15 | (4, −4) | Quadrafoil (obl.) | √10 ρ⁴sin4θ |
| 16 | (5, +1) | Secondary coma (x) | √12 (10ρ⁵−12ρ³+3ρ)cosθ |
| 17 | (5, −1) | Secondary coma (y) | √12 (10ρ⁵−12ρ³+3ρ)sinθ |
| 18 | (5, +3) | Secondary trefoil (vert.) | √12 (5ρ⁵−4ρ³)cos3θ |
| 19 | (5, −3) | Secondary trefoil (obl.) | √12 (5ρ⁵−4ρ³)sin3θ |
| 20 | (5, +5) | Pentafoil (vert.) | √12 ρ⁵cos5θ |
| 21 | (5, −5) | Pentafoil (obl.) | √12 ρ⁵sin5θ |

(Radial cores from `R_n^m`; the √(n+1)/√(2(n+1)) prefactors give Noll RMS=1 normalisation. The cos/sin assignment for ±m follows the even-j↔cos rule above. Some texts swap the *sign labels* of tip/tilt or of the two members within a degenerate ±m pair; this is a pure convention choice — fix one and use it consistently end-to-end.) [Wikipedia "Zernike polynomials"; Noll 1976; aotools]

---

## 2. The eight (+) modal approaches — math, conditioning, accuracy, real-time fit

### Method 1 — Zernike modal expansion (the basis itself)

**Idea.** Represent the wavefront `W(x,y) = Σ_{j} aⱼ Zⱼ(x,y)` truncated at `J` modes. The unknowns are the `J` coefficients `a`. Reconstruction = estimate `a` from slopes; synthesis = `W = Z·a`.

**Conditioning/accuracy.** Excellent for *smooth, low-order, turbulence-like* wavefronts — Kolmogorov phase has a steep `f^(-11/3)` PSD, so the first few dozen Zernikes capture most variance (§3 shows 5–10 modes already remove >90% of the corrected-able variance). Truncation at `J` discards high spatial frequencies → **fitting error** (and, if those frequencies fold back through the finite sub-aperture sampling, **aliasing error**, §2.8/§4).

**Real-time.** Synthesis `W = Z·a` is a `gemv`; coefficients are reused directly for the DM. Best-in-class real-time fit. **Use as the primary basis for PS9.**

[Noll 1976; aotools `phaseFromZernikes`, `zernikeArray`, `zernike_nm`]

---

### Method 2 — Slope/gradient form: Noll analytic derivative (Γ) matrices → the modal interaction matrix

This is the heart of modal SH-WFS reconstruction: an SH-WFS does **not** measure phase, it measures the **average wavefront gradient over each sub-aperture**:

```
 s_x^(k) ≈ (1/A_k) ∫∫_{subap k} ∂W/∂x dA = Σ_j aⱼ · ⟨∂Zⱼ/∂x⟩_k
 s_y^(k) ≈ (1/A_k) ∫∫_{subap k} ∂W/∂y dA = Σ_j aⱼ · ⟨∂Zⱼ/∂y⟩_k
```

So the forward model is **linear**: `s = M·a`, where the **modal interaction matrix** `M` (size `2·N_sub × J`) has entries `M[(k,x), j] = ⟨∂Zⱼ/∂x⟩_k`. Building `M` is the whole game. Three routes:

**(2a) Noll's analytic derivative / Γ matrices.** A central Noll-1976 result: the *derivative of a Zernike is a linear combination of lower-order Zernikes*. Two constant matrices `Γ_x`, `Γ_y` encode this:

```
 ∂Zⱼ/∂x = Σ_i (Γ_x)_{ij} Z_i ,     ∂Zⱼ/∂y = Σ_i (Γ_y)_{ij} Z_i
```

`aotools.makegammas(nzrad)` builds them, returning a `(2, nzmax, nzmax)` array `[Γ_x, Γ_y]`. Its four selection rules:
- (a) RMS-normalisation factor (√2 when an `m` index is 0);
- (b) zero unless the radial-order/parity relation holds;
- (c) zero unless `|mⱼ − mᵢ| = 1` (derivative changes azimuthal order by exactly 1);
- (d) sign of `Γ_y` set by the `m`-sign relationship and index parity.

The sub-aperture-averaged derivatives `⟨∂Zⱼ/∂x⟩_k` then follow from `Γ` times the mean values of the (lower-order) Zernikes over each sub-aperture. [AOtools `makegammas` docstring & source; Noll 1976]

**(2b) Analytic sub-aperture line integrals (recommended, most accurate).** By the divergence theorem the *area* integral of a gradient reduces to a 1-D *contour* integral of the Zernike itself around the sub-aperture perimeter — i.e. you integrate `Zⱼ` (smooth) rather than `∂Zⱼ` (steeper). This is what the **`mshwfs` toolbox** (Antonello) does: "computes the definite integrals of the gradients of the Zernike modes within each subaperture" for an *arbitrary* sub-aperture arrangement and *arbitrary* mode count. More accurate than point-sampling `∂Z` at the sub-aperture centre, especially for high modes / few sub-apertures. [Antonello `mshwfs` (github.com/jacopoantonello/mshwfs); Barwick, "average derivatives via 1-D integrals of Zernikes along subaperture perimeter"]

**(2c) Empirical calibration.** If lab data with a calibrated source/DM exists, "poke" each Zernike (or each DM mode) and record the slopes → measured columns of `M`. Captures real lenslet/detector imperfections but needs a controllable wavefront generator.

**Accuracy caveat — modal cross-coupling.** Zernike *gradients* are **not** mutually orthogonal over the discrete pupil, so `Mᵀ M` is not diagonal: estimated low-order modes get contaminated by un-modelled high-order modes (and vice-versa). This is the dominant *bias* term in modal SH reconstruction and in `r0` estimation (§5). Mitigations: include enough modes, or use the Laplacian-eigenfunction basis whose gradients *are* orthogonal, or iterate the cross-coupling correction (§5). [Gavel/Sciencedirect S0030401812010759; MNRAS 483,1192]

**Real-time.** `M` and its inverse are offline; run time is unaffected.

---

### Method 3 — Least-squares modal fit via the pseudo-inverse

With `s = M·a + noise`, the **ordinary least-squares (OLS)** estimate minimises ‖s − M a‖²:

```
 â = (Mᵀ M)⁻¹ Mᵀ s  ≡  M⁺ s  ≡  R·s        (overdetermined: 2·N_sub > J)
```

`R = M⁺` is the **reconstructor**, precomputed once.

- **Conditioning** is governed by `cond(M) = σ_max/σ_min`. With a sensible mode count it is well-conditioned; pushing `J` toward `2·N_sub` makes `σ_min → 0` and blows up noise.
- **Noise propagation.** `Cov(â) = σ_n² (Mᵀ M)⁻¹` — small singular values ⇒ huge variance on the corresponding modes. This is exactly why mode count must be capped (§2.8).
- **Accuracy.** Optimal (minimum variance, unbiased) *if* the model is complete and noise is white; otherwise cross-coupling biases it.

**Real-time.** Single `gemv`, `2·N_sub·J` MACs. ~Sub-µs for typical sizes; dominates nothing. **This is the baseline reconstructor for PS9.**

[Standard result; reproduced in MNRAS 483,1192 as `H⁺=(HᵀH)⁻¹Hᵀ`]

---

### Method 4 — SVD reconstruction with mode truncation & Tikhonov regularisation

Decompose `M = U Σ Vᵀ`. The pseudo-inverse is `M⁺ = V Σ⁺ Uᵀ` with `Σ⁺ = diag(1/σ_i)`. Two robustifications:

**(4a) Truncated SVD (TSVD).** Zero the reciprocals of singular values below a threshold (`1/σ_i → 0` for `σ_i < ε`). Removes ill-conditioned/"unseen" mode combinations (e.g. waffle-like or near-null modes the geometry can't sense). Equivalent to dropping the noisiest modes.

**(4b) Tikhonov / damped least squares.** Replace `1/σ_i` by `σ_i/(σ_i²+μ²)`:

```
 â = V diag( σ_i/(σ_i²+μ²) ) Uᵀ s  =  (Mᵀ M + μ² I)⁻¹ Mᵀ s
```

`μ` damps small-`σ` modes smoothly (no hard cutoff), trading a little bias for a large variance reduction; it "alleviates ill-conditioning associated with the lowest singular values, responsible for unnecessarily large control voltages." [Tokovinin/Tikhonov, MDPI; ScienceDirect S0377042711005206]

- **Conditioning/accuracy.** Both bound noise amplification; `μ` (or the truncation threshold) is the single knob trading fitting error vs noise. Choose by L-curve or by matching expected `σ_n`.
- **Real-time.** SVD is **offline**; the resulting `R` is again one `gemv`. No runtime penalty.

**For PS9:** compute `R` via SVD and apply Tikhonov damping — it gives a numerically safe reconstructor and lets you tune robustness without changing the basis.

[Poyneer UCRL-TR-204793 "Fast modal wavefront reconstruction"; ScienceDirect SVD-for-SH S0377042711005206]

---

### Method 5 — Karhunen–Loève (KL) modes (statistically optimal for turbulence)

**Idea.** The KL modes `K_i` are the **eigenfunctions of the atmospheric (Kolmogorov) phase covariance**:

```
 ∫ Γ_φ(r, r') K_i(r') dr' = λ_i K_i(r) ,     Γ_φ = ⟨φ(r)φ(r')⟩
```

They are the **optimal** modal basis in the Karhunen–Loève sense: for any fixed number `N` of corrected modes, the residual phase variance is **minimum** compared with *any* other orthogonal set. KL modes are mutually orthogonal **and statistically independent** (decorrelated coefficients ⇒ diagonal covariance), unlike Zernikes whose coefficients are correlated (§3 covariance). [Wang & Markey 1978; KL-for-turbulence search results]

**Construction in practice.** Diagonalise the **Zernike covariance matrix** `C` (Noll/Wang–Markey/Roddier, §3) — its eigenvectors give KL modes as *linear combinations of Zernikes*. So you keep the convenient Zernike machinery (and the analytic Γ derivative matrices ⇒ the KL interaction matrix is just `M_KL = M_Zernike · E`, where `E` is the eigenvector matrix). On an annular/discrete pupil, compute `C` numerically and diagonalise.

- **Conditioning/accuracy.** Best variance-per-mode and *diagonal* coefficient statistics ⇒ cleanest input for `r0` fitting (no cross-mode correlation to undo). Requires *a-priori* turbulence statistics (it bakes in Kolmogorov + the pupil).
- **Real-time.** Same cost as Zernike (it *is* a fixed linear basis); only the offline matrices differ. 

**For PS9:** strong **secondary** choice — use KL when you want statistically-independent coefficients for cleaner `r0`/`τ0` estimation, or as the DM-mode basis. Zernike remains the reporting basis (the evaluation asks specifically for Zernike coefficients).

[Wang & Markey 1978; Roddier 1990; KL testbed patents US8,725,471 / US8,452,574; HAL hal-02887698 modal sensorless AO with KL]

---

### Method 6 — Gram–Schmidt orthonormalisation over the discrete / annular pupil

**Problem.** Zernikes are orthonormal over the *continuous full unit disc*. Over a **discrete sampled grid**, a **central-obscuration annulus** (secondary mirror), or a **vignetted/non-circular** pupil, they lose orthogonality ⇒ coefficient cross-talk and biased fits.

**Fix.** **Gram–Schmidt** orthonormalise the Zernikes over the *actual* pupil domain to obtain a custom orthonormal set, or use the closed-form **Mahajan annular Zernike polynomials** (orthonormal over an annulus of obscuration ratio ε, reducing to circle Zernikes as ε→0). Mahajan's annular set represents *balanced aberrations* with minimum variance over the annulus. Dai–Mahajan generalise to *any* integrable domain via a numerically superior (QR-based) orthonormalisation. [Mahajan, JOSA 71,75 (1981); AO 33,8125 (1994); Dai & Mahajan; ScienceDirect S0143816624006213 Schwarz–Christoffel for non-circular]

- **Conditioning/accuracy.** Restores orthogonality ⇒ uncorrelated, low-bias coefficients on the real pupil. Note the obscuration changes per-mode statistics (spherical/curvature variance ↓, coma/astig/distortion variance ↑ with ε).
- **Real-time.** Orthonormalisation is offline; runtime unchanged.

**For PS9:** apply **only if** the lab pupil is annular/obscured or markedly non-circular. The problem says "pupil size of the turbulated beam" (likely a clear circular pupil) — if circular, plain Zernikes suffice and this is optional polish. Keep it in reserve.

---

### Method 7 — Modal ↔ Zonal trade-offs and hybrid modal–zonal reconstructors

| Aspect | **Modal (Zernike/KL)** | **Zonal (Fried/Hudgin/Southwell)** |
|---|---|---|
| Unknowns | `J` global coefficients | phase at grid points (≈ N_sub) |
| Output | smooth analytic `W`; coefficients for free | detailed point-wise `W`; high spatial detail |
| Turbulence stats | **direct** (coefficients → `r0`) | needs a separate modal fit afterward |
| Noise behaviour | low modes robust; truncation controls noise | waffle/null-space modes; needs regularisation |
| Aliasing | high modes alias into low (cross-coupling) | high freq amplified/attenuated by geometry |
| Fried geometry fit | natural (DM on Fried grid) | **natural** — actuators at sub-aperture corners |
| Compute (runtime) | one `gemv` | one `gemv` (or FFT) |

**Hybrid.** Reconstruct **low orders modally** (Zernike — robust, gives `r0` and the DM low-order command) and **high spatial frequencies zonally** (local detail the few modes miss), then sum. Because the DM and MLA share a **Fried geometry**, the *zonal* reconstructor maps naturally to actuator commands while the *modal* part supplies the turbulence statistics — so a hybrid serves both required outputs cleanly. [WaveTrain/MZA AO geom; Southwell ScienceDirect S0030401816311476; Lumetrics "Zernike vs Zonal"]

**For PS9:** recommend **modal-primary for reconstruction & turbulence, with an optional zonal residual pass** for the actuator map detail (the zonal report covers that side).

---

### Method 8 — Bayesian / MAP (minimum-variance) modal estimation

**Idea.** Treat `a` as random with known prior covariance `C_a` (from Kolmogorov, §3) and noise covariance `C_n`. The **maximum-a-posteriori / minimum-variance / Wiener** estimate is:

```
 â = ( Mᵀ C_n⁻¹ M + C_a⁻¹ )⁻¹ Mᵀ C_n⁻¹ s
```

- This is exactly **Tikhonov with a physically-motivated, mode-dependent regulariser** `C_a⁻¹` — it damps each mode in proportion to how much turbulence *and* noise it carries. Equivalent to a Wiener filter; under ideal conditions the optimal AO controller is an integrator fed by a MAP reconstructor.
- The prior `C_a ∝ (D/r0)^(5/3)` (Noll/§3). So MAP needs an `r0` estimate — natural in closed loop (bootstrap from a least-squares pass, then refine).
- **Conditioning/accuracy.** Lowest mean-square error of all linear reconstructors when the priors are correct; gracefully handles ill-conditioning and noisy high modes (no hard truncation).
- **Real-time.** Reconstructor precomputed ⇒ one `gemv`.

**For PS9:** an **upgrade path** once a working least-squares/SVD pipeline exists and an `r0` estimate is available — squeezes out residual error in the noisy/low-light regime. Not the first thing to build. [Wallner; "minimum-norm ML vs MAP" JOSA-A 26,497; arXiv:1003.0274; arXiv:2301.03478]

---

### 2.8 Cross-cutting practical issues

**Piston / tip / tilt removal.** Piston (`j=1`) is **unobservable** by an SH-WFS (slopes carry no absolute phase) — always drop it from `M`/`R` (start fitting at `j=2`). Tip/tilt (`j=2,3`) *are* measured and carry the **largest** turbulence variance (87% of total, §3), so **keep them for reconstruction and the DM/tip-tilt mirror**, but **exclude them from `r0` fitting** because lab/telescope tip-tilt is polluted by tracking, wind-shake and vibrations (MNRAS sums start at `j=4` or `j=5`). [MNRAS 483,1192]

**Annular / obscured pupils.** See Method 6.

**Aliasing of high modes.** A finite `N_sub` lenslet array cannot distinguish a high Zernike from a lower one with the same sub-aperture-average gradients ⇒ high modes "fold" into the estimated low modes (this *is* modal cross-coupling). Cap `J` and/or use the cross-coupling correction (§5). [MNRAS 483,1192]

**Mode-count vs number of sub-apertures (mode selection).** Hard rank limit: a centroiding SH-WFS yields `2·N_sub` slope numbers but loses piston, so the maximum reconstructable modes ≈ **`2·N_sub − 1`**, and the *useful* number is lower because near-null/aliased combinations are noisy. Empirical anchor: a **7×7 (49) sub-aperture array reconstructs ~35 Zernike modes** by the standard centroid method (only with extra per-sub-aperture information can it reach ~65). Rule of thumb: choose `J` ≈ number of modes the DM can actuate and well below `2·N_sub`; verify with `cond(M)` and the noise-propagation `Tr[(MᵀM)⁻¹]`. For a Fried-geometry DM with `N_act` actuators on the same grid, `J ≈ N_act` is a sensible target. [PubMed 27828187 "More Zernike modes in the sub-aperture"; standard `2N−1` rank argument]

---

## 3. Noll Kolmogorov statistics — the turbulence numbers

### 3.1 Total phase variance and the residual table

For Kolmogorov turbulence over a circular aperture of diameter `D`, the mean-square phase over the pupil (piston removed) is

```
 σ²_total = 1.0299 · (D/r0)^(5/3)   [rad²]          (= Δ₁)
```

After perfectly correcting the **first J Zernike modes**, the **residual** mean-square wavefront error is `Δ_J·(D/r0)^(5/3)`. Noll's coefficients:

| Modes corrected (up to & incl.) | `Δ_J` (×(D/r0)^(5/3), rad²) | Mode just added |
|---|---|---|
| J=1 (piston only) | **1.0299** | piston (= total turbulence) |
| J=2 | **0.582** | tip |
| J=3 | **0.134** | tilt → *tip+tilt removed* |
| J=4 | **0.111** | defocus |
| J=5 | **0.0880** | astig (oblique) |
| J=6 | **0.0648** | astig (vertical) |
| J=7 | **0.0587** | coma y |
| J=8 | **0.0525** | coma x |
| J=9 | **0.0463** | trefoil |
| J=10 | **0.0401** | trefoil |
| J=11 | **0.0377** | spherical |
| J=15 | **0.0279** | through quadrafoil |
| J=21 | **0.0208** | through pentafoil |

**Asymptotic law (large J):**
```
 Δ_J  ≈  0.2944 · J^(−√3/2) · (D/r0)^(5/3)    [rad²],   √3/2 ≈ 0.866
```
i.e. residual falls only slowly (`~J^-0.866`) — diminishing returns, motivating both mode-count caps and the *statistically optimal* KL basis. [Noll 1976, Table; values cross-checked across AO references]

**Per-mode contributions** (difference of consecutive `Δ`): tip = 1.0299−0.582 = **0.448**, tilt = 0.582−0.134 = **0.448** (tip+tilt = **0.896**, i.e. **87%** of the total variance lives in tip/tilt — why a fast tip/tilt stage matters); defocus = 0.134−0.111 = **0.023**; each astigmatism ≈ 0.023/0.013; etc.

### 3.2 The bridge to `r0`: Zernike coefficient variances

Every Zernike coefficient variance scales the same way with turbulence strength:

```
 ⟨aⱼ²⟩  =  c_j · (D/r0)^(5/3)      [rad²]
```

with **mode-specific constants `c_j`** that depend only on `(n,m)` and are given by Noll's closed form (also Wang & Markey 1978, Roddier 1990). The Noll covariance of two coefficients with the same azimuthal order is

```
                       (−1)^[(n+n'−2m)/2] · √((n+1)(n'+1)) · Γ(14/3) · Γ[(n+n'−5/3)/2]
 ⟨aⱼ aⱼ'⟩ ∝ (D/r0)^(5/3) · ─────────────────────────────────────────────────────────────────
                       Γ[(n−n'+17/3)/2] Γ[(n'−n+17/3)/2] Γ[(n+n'+23/3)/2]
```
times the standard 3.895… prefactor, **non-zero only when `m=m'` and `n−n'` is even** (this off-diagonal structure is the Zernike-coefficient *correlation* that KL modes diagonalise). The diagonal (`n=n'`) gives the per-mode `c_j`. Representative diagonal values: tilt (`j=2,3`) `c≈0.448`, defocus (`j=4`) `c≈0.023`, etc. — exactly the per-mode contributions above. [Noll 1976 eq.; Wang & Markey 1978; Roddier 1990; reproduced in arXiv:2004.11210, 1004.3278]

**Estimating `r0`** (this is the turbulence-characterisation deliverable):
1. Reconstruct `aⱼ` for every frame.
2. Compute the *measured* variance `⟨aⱼ²⟩` over the time series for each mode (exclude tip/tilt — §2.8).
3. **Fit** the measured variances to the theoretical `c_j·(D/r0)^(5/3)` (a single free parameter `r0`, since `D` is known); equivalently average `r0 = D·[ c_j / ⟨aⱼ²⟩ ]^(3/5)` over the fitted modes, or do a log–log linear fit `log⟨aⱼ²⟩ vs log c_j`.

**von Kármán / outer-scale `L0` refinement.** Real turbulence has a finite outer scale; the PSD becomes `∝ (f² + 1/L0²)^(−11/6) r0^(−5/3)`, which *reduces* low-order (esp. tilt/defocus) variances. Fit `(r0, L0)` jointly to the modal variances. The MNRAS-2019 method (Andrade et al.) shows that the dominant *bias* is **modal cross-coupling** (not aliasing), and an **iterative** scheme that subtracts the cross-coupling term `σ²_cc,i` (which itself depends on the turbulence params) recovers both `r0` and `L0` to **sub-percent** accuracy over `L0 = 4–32 m`:

```
 ⟨b_i²⟩ = ⟨a_∥i²⟩_vK + σ²_cc,i(r0,L0) + σ²_noise,i      (measured = signal + cross-coupling + noise)
```
iterate: estimate `(r0,L0)` from de-coupled variances → recompute `σ²_cc` → repeat (≈3 iterations). [MNRAS 483, 1192 (academic.oup.com/mnras/article/483/1/1192/5203639); arXiv:1811.08396]

### 3.3 The bridge to `τ0` (coherence time) — from the time series

With wind advecting a frozen screen ("Taylor hypothesis"), temporal statistics give the coherence time:

```
 τ0 = 0.314 · r0 / v̄        (effective wind speed v̄)
 f_G = 0.427 · v̄ / r0       (Greenwood frequency);   τ0 = 0.134 / f_G
```

Estimate `v̄` (hence `τ0`) from the **temporal power spectra of the Zernike coefficients** (especially tip/tilt): the spectrum shows a low-frequency plateau and a mid-frequency `f^(−8/3)` roll-off (tilt-included; `f^(−2/3)` low-freq; tilt-removed modes roll as `f^(−11/3)` at high freq). The **knee** frequency between regimes scales with `v̄/D`, and/or fit the temporal autocorrelation of `aⱼ(t)` whose decorrelation time gives `τ0` directly. [ten Brummelaar 1994 SPIE; arXiv:1906.03128; SOAR 1010.4176]

---

## 4. Error budget for modal reconstruction (what limits accuracy)

```
 σ²_recon  =  σ²_fit (truncation: modes > J)          ~ Δ_J (D/r0)^5/3, falls as J^-0.866
            + σ²_alias (high modes → low via Mᵀ)        ← cross-coupling; cap J, KL, or iterate
            + σ²_noise (centroid noise × Tr[(MᵀM)⁻¹])    ← Tikhonov/MAP, cap J
            + σ²_pupil (non-orthogonality on real pupil) ← Gram–Schmidt / annular Zernike
```
The **`J` knob** trades `σ²_fit` (↓ with J) against `σ²_noise` (↑ with J): there is an optimum `J*` set by `r0`, `N_sub` and centroid SNR. MAP (§8) minimises the *total* analytically.

---

## 5. Method comparison & ranking for PS9

| # | Method | Builds `M` / `R` how | Conditioning | Accuracy on Kolmogorov | Runtime (per frame) | PS9 priority |
|---|---|---|---|---|---|---|
| 1 | Zernike basis | — (basis) | n/a | High for low orders; truncation error | `gemv` `W=Za` | ★★★★★ core |
| 2 | Noll Γ / analytic sub-ap. integral interaction matrix | analytic | good | High; cross-coupling bias if J small | offline | ★★★★★ core |
| 3 | LS pseudo-inverse `M⁺` | `(MᵀM)⁻¹Mᵀ` | good if J≪2N_sub | optimal if model complete | one `gemv` | ★★★★★ baseline |
| 4 | SVD + Tikhonov/TSVD | SVD offline | **robust** (tunable μ) | slight bias, big variance ↓ | one `gemv` | ★★★★★ ship this |
| 5 | Karhunen–Loève | diagonalise `C` | best variance/mode | **optimal** per mode; diagonal stats | one `gemv` | ★★★★ for r0/τ0 |
| 6 | Gram–Schmidt / annular Zernike | orthonorm. offline | restores orthogonality | removes pupil bias | offline | ★★★ if obscured pupil |
| 7 | Hybrid modal+zonal | both | — | best of both (stats + detail) | two `gemv`/FFT | ★★★ for DM detail |
| 8 | Bayesian / MAP | `(MᵀC_n⁻¹M+C_a⁻¹)⁻¹…` | best (priors) | **min MSE** of linear methods | one `gemv` | ★★★ upgrade |

**Real-time verdict:** *every* modal method reduces at run time to a precomputed `gemv` (`a = R·s`, optional `W = Z·a`), `~2·N_sub·J` MACs per frame — microseconds with BLAS/C. The choice of method affects only the **offline** construction of `R` and the **accuracy/robustness**, never the per-frame speed. This is the single biggest reason modal reconstruction suits PS9's ms-cadence requirement.

---

## 6. Recommended approach for PS9

**Primary pipeline (build first):**
1. **Basis:** Zernike, **Noll-indexed, RMS(Noll)-normalised**, **piston excluded**, modes `j = 2…J`. Pick `J ≈ N_actuators` and **`J ≪ 2·N_sub`** (e.g. `J ≈ 35` for a 7×7 array); validate with `cond(M)` and `Tr[(MᵀM)⁻¹]`.
2. **Interaction matrix `M`:** build **analytically** via sub-aperture integrals of Zernike gradients (à la `mshwfs`; divergence-theorem 1-D perimeter integrals) on the **exact Fried sub-aperture geometry**. Cross-check against `aotools.makegammas` Γ matrices. If a calibrated lab source/DM is available, also capture an **empirical** `M` and compare.
3. **Reconstructor `R`:** `R = M⁺` via **SVD with Tikhonov damping** (or TSVD) — numerically safe, tunable. Precompute once. Run time: `a = R·s` (one `gemv`).
4. **Wavefront map:** `W = Z·a` (precomputed `Z`); the **conjugate** `−W` feeds the DM (actuator-map / coupling handled in the DM report).

**Turbulence characterisation (the second deliverable):**
5. Per frame → `aⱼ`. Over the time series, form `⟨aⱼ²⟩` for `j ≥ 4` (drop tip/tilt). **Fit** to `c_j·(D/r0)^(5/3)` with the **von Kármán `L0`** extension and the **iterative cross-coupling correction** (MNRAS 483,1192) to get **`r0`** (and `L0`) to ~percent accuracy.
6. **`τ0`:** from temporal power spectra / autocorrelation of `aⱼ(t)` (tip/tilt knee → `v̄`), then `τ0 = 0.314 r0/v̄` (or via Greenwood `f_G`).

**Recommended upgrades (if time permits):**
- **KL basis** (diagonalise the Noll/Wang–Markey covariance) for statistically-independent coefficients → cleaner `r0`/`τ0` and better variance-per-mode; report Zernike coefficients as required, compute stats in KL.
- **MAP/Wiener reconstructor** once an `r0` estimate exists → minimum-MSE in the noisy/low-light regime.
- **Annular/Gram–Schmidt Zernikes** *only if* the lab pupil is obscured/non-circular.
- **Hybrid modal+zonal** if the DM needs higher spatial detail than `J` modes provide.

**One-line summary:** *Noll-indexed Zernike modal fit with an analytically-built interaction matrix, inverted offline by SVD/Tikhonov, applied per frame as a single matrix–vector product `a = R·s` → `W = Σ aⱼ Zⱼ`; the same coefficients' variances are fit to `c_j·(D/r0)^{5/3}` (von Kármán + iterative cross-coupling correction) for `r0`, and their temporal spectra give `τ0`.*

---

## 7. Sources (URLs)

**Foundational**
- Noll, R. J. (1976), "Zernike polynomials and atmospheric turbulence," *JOSA* 66(3), 207–211 — abstract: https://opg.optica.org/josa/abstract.cfm?uri=josa-66-3-207 ; ADS: https://ui.adsabs.harvard.edu/abs/1976JOSA...66..207N/abstract ; full PDF: https://e-l.unifi.it/pluginfile.php/1055871/mod_resource/content/1/Appunti_2020_Lezione%2014_3_NOLL1976.pdf ; citation: https://www.scirp.org/reference/ReferencesPapers.aspx?ReferenceID=1793169
- Zernike polynomials (Noll vs OSA indexing, radial formula, normalisation): https://en.wikipedia.org/wiki/Zernike_polynomials
- Zernike polynomials and applications (review, IOP/Optica): https://iopscience.iop.org/article/10.1088/2040-8986/ac9e08
- Wyant, Arizona, "Zernike Polynomials": https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/Zernike_Polynomials_For_The_Web.pdf
- Telescope-optics.net, Zernike coefficients & aberration table: https://www.telescope-optics.net/zernike_coefficients.htm

**Slope/gradient form & modal interaction matrix**
- AOtools `zernike` module (source: `zernIndex`, `makegammas` Γ matrices, `zernike_nm`, `phaseFromZernikes`, `zernikeArray`): https://aotools.readthedocs.io/en/v1.0/_modules/aotools/functions/zernike.html ; GitHub: https://github.com/AOtools/aotools/blob/main/aotools/functions/zernike.py ; circular-functions docs: https://aotools.readthedocs.io/en/latest/zernike.html
- Antonello, `mshwfs` Modal Shack–Hartmann toolbox (analytic sub-aperture gradient integrals, arbitrary modes): https://github.com/jacopoantonello/mshwfs
- "Analytical calibration of slope response of Zernike modes in a SH-WFS based on matrix product," *Opt. Lett.* 47, 1466 (2022): https://opg.optica.org/ol/abstract.cfm?uri=ol-47-6-1466
- "Modal wavefront reconstruction with Zernike polynomials and eigenfunctions of Laplacian" (orthogonal gradients, cross-coupling): https://www.sciencedirect.com/science/article/abs/pii/S0030401812010759
- "More Zernike modes' open-loop measurement in the sub-aperture of the SH-WFS" (mode count vs sub-apertures): https://pubmed.ncbi.nlm.nih.gov/27828187/

**LS / SVD / regularisation / fast modal**
- Poyneer, "Fast modal wave-front reconstruction," UCRL-TR-204793: https://www.osti.gov/servlets/purl/15014303
- "A singular value decomposition for the Shack–Hartmann based wavefront reconstruction," *J. Comp. Appl. Math.*: https://www.sciencedirect.com/science/article/pii/S0377042711005206
- "Estimation of the total error of modal wavefront reconstruction with Zernike polynomials and Hartmann–Shack test": https://www.researchgate.net/publication/44387187

**Karhunen–Loève**
- KL-based turbulence simulation testbed (US Navy patents): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8725471 ; https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8452574
- "Modal wavefront sensorless AO with Karhunen–Loève functions" (COAT-2019): https://hal.science/hal-02887698/document

**Annular / non-circular pupils**
- Mahajan, "Zernike annular polynomials for imaging systems with annular pupils," *JOSA* 71, 75 (1981): https://opg.optica.org/josa/abstract.cfm?uri=josa-71-1-75
- Mahajan, "Zernike annular polynomials and optical aberrations of systems with annular pupils," *Appl. Opt.* 33, 8125 (1994): https://opg.optica.org/ao/abstract.cfm?uri=ao-33-34-8125 ; https://pubmed.ncbi.nlm.nih.gov/20963042/
- "Modal wavefront reconstruction by Schwarz–Christoffel mapping … for non-circular pupils": https://www.sciencedirect.com/science/article/abs/pii/S0143816624006213

**Bayesian / MAP / minimum-variance**
- "Comparison of minimum-norm ML and MAP wavefront reconstructions for large AO systems," *JOSA A* 26, 497 (2009): https://opg.optica.org/josaa/abstract.cfm?uri=josaa-26-3-497
- "Fast minimum variance wavefront reconstruction for extremely large telescopes," arXiv:1003.0274: https://arxiv.org/pdf/1003.0274
- "Inverse problem approach in Extreme AO: fitting error and lowering aliasing," arXiv:2301.03478: https://arxiv.org/pdf/2301.03478

**Modal vs zonal / hybrid / Fried geometry**
- "Zernike vs. Zonal Matrix Iterative Wavefront Reconstructor" (Lumetrics): https://www.lumetrics.com/hubfs/Lumetrics_August2021/PDF/Zernike-vs-Zonal.pdf
- WaveTrain/MZA AO geometry (Fried/Hudgin/Southwell): https://www.mza.com/doc/wavetrain/aogeom/main.htm
- "AO system based on the Southwell geometry…": https://www.sciencedirect.com/science/article/abs/pii/S0030401816311476

**Turbulence parameters (r0, L0, τ0) from SH-WFS / modal coefficients**
- Andrade et al., "Estimation of atmospheric turbulence parameters from Shack–Hartmann WFS measurements," *MNRAS* 483, 1192 (2019): https://academic.oup.com/mnras/article/483/1/1192/5203639 ; arXiv:1811.08396: https://arxiv.org/pdf/1811.08396
- "Integrated turbulence parameters' estimation from NAOMI AO telemetry," arXiv:2307.15178: https://arxiv.org/pdf/2307.15178
- "Simulating Anisoplanatic Turbulence by Sampling Inter-modal and Spatially Correlated Zernike Coefficients" (Noll covariance reproduced), arXiv:2004.11210: https://arxiv.org/pdf/2004.11210
- ten Brummelaar, "Temporal power spectra of Zernike coefficients," SPIE (1994): https://spie.org/Publications/Proceedings/Paper/10.1117/12.177258
- "The minimum of the time-delay wavefront error in AO" (Greenwood freq, τ0), arXiv:1906.03128: https://arxiv.org/pdf/1906.03128
- Tokovinin, "AO tutorial: turbulence" (turbulence stats, Δ_J context): https://www.ctio.noirlab.edu/~atokovin/tutorial/part1/turb.html

**Software references**
- HCIPy (open-source AO/coronagraph simulator; SH-WFS, modal bases): https://docs.hcipy.org/dev/
- POPPY / Zernike (note: POPPY uses standard circle Zernikes) — via Zernike review above and aotools.
