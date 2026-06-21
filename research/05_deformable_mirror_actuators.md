# Deformable-Mirror Actuator-Map Computation (with Inter-Actuator Coupling)

**ISRO BAH 2026 — Problem Statement 9 (Shack-Hartmann WFS reconstruction)**
**Research domain:** Deriving the actuator map `A(x,y)` (in units of actuator stroke) for a deformable mirror (DM) from the reconstructed wavefront, *explicitly incorporating inter-actuator coupling*, as required by PS9.

---

## 0. Where this fits in the PS9 pipeline

```
SH-WFS frames (.bmp)
   -> centroiding -> spot slopes s_x, s_y
   -> wavefront reconstruction -> W(x,y)         [other research module]
   -> turbulence params r0, tau0                 [other research module]
   -> CONJUGATE the wavefront: W_corr = -W        [THIS module starts here]
   -> solve for actuator commands a (stroke units) that make the DM
      surface reproduce W_corr/2, ACCOUNTING FOR inter-actuator coupling
   -> A(x_i, y_i) = actuator map fed to the DM
```

PS9 (idea.md, lines 5, 11, 27) states three things this module must satisfy:
1. *"The conjugate of this reconstructed wavefront is typically used to derive an actuator map in units of the actuator stroke length."*
2. *"The effect of inter-actuator coupling needs to be incorporated while deriving these actuator maps."*
3. The actuator grid of the DM and the lenslet grid of the MLA are arranged in **Fried geometry**, and the correction must run faster than the ~10 ms atmospheric timescale (so the per-frame computation must reduce to a **matrix-vector multiply**).

The central message of this report: **you cannot just sample the conjugate wavefront at the actuator positions and send those values as strokes.** Because each actuator's deformation spreads onto its neighbours (coupling), naive sampling over-corrects. You must **invert the coupling** — i.e. solve a linear system whose system matrix encodes the influence functions. The good news: that inverse is computed *once* (calibration / offline), and per-frame you only do a matrix-vector product, which is real-time-friendly.

---

## 1. DM influence functions and the Gaussian / coupling-coefficient model

### 1.1 Definition
The **influence function** (IF) of actuator *j* is the static surface shape the mirror takes when actuator *j* is poked by one unit of command while all others are held at zero (or at half-bias):

```
IF_j(x,y) = mirror surface produced by unit command on actuator j alone.
```

It is the "characteristic shape corresponding to the mirror response to the action of a single actuator" (Wikipedia, *Deformable mirror*; Tokovinin AO tutorial). For a continuous-facesheet DM the IF is a smooth bump centred on the actuator; for a segmented DM it is a localized piston/tip/tilt.

### 1.2 The coupling coefficient `c` (a.k.a. `w`, `omega`)
The **inter-actuator coupling coefficient** is the height of one actuator's influence function, sampled at the position of its *nearest neighbour*, expressed as a fraction of the peak:

```
c  =  IF_j(neighbour position) / IF_j(peak)         (typically 0.1 - 0.35)
```

Equivalently it is "the ratio of the deflection induced on a neighbour of a powered actuator to the maximum deflection of the powered actuator" (Boston Micromachines DM FAQ). Measured/representative values:

| DM technology | Coupling `c` | Source |
|---|---|---|
| Generic **Gaussian** IF (half-width ≈ pitch) | **≈ 0.27** | LLNL/Beamlet modeling (OSTI) |
| Boston Micromachines MEMS, 1.5 µm stroke | **0.15** | BMC DM FAQ |
| Boston Micromachines MEMS, 3.5 µm stroke | **0.13** | BMC DM FAQ |
| Boston Micromachines MEMS, 5.5 µm stroke | **0.22** | BMC DM FAQ |
| **ALPAO** magnetic (DM192) | **≈ 0.34–0.37** (per-actuator 0.32–0.37) | ALPAO/NAOMI characterisation |
| Classic continuous facesheet (textbook) | **0.1 – 0.2** | Hardy, *AO for Astronomical Telescopes* |

> **PS9 note:** the problem statement says *"DM information and inter-actuator coupling … shall be provided."* So `c` (and likely the measured IF or actuator pitch) is a **given dataset value** — read it from the provided DM spec and use it to build the IF model below. Do not assume a number; use theirs.

### 1.3 Gaussian influence-function model and how `c` sets the width
The most common analytic IF is a 2-D Gaussian (axisymmetric):

```
IF_j(x,y) = exp( - r^2 / (2 sigma^2) ),     r = sqrt((x-x_j)^2 + (y-y_j)^2)
```

To make this consistent with a *specified coupling* `c` at the actuator pitch `d` (nearest-neighbour spacing), set the Gaussian width `sigma` so that the bump has dropped to `c` at distance `d`:

```
exp( - d^2 / (2 sigma^2) ) = c
=>   sigma = d / sqrt( -2 ln c )                       (KEY relation)
```

Worked widths (pitch d = 1):

| coupling c | sigma = d / sqrt(-2 ln c) | width / pitch |
|---|---|---|
| 0.05 | 0.408 d | 0.41 |
| 0.10 | 0.466 d | 0.47 |
| 0.15 | 0.519 d | 0.52 |
| 0.20 | 0.557 d | 0.56 |
| 0.27 | 0.598 d | 0.60 |
| 0.35 | 0.673 d | 0.67 |

Note: **Soapy's `GaussStack`** simply hard-codes `width = actSpacing/2` (i.e. `sigma = d/2`), which corresponds to coupling `c = exp(-d^2/(2(d/2)^2)) = exp(-2) ≈ 0.135` — right in the typical band. (Source: `soapy/DM.py`, `GaussStack.makeIMatShapes`, uses `aotools.gaussian2d(N, width, cent=(x,y))` with `width = pupil_size/(nxActuators-1)/2`.)

### 1.4 More general parametric models (when a single Gaussian is a poor fit)
Real measured IFs are not perfectly Gaussian. Common refinements found in the literature:

- **Power-law (stretched-)exponential** — directly parameterised by coupling:
  ```
  IF(rho) = exp( ln(c) * (rho/d0)^alpha )
  ```
  where `c` = inter-actuator coupling, `d0` = pitch, and `alpha` (power index, typ. 1.5–4) controls the skirt steepness. With `alpha = 2` this *is* the Gaussian above. (Huang et al.; OOPAO/soapy-style models.)
- **Modified / double-Gaussian** — `A·exp(-(r/σ1)²) − B·exp(-(r/σ2)²)`, captures the slight negative "moat" some membrane/MEMS mirrors show. (Huang, *Modified Gaussian influence function of deformable mirror actuators*, Opt.\ Express 2008; example fit `A=0.9841, c=0.15, σ=8 px`.)
- **Thin-plate / plate-equation (Kirchhoff) model** — physically derived from the biharmonic plate equation `D ∇⁴ w = load`; IF ∝ Kelvin functions / `r²ln r` terms. Used for bimorph and continuous-facesheet mirrors when an FEM-grade model is needed. (Tokovinin tutorial; Hardy.)
- **Measured IFs** — the gold standard: poke each actuator, record the surface (interferometer or the SH-WFS itself), and store the sampled IF directly as a matrix column (Section 2). This automatically captures the *true* coupling and any asymmetry, at the cost of a calibration step.

---

## 2. Building the influence-function matrix `H` (surface = H·a)

Discretise the pupil/wavefront onto `M` sample points `(x_i, y_i)` (e.g. the WFS grid, or a dense map). For `N` actuators, define the **influence-function matrix** `H` (also written `F` or `B`):

```
H  is  M x N ,    H[i, j] = IF_j(x_i, y_i)
```

Each **column** of `H` is one actuator's influence function sampled on the grid; each **row** is "how all actuators contribute to surface point i."

The DM surface produced by a command vector `a ∈ R^N` (in stroke units) is the **linear superposition**:

```
S(x_i,y_i) = sum_j a_j * IF_j(x_i,y_i)        <=>        s = H a
```

with `s ∈ R^M` the sampled mirror surface. This is exactly the relation implemented in Soapy:
`dm_shape = (iMatShapes.T * actCoeffs.T).T.sum(0)` i.e. `s = Σ_j a_j · IF_j` (source: `soapy/DM.py`, `DM.makeDMFrame`). Wikipedia states the same as `s = F a` with "F a matrix describing each actuator's influence function."

Complexity to *form* `H`: O(M·N) storage, built once. For a continuous DM, IFs overlap only locally, so `H` is **banded/sparse** (each surface point sees only the few nearest actuators) — this is exploitable for speed and memory.

---

## 3. Solving for actuator commands: deconvolving the coupling (`a = H⁺·s_target`)

### 3.1 The target surface (conjugate + factor of two)
PS9 wants the DM to *cancel* the reconstructed wavefront `W`. Phase conjugation means the DM must imprint `-W` on the beam. **Reflection doubles the optical path**: a mirror surface displaced by `z` changes the wavefront OPD by `2z` (for near-normal incidence). Therefore the required *mechanical surface* (stroke) target is **half** the conjugate wavefront:

```
target surface:   s_target(x,y) = - W(x,y) / 2            (mechanical stroke units)
```

(RP-Photonics, Wikipedia: "the local displacement needed of the DM surface is approximately equal to half the path-length variations of the aberrated wavefront." The exact factor is `1/(2 cos θ)` for incidence angle θ.) Keep this factor of 2 explicit — it is a common bug and it directly affects the *stroke-length units* PS9 asks for.

### 3.2 Why naive sampling is WRONG (the heart of the coupling requirement)
Tempting (and incorrect) approach: set `a_j = s_target(x_j,y_j)` — just read the conjugate map at each actuator. This ignores that neighbours also push surface point `x_j`. Because of coupling, the *actual* surface at `x_j` becomes `a_j + c·(sum of neighbour commands) + ...`, which **over-shoots**. The map you sent is not the map you get. To get the surface you want, you must **invert** the coupling operator `H`.

### 3.3 Least-squares / pseudo-inverse solution
Solve `H a = s_target` in the least-squares sense (usually `M ≥ N`, overdetermined):

```
minimize || H a - s_target ||^2
=>  a_hat = H^+ s_target ,     H^+ = (H^T H)^{-1} H^T   (Moore-Penrose pseudo-inverse)
```

Substituting the PS9 target:

```
==========================================================
   a  =  H^+ * ( - W / 2 )           [ACTUATOR MAP, stroke units]
==========================================================
```

- `H^+` is `N x M`, computed **once** (offline) via SVD.
- Per frame: one matrix-vector multiply `H^+ · s_target` — O(M·N) flops, microseconds on a CPU/GPU. This is the real-time path PS9 needs.
- This is sometimes phrased as the **command matrix** `C = H^+` (model-based) or, in calibration-based systems, `C` = pseudo-inverse of the measured *interaction* matrix (Section 6).

If the actuator grid coincides with the surface grid (square `N=M`) the coupling reduces to a square **coupling matrix** `C_coup` and the solve is a clean `a = C_coup^{-1} s_target` — this is the deconvolution-of-coupling step made explicit (worked example in Section 9).

### 3.4 Modal alternative (Zernike route)
Equivalently, since PS9 already produces Zernike coefficients, you can build a **mode-to-actuator** matrix once: project each Zernike mode `Z_k` onto the actuators (`a^(k) = H^+ Z_k`), assemble `M2A = [a^(1) ... a^(K)]`, then per frame `a = M2A · z` where `z` is the Zernike vector of `-W/2`. Same MVM cost; numerically nicer because you can truncate poorly-corrected high-order modes.

---

## 4. Tikhonov / regularized inversion, conditioning, and stroke saturation

### 4.1 Ill-conditioning of `H`
`H^T H` can be ill-conditioned: overlapping IFs make some actuator combinations nearly degenerate (notably **waffle / piston-like** patterns), so the plain pseudo-inverse amplifies noise into huge, useless commands. Diagnose via the **condition number** `κ(H) = σ_max/σ_min` and the singular-value spectrum.

### 4.2 Tikhonov (damped least squares)
Add a penalty on command magnitude:

```
minimize  || H a - s_target ||^2 + mu^2 || a ||^2
=>  a = (H^T H + mu^2 I)^{-1} H^T s_target
```

`mu` (the regularization / damping parameter) suppresses the smallest singular values that "are responsible for unnecessarily large control voltages without any benefit to the fitting error" (this is exactly the damped-least-squares / Tikhonov method, widely used for DM control — see MDPI unimorph-DM paper, dual-DM AO). In SVD form:

```
a = sum_k [ sigma_k / (sigma_k^2 + mu^2) ] (u_k^T s_target) v_k
```

Choose `mu` by an L-curve, GCV, or simply "truncate modes with `σ_k < threshold`" (**truncated SVD**, the most common AO practice — set low-response modes to zero). A weighted prior `Σ^{-1}` instead of `I` yields the **MMSE / minimum-variance** reconstructor when turbulence + noise statistics are known.

### 4.3 Stroke saturation / clipping (edge actuators)
Physical actuators have a finite stroke `±a_max` (e.g. ±1.5 to ±5.5 µm — see Section 7). After computing `a`:
- **Clip:** `a_j <- clip(a_j, -a_max, +a_max)`. Simple but introduces a fitting residual and can excite edge ringing.
- **Constrained least squares / projection:** re-solve with box constraints (NNLS-style or active-set) so the *clipped* solution still best fits `s_target`. Better when many actuators saturate.
- **Remove un-correctable content first:** subtract global piston (DM cannot do piston usefully) and, if needed, tip/tilt handled by a separate TT stage; this lowers required stroke. Edge/slaved actuators (just outside the pupil) are often "slaved" to their inner neighbours to avoid runaway. Regularization (4.2) also keeps commands inside the stroke envelope and is the first line of defence against saturation.

---

## 5. Fried geometry: actuator-grid ↔ lenslet-grid registration

PS9 explicitly fixes the **Fried geometry** (idea.md line 25). Definitions and consequences:

- **Layout:** SH-WFS sub-apertures (lenslets) and DM actuators both on square grids, but offset so that **actuators sit at the *corners* of the sub-apertures**. A grid of `n × n` lenslets is bordered by `(n+1) × (n+1)` actuators. (E.g. 15×11 sub-apertures ↔ 16×12 actuators.)
- **Why Fried:** each lenslet measures the *average slope* over its sub-aperture; placing actuators at the corners means a sub-aperture's two diagonal slope measurements relate cleanly to the four surrounding actuator heights — the geometry that the classic zonal reconstructors (Fried 1977) were derived for. It gives a well-posed slope→phase relation and matches how the DM influences the measured slopes.
- **Actuator pitch vs sub-aperture size:** in Fried geometry the **actuator pitch `d` equals the sub-aperture (lenslet) pitch** projected onto the DM. This single number `d` is what enters both the fitting-error formula and the Gaussian-width relation (Section 1.3).
- **Number of actuators vs sub-apertures:** roughly `N_act ≈ N_subap`, and both scale as `(D/r0)^2` to keep fitting error low (Section 8). For a `D`-diameter pupil, `n ≈ D/d`.
- **Registration / alignment matters:** the IF matrix `H` (and any interaction matrix) is only valid if the lenslet spots and actuator footprints are co-registered (translation, rotation, magnification, and pupil clocking all calibrated). Mis-registration is a leading error term; in practice you measure the registration during the interaction-matrix calibration (Section 6).
- **Caveat — waffle mode:** the Fried geometry is *blind* to the "waffle" pattern (checkerboard +/- on the two diagonals): the SH-WFS slopes for that DM mode are ~zero, so it is unsensed and can build up. Standard fix: **filter/penalise the waffle mode** in the reconstructor (remove it from `H^+` via SVD, or add an anti-waffle penalty). Southwell (actuators co-located with sub-aperture centres) and Hudgin geometries avoid waffle but have other downsides (poorer actuator sensing / need overlapping WFS). Since PS9 mandates Fried, **explicitly null the waffle mode in the command matrix.**

---

## 6. Interaction matrix + command matrix (calibration-based alternative to the model)

Instead of an analytic `H`, measure the DM↔WFS response directly:

1. **Poke** actuator `j` by a known command (push–pull: `+δ` then `−δ`, average to cancel bias/nonlinearity).
2. Record the resulting **WFS slope vector** `Δs_j` (length `2·N_subap`).
3. Assemble the **interaction matrix** `D` (a.k.a. `IM`): `D[:, j] = Δs_j / δ`. So `slopes = D · a`.
4. Invert (regularised SVD / truncated SVD) to get the **command (control) matrix**:
   ```
   C = D^+        (pseudo-inverse, N x 2N_subap)
   a = C · s_meas        (commands from measured slopes)
   ```

This is the workhorse of real telescope AO: the per-frame operation `a = C · s` is the matrix-vector multiply that runs at kHz rates (e.g. NAOMI, Gemini, THEMIS RTCs all do "slopes → commands" by MVM). Compared with the model-based `H`:
- **Pros:** captures *real* coupling, IF asymmetry, and optical registration automatically; no model error.
- **Cons:** needs a calibration source/procedure; push–pull is noisy for high-order systems (mitigated by Hadamard-multiplexed pokes, sinusoidal modulation, or on-sky methods like DO-CRIME).

For PS9, since the dataset is laboratory frames with a *given* DM coupling, you can use **either**: build `H` from the provided coupling/IF model (Section 2), *or*, if poke frames are provided, build `D` empirically. The two converge; the interaction-matrix route is closest to operational practice and inherently "incorporates inter-actuator coupling."

> **Important distinction:** the calibration matrix `C = D^+` maps *measured slopes → commands* in closed loop. The model matrix `H^+` maps a *desired surface (−W/2) → commands*. PS9 phrasing ("use the conjugate of the reconstructed wavefront to derive an actuator map") matches the **`H^+` surface-fitting route**; the interaction-matrix route is the closed-loop equivalent. Both are valid; state which you use.

---

## 7. Conversion to physical stroke-length units (µm / volts)

PS9 wants the actuator map *in units of actuator stroke length*. Steps:

1. **Normalise the IF model:** define IF peak = 1 so that command `a_j` is "fraction of unit stroke," OR define the IF in physical units directly (peak = measured µm/command).
2. **Stroke calibration constant:** from the DM spec/dataset, the per-actuator gain `g` (e.g. µm of surface per volt, or µm per DAC count). Then physical surface stroke `z_j[µm] = g · a_j` (linear-DM approximation; ALPAO/PZT roughly linear, MEMS quadratic in voltage so a `volt = sqrt(stroke)` lookup may be needed).
3. **Factor of two:** remember `surface_stroke = OPD/2` (Section 3.1). So if `W` is given as wavefront (OPD) in µm or in radians, convert: `OPD[µm] = W[rad]·λ/(2π)`, then `z_target = -OPD/2`.
4. **Output the map** either as (a) mechanical stroke in µm, (b) fraction of max stroke (−1…+1), or (c) drive voltage/DAC counts via the calibration curve — whichever the DM dataset format requires. Report units explicitly.

Typical stroke envelopes to sanity-check against (and to set `a_max` for clipping): Boston MEMS 1.5 / 3.5 / 5.5 µm; ALPAO up to ~30 µm tip/tilt and ~5 µm inter-actuator; classic PZT stack ~ a few µm. (Sources: BMC FAQ, ALPAO product pages.)

---

## 8. Fitting error, coupling-induced residual, achievable correction

Even a perfect inversion leaves a **fitting error** because a finite actuator grid cannot represent spatial frequencies above the actuator Nyquist (`1/2d`):

```
sigma_fit^2 = mu * (d / r0)^{5/3}        [rad^2, Kolmogorov turbulence]
```

- `d` = actuator pitch (= sub-aperture pitch in Fried geometry), `r0` = Fried parameter (from the turbulence module).
- `mu` depends on the IF shape (i.e. on coupling!):
  - `mu ≈ 0.28–0.29` for an idealized/continuous corrector (Hudgin/Fried fits, "0.27(d/r0)^{5/3}");
  - `mu ≈ 0.34` for a continuous DM with realistic Gaussian-like IFs;
  - `mu ≈ 1.26` for pure piston (segmented) correction.
  (Tokovinin; Hardy; multiple AO error-budget references.)
- **Implication:** choosing `d` (hence `N_act ≈ (D/d)^2 ≈ (D/r0)^2`) trades correction quality against actuator count. To reach a target Strehl `S ≈ exp(-σ²)`, size the grid so `σ_fit²` is acceptable.

**Coupling's specific effect on achievable correction (the part PS9 cares about):**
- Coupling makes IFs *broader/smoother* → the DM is good at low spatial frequencies but the IF basis is **less able to represent sharp, high-order shapes**; the effective `mu` rises slightly and high-frequency fitting residual grows.
- Coupling makes `H` **more ill-conditioned** (neighbouring IFs more similar → smaller singular values), so more regularization is needed, which itself leaves residual. Too *little* coupling (very peaked IFs) leaves "bumpy" inter-actuator residual ("print-through"); too *much* coupling reduces independent degrees of freedom and hurts stability/correction quality. There is an optimum; literature finds the slope-response matrix sparseness and AO stability both degrade at coupling extremes (Wang/Wulixb studies; OSTI). Practically, **the inversion in Section 3 already compensates the *mean* coupling**; what remains is the high-order residual set by `mu(d/r0)^{5/3}` plus regularization/saturation residual.

---

## 9. Worked example: small actuator grid, Gaussian IF, explicit coupling matrix & inverse

To make the deconvolution concrete, take the simplest illustrative case: **actuator grid = surface sample grid** (so `M = N`, square system), coupling only to *nearest neighbours*, and a Gaussian IF with coupling `c`.

### 9.1 1-D, 3-actuator example (clearest)
Actuators at positions 0,1,2 (pitch d=1). Gaussian IF with coupling `c` at unit spacing. The **coupling matrix** `H` (here a square "coupling matrix" `C_coup`) has entry `H[i,j] = IF_j(position i) = c^{(i-j)^2}` (since Gaussian → `exp(-Δ²/(2σ²))` and `exp(-1/(2σ²))=c`, so distance-2 gives `c^4`):

Take `c = 0.15` (Boston-like). Then `c = 0.15`, `c^4 = 0.000506 ≈ 0`:

```
        a0     a1     a2
   p0 [ 1.000  0.150  0.001 ]
H= p1 [ 0.150  1.000  0.150 ]
   p2 [ 0.001  0.150  1.000 ]
```

Desired surface (conjugate, after the /2 and unit-stroke normalisation), say `s_target = [0.5, 1.0, 0.5]^T`.

**Naive (wrong):** `a = s_target = [0.5, 1.0, 0.5]`. Check the surface it actually makes: `H·s_target = [0.5+0.15+0.0005, 0.075+1.0+0.075, ...] = [0.650, 1.150, 0.650]` — **over-shot by ~15–30%** because neighbours added in. This is precisely the coupling error PS9 warns about.

**Correct (deconvolve coupling):** invert `H`.
```
H^{-1} ≈
   [  1.0234  -0.1539   0.0215 ]
   [ -0.1539   1.0470  -0.1539 ]
   [  0.0215  -0.1539   1.0234 ]

a = H^{-1} s_target
  = [ 1.0234*0.5 -0.1539*1.0 +0.0215*0.5,
     -0.1539*0.5 +1.0470*1.0 -0.1539*0.5,
      0.0215*0.5 -0.1539*1.0 +1.0234*0.5 ]
  = [ 0.369 , 0.893 , 0.369 ]
```
So the correct commands are **smaller** than the naive samples (0.369 vs 0.5, 0.893 vs 1.0): each actuator is *backed off* to leave room for what its neighbours add. Verify: `H·a = [0.369+0.134+~0, 0.055+0.893+0.055, ...] ≈ [0.503, 1.003, 0.503] ✓` reproduces `s_target`. The off-diagonal `−0.154` terms in `H^{-1}` are the **explicit "anti-coupling"** — the inverse subtracts neighbour contributions. That is the deconvolution of inter-actuator coupling, in numbers.

### 9.2 2-D, 3×3 grid sketch
For a 3×3 actuator grid (9 actuators), order them row-major. The 9×9 coupling matrix `H` has:
- diagonal = 1 (self),
- `c` for the 4-neighbour (edge-adjacent) couplings,
- `c²` for the diagonal neighbours (distance √2 → Gaussian gives `c^{(√2)²}=c²`),
- ≈0 beyond.

With `c = 0.15`: edge neighbours 0.15, diagonal neighbours 0.0225. `H` is a sparse, symmetric, banded (block-tridiagonal) matrix. The actuator map is again `a = H^{-1} s_target` (or `H^+ s_target` if you sample the surface on a denser grid than the actuators). For real `N`~hundreds, `H^{-1}` (or `H^+`) is precomputed once; per-frame cost = one `N×M` MVM.

### 9.3 Effect of larger coupling
Repeat 9.1 with ALPAO-like `c = 0.35`: off-diagonal of `H` = 0.35, the inverse's anti-coupling terms grow to ≈ −0.45, the correct commands shrink further (≈ 0.30, 0.78, 0.30), and `κ(H)` rises — illustrating why higher coupling needs more care / regularization (Section 4, 8).

---

## 10. Comparison table of methods/components

| # | Method / component | Core equation | Matrices (size) | Build cost | Per-frame (real-time) cost | Notes / suitability for PS9 |
|---|---|---|---|---|---|---|
| 1 | Gaussian IF model | `IF=exp(-r²/2σ²)`, `σ=d/√(-2ln c)` | per-actuator scalar params | trivial | n/a (offline) | Simple, needs only `c`,`d` (PS9 provides `c`). Good default. |
| 1b| Power-law / modified-Gaussian / thin-plate IF | `IF=exp(ln c·(r/d)^α)` etc. | params | trivial | n/a | Use if measured IF ≠ Gaussian. |
| 2 | Influence matrix `H` (surface=H·a) | `s = H a` | `H: M×N` (sparse/banded) | O(MN) once | — | Foundation for everything below. |
| 3 | Least-squares command (model) | `a = H⁺(−W/2)` | `H⁺: N×M` | SVD once | **MVM O(MN)** | The PS9 "conjugate→actuator map" route. |
| 4 | Tikhonov / truncated-SVD | `a=(HᵀH+μ²I)⁻¹Hᵀs` | same | SVD once | MVM | Needed for conditioning + stroke control. **Use it.** |
| 4b| Stroke clipping / constrained LS | `clip(a,±a_max)` / box-LS | — | — | O(N) clip | Handle saturation, edge actuators. |
| 5 | Fried-geometry registration | actuators at sub-ap corners; pitch `d` | mapping/index tables | once | — | **Mandated by PS9**; null waffle mode. |
| 6 | Interaction + command matrix (calibration) | `D=∂s/∂a`, `C=D⁺`, `a=C·s_meas` | `D: 2N_sub×N`, `C: N×2N_sub` | poke + SVD once | **MVM O(N·N_sub)** | Operational standard; auto-includes real coupling. Use if poke data given. |
| 7 | Stroke-unit conversion | `z[µm]=g·a`, `z=−OPD/2` | gain `g`, λ | calib once | O(N) | Produces the *required units*; keep factor 2. |
| 8 | Fitting / coupling residual | `σ²_fit=μ(d/r0)^{5/3}` | — | — | — | Error-budget metric; ties to r0 module. |

---

## 11. Recommended approach for PS9 (with inter-actuator coupling)

**Pipeline (offline calibration once → real-time per frame):**

**A. Offline / calibration (run once, using the provided DM info):**
1. Read the **provided DM dataset**: number of actuators `N`, actuator pitch `d`, and the **inter-actuator coupling `c`** (and the measured IF if supplied). Read MLA geometry to confirm **Fried** registration (actuators at sub-aperture corners; `d` = lenslet pitch on the DM).
2. Build the **influence-function matrix `H`** (M×N), each column a Gaussian IF `exp(-r²/2σ²)` with `σ = d/√(-2 ln c)` (use the provided `c`). If poke/interaction data is provided instead, build the empirical interaction matrix `D` and use `C=D⁺` — this is preferred because it captures *real* coupling and registration. If only `c` is given, the model `H` is the way.
3. Compute the **regularized command matrix once**: `C_cmd = (HᵀH + μ²I)⁻¹ Hᵀ` (Tikhonov) or truncated-SVD pseudo-inverse `H⁺`. While doing the SVD, **explicitly remove/penalise the waffle mode** (Fried-geometry blind spot) and any modes with singular value below threshold.
4. Store the per-actuator **stroke calibration** `g` (µm or volts per unit command) and the **stroke limit `a_max`**.

**B. Real-time, per SH-WFS frame (the ~10 ms budget):**
5. From the reconstructed wavefront `W` (other module), form the **conjugate target surface**: `s_target = −W/2` (factor 2 for reflection; convert radians→µm via `λ/2π` if needed).
6. **One matrix-vector multiply:** `a = C_cmd · s_target`. *(If using the closed-loop interaction-matrix form, instead apply `a = C·s_meas` directly on slopes with an integrator gain.)* This single MVM is the only heavy per-frame op → meets the real-time requirement; implement in C/BLAS or GPU as PS9 suggests.
7. **Saturation handling:** `a ← clip(a, ±a_max)` (or constrained re-solve if many saturate); subtract uncontrollable piston.
8. **Convert to stroke units:** `A(x_i,y_i) = g · a_i` → output the **actuator map in stroke length** (the PS9 deliverable). Optionally also emit voltage/DAC via the calibration curve (MEMS: quadratic).

**Why this satisfies the coupling requirement:** the actuator map is *not* the sampled conjugate wavefront — it is `H⁺(−W/2)`, where `H⁺` contains the **explicit anti-coupling off-diagonal terms** (Section 9). Each actuator is automatically backed off to account for what its neighbours add, i.e. the inter-actuator coupling is *inverted/deconvolved*, exactly as PS9 mandates. The residual that remains is the fundamental fitting error `σ²_fit=μ(d/r0)^{5/3}` (report it as a quality metric, linking to the `r0` from the turbulence module).

**Concrete defaults if a value is missing from the dataset:** coupling `c≈0.15` (≙ Soapy `σ=d/2`, ≙ Boston low-stroke); Tikhonov `μ` from an L-curve (start at `μ ≈ 0.01·σ_max`); null waffle; `a_max` from DM spec. State every assumption.

---

## 12. Sources (URLs)

**Influence functions, Gaussian model, coupling coefficient**
- Tokovinin, *AO tutorial part 2 — Deformable mirrors*, CTIO/NOIRLab: https://www.ctio.noirlab.edu/~atokovin/tutorial/part2/dm.html
- LLNL/Beamlet, *Modeling for Deformable Mirrors and the Adaptive Optics System* (OSTI): https://www.osti.gov/servlets/purl/492017
- Huang et al., *Modified Gaussian influence function of deformable mirror actuators*, Opt. Express (PubMed): https://pubmed.ncbi.nlm.nih.gov/18521137/
- *A novel model of influence function: calibration of a continuous membrane DM* (ResearchGate): https://www.researchgate.net/publication/216852286
- *Influence Function Measurement of Continuous Membrane DM Actuators Using Shack-Hartmann Sensor* (ResearchGate): https://www.researchgate.net/publication/215880005
- *Influence of coupling coefficient on sparseness of slope response matrix and iterative matrix* (ResearchGate): https://www.researchgate.net/publication/278093616
- *Influence of Gaussian function index of DM on iterative algorithm AO system*, Acta Phys. Sinica: https://wulixb.iphy.ac.cn/en/article/id/63986
- *Characterisation of the influence-function non-additivities for a 1024-actuator MEMS DM* (arXiv): https://arxiv.org/pdf/1001.5048
- Wikipedia, *Deformable mirror* (IF, coupling, `s=Fa`, `a=F⁺s`): https://en.wikipedia.org/wiki/Deformable_mirror

**Command computation / pseudo-inverse / interaction & command matrices**
- Soapy DM module docs (GaussStack): https://soapy.readthedocs.io/en/latest/dms.html
- Soapy `DM.py` source (GaussStack, `gaussian2d`, `makeDMFrame`): https://github.com/AOtools/soapy/blob/master/soapy/DM.py
- AOtools paper (Python AO package): https://www.researchgate.net/publication/336543057
- *Analytical modelling of AO systems: role of the influence function* (arXiv HTML): https://arxiv.org/html/2306.10803
- OOPAO (Object-Oriented Python AO): https://github.com/cheritier/OOPAO
- *Real-time modal control implementation for AO* (ApOpt 1998): http://www.caha.es/CAHA/Instruments/ALFA/PAPERS/aoVol37N21.pdf
- *Closing the loop as an inverse problem: real-time control of THEMIS AO* (arXiv): https://arxiv.org/pdf/2311.17779
- *DO-CRIME: dynamic on-sky interaction-matrix evaluation* (arXiv): https://arxiv.org/pdf/2011.14705
- *High-precision system identification method for a DM in wavefront control* (Optica/AO): https://opg.optica.org/ao/abstract.cfm?uri=ao-54-14-4313

**Fried geometry / reconstruction geometries / waffle**
- Gemini AO reconstructor design (Fried geometry): https://www.gemini.edu/sciops/instruments/altair/les_spie98.pdf
- *AO system based on Southwell geometry & control stability* (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S0030401816311476
- Prabhudesai, *AO reconstruction methods* (IfA Hawaii, Fried/Hudgin/Southwell + waffle): https://www2.ifa.hawaii.edu/Robo-AO/docs/smp_ao_22aug2011_talk.pdf
- *Analysis of Fundamental Waffle Mode in AEOS AO Images* (arXiv): https://arxiv.org/pdf/astro-ph/0505195
- Max, *Adaptive Optics: An Introduction* (UCSC): https://www.ucolick.org/~max/289/Assigned%20Readings/Max_Adaptive_Optics_Intro_v1.pdf

**Tikhonov / regularization / conditioning / saturation**
- *Shape Control of a Unimorph DM under Uncertainties* (MDPI, Tikhonov/damped LS): https://doi.org/10.3390/mi14091756
- *Wavefront-aberration sorting & correction for dual-DM AO* (PMC, damped LS, strokes 2.5/50 µm): https://pmc.ncbi.nlm.nih.gov/articles/PMC2738988/

**DM hardware: coupling, pitch, stroke (units)**
- Boston Micromachines, *Deformable Mirror FAQ* (coupling 13/15/22%, IF def., pitch/stroke): https://bostonmicromachines.com/products/deformable-mirrors/deformable-mirror-faq/
- ALPAO Deformable Mirrors (product, stroke/pitch/coupling): https://www.alpao.com/products-and-services/deformable-mirrors/
- *Tests and characterisations of the ALPAO 64×64 DM* (AO4ELT6): https://ao4elt6.copl.ulaval.ca/proceedings/401-PhKb-251.pdf
- *Characterisation of ALPAO DMs for NAOMI VLTI ATs* (arXiv, coupling ≈34.5%): https://arxiv.org/pdf/1806.10552
- *Characterization of deformable mirrors for MagAO-X* (arXiv): https://arxiv.org/pdf/1807.04370
- Thorlabs MEMS / PZT DM specs: https://www.thorlabs.de/newgrouppage9.cfm?objectgroup_id=3258
- RP-Photonics, *Deformable mirrors* (½-OPD reflection factor): https://www.rp-photonics.com/deformable_mirrors.html

**Fitting error / error budget**
- *Numerical estimation of wavefront error breakdown in AO* (A&A): https://www.aanda.org/articles/aa/full_html/2018/08/aa32579-18/aa32579-18.html
- *Deformable mirror fitting error by correcting the segmented wavefronts* (AO4ELT): https://ao4elt.edpsciences.org/articles/ao4elt/pdf/2010/01/ao4elt_06008.pdf
- Hardy, *Adaptive Optics for Astronomical Telescopes* (Oxford, 1998) — standard text for `σ²_fit=μ(d/r0)^{5/3}`, coupling 0.1–0.2 (book; see UCSC Astro 289 notes: https://www.ucolick.org/~max/289/ ).

---

*Prepared for BAH 2026 PS9. Key takeaways: (1) actuator map = `H⁺·(−W/2)`, not sampled conjugate; (2) `H⁺` (or `C=D⁺`) deconvolves inter-actuator coupling via explicit anti-coupling off-diagonal terms; (3) precompute the (regularized) command matrix once → per-frame is a single matrix-vector multiply (real-time); (4) keep the factor-of-2 reflection and the stroke-calibration gain to output the map in true stroke-length units; (5) Fried geometry ⇒ null the waffle mode.*
