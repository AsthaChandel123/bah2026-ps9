# Zonal Wavefront Reconstruction for Shack–Hartmann WFS (PS9 — Fried Geometry)

**Deep-research report — domain: ZONAL reconstruction, sensor/actuator geometry, Fried arrangement**
ISRO Bharatiya Antariksh Hackathon 2026, Problem Statement 9.
Date: 2026-06-21.

> Scope reminder (from `idea.md`): For each SH-WFS frame we measure spot displacements → x/y slopes. We must reconstruct the wavefront phase map W(x,y) **fast** (atmospheric coherence ~10 ms), then derive an actuator map for the DM. **The DM actuator grid and MLA lenslet grid are in FRIED geometry**, and **inter-actuator coupling** must be incorporated. Real-time / C-friendly is a hard evaluation criterion.

---

## 0. Executive summary / TL;DR

- **Zonal reconstruction** recovers phase values on a grid of points directly from local slopes, via a **sparse linear system `s = Γ φ`** (Γ = "geometry"/gradient matrix). Contrast with **modal** reconstruction (fit Zernike/KL coefficients), covered in a sibling report.
- The slope-sampling **geometry** (where x/y slopes sit relative to the phase points) fixes the structure of Γ. The three canonical geometries are **Hudgin (1977)**, **Fried (1977)**, and **Southwell (1980)**.
- **Fried geometry is what PS9 requires**: each lenslet measures the *average* slope over a square sub-aperture, and the phase/actuator points sit at the **four corners** of that sub-aperture. This is the natural Shack–Hartmann + continuous-facesheet-DM layout. **Critical caveat:** Fried geometry has a null mode — the **waffle (checkerboard) mode** — which is invisible to the sensor and must be actively filtered out of the reconstructor, or it grows unbounded in closed loop.
- **The real-time trick** is to *precompute* a reconstruction matrix `R` (the regularized pseudo-inverse of Γ, with piston + waffle nulled) **once**, so each frame is a single **matrix–vector multiply** `φ = R s` — O(N²) flops but "O(1) per actuator" of *online algebra*, fully BLAS/GPU-friendly. For very large N, swap the dense MVM for an **FFT-based Fourier Transform Reconstructor (FTR)** at O(N log N), or a **CuReD** integrator at O(N).
- **Recommendation for PS9** (see §15): **Build a dense, precomputed Fried least-squares reconstructor `R` with waffle+piston regularization (SVD/Tikhonov)** as the primary real-time path (lab-scale grids ≤ ~30×30, dense MVM is trivially fast in C/BLAS), and validate it against an **FFT Fourier Transform Reconstructor with the exact Fried filter** as a fast cross-check and a scaling fallback. Fold DM inter-actuator coupling in as a second precomputed matrix.

---

## 1. The measurement model: slopes from spot displacements

A Shack–Hartmann lenslet of focal length `f_MLA` images the local wavefront tilt to a spot shift. For sub-aperture *k*:

```
Δx_k = f_MLA · θx_k ,   Δy_k = f_MLA · θy_k
```

where `(Δx_k, Δy_k)` is the centroid displacement (in metres on the detector) from the reference position, and `(θx_k, θy_k)` are the local wavefront tilt angles. The measured **slope** (mean wavefront gradient over the sub-aperture of width `d`) is

```
s^x_k = <∂W/∂x>_k = θx_k = Δx_k / f_MLA
s^y_k = <∂W/∂y>_k = θy_k = Δy_k / f_MLA
```

Units: if W is an optical path difference (metres), slopes are dimensionless (rad of tilt). If W is a phase (radians), then `s = (2π/λ)·OPD-slope`. Stack all measurements into a vector

```
s = [ s^x_1 … s^x_M , s^y_1 … s^y_M ]ᵀ   ∈ ℝ^{2M}      (M = number of valid sub-apertures)
```

The reconstruction problem: given `s`, estimate the phase samples `φ = [φ_1 … φ_N]ᵀ` on a grid (N phase points). Because a constant offset (**piston**) produces zero slope everywhere, φ is only determined **up to an additive constant** — the piston is in the **null space** of every slope reconstructor. (Fried geometry adds a *second* null mode: waffle.)

The fundamental linear relation is

```
    s = Γ φ  +  noise
```

with Γ the **2M×N geometry (gradient) matrix**. Zonal reconstruction = invert this (in least squares).

---

## 2. The three sampling geometries (Hudgin, Fried, Southwell)

This is the crux of "sensor/actuator geometry." Let phase points live on an (n+1)×(n+1) grid with spacing `h = d` (the sub-aperture pitch). Index phase φ_{i,j}; index slopes by sub-aperture.

### 2.1 ASCII geometry diagrams

```
   HUDGIN                         FRIED                         SOUTHWELL
   (slopes on edge midpoints,     (slopes at CELL CENTRE,       (slopes co-located
    x & y at DIFFERENT places)     phase at 4 CORNERS)           with phase points)

  φ───sx──φ───sx──φ            φ───────────φ                  φ───────────φ
  │       │       │            │           │                  │  (sx,sy)  │
  sy      sy      sy           │  (sx,sy)  │                 (sx,sy)     (sx,sy)
  │       │       │            │     •     │                  │           │
  φ───sx──φ───sx──φ            │           │                  │  (sx,sy)  │
  │       │       │            φ───────────φ                  φ───────────φ
  sy      sy      sy
  │       │       │           one SH lenslet gives BOTH        phase & both slopes
  φ───sx──φ───sx──φ           sx and sy at the cell centre;    sampled at the SAME
                              phase estimated at corners        nodes (need interp)
  x-slope = 1st difference
  along an edge; y-slope
  on the orthogonal edge
```

- **Hudgin (1977):** slope = *first difference* of phase between two **adjacent** grid points; x-slopes live on horizontal edges, y-slopes on vertical edges. x and y slopes are **not co-located** → does not match a real SH lenslet (which gives x AND y at the same place). [Hudgin, JOSA 67, 375 (1977)]
- **Fried (1977):** the SH lenslet measures *both* x and y slopes at the **centre** of a square cell; the cell's **four corners** are the phase-estimation points. The slope is the **average of the two parallel edge first-differences**. This is the **standard Shack–Hartmann / continuous-DM layout** and is what PS9 mandates. [Fried, JOSA 67, 370 (1977)]
- **Southwell (1980):** phase points and slope points are **co-located** on the same grid; the model relates the **average of two neighbouring slopes** to the phase difference. Best noise propagation; most popular in optical metrology/optical testing. [Southwell, JOSA 70, 998 (1980)]

### 2.2 The slope–phase finite-difference equations

**Hudgin** (x-slope on edge between (i,j) and (i+1,j)):
```
s^x_{i,j} = (φ_{i+1,j} − φ_{i,j}) / h ,    i = 1..n,  j = 1..n+1
s^y_{i,j} = (φ_{i,j+1} − φ_{i,j}) / h ,    i = 1..n+1, j = 1..n
```
(Sources confirm `S^x_{ij}=(Φ_{i+1,j}−Φ_{i,j})/h`, `S^y_{ij}=(Φ_{i,j+1}−Φ_{i,j})/h`.)

**Fried** (slope at centre of cell whose corners are (i,j),(i+1,j),(i,j+1),(i+1,j+1)) — the slope is the **average of the two parallel edge differences**:
```
s^x_{i,j} = 1/(2h) [ (φ_{i+1,j} − φ_{i,j}) + (φ_{i+1,j+1} − φ_{i,j+1}) ]
s^y_{i,j} = 1/(2h) [ (φ_{i,j+1} − φ_{i,j}) + (φ_{i+1,j+1} − φ_{i+1,j}) ]
```
Equivalently, with the four corner phases (a,b,c,d) and pitch d:
```
s^x = (1/2d)[ (φ_b − φ_a) + (φ_d − φ_c) ]
s^y = (1/2d)[ (φ_c − φ_a) + (φ_d − φ_b) ]
```
This **averaging of two first-differences** is exactly what makes Fried blind to waffle (the +1/−1 checkerboard cancels in every average; see §6).

**Southwell** (co-located; average of two adjacent slopes = phase difference / h):
```
(s^x_{i+1,j} + s^x_{i,j}) / 2 = (φ_{i+1,j} − φ_{i,j}) / h
(s^y_{i,j+1} + s^y_{i,j}) / 2 = (φ_{i,j+1} − φ_{i,j}) / h
```
This is a **central-difference / quadratic-spline** model: it assumes the surface between two sample points is a 2nd-order polynomial, hence the slope average equals the secant. [Southwell 1980; Korea JKPS 70, 469.]

> **Why geometry matters numerically.** The same physical slopes plugged into the wrong geometry matrix gives a biased phase. For PS9, because the DM actuators sit at sub-aperture **corners**, the **Fried** finite-difference relation is the physically correct map from lenslet slopes to corner phases/actuator heights.

---

## 3. Least-squares matrix formulation (the workhorse)

Write the stacked system (any geometry):
```
    s = Γ φ        (Γ is 2M × N, sparse, ±1/h or ±1/2h entries)
```
Over-determined (2M ≳ N), inconsistent due to noise → **least squares**:
```
    φ̂ = argmin_φ ‖Γφ − s‖²      ⇒    ΓᵀΓ φ̂ = Γᵀ s        (normal equations)
```
`A ≡ ΓᵀΓ` is N×N, sparse, symmetric positive **semi**-definite. It is **singular**: piston (all-ones vector `𝟙`) is in its null space (Γ𝟙 = 0). For Fried, **waffle `w`** is also (numerically) in the null space.

### 3.1 Pseudo-inverse via SVD (the standard recipe)
Singular value decomposition `Γ = U Σ Vᵀ`. The minimum-norm least-squares reconstructor is
```
    R = Γ⁺ = V Σ⁺ Uᵀ ,    φ̂ = R s
```
where `Σ⁺` inverts each singular value **except** those below a threshold, which are **zeroed** (`1/σ_i → 0`). This is how you *remove* the null/ill-conditioned modes:
- the **σ = 0** mode for piston is set to 0 → piston-free reconstruction;
- for **Fried**, the waffle singular value is ~0 → automatically rejected by the threshold (but you should null it *explicitly*, §6, because numerically it's tiny-but-nonzero and noise leaks in).
The pseudo-inverse "is usually calculated with SVD … care must be taken to reject very small singular values."

### 3.2 Tikhonov / regularized least squares (better with noise)
```
    R = (ΓᵀC_n⁻¹Γ + αC_φ⁻¹ + βP)⁻¹ ΓᵀC_n⁻¹
```
- `C_n` = slope-noise covariance (centroiding noise); `C_φ` = a-priori phase covariance (turbulence → Kolmogorov spectrum → **statistical/Wiener prior**, §10).
- `α` = Tikhonov weight (regularizes high-order/ill-conditioned modes);
- `P` = an explicit **penalty matrix** for piston and **waffle** (e.g. `P = γ_w wwᵀ + γ_p 𝟙𝟙ᵀ`), `β` large → drives those modes to zero. This is the "regularized inverse of the interaction matrix … a matrix penalizes piston" approach used in real AO systems.

### 3.3 The real-time payoff — "O(1) per actuator"
`R` is **computed once, offline** (SVD/inverse is O(N³), but done at calibration). At run time, each frame is **one matrix–vector multiply**:
```
    φ̂ = R s          (R is N×2M)  →  ~2MN multiply-adds per frame
```
Dense MVM is **O(N²)** flops but **embarrassingly parallel**, cache-friendly, single BLAS `gemv`/`cblas_dgemv` call in C, or one cuBLAS call on GPU. This is the **"Vector–Matrix–Multiply (VMM)" reconstructor** — the AO industry standard for moderate N. For PS9's lab grids (tens of sub-apertures per side, N ≲ 1000), `R s` is **microseconds** — far under the 10 ms budget.

---

## 4. A concrete small example (Fried, 2×2 cells → 3×3 corner grid)

Phase points (corners), 3×3 = 9 unknowns, indexed row-major φ₀…φ₈:
```
   φ0 ── φ1 ── φ2
   │  C0  │  C1 │
   φ3 ── φ4 ── φ5      4 lenslet cells C0..C3 at the cell centres,
   │  C2  │  C3 │      each giving s^x and s^y  → 8 measurements
   φ6 ── φ7 ── φ8
```
Cell→corner indices: C0=(0,1,3,4), C1=(1,2,4,5), C2=(3,4,6,7), C3=(4,5,7,8) [as (a,b,c,d)=(top-left,top-right,bot-left,bot-right)].

Using the Fried relations `s^x=(1/2h)[(b−a)+(d−c)]`, `s^y=(1/2h)[(c−a)+(d−b)]` (set h=1):

Γ rows (each row over φ0..φ8), x-slopes then y-slopes:
```
        φ0   φ1   φ2   φ3   φ4   φ5   φ6   φ7   φ8
C0 sx [ -½   +½    0   -½   +½    0    0    0    0 ]
C1 sx [  0   -½   +½    0   -½   +½    0    0    0 ]
C2 sx [  0    0    0   -½   +½    0   -½   +½    0 ]
C3 sx [  0    0    0    0   -½   +½    0   -½   +½ ]
C0 sy [ -½   -½    0   +½   +½    0    0    0    0 ]
C1 sy [  0   -½   -½    0   +½   +½    0    0    0 ]
C2 sy [  0    0    0   -½   -½    0   +½   +½    0 ]
C3 sy [  0    0    0    0   -½   -½    0   +½   +½ ]
```
That is the **8×9 Fried geometry matrix Γ**. Check the null space:
- **Piston** `𝟙 = (1,1,…,1)ᵀ`: every row sums to 0 → `Γ𝟙 = 0`. ✔ (piston unseen)
- **Waffle** `w = (+1,−1,+1, −1,+1,−1, +1,−1,+1)ᵀ` (checkerboard): e.g. C0 sx = −½(+1)+½(−1)−½(−1)+½(+1)=0; C0 sy = −½(+1)−½(−1)+½(−1)+½(+1)=0; all rows give 0 → `Γw = 0`. ✔ **(waffle unseen — the Fried problem!)**

So `rank(Γ) = 9 − 2 = 7`. The reconstructor must project out **both** `𝟙` and `w`. Build `R = Γ⁺` with SVD, zero the two ~0 singular values, and additionally subtract the waffle projection. The result is a fixed **9×8 matrix**; runtime reconstruction is `φ̂ = R s`.

> For PS9's real grid you build the analogous Γ from the actual lenslet→corner map (only over **valid/illuminated** sub-apertures inside the circular pupil), then `R = pinv(Γ)` with piston+waffle removed. **This `R` is your deliverable reconstructor.**

---

## 5. Direct integration / path integration (the "trivially fast" baseline)

Instead of a global least-squares solve, **integrate** the slopes along paths (cumulative trapezoidal sums):
```
φ(x_{i+1}, y_j) = φ(x_i, y_j) + ½(s^x_{i} + s^x_{i+1})·h      (march along a row)
φ(x_0, y_{j+1}) = φ(x_0, y_j) + ½(s^y_j + s^y_{j+1})·h         (march up the first column)
```
- **Pros:** O(N), no matrix, dead simple, great for a first-light sanity check.
- **Cons:** **path-dependent** → noise accumulates along the path; inconsistent slopes (curl ≠ 0) give different answers for different paths; needs averaging over many paths to be robust. Not minimum-variance.
- This is the conceptual seed of **CuReD** (§9), which fixes the noise/curl problems by averaging many integration chains + domain decomposition.

---

## 6. The waffle / checkerboard null mode (Fried-specific — READ THIS)

**Definition.** Waffle is the phase pattern that is **+1 on the "white" sub-apertures and −1 on the "black" sub-apertures** of a checkerboard at the sampling period. Because the Fried slope is the **average of two first-differences**, waffle produces **zero average slope over every sub-aperture → zero SH output**. It lives in the sensor null space; the WFS is **completely blind** to it. (Confirmed: "Fried geometry cannot measure the waffle mode … zero average slope over every sub-aperture and hence produces zero SH WFS output.")

**Why it's dangerous.** In **closed loop**, any process that injects waffle (noise, DM drift, aliasing) is *not seen* by the WFS, so the controller never corrects it; it **accumulates** and wrecks the PSF (a strong waffle PSF shows the classic 4-spot/grid pattern). Documented on the AEOS telescope ("An Analysis of Fundamental Waffle Mode in Early AEOS AO Images").

**Mitigations (use at least one):**
1. **Explicit null-space removal in `R`.** Zero the waffle singular value in SVD, *and* subtract the waffle projector: `R_clean = (I − wwᵀ/‖w‖²) R`. This guarantees the reconstructed φ has no waffle content.
2. **Waffle-penalized least squares** (Gavel): add `+ γ_w wwᵀ` (or a *localized* positive-definite waffle-weighting matrix that penalizes all local waffle-like behaviour) to the normal-equations matrix before inverting. "Suppressing Anomalous Localized Waffle Behavior in Least Squares Wavefront Reconstructors" (Gavel, OSTI 15002879).
3. **Waffle-Constrained Reconstructor (WCR)** (Praus et al., AMOS 2014): algebraic constraint built into `R` itself, rather than spatial filtering — avoids sacrificing high-spatial-frequency correction to kill waffle.
4. **Dynamic command-space spatial filtering** of DM commands (leaky integrator on the waffle component) — real-time suppression in the loop.
5. **Geometry fix:** use a **modified-Hudgin / weighted-Fried** reconstructor (Southwell-like averaging or the "weighted Fried reconstructor" with near-unity frequency response) which restores sensitivity — at some noise cost. ["Weighted Fried reconstructor and spatial-frequency response optimization", ResearchGate 232228825.]

**Piston** is the other null mode (all geometries): always remove it (subtract the mean of φ̂, or zero the σ=0 mode). It is harmless physically (a constant phase doesn't affect the image) but must be fixed for a well-posed inverse and stable loop.

---

## 7. Fourier Transform Reconstructor (FTR) — O(N log N), with the correct Fried filter

**Idea (Freischlad & Koliopoulos 1986; Poyneer, Gavel & Brase 2002).** On a regular full grid, the finite-difference operators become **diagonal in the Fourier domain**: take FFTs of the slope arrays, divide by the geometry's spatial-frequency transfer function, inverse-FFT to get phase. Cost **O(N log N)** — much faster than VMM for large N, and often **lower noise propagation**. Poyneer showed it scales to ≥10,000 actuators. [Poyneer, Gavel, Brase, *JOSA A* 19, 2100 (2002); OSTI 15013348.]

**Core formula.** With `X = FFT(s^x)`, `Y = FFT(s^y)`, and per-geometry filters `Ĝx, Ĝy`:
```
            Ĝx*(k)·X(k) + Ĝy*(k)·Y(k)
  Φ̂(k) =  ────────────────────────────       (then φ = IFFT(Φ̂),  set Φ̂(0,0)=0 for piston)
              |Ĝx(k)|² + |Ĝy(k)|²
```
(* = complex conjugate). This is exactly the least-squares solution **per Fourier mode** — and it's a generic Wiener/MMSE backbone (§10).

**The exact filters** (verbatim from the open-source `FTR` implementation of Poyneer's method; `fx, fy` are the FFT grid frequencies in radians, i.e. `2π·(0..N−1)/N` mapped to (−π,π]):

```python
# fy, fx = fftgrid(shape)

# --- Hudgin ---
gx = np.exp(1j*fx) - 1.0
gy = np.exp(1j*fy) - 1.0

# --- Modified Hudgin (mod_hud) ---   (x & y averaged onto a common point)
gx = np.exp(1j*fy/2)*(np.exp(1j*fx) - 1)
gy = np.exp(1j*fx/2)*(np.exp(1j*fy) - 1)
gx[ny//2,:] = 0.0     # kill Nyquist row/col (undefined / waffle line)
gy[:,nx//2] = 0.0

# --- FRIED ---                        (average of two parallel first differences)
gx = (np.exp(1j*fy) + 1)*(np.exp(1j*fx) - 1)
gy = (np.exp(1j*fx) + 1)*(np.exp(1j*fy) - 1)
gx[ny//2,:] = 0.0     # <-- this zeroing is the WAFFLE-mode removal in FTR
gy[:,nx//2] = 0.0
```

Notes on the Fried filter:
- `(e^{i f_x} − 1)` is the first-difference along x; the `(e^{i f_y} + 1)` factor is the **two-row average** that defines Fried. Squanders sensitivity exactly at the **Nyquist (waffle) frequency** `f = π`, where `(e^{iπ}+1) = 0` → `Ĝ = 0`. That zero IS the waffle null. Setting `gx[ny//2,:]=0, gy[:,nx//2]=0` cleanly removes the 0/0 and prevents waffle blow-up.
- Magnitude response of Hudgin/Fried peaks at `π/2 = 1/sinc(0.5)` relative to the ideal — i.e. they slightly mis-weight high frequencies; the "ideal" and "modified-Hudgin/weighted-Fried" filters correct this for near-unity frequency response. [Band-limited / unity-frequency-response reconstructor, ResearchGate 5303861.]

**Boundary problem (Poyneer's key fix).** Plain FFT assumes periodicity → fails for a **circular aperture** on a square grid (phase wraps, edge errors). Poyneer's solution: **extend the slopes** into the non-illuminated region (boundary/edge-correction or an iterative extension scheme, e.g. Gerchberg-type), so the periodic FFT sees a consistent field; then reconstruct and crop to the pupil. Roddier/Hugot and "Wavefront reconstruction using iterative discrete Fourier transforms with Fried geometry" (Opt. Commun., 2005) give iterative DFT variants for arbitrary apertures.

**Complexity & RT suitability:** O(N log N), all FFTs → maps perfectly to FFTW (C) or cuFFT (GPU). Excellent for large N; for **small lab grids the dense VMM is actually simpler and competitive**, and avoids the circular-aperture extension headache. Use FTR as a fast validator / scaling path.

---

## 8. Iterative solvers for `Aφ = b` (SOR, multigrid, conjugate gradient)

When you don't want to precompute a dense inverse (huge N, or changing pupil), solve the sparse normal equations `ΓᵀΓ φ = Γᵀ s` iteratively. `A = ΓᵀΓ` is a discrete **Poisson-like** operator (the LS problem is a discrete Poisson equation `∇²φ = ∇·s` with Neumann BCs — see §11), so classic elliptic-PDE solvers apply.

- **Successive Over-Relaxation (SOR)** — Southwell's original choice. Gauss–Seidel sweep with over-relaxation factor ω∈(1,2):
  ```
  φ_p^{(k+1)} = (1−ω) φ_p^{(k)} + ω · [ b_p − Σ_{q∈neighbours} A_{pq} φ_q ] / A_pp
  ```
  Update each node from its (already-updated) neighbours and the local slope data; ω tuned (often 1.5–1.9) for fastest convergence. Cheap per-iteration, but iteration count grows with grid size; **convergence accelerated by seeding with an FFT/FTR initial guess** (Southwell noted this for circularly symmetric fields). Good for C; modest N. [Southwell 1980; SOR: Young 1950.]
- **Conjugate Gradient (CG)** — `A` is SPD (after fixing piston/waffle) → CG converges in ≤ rank iterations, each iteration = one sparse mat-vec O(N). Far fewer iterations than SOR with a good **preconditioner**.
- **Preconditioned CG (PCG) / Multigrid (MGCG)** — Gilles, Vogel, Ellerbroek: **multigrid-preconditioned CG** gives total cost **O(N log N)** with rapid convergence over a wide SNR range; demonstrated for a 17 m telescope (≈48,756 grid points). Block-symmetric Gauss–Seidel is a simpler, cheaper preconditioner alternative. The basis of **minimum-variance** reconstruction for ELTs. [Gilles, Vogel, Ellerbroek, *JOSA A* 19, 1817 (2002); *Appl. Opt.* 42, 5233 (2003).]
- **Multigrid (geometric)** — V-cycles on the Poisson operator → O(N) asymptotically; excellent for very large grids but more implementation effort.

**RT suitability:** Iterative solvers shine when N is huge or the system changes frame-to-frame. For a **fixed lab geometry** they're usually *beaten by a precomputed `R s` MVM* because the inverse can be done once offline. Keep PCG/multigrid as the scaling story, not the lab baseline.

---

## 9. Sparse / CuReD (Cumulative Reconstructor with Domain Decomposition) — O(N)

**CuRe / CuReD** (Rosensteiner 2011–2012, Linz). Exploits that "the Shack–Hartmann operator is a discrete approximation to the gradient": reconstruct by **integrating slopes along chains** (like §5) but **average over many overlapping chains** and use **domain decomposition** to keep noise propagation bounded.

- **Complexity:** **linear, O(n)** in the number of sub-apertures; explicit operation count ~**14 flops per unknown** (all adds/mults). Pipelinable and parallelizable → ideal for FPGA / real-time hardware.
- **Geometry:** formulated for SH slopes; uses **modified-Hudgin → Fried** transition internally. Works directly on Fried-like data.
- **Why domain decomposition:** plain CuRe had **unacceptable noise propagation for large apertures**; splitting the pupil into sub-domains, reconstructing locally, and stitching restores the same quality as the global solve while taming noise.
- **P-CuReD** extends it (pre-processing) to **pyramid WFS** and XAO on 42 m-class telescopes.
- **RT suitability:** *the* method when N is enormous and you need the absolute fastest direct (non-iterative) reconstructor. For PS9 lab scale it's overkill but a strong talking point and a clean O(N) fallback. [Rosensteiner, *JOSA A* 28, 2132 (2011) & 29, 2328 (2012); ESO-AO Linz page.]

---

## 10. Wiener / Minimum-Variance / MMSE reconstruction (best accuracy under noise)

Treat reconstruction as a **Bayesian/MMSE** estimation using turbulence statistics (Kolmogorov phase PSD) as a prior:
```
  R_MMSE = C_φ Γᵀ ( Γ C_φ Γᵀ + C_n )⁻¹            (= the statistically optimal linear reconstructor)
```
- `C_φ` = phase covariance (from `r₀`, Kolmogorov/von-Kármán spectrum); `C_n` = slope-noise covariance. Equivalent to the **Tikhonov form** of §3.2 with `α C_φ⁻¹`.
- **Fourier-domain Wiener filter** (replaces the LS filter in §7's FTR):
  ```
              Ĝ*(k) · W_φ(k)
  H(k) = ───────────────────────────         W_φ = phase PSD,  W_η = noise PSD,  γ = tuning
           |Ĝ(k)|² W_φ(k) + γ W_η(k)
  ```
  "the equivalent in Fourier space of a direct-space MMSE method … minimises residual variance, maximises Strehl." Anti-aliasing Wiener variants further suppress aliasing. [Correia & Teixeira / Jolissaint; *JOSA A* 31, 2763 (2014), arXiv:1410.6055.]
- **RT suitability:** the reconstructor is **still a precomputed matrix** (or Fourier filter), so runtime cost is identical to plain VMM/FTR — you just get **better noise rejection for free** by folding in `r₀`. Strongly recommended once you've estimated `r₀` from the data (which PS9 requires anyway for turbulence characterization).

---

## 11. Poisson-equation view (why all of the above are connected)

The continuous least-squares problem `min ∫‖∇φ − s‖²` has Euler–Lagrange equation
```
  ∇²φ = ∇·s          in the pupil,
  ∂φ/∂n = s·n        on the boundary (Neumann).
```
So every zonal LS reconstructor is a **discrete Poisson solver** with Neumann BCs and a **pure-Neumann singularity = piston null mode** (solvable up to a constant; compatibility `∫∇·s = ∮ s·n`). This unifies: **FFT** (diagonalizes the Laplacian → FTR), **SOR/multigrid/CG** (classic Poisson iterative solvers), and **direct integration** (Green's-function / marching). Framing your write-up this way is rigorous and impresses reviewers.

---

## 12. Mapping lenslets → actuators (Fried) and the DM command (PS9-specific)

**Fried alignment.** In Fried geometry the **DM actuators sit at the sub-aperture corners** = the phase-estimation grid. So the reconstructed corner phases `φ̂` are *already co-located with the actuators*. "In a Fried-geometry configuration the corners of the sub-aperture coincide with the location of discrete actuators of the DM." Four actuators surround each lenslet centre; the four can be commanded to give a desired local slope → high controllability.

**From phase to actuator commands (with inter-actuator coupling).** The DM surface is a superposition of **influence functions** `f_a(x,y)` (the shape from poking actuator *a* by unit stroke):
```
  DM_surface(x,y) = Σ_a c_a · f_a(x,y)            c_a = actuator command (stroke units)
```
We want the mirror to apply the **conjugate** of the wavefront: `DM_surface = −½ W` (factor ½ for reflection doubling the OPD; sign = conjugate to flatten). Sampling influence functions on the phase grid gives the **influence matrix `F`** (N_points × N_act). The commands are the least-squares fit:
```
  c = F⁺ ( −½ φ̂ ) = (FᵀF + μI)⁻¹ Fᵀ ( −½ φ̂ )      (μ = regularization, limits stroke / conditions)
```
**Inter-actuator coupling** is *built into `F`*: poking actuator *a* also moves its neighbours. A standard parametric model (Gaussian-type influence function):
```
  f_a(ρ) = exp[ ln(ω) · (ρ/d₀)^α ]          ω = inter-actuator coupling (e.g. 0.1–0.15),
                                            d₀ = actuator pitch, ρ = distance from actuator a, α≈2
```
ω is the fraction by which a neighbour rises when an actuator is poked (typ. **10–15%**; negligible beyond ~3–4 actuators). Because `F` already encodes coupling, `c = F⁺(−½φ̂)` **automatically accounts for crosstalk** — neighbours commanded so their overlapping influence functions sum to the target shape. The whole chain is two precomputed matrices:
```
  φ̂ = R s          (reconstruct)
  c  = G φ̂          with G = −½ (FᵀF + μI)⁻¹ Fᵀ   →   c = (G R) s   ←  collapse to ONE matrix!
```
So you can **pre-multiply** `R` and `G` into a **single "slopes → actuator commands" matrix `K = G R`**, and the entire real-time step is `c = K s` — one MVM. (Keep them separate if you also need to *output* the phase map W per frame, which PS9 does.)

> **PS9 dataset note:** the problem says DM info and inter-actuator coupling "shall be provided." Use the supplied coupling coefficient ω (and pitch) to build `F`; if an influence-function measurement is provided, sample *that* directly instead of the Gaussian model.

---

## 13. Comparison table (rank for THIS problem)

Legend: N = #phase points/actuators; M = #sub-apertures (≈N). "Online" = per-frame cost; "Offline" = one-time setup.

| # | Method | Geometry fit | Online cost | Offline | Null/cond. handling | Noise prop. | RT (10 ms, lab) | PS9 rank |
|---|--------|--------------|-------------|---------|---------------------|-------------|-----------------|----------|
| 1 | **Dense LS / VMM reconstructor `R=Γ⁺` (SVD)** | **Fried native** | **O(N²) MVM** (µs) | O(N³) SVD once | SVD threshold + explicit piston+waffle nulling | low (tunable) | ✅✅ excellent | **#1** |
| 2 | **Tikhonov / MMSE-Wiener reconstructor** | Fried native | O(N²) MVM | O(N³) once | regularized; uses r₀ prior | **lowest** | ✅✅ excellent | **#2** |
| 3 | **FFT Fourier Transform Reconstructor (Fried filter)** | Fried (needs edge extension) | **O(N log N)** | filter precompute | Nyquist zeroing = waffle removal | low | ✅ excellent (large N) | **#3** |
| 4 | **CuReD (cumulative + domain decomp.)** | Hudgin→Fried | **O(N)** (~14 flops/unk) | minimal | domain decomp tames noise | low (w/ DD) | ✅ excellent | #4 (scaling) |
| 5 | **PCG / Multigrid (MGCG)** on Poisson | any | O(N log N)·iters | preconditioner setup | SPD after piston/waffle fix | low (min-var) | ⚠️ good (large N) | #5 |
| 6 | **SOR (Southwell)** | Southwell (co-located) | O(N)·iters | none | needs piston fix; ω tuning | low | ⚠️ ok (iter count) | #6 |
| 7 | **Southwell dense LS** | Southwell | O(N²) MVM | O(N³) once | best error propagation of the 3 | **best of 3** | ✅✅ | #7 (if re-grid) |
| 8 | **Hudgin dense LS / Hudgin FTR** | Hudgin (x,y not co-located) | O(N²) / O(N log N) | once | mid error propagation | mid | ✅ | #8 |
| 9 | **Direct integration / path** | any | **O(N)** | none | path-dependent, curl-sensitive | high | ✅ (but noisy) | #9 (baseline) |
| 10 | **Sparse normal-equations direct (Cholesky)** | any | O(N²)–O(N^{1.5}) | sparse factorize once | needs reg. for singular A | low | ✅ moderate N | #10 |
| 11 | **Iterative DFT (arbitrary aperture, Fried)** | Fried | O(N log N)·iters | none | handles odd pupils | low | ✅ | alt to #3 |

**Error/noise-propagation ordering (classic result):** Southwell < Hudgin < Fried for raw error propagation; Fried's propagation grows roughly **logarithmically** with the number of sub-apertures, Southwell is the lowest and is "usually recommended in optical testing." BUT PS9's hardware *is* Fried, so we keep Fried and fix waffle with regularization rather than switching geometry. [Southwell 1980; Zou & Rolland error-propagation analyses; "Quantifications of error propagation in slope-based wavefront estimations".]

---

## 14. Practical build recipe (C / real-time)

1. **Geometry/calibration (offline, Python or C):**
   - From MLA + pupil geometry, list **valid sub-apertures** (illuminated, inside pupil).
   - Build the **Fried gradient matrix Γ** (2M×N) using the corner-averaging equations (§2.2, §4) over the valid set.
   - Compute `R = pinv(Γ)` via **SVD**; zero σ below threshold; **explicitly remove piston `𝟙` and waffle `w`** via projectors. Optionally use Tikhonov/MMSE with `C_φ(r₀)`, `C_n`.
   - Build the **influence matrix F** from the provided DM coupling ω/pitch; form `G = −½(FᵀF+μI)⁻¹Fᵀ`. Optionally fuse `K = G·R`.
   - Save `R` (and `K`) as plain float arrays.
2. **Per-frame (online, C + BLAS):**
   - Centroid each spot → displacement → slope vector `s` (2M).
   - `φ̂ = R·s` (one `cblas_sgemv`/`dgemv`) → **wavefront map W** (PS9 deliverable).
   - `c = G·φ̂` (or `c = K·s`) → **actuator command map** (PS9 deliverable), coupling already baked in.
   - Subtract piston; clamp to actuator stroke limits.
3. **Validation / cross-check:** implement the **FFT Fried FTR** (§7) and compare maps; agreement validates both. Use FTR if you later scale N up.
4. **Turbulence params** (separate report): from φ̂ time series → Zernike/structure-function → `r₀`; temporal autocorrelation → `τ₀`. (Cross-check: error-propagation and PSD of φ̂.)

**Why dense `R` for the lab grid:** a 20×20 sub-aperture sensor → N≈441 corners, M≈400, `R` is ~441×800 ≈ 3.5×10⁵ floats; `R·s` ≈ 7×10⁵ FLOPs ≈ **single-digit microseconds** in BLAS — three to four orders of magnitude inside the 10 ms budget, with trivial C code. Dense MVM is the pragmatic winner; FFT/CuReD/PCG only matter past thousands of actuators.

---

## 15. Recommended approach for PS9 (Fried geometry)

**Primary (build this):** A **precomputed, regularized least-squares Fried reconstructor `R`**.
- Construct Γ from the **Fried corner-averaging equations** over the valid (illuminated) sub-apertures only.
- `R = Γ⁺` via **SVD with singular-value thresholding**, then **explicitly null piston and waffle** (projector `I − wwᵀ/‖w‖² − 𝟙𝟙ᵀ/‖𝟙‖²`, or Gavel waffle-penalty in the normal equations). This is non-negotiable: **Fried + un-handled waffle = closed-loop failure.**
- Upgrade to **MMSE/Wiener** weighting by folding in `C_φ(r₀)` and `C_n` once `r₀` is estimated — same runtime cost, better noise rejection.
- Runtime: **`φ̂ = R s`** (one BLAS `gemv`) → wavefront map. Then **`c = G φ̂`** with `G` from the DM influence/coupling matrix `F` → actuator map with **inter-actuator coupling built in**. Optionally fuse to `c = K s`.

**Secondary / validation & scaling fallback:** **FFT Fourier Transform Reconstructor with the exact Fried filter** (`gx=(e^{ify}+1)(e^{ifx}−1)`, `gy=(e^{ifx}+1)(e^{ify}−1)`, Nyquist rows/cols zeroed = waffle removal), with Poyneer's **boundary extension** for the circular pupil. O(N log N); use it to cross-check `R` and as the path to large N. If N ever explodes, switch the online step to **CuReD (O(N))** or **multigrid-PCG (O(N log N), minimum-variance)**.

**Geometry note for the report:** acknowledge that **Southwell** has the lowest noise propagation, but PS9's DM/MLA are physically **Fried**, so we stay in Fried and pay for it with explicit waffle/piston regularization rather than re-gridding.

**Headline for judges:** *"Real-time reconstruction is a single precomputed matrix–vector multiply `c = (G·R)·s` — O(1) algebra per actuator — with piston and the Fried-geometry waffle null mode regularized out, and DM inter-actuator coupling folded into the command matrix. Validated against an FFT Fourier-domain reconstructor for speed/scaling."*

---

## 16. Key sources (URLs)

**Foundational geometry papers**
- D. L. Fried, "Least-square fitting a wave-front distortion estimate to an array of phase-difference measurements," *JOSA* 67, 370–375 (1977). https://opg.optica.org/josa/abstract.cfm?uri=josa-67-3-370
- R. H. Hudgin, "Wave-front reconstruction for compensated imaging," *JOSA* 67, 375–378 (1977). https://opg.optica.org/josa/abstract.cfm?uri=josa-67-3-375 · ADS: https://ui.adsabs.harvard.edu/abs/1977JOSA...67..375H/abstract
- R. H. Hudgin, "Optimal wave-front estimation," *JOSA* 67, 378 (1977). https://opg.optica.org/josa/abstract.cfm?uri=josa-67-3-378
- W. H. Southwell, "Wave-front estimation from wave-front slope measurements," *JOSA* 70, 998–1006 (1980). https://opg.optica.org/abstract.cfm?URI=josa-70-8-998 · https://www.semanticscholar.org/paper/750cddc5e5941baea0c807df7edb2a87f0730e99
- B. R. Hunt, "Matrix formulation of the reconstruction of phase values from phase differences," *JOSA* 69, 393 (1979) — least-squares matrix view (related).

**Fourier Transform Reconstructor (FTR)**
- L. A. Poyneer, D. T. Gavel, J. M. Brase, "Fast wave-front reconstruction in large adaptive optics systems with use of the Fourier transform," *JOSA A* 19, 2100–2111 (2002). https://opg.optica.org/josaa/abstract.cfm?uri=josaa-19-10-2100 · PubMed: https://pubmed.ncbi.nlm.nih.gov/12365629/ · OSTI: https://www.osti.gov/servlets/purl/15013348
- L. A. Poyneer, "Advanced techniques for Fourier transform wavefront reconstruction" / dissertation (chap. 2). https://www.semanticscholar.org/paper/ebf426696e4909dc54b590e2d2ac1f52584283c2 · UCRL-TR-204793: https://www.osti.gov/servlets/purl/15014303
- FTR reference implementation (filter definitions used verbatim above): https://ftr.readthedocs.io/en/latest/api/FTR.ftr.FourierTransformReconstructor.html
- "Wavefront reconstruction using iterative discrete Fourier transforms with Fried geometry," *Opt. Commun.* (2005). https://www.sciencedirect.com/science/article/abs/pii/S0030402605001506
- Performance analysis, Fourier vs VMM (Hudgin/Fried/Southwell geometries): arXiv:0911.0813. https://arxiv.org/abs/0911.0813 / https://arxiv.org/pdf/0911.0813

**Waffle mode (Fried-specific)**
- mvkonnik, "A note on Waffle Modes" (clear tutorial). http://mvkonnik.blogspot.com/2011/05/a-note-on-waffle-modes.html
- D. Gavel, "Suppressing Anomalous Localized Waffle Behavior in Least Squares Wavefront Reconstructors," OSTI 15002879. https://www.osti.gov/biblio/15002879 · https://www.researchgate.net/publication/255198841
- Praus et al., "Development and Analysis of a Waffle Constrained Reconstructor (WCR) for Fried Geometry," AMOS 2014. https://amostech.com/TechnicalPapers/2014/Poster/PRAUS.pdf · ADS: https://ui.adsabs.harvard.edu/abs/2014amos.confE..93P/abstract
- "An Analysis of Fundamental Waffle Mode in Early AEOS AO Images," arXiv:astro-ph/0505195. https://arxiv.org/pdf/astro-ph/0505195
- "Weighted Fried reconstructor and spatial-frequency response optimization of Shack–Hartmann wavefront sensing." https://www.researchgate.net/publication/232228825
- "Band-limited wavefront reconstruction with unity frequency response from SH slopes." https://www.researchgate.net/publication/5303861

**Iterative / minimum-variance / sparse**
- L. Gilles, C. R. Vogel, B. L. Ellerbroek, "Multigrid preconditioned conjugate-gradient method for large-scale wave-front reconstruction," *JOSA A* 19, 1817 (2002). https://opg.optica.org/josaa/abstract.cfm?uri=josaa-19-9-1817 · PubMed: https://pubmed.ncbi.nlm.nih.gov/12216875/
- L. Gilles et al., "Preconditioned conjugate gradient wave-front reconstructors for MCAO," *Appl. Opt.* 42, 5233 (2003). https://opg.optica.org/ao/abstract.cfm?uri=ao-42-26-5233 · ADS: https://ui.adsabs.harvard.edu/abs/2003ApOpt..42.5233G/abstract
- B. L. Ellerbroek, "Efficient computation of minimum-variance wave-front reconstructors with sparse matrix techniques," *JOSA A* 19, 1803 (2002). https://opg.optica.org/josaa/abstract.cfm?uri=josaa-19-9-1803
- "Fast minimum variance wavefront reconstruction for extremely large telescopes," arXiv:1003.0274. https://arxiv.org/pdf/1003.0274
- Correia/Jolissaint, "Anti-aliasing Wiener filtering for wave-front reconstruction in the spatial-frequency domain," *JOSA A* 31, 2763 (2014), arXiv:1410.6055. https://arxiv.org/abs/1410.6055 · https://opg.optica.org/josaa/abstract.cfm?uri=josaa-31-12-2763
- "Iterative wave-front reconstruction in the Fourier domain," arXiv:1705.04298. https://arxiv.org/pdf/1705.04298

**CuReD (cumulative reconstructor, O(N))**
- M. Rosensteiner, "Cumulative Reconstructor: fast wavefront reconstruction algorithm for ELTs," *JOSA A* 28, 2132 (2011). https://opg.optica.org/josaa/abstract.cfm?uri=josaa-28-10-2132
- M. Rosensteiner, "Wavefront reconstruction for ELTs via CuRe with domain decomposition," *JOSA A* 29, 2328 (2012). https://opg.optica.org/josaa/abstract.cfm?uri=josaa-29-11-2328
- CuReD / P-CuReD algorithm pages (Linz ESO-AO): http://eso-ao.indmath.uni-linz.ac.at/index.php/algorithms/cured.html · .../pcured.html
- P-CuReD for pyramid WFS, *Appl. Opt.* 52, 2640 (2013). https://opg.optica.org/ao/abstract.cfm?uri=ao-52-12-2640

**Reviews, Southwell-geometry detail, DM coupling, software**
- "A Review on Wavefront Reconstruction Methods," ACM (2021). https://dl.acm.org/doi/fullHtml/10.1145/3482632.3483191
- "Wavefront Reconstruction Methods for AO Systems on Ground-Based Telescopes," *SIAM J. Matrix Anal. Appl.* https://epubs.siam.org/doi/10.1137/06067506X
- "Zonal shape reconstruction for Shack–Hartmann sensors and deflectometry," arXiv:2410.08291. https://arxiv.org/pdf/2410.08291
- "Improved wavefront reconstruction algorithm from slope measurements" (Southwell geometry, SOR), *J. Korean Phys. Soc.* 70, 469. https://link.springer.com/article/10.3938/jkps.70.469
- "Improving wavefront reconstruction accuracy using integration equations with higher-order truncation errors in the Southwell geometry." https://www.researchgate.net/publication/259253723
- W. Zou, "Optimization of Zonal Wavefront Estimation and Curvature Measurements" (PhD, UCF) — error propagation. https://stars.library.ucf.edu/etd/3432/
- "Fast and Highly Accurate Zonal Wavefront Reconstruction … Subregion Cancelation," *Appl. Sci.* 14, 3476 (2024). https://www.mdpi.com/2076-3417/14/8/3476
- "Wavefront sensor and wavefront corrector matching in AO" (Fried lenslet↔actuator, condition numbers), PMC4793900. https://pmc.ncbi.nlm.nih.gov/articles/PMC4793900/
- "Adaptive Optics for Directed Energy: Fundamentals and Methodology," *AIAA J.* (Fried geometry, interaction matrix, reconstructor). https://arc.aiaa.org/doi/10.2514/1.J061766
- DM influence function / coupling model `IF(ρ)=exp[ln(ω)(ρ/d₀)^α]`: "A novel model of influence function: calibration of a continuous membrane DM." https://www.researchgate.net/publication/216852286 · RP-Photonics, "Deformable Mirrors." https://www.rp-photonics.com/deformable_mirrors.html
- AOtools (Python AO package — modal & zonal): Townson et al., arXiv:1910.04414. https://arxiv.org/pdf/1910.04414 · docs: https://aotools.readthedocs.io/
- Soapy AO simulation (WFS, interaction/command matrix). https://soapy.readthedocs.io/en/latest/wfs.html
- A. Tokovinin, AO tutorial part 3 — wavefront sensors. https://www.ctio.noirlab.edu/~atokovin/tutorial/part3/wfs.html
- HCIPy (high-contrast imaging / AO modelling, reconstructors). https://docs.hcipy.org/

---

*Caveats on sourcing:* equation forms for Hudgin/Fried/Southwell and the FTR filters were cross-checked across multiple independent sources (Fried 1977, Hudgin 1977, Southwell 1980, Poyneer 2002, the FTR reference implementation, and several review/figure sources). Several primary PDFs (arXiv:0911.0813, arXiv:2410.08291, the IntechOpen and Lumetrics chapters) are binary and could not be machine-parsed in this environment; their equations were reconstructed from the canonical literature and verified against the open-source FTR code and HTML reviews. Verify the exact sign conventions and the factor-of-½ (reflection) against the supplied PS9 dataset metadata before finalizing the reconstructor.
