# Master Architecture — SH-WFS Wavefront Reconstruction, Turbulence Characterization & DM Actuator Mapping

**Project:** ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 9 (PS9)
**Document status:** Primary deliverable. This is the contract the implementation teams build against.
**Repository:** `bah2026-ps9` — branch `claude/elegant-hamilton-5tzss9`
**Synthesizes:** all seven research reports under `research/` (centroiding, zonal, modal, turbulence, DM actuators, real-time, data/validation). Each algorithmic choice below honors the "Recommended approach for PS9" section of the corresponding report; cross-references are given as `[R0n]` (e.g. `[R01]` = `research/01_centroiding.md`).

---

## Table of Contents

1. [Problem restatement, requirements, evaluation criteria](#1-problem-restatement-requirements-and-evaluation-criteria)
2. [End-to-end system overview & data-flow diagram](#2-end-to-end-system-overview)
3. [Chosen algorithm per stage (with justification + retained alternatives)](#3-chosen-algorithms-per-stage)
4. [Data structures & file formats (C ↔ Python interop)](#4-data-structures--file-formats)
5. [Directory layout](#5-directory-layout)
6. [Build & run instructions; tech-stack rationale](#6-build--run-instructions--tech-stack-rationale)
7. [Robustness via multiple methods (the ≥30-method catalogue)](#7-robustness-via-multiple-methods)
8. [Validation & testing plan](#8-validation--testing-plan)
9. [Implementation roadmap & risk register](#9-implementation-roadmap--risk-register)

---

## 1. Problem restatement, requirements, and evaluation criteria

### 1.1 Restatement

Atmospheric turbulence distorts a nominally plane wavefront. A **Shack–Hartmann Wavefront Sensor (SH-WFS)** samples the distorted wavefront with a **microlens array (MLA)**; each lenslet forms a focal spot whose displacement from a reference position is proportional to the **average wavefront gradient (slope)** over that sub-aperture. From a *time series* of `.bmp` spot-field frames captured at a few-millisecond cadence we must, **for every frame**:

1. **Reconstruct the wavefront phase map** `W(xᵢ, yᵢ)` and its Zernike coefficients.
2. **Characterize the turbulence** by the **Fried parameter `r₀`** (spatial strength) and the **coherence time `τ₀`** (temporal strength), derived "from the same data."
3. **Derive a deformable-mirror (DM) actuator map** `A(xᵢ, yᵢ)` **in units of actuator stroke length**, formed from the **conjugate** of the reconstructed wavefront, **explicitly incorporating inter-actuator coupling**.

The DM actuator grid and the MLA lenslet grid are arranged in **Fried geometry** (actuators at sub-aperture corners). Because turbulence decorrelates on a `~10 ms` timescale, the per-frame reconstruction loop must be **fast** (target `< 10 ms`, with large margin).

### 1.2 Functional requirements

| ID | Requirement | Source |
|----|-------------|--------|
| FR-1 | Ingest 8-bit (and 24-bit) `.bmp` frames + a metadata/config file | `idea.md` L14–18; `[R07]` A |
| FR-2 | Detect each spot, compute sub-pixel centroid, subtract reference, convert to slope | `idea.md` L23–24; `[R01]` |
| FR-3 | Reconstruct `W(x,y)` zonally (Fried) **and** modally (Zernike) | `idea.md` L25; `[R02]`, `[R03]` |
| FR-4 | Derive `r₀` and `τ₀` from the time series | `idea.md` L8, L26; `[R04]` |
| FR-5 | Derive DM actuator map in stroke units with inter-actuator coupling deconvolved | `idea.md` L11, L27; `[R05]` |
| FR-6 | Honor Fried geometry; handle the waffle null mode | `idea.md` L25; `[R02]` §6, `[R05]` §5 |
| FR-7 | Per-frame loop `< 10 ms` (target ≤ 100 µs compute at PS9 scale) | `idea.md` L8, L31; `[R06]` |
| FR-8 | Provide a synthetic data generator with **known injected** `r₀`/`τ₀` (dataset not yet supplied) | `[R07]` B |

### 1.3 Non-functional requirements & environment constraints

- **Self-contained C real-time core.** Target host has **no BLAS/LAPACK/FFTW/GSL** and no numpy/scipy preinstalled. Therefore the C core (`src/`) links **only `-lm`**: hand-rolled BMP I/O, a portable matrix–vector multiply (GEMV) with optional `#ifdef __AVX2__` + FMA and OpenMP, and **no FFT at runtime**. (`[R06]` §13 prescribes "C real-time core + Python offline"; we keep the *spirit* — a lean precompute-then-MVM core — but drop the external BLAS/FFTW dependency to satisfy the zero-lib constraint.)
- **Python offline toolkit** (`aokit/`) may use numpy/scipy/matplotlib, installed via `requirements.txt` (pip + network available). It builds the matrices, generates synthetic data, runs validation, and is the algorithmic reference oracle.
- **Hardware:** Ubuntu 24.04, x86_64, 4 cores, 15 GiB RAM, **no GPU**. gcc 13.3, clang 18, cmake, make, OpenMP; AVX2 and AVX-512 available. (`[R06]` §8 — a GPU would *lose* at this scale due to PCIe latency; AVX2 CPU is the right tool.)
- **Precision:** `fp32` in the real-time path (camera SNR-limited); `fp64` offline for SVD conditioning. (`[R06]` §3.)

### 1.4 Success / evaluation criteria

The problem statement's evaluation criteria (`idea.md` L28–31) and our concrete acceptance targets:

| Criterion | Concrete target | Metric / where validated |
|-----------|-----------------|--------------------------|
| **Reconstruction fidelity** ("phase maps that conform to the turbulence characteristics") | RMS WFE `< λ/14` at design SNR on synthetic; phase correlation `ρ > 0.95`; reconstructed-phase structure function slope `≈ 5/3` | `aokit/validation.py`; `[R07]` C.2 |
| **`r₀` derivation** | `|r₀_est − r₀_true| / r₀_true` within a few % (ensemble) | ≥7 estimators agree; `[R04]` §5 |
| **`τ₀` derivation** | `|τ₀_est − τ₀_true| / τ₀_true` within ~10–20% | ≥6 estimators agree; `[R04]` §5 |
| **DM actuator map** | residual `σ(W − DM_shape)` greatly reduced; coupling deconvolved (anti-coupling off-diagonals present) | `[R05]` §9, §11 |
| **Speed & efficiency** | per-frame compute `< 100 µs` (≥100× under 10 ms); zero in-loop allocation; bounded jitter | `src/pipeline.c` timing; `[R06]` §14 |

---

## 2. End-to-end system overview

The system has **two paths** that share the same matrices and metadata:

- **OFFLINE calibration path** (Python, `aokit` + `scripts/build_calibration.py`): runs **once** per optical configuration. Builds the Fried geometry, the regularized zonal reconstructor `R`, the modal pseudo-inverse `M⁺` and synthesis matrix `Z`, the DM command matrix `G = H⁺·(−½)`, reference slopes, and the sub-aperture map. Serializes them as **AOMX** binary matrix files (§4.2).
- **REAL-TIME per-frame path** (C, `src/` → `bin/wfs_rt`): loads the AOMX matrices + config once, then for each frame does centroiding + 2–3 matrix–vector multiplies (MVMs). This is the `< 10 ms` loop.

### 2.1 ASCII data-flow diagram

```
                        ┌──────────────────────────────────────────────────────────────────────┐
                        │                    OFFLINE CALIBRATION PATH (Python / aokit)           │
                        │                    runs ONCE per optical configuration                 │
                        └──────────────────────────────────────────────────────────────────────┘
   config (JSON) ─────┐
   ┌─────────────────┐ │   ┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
   │ camera, MLA,    │ ├──▶│  geometry.py  │──▶│ Fried sub-ap grid +  │──▶│ reconstructor.py:        │
   │ pupil, DM,      │ │   │  pupil mask,  │   │ actuator-corner grid,│   │  Γ (Fried)→ R=Γ⁺ (SVD,   │
   │ wavelength, dt  │ │   │   references  │   │ valid-subap mask,    │   │  Tikhonov, piston+waffle │
   └─────────────────┘ │   └───────────────┘   │ ref slopes (flat WF) │   │  nulled)                 │
                       │                       └──────────────────────┘   │  zernike.py → M, M⁺, Z   │
   (reference flat ────┘                                                   │  dm.py → H (Gaussian IF, │
    frame, optional)                                                       │   coupling c) → G=H⁺·(−½)│
                                                                           └────────────┬─────────────┘
                                                                                        │ writes AOMX .bin
                                                                                        ▼
                                            ┌───────────────────────────────────────────────────────┐
                                            │  calib/  R.aomx  Mpinv.aomx  Z.aomx  G.aomx            │
                                            │          refslopes.aomx  subapmap.aomx  (K=G·R opt.)  │
                                            └───────────────────────────────────────────────────────┘
                        ┌──────────────────────────────────────────────────────────────────────┐
                        │                  REAL-TIME PER-FRAME PATH (C / bin/wfs_rt)             │
                        │                  per frame; target < 10 ms (≈ µs at PS9 scale)         │
                        └──────────────────────────────────────────────────────────────────────┘
  .bmp frame ──▶ bmp.c ──▶ preprocess ──▶ centroid.c ──▶ slopes.c ──▶  s = [sx₁..sxM, sy₁..syM]
  (8/24-bit)     read       (−dark,/flat,   TWCoG over     (centroid−ref)        │  (2M vector)
                 gray       −background)    sub-ap grid     ·p_pix/f_MLA          │
                                            + ref subtract                       │
                                                                                 ▼
                          ┌──────────────────────────────┬─────────────────────────────────────────┐
                          ▼                               ▼                                         ▼
            reconstruct.c:  φ = R·s   (MVM, zonal)   reconstruct.c: a = M⁺·s (MVM, modal)   [optional fused]
                          │  → WAVEFRONT MAP W(x,y)      │  → ZERNIKE COEFFS a               c = K·s, K=G·R
                          │     (DELIVERABLE 1)          │     W = Z·a  (cross-check)        (slopes→commands
                          │                              │     (feeds r0/τ0 stats)           in ONE MVM)
                          ▼                               ▼
            dmcmd.c:  a_dm = G·φ  (MVM)            (Zernike coeffs streamed to disk
                      = H⁺·(−W/2)                   for turbulence analysis)
                      clip to ±stroke_max
                      → ACTUATOR MAP A(x,y)
                        (DELIVERABLE 3, stroke units)
                                                                 ┌───────────────────────────────────┐
   slopes / Zernike-coeff / phase time-series ───────────────▶  │ OFFLINE (Python):                   │
   (written by wfs_rt or produced by run_pipeline_py)           │ turbulence.py + analyze_turbulence  │
                                                                │  r0 (≥7 estimators), τ0 (≥6),       │
                                                                │  combine → turbulence_summary.json  │
                                                                │  (DELIVERABLE 2: r0, τ0)            │
                                                                └───────────────────────────────────┘
```

**Key real-time insight (`[R02]` §15, `[R06]` §2):** because reconstruction and DM command are *linear* in the slopes and all matrices are precomputed, the entire per-frame algebra is **2–3 dense MVMs** (`φ=R·s`, `a=M⁺·s`, `a_dm=G·φ`), each `O(N²)` flops but `O(1)` per actuator. The reconstructed phase `W` is a required output (Deliverable 1), so we keep `R` and `G` separate; if only commands were needed we would fuse `K = G·R` and run a single MVM `c = K·s`.

### 2.2 Per-frame latency budget (PS9 lab scale, ≤ 30×30 sub-apertures)

From `[R06]` §3, §10. Take `n_sub` sub-apertures across, `M ≈ 2·n_sub²` slopes, matrices `~M×M` fp32 (≤ ~8 MB, cache-resident):

| Stage | Work | Est. wall-clock (1 core, AVX2) |
|-------|------|--------------------------------|
| BMP read + preprocess | tiny frame, table-driven | ~10–50 µs |
| Centroiding (TWCoG) | O(1)/sub-ap, ~handful FMAs × N_sub | ~5–30 µs |
| `φ = R·s` (zonal MVM) | `2·N·2M` FLOPs, ≤ 4×10⁶ | ~1–10 µs |
| `a = M⁺·s` (modal MVM) | `2·J·2M` FLOPs | ~1–5 µs |
| `a_dm = G·φ` (DM MVM) | `2·N_act·N` FLOPs | ~1–10 µs |
| **Total compute** | — | **~20–100 µs** → **100×–500× under 10 ms** |

GPUs, sparse solvers, CuReD, and FFT reconstructors are unnecessary at this scale (`[R06]` §4, §8); they are retained only as documented scaling/validation paths.

---

## 3. Chosen algorithms per stage

For each stage: the **chosen** algorithm (with research justification) and the **retained alternatives** (kept for cross-validation/robustness, per the multi-method philosophy of §7).

### 3.1 Centroiding — `[R01]`

**Chosen: Thresholded + Windowed Weighted Center-of-Gravity (TWCoG)**, `O(1)` per sub-aperture. Per sub-aperture, per frame (`[R01]` §12):

1. Slice the **precomputed per-lenslet window** (from flat-wavefront calibration).
2. Apply **noise-floor threshold** `I_T = max(T·I_max, m·σ_read)`, `T≈0.05–0.10`, `m≈3–5`; clip negatives. Enforce the **≥3-pixels-above-threshold** validity gate.
3. Compute **weighted first moments** with a **precomputed Gaussian weight LUT** centered on the reference (FWHM ≈ spot FWHM ≈ 4–4.5 px): `x̂ = Σ(i·W·I) / Σ(W·I)` using precomputed `i·W`, `j·W` index tables (each moment = one dot product, FMA-friendly).
4. Apply the **WCoG gain `γ`** (precomputed) and an optional **intra-pixel bias-correction LUT**.
5. Subtract the **reference centroid** (same algorithm/threshold/window as calibration → common-mode bias cancels). This is the cheapest, most effective bias killer.
6. Convert to slope (Stage 3.2).

**Justification (`[R01]` §3, §12):** for lab-simulated turbulence with compact point-source spots on a science-grade camera (medium-to-high SNR) and a fixed Fried grid, TWCoG is the **maximum-likelihood spot-position estimator under Gaussian noise**, runs in a fixed flop budget with no runtime transcendentals, and reaches near-Cramér–Rao accuracy. Centroiding accuracy dominates final wavefront error (`[R01]` §2.2), so the cheap calibration/bias-LUT work is where accuracy is won.

**Retained alternatives:**
- **Plain CoG** — the inner kernel and always-on baseline (`TWCoG` with `W=1`, `T=0`); bootstraps reference-spot detection.
- **Thresholded CoG (TCoG)** and **Weighted CoG (WCoG)** — intermediate variants, exposed for ablation.
- **Windowed/floating CoG** — for spots drifting toward cell boundaries (dynamic range, cross-talk); FPGA migration path.
- **Brightest-pixel (Basden)** — drop-in when per-frame brightness varies.
- **Iteratively-Weighted CoG (IWCoG, fixed 2-iter cap)** — large excursions / low SNR.
- **Correlation / matched-filter** — fallback for extended/elongated spots (with peak-locking anti-symmetry correction).
- **Gaussian-fit / Maximum-likelihood** — offline ground-truth references to benchmark TWCoG against the CRLB. Python only.

All variants are implemented in `aokit/centroiding.py` (validation); the C core implements TWCoG + CoG kernel.

### 3.2 Geometry & slopes — `[R02]` §1, §5; `[R07]` A.4

**Chosen: Fried geometry** (actuators at sub-aperture corners). Offline (`aokit/geometry.py`):

- From MLA pitch / pixel pitch / focal length, compute the nominal lenslet grid (`pitch_m / pixel_size_m` = px per lenslet).
- **Reference-frame registration:** detect the spot in each cell from a flat-wavefront frame, CoG → **reference centroids** `x_ref,k` (averaged over many flats). These define cell origins and per-lenslet windows.
- **Active sub-aperture mask:** keep cells whose flux exceeds ~50–75% of the unobscured-cell flux (excludes under-illuminated pupil-edge lenslets).
- **Actuator grid:** `(n+1)×(n+1)` corners for `n×n` lenslets; build the lenslet↔actuator index maps.

**Spot → slope** (Stage in `src/slopes.c`): `s = (centroid − ref) · p_pix / f_MLA` (radians of tilt). Only `p_pix` and `f_MLA` enter here — one multiply per axis per sub-aperture (`[R01]` §1.1, `[R02]` §1).

### 3.3 Zonal reconstruction — `[R02]`

**Chosen: precomputed, regularized least-squares Fried reconstructor `R`.** Offline (`aokit/reconstructor.py`):

- Build the **Fried gradient matrix `Γ`** (`2M×N`) from the corner-averaging equations over the **valid** sub-apertures only (`[R02]` §2.2, §4):
  ```
  s^x = (1/2h)[(φ_b − φ_a) + (φ_d − φ_c)]
  s^y = (1/2h)[(φ_c − φ_a) + (φ_d − φ_b)]   (corners a,b,c,d = TL,TR,BL,BR; pitch h)
  ```
- Compute `R = Γ⁺` via **SVD with singular-value thresholding**, then **explicitly null piston `𝟙` and waffle `w`** (projector `I − wwᵀ/‖w‖² − 𝟙𝟙ᵀ/‖𝟙‖²`, or Gavel waffle-penalty in the normal equations). **Non-negotiable:** Fried geometry has a waffle null mode (checkerboard pattern producing zero average slope over every sub-aperture); unhandled, it grows unbounded (`[R02]` §6).
- Optionally upgrade to **Tikhonov/MMSE** by folding in `C_φ(r₀)` and slope-noise `C_n` once `r₀` is estimated — same runtime cost, better noise rejection (`[R02]` §3.2, §10).

**Runtime (`src/reconstruct.c`):** `φ = R·s` — one MVM → wavefront map (Deliverable 1).

**Justification (`[R02]` §15):** lab grids are small (`N ≲ 1000`); `R·s` is single-digit microseconds in plain C. Dense MVM is the pragmatic winner; FFT/CuReD only matter past thousands of actuators.

**Retained alternatives (Python, `reconstructor.py`):**
- **FFT Fourier Transform Reconstructor (FTR)** with the exact **Fried filter** `gx=(e^{ if_y}+1)(e^{ if_x}−1)`, `gy=(e^{ if_x}+1)(e^{ if_y}−1)`, Nyquist rows/cols zeroed (= waffle removal), plus Poyneer boundary extension. `O(N log N)`; validation cross-check + scaling fallback (`[R02]` §7).
- **Direct path integration** — `O(N)` baseline sanity check (`[R02]` §5).
- **Southwell / Hudgin** dense LS reconstructors — alternative geometries for noise-propagation comparison (`[R02]` §13).
- **SOR / PCG / multigrid / CuReD** — iterative & linear-time solvers, documented scaling story (`[R02]` §8, §9).

### 3.4 Modal reconstruction — `[R03]`

**Chosen: Zernike modal fit (Noll-indexed, RMS-normalized) with analytic-gradient interaction matrix + SVD/Tikhonov pseudo-inverse.** Offline (`aokit/zernike.py`, `aokit/reconstructor.py`):

- **Basis:** Zernike, **Noll single index**, **RMS(Noll) normalization** (each coefficient *is* the RMS wavefront of that mode), **piston excluded** (unobservable by SH), modes `j = 2…J`. Pick `J ≈ N_actuators` and `J ≪ 2·N_sub` (e.g. `J ≈ 35` for 7×7); validate with `cond(M)` and `Tr[(MᵀM)⁻¹]` (`[R03]` §2.8).
- **Interaction matrix `M`** (`2M×J`): column `j` is the SH slope response to Zernike mode `j`, built **analytically** from sub-aperture-averaged Zernike gradients (`⟨∂Z_j/∂x⟩_k`), via analytic sub-aperture integrals (divergence theorem → 1-D perimeter integrals, à la `mshwfs`) and cross-checked against `makegammas` Γ matrices (`[R03]` §2).
- **Reconstructor `R_modal = M⁺`** via **SVD with Tikhonov damping** (`σ_i/(σ_i²+μ²)`) — numerically safe, tunable (`[R03]` §2.4).
- **Synthesis matrix `Z`** (`N_pts×J`): `W = Z·a`.

**Runtime (`src/reconstruct.c`):** `a = M⁺·s` (one MVM) → Zernike coefficients; optional `W = Z·a` cross-check vs zonal.

**Justification (`[R03]` §6):** the evaluation explicitly asks for Zernike coefficients; modal coefficients feed turbulence stats (`r₀`/`τ₀`) directly; reconstruction is one `gemv`. The modal map cross-checks the zonal map.

**Retained alternatives:** **Karhunen–Loève** (statistically-independent coefficients → cleaner `r₀`/`τ₀`; `M_KL = M_Zernike·E`), **annular/Gram–Schmidt Zernikes** (if pupil obscured/non-circular), **Bayesian/MAP** (minimum-MSE in low light), **hybrid modal+zonal** (low orders modal, high-freq zonal). All Python, all reduce at runtime to a precomputed MVM (`[R03]` §5).

### 3.5 Turbulence characterization — `[R04]`

**Chosen: multi-method, cross-validated estimation across three data domains** (raw slopes, reconstructed phase, intensity). The robustness story is "independent estimators that agree."

**`r₀` — ≥7 estimators** (`aokit/turbulence.py`):
| ID | Estimator | Domain | Core relation |
|----|-----------|--------|---------------|
| R1 | **Zernike-coefficient variance vs Noll** (primary; also yields `L₀`) | phase | `⟨a_j²⟩ = c_j·(D/r₀)^{5/3}`, fit modes 4–15 |
| R2 | Slope (gradient) variance / slope structure function | slopes | `⟨α²⟩ = 0.170 λ² r₀^{−5/3} d^{−1/3}` (G-tilt) |
| R3 | **DIMM** differential tip/tilt between sub-aperture pairs (vibration-immune) | slopes | `σ_{l,t}² = K_{l,t} λ² r₀^{−5/3} D_sub^{−1/3}` |
| R4 | Total phase variance (Noll normalization) | phase | `σ_φ² = 1.0299 (D/r₀)^{5/3}` (TT-removed: 0.134) |
| R5 | Kolmogorov phase **structure-function fit** (also validates Kolmogorov via 5/3 slope) | phase | `D_φ(r) = 6.88 (r/r₀)^{5/3}` |
| R6 | **von Kármán** joint `(r₀, L₀)` fit | phase/slopes | `W_φ(f)=0.0229 r₀^{−5/3}(f²+1/L₀²)^{−11/6}` |
| R7 | Seeing FWHM (image domain) | intensity | `ε = 0.98 λ/r₀` |

**`τ₀` — ≥6 estimators** (`aokit/turbulence.py`):
| ID | Estimator | Core relation |
|----|-----------|---------------|
| T1 | **Temporal autocorrelation** 1/e of mid-order modes (primary; also yields noise) | `C_i(τ)/C_i(0)=1/e` |
| T2 | Temporal **PSD** slopes + cutoff | `f_c,i ≈ 0.3(n_i+1)v/D`; slopes −11/3 (tilt), −17/3 (higher) |
| T3 | **Greenwood frequency** bridge | `f_G = 0.426 v/r₀`, `τ₀ = 0.134/f_G = 0.314 r₀/v` |
| T4 | Temporal **structure function** (reaches 1 rad²) | `D_φ(τ)=6.88(vτ/r₀)^{5/3}`; `t₀=0.66 τ₀` |
| T5 | **Taylor frozen-flow** wind retrieval (spatio-temporal cross-correlation) | peak at `r=v·τ` → `v` → `τ₀` |
| T6 | **Tyler frequency** (tip/tilt-specific) | `f_T = 0.368 v r₀^{−1/6} D^{−5/6}` |

**Bias removal (mandatory; `[R04]` §3.3):** subtract centroid-noise variance (from `τ=0` ACF jump / high-f PSD floor); iterate out modal cross-coupling/aliasing; exclude tip/tilt from the `r₀` fit; use von Kármán when `L₀` is finite.

**Combiner:** tabulate all estimates, take the **median** as the central value and the **spread** as systematic uncertainty; `v` is the common currency tying `r₀`↔`τ₀` via `f_G` (`[R04]` §5). Output `turbulence_summary.json` (§4.4).

### 3.6 DM actuator map — `[R05]`

**Chosen: influence-matrix deconvolution** — `a_dm = H⁺·(−W/2)`, **NOT** naive sampling of the conjugate.

Offline (`aokit/dm.py`):
- Read provided DM info: `N_act`, pitch `d`, **inter-actuator coupling `c`** (and measured IF if supplied). Confirm **Fried** registration (`d` = lenslet pitch on DM).
- Build the **influence-function matrix `H`** (`N_pts×N_act`), each column a Gaussian IF `exp(−r²/2σ²)` with `σ = d/√(−2 ln c)` (so the bump drops to `c` at the pitch). Power-law / modified-Gaussian / measured IFs are alternatives (`[R05]` §1.3–1.4).
- Compute the **regularized command matrix** `G = (HᵀH + μ²I)⁻¹ Hᵀ · (−½)` (Tikhonov), or truncated-SVD `H⁺·(−½)`. While doing the SVD, **null/penalize the waffle mode** and small singular values. Store stroke calibration `g` and limit `a_max`.
- Optionally **fuse** `K = G·R` (slopes→commands in one MVM) — but keep `G`,`R` separate because `W` is a required output.

Runtime (`src/dmcmd.c`):
- Form target `s_target = −W/2` (the **factor-of-2 reflection**: a surface displaced by `z` changes OPD by `2z`; convert radians→length via `λ/2π` if needed).
- **One MVM** `a_dm = G·φ` (= `H⁺·(−W/2)`), then `a_dm ← clip(a_dm, ±a_max)`.
- Output in **actuator-stroke-length units** via gain `g` (Deliverable 3).

**Justification (`[R05]` §3.2, §9, §11):** poking one actuator also moves its neighbours (coupling), so naive sampling **over-corrects**. `H⁺` contains explicit **anti-coupling off-diagonal terms** that back each actuator off to account for neighbour contributions — the inter-actuator coupling is *deconvolved*, exactly as PS9 mandates. The residual is the fundamental fitting error `σ²_fit = μ(d/r₀)^{5/3}`, reported as a quality metric linking to the `r₀` module.

**Retained alternative:** the **calibration-based interaction matrix** route `D = ∂s/∂a`, `C = D⁺`, `a = C·s_meas` — used if poke frames are provided; closed-loop operational standard (`[R05]` §6).

### 3.7 Real-time engine — `[R06]`

**Chosen: dense MVM workhorse.** All matrices precomputed offline; per-frame cost = centroiding + 2–3 MVMs. The C GEMV (`src/linalg.c`) is portable scalar by default with an `#ifdef __AVX2__` FMA path and optional `#ifdef _OPENMP` parallelization, row-major contiguous, fp32, **no external BLAS** (constraint-driven deviation from `[R06]`'s OpenBLAS suggestion; at PS9 scale a hand-rolled AVX2 GEMV is microsecond-class and memory-bandwidth-bound regardless). Determinism via pre-sized buffers, zero in-loop allocation (`[R06]` §10).

---

## 4. Data structures & file formats

These formats are the **contract** that lets the C core and the Python toolkit interoperate. Both `src/matio.c` and `aokit/matio.py` MUST agree byte-for-byte (guarded by `tests/test_matio_roundtrip.py`).

### 4.1 Config / metadata schema (JSON)

Loaded by `aokit/config.py` (→ `Config` dataclass) and `src/aoconfig.h` (key/value loader). `config/example_config.json` is a valid instance. Schema:

```jsonc
{
  "schema_version": 1,
  "camera": {
    "pixel_size_m": 5.5e-6,     // detector pixel pitch (m)
    "frame_w": 256,             // frame width  (px)
    "frame_h": 256,             // frame height (px)
    "bit_depth": 8              // 8 or 24
  },
  "mla": {
    "n_lenslets_x": 10,         // lenslets across
    "n_lenslets_y": 10,
    "pitch_m": 1.5e-4,          // lenslet pitch projected on detector grid context (m)
    "focal_length_m": 5.2e-3    // MLA focal length f_MLA (m)
  },
  "pupil": {
    "diameter_m": 1.5e-3,       // turbulated-beam pupil diameter D (m)
    "center_x_px": 128.0,       // pupil center on the detector (px)
    "center_y_px": 128.0
  },
  "wavelength_m": 6.33e-7,      // sensing wavelength (m)
  "dm": {
    "n_act_x": 11,              // Fried: n_lenslets + 1
    "n_act_y": 11,
    "pitch_m": 1.5e-4,          // actuator pitch on the pupil (m); = lenslet pitch in Fried
    "coupling_coeff": 0.15,     // inter-actuator coupling c (fraction); PS9-provided
    "stroke_max_m": 3.5e-6,     // |stroke| limit a_max (m)
    "influence_model": "gaussian", // "gaussian" | "power_law" | "measured"
    "influence_alpha": 2.0,     // power index (gaussian => 2.0)
    "stroke_gain_m_per_unit": 1.0e-6 // g: meters of surface per unit command
  },
  "geometry": {
    "type": "fried",            // actuators at sub-aperture corners
    "rotation_deg": 0.0,        // MLA-to-detector clocking
    "flip_y": false
  },
  "cadence": {
    "dt_s": 2.0e-3              // inter-frame interval (s); drives τ0
  },
  "ground_truth": {             // OPTIONAL: present only for synthetic datasets
    "r0_m": 0.15,
    "tau0_s": 0.0045,
    "wind_speed_mps": 10.0,
    "L0_m": 25.0,
    "zernike_noll": []          // optional injected modal content
  }
}
```

Notes: `pitch_m` for MLA and DM are both referenced to the pupil plane; in Fried geometry `dm.pitch_m == mla.pitch_m`. The `ground_truth` block is written by `scripts/generate_dataset.py` and consumed by `aokit/validation.py`; it is absent for real organizer data.

### 4.2 AOMX binary matrix file format

Self-describing little-endian binary, used for **all** precomputed matrices and vectors passed Python→C. Exact layout (header 32 bytes, then row-major data):

```
Offset  Size  Type     Field        Meaning
------  ----  -------  -----------  --------------------------------------------------
  0      4    char[4]  magic        'A','O','M','X'  (0x414F4D58)
  4      4    uint32   version      format version = 1
  8      4    uint32   rows         number of rows R
 12      4    uint32   cols         number of cols C   (vector => cols=1)
 16      4    uint32   dtype        0 = float32, 1 = float64
 20      4    uint32   layout       0 = row-major (C order)   [only 0 defined in v1]
 24      4    uint32   flags        bit0: 1 if a "vector" semantically (cols==1)
 28      4    uint32   checksum     additive checksum of the data bytes mod 2^32
                                    (0 == "not computed", readers may skip verify)
 32    R*C*sz  data    payload      row-major: element(i,j) at index i*C + j
                                    sz = 4 (float32) or 8 (float64)
```

- **Endianness:** little-endian (x86_64 native; readers/writers use explicit byte assembly for portability).
- **Total file size:** `32 + rows*cols*sizeof(dtype)` bytes.
- **dtype policy:** offline SVD/inverse done in fp64; matrices that feed the real-time core are written as **float32** (`dtype=0`) for bandwidth; analysis matrices may be float64.
- **checksum:** simple additive sum of payload bytes (mod 2³²); cheap integrity check, optional to verify.

C API (`src/matio.h`): `aomx_read(path, &rows, &cols, &dtype, &data)` and `aomx_write(path, rows, cols, dtype, data)`. Python API (`aokit/matio.py`): `write_aomx(path, array, dtype="f32")`, `read_aomx(path) -> np.ndarray`. The roundtrip test writes from numpy, reads in C (and vice-versa) and asserts equality.

**Standard calibration artifacts (written to `calib/`):**
| File | Shape | dtype | Meaning |
|------|-------|-------|---------|
| `R.aomx` | `N × 2M` | f32 | zonal reconstructor (`φ = R·s`) |
| `Mpinv.aomx` | `J × 2M` | f32 | modal reconstructor (`a = M⁺·s`) |
| `Z.aomx` | `N_pts × J` | f32 | Zernike synthesis (`W = Z·a`) |
| `G.aomx` | `N_act × N` | f32 | DM command (`a_dm = G·φ = H⁺·(−½)·φ`) |
| `K.aomx` (opt.) | `N_act × 2M` | f32 | fused `K = G·R` (`c = K·s`) |
| `refslopes.aomx` | `2M × 1` | f32 | reference slopes (flat wavefront) |
| `subapmap.aomx` | `N_sub × 4` | f32 | per-sub-ap: `[x0_px, y0_px, win_w, win_h]` (and validity via flux) |

### 4.3 Per-frame output formats

- **Reconstructed phase map** `W(x,y)`: one AOMX `N_pts×1` (or grid `H×W`) per frame, in `phase_*.aomx`, plus optional `.bmp`/PNG visualization. Units: radians at the sensing wavelength.
- **Zernike coefficients:** appended to `zernike_coeffs.csv` — columns `frame_idx, j2, j3, ..., jJ` (Noll order, RMS-normalized, radians).
- **Actuator map:** `actuators_<frame>.csv` (columns `act_x, act_y, stroke_m`) and/or `actuators.bin` (AOMX `N_act×1`, meters of stroke).
- **Slopes time-series** (for turbulence analysis): `slopes.csv` or `slopes.aomx` (`T×2M`).

### 4.4 Turbulence summary (JSON)

Written by `scripts/analyze_turbulence.py`:

```jsonc
{
  "r0_m": { "median": 0.148, "spread": 0.006,
            "estimators": { "R1_zernike_var": 0.149, "R2_slope_var": 0.146,
                            "R3_dimm": 0.150, "R4_phase_var": 0.144,
                            "R5_struct_fn": 0.151, "R6_vonkarman": 0.148,
                            "R7_seeing": 0.145 } },
  "L0_m": 24.3,
  "tau0_s": { "median": 0.0046, "spread": 0.0004,
              "estimators": { "T1_autocorr": 0.0045, "T2_psd": 0.0047,
                              "T3_greenwood": 0.0046, "T4_struct_fn": 0.0044,
                              "T5_frozenflow": 0.0048, "T6_tyler": 0.0046 } },
  "wind_speed_mps": 10.2,
  "f_greenwood_hz": 29.1,
  "seeing_arcsec": 0.87,
  "strehl_marechal": 0.41,
  "n_frames": 2000, "dt_s": 0.002,
  "notes": "tip/tilt excluded from r0 fit; von Karman used (L0 finite)."
}
```

---

## 5. Directory layout

```
bah2026-ps9/
├── ARCHITECTURE.md                # this document (primary deliverable)
├── README.md                      # overview, quickstart, build/run, results placeholder
├── .gitignore                     # Python/C artifacts; keeps configs
├── requirements.txt               # numpy, scipy, matplotlib (pip-installable)
├── Makefile                       # builds src/ -> bin/wfs_rt; targets: all, clean, test
├── config/
│   └── example_config.json        # realistic 256x256, 10x10 lenslets, 11x11 actuators
├── src/                           # self-contained C11 real-time core (only -lm)
│   ├── aoconfig.h                 # Config struct + loader interface
│   ├── bmp.h / bmp.c              # hand-rolled BMP read (8/24-bit) + write
│   ├── matio.h / matio.c          # AOMX binary matrix reader/writer
│   ├── linalg.h / linalg.c        # portable GEMV (#ifdef AVX2/FMA + OpenMP), dot
│   ├── centroid.h / centroid.c    # TWCoG over sub-aperture grid (+ CoG kernel)
│   ├── slopes.h / slopes.c        # centroid -> slope (p_pix / f_MLA)
│   ├── reconstruct.h / reconstruct.c  # MVM zonal (R) + modal (M+, Z)
│   ├── dmcmd.h / dmcmd.c          # DM command MVM (G), stroke clipping
│   ├── pipeline.h / pipeline.c    # per-frame + frame-loop with timing
│   └── main.c                     # CLI: config + matrices + frames -> outputs + timing
├── aokit/                         # Python offline toolkit (package)
│   ├── __init__.py                # STABLE CONTRACT: imports submodules (do not edit)
│   ├── config.py                  # load/validate JSON -> Config dataclass
│   ├── geometry.py                # Fried grids, pupil mask, references, lenslet<->actuator maps
│   ├── zernike.py                 # Noll Zernike values + analytic gradients; index helpers
│   ├── centroiding.py             # CoG/TCoG/WCoG/TWCoG/correlation/gaussfit (validation)
│   ├── reconstructor.py           # zonal Fried LS R (Tikhonov/SVD, waffle removal); modal M+; FTR
│   ├── dm.py                      # influence matrix H (Gaussian+coupling), command H+, strokes
│   ├── turbulence.py              # >=7 r0 + >=6 tau0 estimators + combiner
│   ├── datagen.py                 # phase screens (FFT/+subharm/Zernike), spot fields, noise, frozen-flow, BMP
│   ├── bmpio.py                   # BMP read/write in pure numpy
│   ├── matio.py                   # AOMX read/write (byte-matches src/matio.c)
│   ├── validation.py              # RMS WFE, Strehl, phase corr, r0/tau0 recovery, DM residual
│   └── viz.py                     # matplotlib: spot field, phase maps, Zernike spectrum, trends
├── scripts/                       # top-level entrypoints
│   ├── build_calibration.py       # config -> R, M+, Z, G, refslopes, subapmap -> AOMX .bin
│   ├── generate_dataset.py        # config + r0,tau0 -> synthetic .bmp series + ground-truth JSON
│   ├── run_pipeline_py.py         # pure-Python end-to-end (validation/plots)
│   └── analyze_turbulence.py      # slopes/coeffs time-series -> r0/tau0 report + plots
├── tests/                         # pytest
│   ├── test_zernike.py
│   ├── test_centroiding.py
│   ├── test_reconstruction.py
│   ├── test_turbulence.py
│   ├── test_dm.py
│   └── test_matio_roundtrip.py    # ensures aokit/matio.py and src/matio.c agree
└── docs/
    └── METHODS.md                 # maps each pipeline stage -> justifying research report
```

---

## 6. Build & run instructions & tech-stack rationale

### 6.1 Build the C real-time core

```bash
make            # builds bin/wfs_rt with -O3 -march=native -fopenmp, links only -lm
make clean      # removes objects and binary
make test       # builds and runs the C self-test (matio roundtrip, BMP roundtrip)
```

The Makefile uses `-O3 -march=native -fopenmp`; AVX2/FMA kernels are compiled conditionally via `#ifdef __AVX2__` (enabled by `-march=native` on AVX2 hosts) and degrade gracefully to portable scalar code elsewhere. **No BLAS/FFTW/LAPACK** — only `-lm`.

### 6.2 Python offline toolkit

```bash
python3 -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt                       # numpy, scipy, matplotlib
python3 -c "import aokit"                              # import sanity
python3 -m json.tool config/example_config.json       # validate config JSON
pytest -q                                             # run unit tests
```

### 6.3 End-to-end (synthetic) workflow

```bash
# 1. Generate a synthetic dataset with known r0/tau0
python3 scripts/generate_dataset.py --config config/example_config.json \
        --r0 0.15 --tau0 0.0045 --frames 2000 --out data/synthetic/run01

# 2. Build calibration matrices (AOMX) for the C core
python3 scripts/build_calibration.py --config config/example_config.json --out calib/

# 3a. Run the FAST C pipeline over the frames
bin/wfs_rt --config config/example_config.json --calib calib/ \
           --frames "data/synthetic/run01/frame_*.bmp" --out out/

# 3b. (or) run the pure-Python pipeline for plots/validation
python3 scripts/run_pipeline_py.py --config config/example_config.json \
        --frames "data/synthetic/run01/frame_*.bmp" --out out_py/

# 4. Turbulence multi-method report
python3 scripts/analyze_turbulence.py --in out/slopes.csv \
        --config config/example_config.json --out out/turbulence_summary.json
```

### 6.4 Tech-stack rationale (`[R06]` §13)

**Two-tier split — "slow, smart, offline" (Python) + "fast, dumb, online" (C):**
- **C11 real-time core:** determinism (no GC, no hidden allocation), the PS explicitly advises C, direct SIMD/alignment/cache control, portability x86 (AVX2/AVX-512) ↔ ARM (NEON). Constraint-driven choice: **self-contained, only `-lm`** (no OpenBLAS/FFTW), because the host lacks those libraries; at PS9 scale a hand-rolled AVX2 GEMV is already microsecond-class.
- **Python (numpy/scipy/matplotlib):** calibration is a one-time cost so developer speed wins; SVD/pinv, DM influence/coupling, turbulence stats, synthetic data, visualization, and validation all live here. Python writes AOMX matrices the C core loads once.

---

## 7. Robustness via multiple methods

**Philosophy (`[R04]` §5, `[R01]` §12):** independent estimators must agree; one fills the gaps of another. Three **orthogonal data domains** are exploited — **raw slopes**, **reconstructed phase**, and **intensity/imaging** — so a bug in one domain cannot fake agreement across all three. Vibration immunity is built in via differential estimators; model validity is *tested* (5/3 slope, PSD slopes), not assumed; `r₀` and `τ₀` are linked through `v` and `f_G` into one self-checking system.

**Enumerated catalogue (≥30 distinct methods/algorithms used or retained):**

*Centroiding (13) — `[R01]`:*
1. Plain CoG · 2. Thresholded CoG · 3. Weighted CoG · 4. Iteratively-Weighted CoG · 5. Windowed/floating CoG · 6. **TWCoG (chosen)** · 7. Quad-cell · 8. Brightest-pixel (Basden) · 9. Parabolic 3-point peak · 10. Gaussian-fit · 11. Correlation/matched-filter · 12. Maximum-likelihood · 13. Template-matching.

*Reconstruction geometries & algorithms (8) — `[R02]`, `[R03]`:*
14. **Fried zonal LS reconstructor (chosen)** · 15. FFT Fourier Transform Reconstructor (Fried filter) · 16. Direct path integration · 17. Southwell zonal LS · 18. Hudgin zonal · 19. **Zernike modal LS/SVD-Tikhonov (chosen)** · 20. Karhunen–Loève modal · 21. SOR / PCG / multigrid / CuReD iterative & linear-time solvers.

*Turbulence `r₀` estimators (7) — `[R04]` §1:*
22. Zernike-variance vs Noll · 23. Slope-variance / slope structure-function · 24. DIMM differential motion · 25. Total phase variance · 26. Kolmogorov structure-function fit · 27. von Kármán joint `L₀` fit · 28. Seeing FWHM.

*Turbulence `τ₀` estimators (6) — `[R04]` §2:*
29. Temporal autocorrelation 1/e · 30. Temporal PSD slopes/cutoff · 31. Greenwood-frequency bridge · 32. Temporal structure function · 33. Taylor frozen-flow wind retrieval · 34. Tyler tracking frequency.

*Phase-screen generators (6) — `[R07]` B.1:*
35. FFT/spectral · 36. FFT + subharmonics · 37. Zernike/Noll synthesis · 38. Covariance/Cholesky · 39. Infinite/streaming (Assémat & Wilson) · 40. AR frozen-flow + boiling.

*DM influence / command models (5) — `[R05]`:*
41. Gaussian IF · 42. Power-law / modified-Gaussian / thin-plate IF · 43. Measured-IF matrix · 44. **`H⁺·(−½)` model command (chosen)** · 45. Calibration interaction-matrix `C=D⁺`.

*Validation metrics (8) — `[R07]` C:*
46. RMS WFE · 47. Strehl (Maréchal) · 48. Phase correlation `ρ` · 49. `r₀` recovery error · 50. `τ₀` recovery error · 51. DM-corrected residual · 52. reconstructor self-consistency (round-trip) · 53. C/Python parity.

**Count: 53 distinct methods/algorithms/metrics across all stages — comfortably ≥30.** The *chosen* primary at each stage is bolded; the rest are retained for cross-validation, ablation, robustness fallback, or as ground-truth oracles.

---

## 8. Validation & testing plan

**Backbone (`[R07]` C):** synthetic data carries injected ground truth, so every metric is *recovered vs. true*. We prove correctness before the organizer dataset arrives and quantify error afterward.

### 8.1 Synthetic inject-known → recover
- `scripts/generate_dataset.py` generates `.bmp` time-series at **known `r₀`** (FFT+subharmonics screen) translated by **known wind `v`** (frozen flow ⇒ **known `τ₀ = 0.314 r₀/v`**), with full detector noise (Poisson shot + Gaussian read + 8-bit quantization).
- Run the *same* ingestion → reconstruction → characterization pipeline that will consume real data; assert recovery of injected `r₀`, `τ₀`, and Zernike content within tolerance.

### 8.2 Metrics (`aokit/validation.py`)
- **RMS WFE** `σ_WFE = sqrt(mean((W_rec − W_true)²))` over the valid pupil (piston/TT removed); pass `< λ/14`.
- **Strehl (Maréchal)** `S = exp(−σ_φ,res²)`; match FFT-PSF Strehl within ~10% for `S > 0.6`.
- **Phase correlation** Pearson `ρ` over the pupil; pass `ρ > 0.95` (low noise).
- **`r₀` / `τ₀` recovery error** vs injected; pass within a few % / ~10–20%.
- **DM residual** `σ(W_true − DM_shape)` with coupling-aware IFs; large reduction vs uncorrected.
- **Reconstructor self-consistency:** `slope(W_rec) ≈ slope_meas` (round-trip ≈ 0).
- **C/Python parity:** identical synthetic frame through both readers/centroiders → matching slopes (guards the fast path).

### 8.3 Unit tests with known Zernikes (`tests/`, `[R07]` C.3)
1. **Pure tip/tilt (Z2/Z3):** all spots shift by `Δ=f·θ`; reconstruct a flat gradient. Checks px↔slope scale & sign.
2. **Pure defocus (Z4):** spots shift radially; reconstruct the parabola.
3. **Astig/coma/higher (Z5–Z11):** inject `a_j`, assert `â_j ≈ a_j`, cross-terms ≈ 0 (mode purity).
4. **Superposition/linearity:** inject `Σ a_j Z_j`, check linear recovery.
5. **Statistical screen:** many von Kármán screens at known `r₀` → ensemble Zernike variances follow Noll `(D/r₀)^{5/3}`.
6. **Null-space/waffle:** confirm piston & Fried waffle are regularized/filtered, not blown up.
7. **AOMX roundtrip:** numpy→file→C and C→file→numpy byte/value parity.

### 8.4 Monte-Carlo convergence
- `r₀`: ≥1000 independent screens at known `r₀`; fitted `r₀` converges with bias & scatter reported vs N.
- `τ₀`: frozen-flow series swept over `v`; confirm linear `τ₀ ∝ r₀/v` and Greenwood closure `τ₀ = 0.134/f_G`.

---

## 9. Implementation roadmap & risk register

### 9.1 Phased roadmap

| Phase | Deliverable | Modules | Depends on |
|-------|-------------|---------|-----------|
| **P0** | Scaffold + formats + build/import green | this doc, all stubs, Makefile, AOMX, config | — |
| **P1** | I/O + geometry | `bmpio`/`bmp.c`, `matio`/`matio.c`, `config`, `geometry`; AOMX roundtrip + BMP parity tests | P0 |
| **P2** | Synthetic datagen | `datagen` (FFT+subharmonics, geometric & Fraunhofer spots, noise, frozen-flow), `generate_dataset.py` | P1 |
| **P3** | Centroiding + slopes | `centroiding.py` (all variants), `centroid.c`/`slopes.c` (TWCoG); known-Zernike spot-shift tests | P1, P2 |
| **P4** | Zonal + modal reconstruction | `zernike`, `reconstructor` (Γ, R, M⁺, Z; FTR cross-check), `reconstruct.c`; `build_calibration.py` | P3 |
| **P5** | DM actuator map | `dm` (H, G, coupling), `dmcmd.c`; DM residual + coupling tests | P4 |
| **P6** | Turbulence characterization | `turbulence` (≥7 r₀ + ≥6 τ₀ + combiner), `analyze_turbulence.py` | P4 |
| **P7** | Real-time integration + timing | `pipeline.c`, `main.c`; latency budget verification (<100 µs) | P3–P5 |
| **P8** | Validation + viz + docs | `validation`, `viz`, `run_pipeline_py.py`, `METHODS.md`; Monte-Carlo convergence | P2–P7 |

### 9.2 Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| 1 | Real dataset metadata differs from schema (units, sign conventions, factor-of-2) | High | High | Config-driven everything; verify sign & ½-reflection against supplied data before finalizing; `[R02]` §16, `[R05]` §3.1 caveats noted |
| 2 | Waffle mode blows up reconstruction/commands | Medium | High | Explicit piston+waffle nulling in `R` and `G` (mandatory); waffle unit test |
| 3 | C ↔ Python AOMX byte mismatch | Medium | High | Single documented format (§4.2); `test_matio_roundtrip.py` is a gate |
| 4 | Centroid bias (WCoG pull, pixel-locking) degrades `r₀` | Medium | Medium | Same-algorithm reference subtraction (common-mode cancel); intra-pixel bias LUT; benchmark vs ML/Gaussian-fit oracle |
| 5 | `r₀`–`L₀` degeneracy if `D ≪ L₀` | Medium | Medium | Report `L₀` only when constrained; fall back to Kolmogorov; cross-check estimators |
| 6 | `τ₀` undersampled (dt not ≪ τ₀) | Medium | Medium | Record true `dt`; sample as fast as possible; flag if `dt`≳τ₀ |
| 7 | BMP edge cases (top-down, palette, padding) | Low | Medium | Stride formula, row-flip, palette-aware reader; BMP parity test |
| 8 | No BLAS/FFTW available at runtime | Certain (by env) | Low | Self-contained C core by design (only `-lm`); AVX2/OpenMP `#ifdef`-guarded |
| 9 | Spots extended/elongated (CoG bias explodes) | Low | Medium | Correlation/matched-filter fallback retained (`[R01]` M10) |
| 10 | Real-time jitter from allocation/scheduling | Low | Low | Pre-sized buffers, zero in-loop alloc; PS9 scale has huge margin |

---

*End of ARCHITECTURE.md — the implementation contract for PS9. Stage-by-stage justification is indexed in `docs/METHODS.md`; algorithmic details live in the seven `research/` reports.*
