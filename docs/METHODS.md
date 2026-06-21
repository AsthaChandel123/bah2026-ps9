# METHODS index — pipeline stage → justifying research report

This index maps each pipeline stage and module to the research report (under
`research/`) that justifies its chosen method. The deep rationale, equations,
and references live in those reports; the integrated design is in
[`ARCHITECTURE.md`](../ARCHITECTURE.md).

| Pipeline stage | Chosen method | Module(s) | Research report |
|----------------|---------------|-----------|-----------------|
| **BMP / data ingestion** | Hand-rolled BMP reader (8/24-bit, palette-aware, stride+row-flip) | `src/bmp.c`, `aokit/bmpio.py` | [`07_data_formats_datasets_validation.md`](../research/07_data_formats_datasets_validation.md) PART A |
| **Config / metadata** | JSON schema ↔ C struct; AOMX binary matrices | `aokit/config.py`, `src/aoconfig.h`, `aokit/matio.py`, `src/matio.c` | [`07`](../research/07_data_formats_datasets_validation.md) A.4; ARCHITECTURE §4 |
| **Synthetic data generation** | FFT+subharmonics screens, geometric+Fraunhofer spots, frozen-flow τ₀ | `aokit/datagen.py`, `scripts/generate_dataset.py` | [`07`](../research/07_data_formats_datasets_validation.md) PART B |
| **Spot detection & centroiding** | **TWCoG** (thresholded + windowed weighted CoG), O(1)/sub-ap | `src/centroid.c`, `aokit/centroiding.py` | [`01_centroiding.md`](../research/01_centroiding.md) §12 |
| **Geometry & references** | Fried geometry; reference-frame registration; active-subap flux mask | `aokit/geometry.py` | [`01`](../research/01_centroiding.md) §4–5; [`02`](../research/02_zonal_reconstruction.md) §1, §5 |
| **Centroid → slope** | `s = (centroid − ref)·p_pix/f_MLA` | `src/slopes.c` | [`01`](../research/01_centroiding.md) §1.1; [`02`](../research/02_zonal_reconstruction.md) §1 |
| **Zonal reconstruction** | Precomputed regularized **Fried LS reconstructor R** (SVD/Tikhonov, piston+waffle nulled); runtime `φ=R·s` | `aokit/reconstructor.py`, `src/reconstruct.c` | [`02_zonal_reconstruction.md`](../research/02_zonal_reconstruction.md) §15 |
| **Zonal cross-check** | FFT Fourier Transform Reconstructor (Fried filter) | `aokit/reconstructor.py` (`ftr_*`) | [`02`](../research/02_zonal_reconstruction.md) §7 |
| **Modal reconstruction** | **Zernike (Noll)** analytic-gradient interaction matrix + SVD pinv; `a=M⁺·s`, `W=Z·a` | `aokit/zernike.py`, `aokit/reconstructor.py`, `src/reconstruct.c` | [`03_modal_reconstruction.md`](../research/03_modal_reconstruction.md) §6 |
| **Turbulence r₀** | **≥7 estimators** (Zernike-var, slope-var, DIMM, phase-var, structure-fn, von Kármán, seeing), median-combined | `aokit/turbulence.py`, `scripts/analyze_turbulence.py` | [`04_turbulence_characterization.md`](../research/04_turbulence_characterization.md) §1, §5 |
| **Turbulence τ₀** | **≥6 estimators** (autocorr, PSD, Greenwood, structure-fn, frozen-flow, Tyler), median-combined | `aokit/turbulence.py`, `scripts/analyze_turbulence.py` | [`04`](../research/04_turbulence_characterization.md) §2, §5 |
| **DM actuator map** | Influence-matrix deconvolution `a=H⁺·(−W/2)`, coupling baked in, reflection ½, stroke clip/units | `aokit/dm.py`, `src/dmcmd.c` | [`05_deformable_mirror_actuators.md`](../research/05_deformable_mirror_actuators.md) §11 |
| **Real-time engine** | Precompute-then-**MVM**; per-frame = centroiding + 2–3 GEMVs; self-contained C (AVX2/OpenMP, only `-lm`) | `src/linalg.c`, `src/pipeline.c`, `src/main.c` | [`06_realtime_performance.md`](../research/06_realtime_performance.md) §2, §14 |
| **Validation & metrics** | RMS WFE, Strehl (Maréchal), phase corr, r₀/τ₀ recovery, DM residual, C/Python parity | `aokit/validation.py`, `tests/` | [`07`](../research/07_data_formats_datasets_validation.md) PART C |
| **Visualization** | Spot field, phase maps, Zernike spectrum, r₀/τ₀ trends, residuals | `aokit/viz.py` | [`06`](../research/06_realtime_performance.md) §13 |

## Robustness / multi-method philosophy

The pipeline deliberately retains **alternatives at every stage** (53 distinct
methods/algorithms/metrics; see ARCHITECTURE §7) across **three orthogonal data
domains** — raw slopes, reconstructed phase, intensity. Independent estimators
must agree; one fills the gaps of another. This is the headline robustness claim
for the turbulence parameters in particular (`research/04` §5). The turbulence
bank alone supplies **≥7 r₀ + ≥6 τ₀ = ≥13 independent estimators**; counting the
centroiders, reconstructors, screen generators, spot models and metrics the
pipeline exposes **>30 distinct methods** end-to-end.

## End-to-end integration (calibration → C core ↔ Python parity)

The two tiers are wired by `scripts/build_calibration.py`, which serializes the
AOMX matrices the C core (`bin/wfs_rt`) loads. Two integration conventions are
fixed there so the C real-time path and the Python reference agree:

* **Valid-node alignment** — the zonal reconstructor `R` (and synthesis `Z`) are
  built on the **valid Fried corner nodes** (`N_phase` of them), so `φ = R·s` has
  length `N_phase`. The DM command matrix `G` is therefore built on the *same*
  valid nodes (shape `N_phase × N_phase`), so `a = G·φ` is dimensionally
  consistent with `R`. (`dm.build_dm`'s default full `(n+1)²` grid would mismatch
  `R`; the builder restricts the influence matrix to the valid nodes.)
* **Modal slope units** — the analytic Zernike modal interaction matrix `M` is in
  *normalized-coordinate gradient* units; the measured slope vector is in
  **radians of tilt**. The builder scales `M` (hence `M⁺` and every Zernike-
  variance r₀ estimator) by the physical factor `λ/(π·D) = (λ/2π)/(D/2)` so the
  modal coefficients are correctly normalized. (The working `reconstructor`
  module is left untouched; the documented scaling is applied in the integration
  layer.)

`scripts/run_pipeline_py.py` is the byte-faithful Python mirror of the C core
(same window rounding, window-center references, thresholded-CoG centroider, and
AOMX matrices), so the two produce slopes/phase/coefficients agreeing to float32
round-off (≈1e-5). `tests/test_integration.py` automates the whole loop and the
C↔Python parity check (`research/07` PART C).
