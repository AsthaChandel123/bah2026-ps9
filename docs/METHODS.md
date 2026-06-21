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
for the turbulence parameters in particular (`research/04` §5).
