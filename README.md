# bah2026-ps9 — SH-WFS Wavefront Reconstruction, Turbulence Characterization & DM Actuator Maps

**ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 9.**

Given a time-series of Shack–Hartmann Wavefront Sensor (`.bmp`) frames, this
project (for every frame) reconstructs the **wavefront phase map** `W(x,y)` and
its **Zernike coefficients**, derives the turbulence **Fried parameter `r₀`** and
**coherence time `τ₀`**, and computes a **deformable-mirror actuator map** in
stroke-length units with **inter-actuator coupling deconvolved** — fast enough
(`< 10 ms`/frame, target ≈ µs) to correct turbulence in real time.

> **Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first** — it is the primary
> deliverable and the implementation contract (problem analysis, algorithms,
> file formats, directory layout, validation plan, roadmap). Stage→research
> mapping is in [`docs/METHODS.md`](docs/METHODS.md).

## Design in one paragraph

Two tiers (research note 06): a **self-contained C11 real-time core** (`src/` →
`bin/wfs_rt`) that links **only `-lm`** (no BLAS/FFTW) — hand-rolled BMP I/O, a
portable AVX2/OpenMP-optional GEMV, TWCoG centroiding, and 2–3 precomputed
matrix–vector multiplies per frame; and a **Python offline toolkit** (`aokit/`)
that builds the calibration matrices, generates synthetic datasets with known
`r₀`/`τ₀`, runs the multi-method turbulence estimation, and validates everything.
The two communicate through a small self-describing binary matrix format
(**AOMX**) that both sides implement byte-for-byte.

```
.bmp ─▶ centroid (TWCoG) ─▶ slopes ─▶ { φ = R·s  (zonal),  a = M⁺·s (modal) } ─▶ a_dm = H⁺·(−W/2)
        precomputed offline:  R (Fried, waffle-nulled),  M⁺/Z (Zernike),  G = H⁺·(−½) (coupling deconvolved)
```

## Quickstart

### 1. Build the C real-time core (only needs a C compiler + libm)
```bash
make            # -> bin/wfs_rt  (-O3 -march=native -fopenmp, links only -lm)
make test       # built-in self-test: AOMX + BMP roundtrips, GEMV sanity
```

### 2. Python offline toolkit
```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib (network OK)
python3 -c "import aokit"               # import sanity
python3 -m json.tool config/example_config.json   # validate config JSON
pytest -q                              # unit tests
```

### 3. End-to-end (synthetic) workflow
```bash
# Generate a synthetic dataset with known r0/tau0
python3 scripts/generate_dataset.py --config config/example_config.json \
        --r0 0.15 --tau0 0.0045 --frames 2000 --out data/synthetic/run01

# Build calibration matrices (AOMX) for the C core
python3 scripts/build_calibration.py --config config/example_config.json --out calib/

# Run the fast C pipeline over the frames
bin/wfs_rt --config config/example_config.json --calib calib/ \
           --frames data/synthetic/run01/frame_00001.bmp data/synthetic/run01/frame_00002.bmp \
           --out out/

# (or) pure-Python pipeline for plots/validation
python3 scripts/run_pipeline_py.py --config config/example_config.json \
        --frames "data/synthetic/run01/frame_*.bmp" --out out_py/

# Multi-method turbulence report (>=7 r0, >=6 tau0 estimators)
python3 scripts/analyze_turbulence.py --in out/slopes.csv \
        --config config/example_config.json --out out/turbulence_summary.json
```

## Repository layout

```
ARCHITECTURE.md     # master architecture (primary deliverable)
docs/METHODS.md     # stage -> research-report map
config/             # example_config.json (schema in ARCHITECTURE §4.1)
src/                # self-contained C11 real-time core (links only -lm)
aokit/              # Python offline toolkit (numpy/scipy/matplotlib)
scripts/            # build_calibration / generate_dataset / run_pipeline_py / analyze_turbulence
tests/              # pytest (matio roundtrip is live; algorithmic tests scaffolded)
research/           # seven deep-research reports backing every design choice
```

## Status

This is the **architecture + scaffold** milestone. Implemented and verified:
- `ARCHITECTURE.md` (deep, synthesizing all 7 research reports).
- C core **builds clean** (`-Wall -Wextra`, links only `-lm`) and **self-test passes**.
- **AOMX format** implemented on both sides (`src/matio.c` ↔ `aokit/matio.py`) and
  **C↔Python byte parity verified**; BMP and GEMV roundtrips pass.
- `aokit` **imports cleanly**; `config`, `matio`, `bmpio` are functional.
- Algorithmic stages are **documented stubs** (`TODO(impl)` in C, `NotImplementedError`
  in Python) with precise signatures + the unit-test names that will exercise them.

### Results (placeholder — filled in after implementation)

| Metric | Target | Achieved |
|--------|--------|----------|
| Per-frame compute | `< 100 µs` (≥100× under 10 ms) | _TBD_ |
| RMS WFE (synthetic) | `< λ/14` | _TBD_ |
| `r₀` recovery error | few % | _TBD_ |
| `τ₀` recovery error | ~10–20 % | _TBD_ |
| Phase correlation `ρ` | `> 0.95` | _TBD_ |
| DM residual reduction | large vs uncorrected | _TBD_ |

## Environment

Ubuntu 24.04, x86_64, 4 cores, no GPU; gcc 13 / clang 18, OpenMP, AVX2/AVX-512.
Python 3.11. The C core has **no external library dependency** (only libm); the
Python toolkit installs numpy/scipy/matplotlib from wheels.

## License / attribution

Prepared for ISRO BAH 2026 PS9. Each algorithmic choice is justified in the
`research/` reports (with primary-source URLs) and integrated in
`ARCHITECTURE.md`.
