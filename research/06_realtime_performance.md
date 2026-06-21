# Real-Time Computing Architecture & Performance for SH-WFS Wavefront Reconstruction
### ISRO BAH 2026 — Problem Statement 9 — Research Note 06

**Domain:** real-time computing architecture, performance optimization, language/library selection for the adaptive-optics (AO) control loop.

**The hard constraint.** Atmospheric turbulence has a coherence time `τ0 ~ a few ms` (the PS quotes ~10 ms). The correction must be *measured and applied faster than the atmosphere changes*, so the **end-to-end loop period must sit comfortably below `τ0`** — in practice we target a control loop running at **≥ 1 kHz (period ≤ 1 ms)**, leaving a 10× margin against a 10 ms coherence time. The PS explicitly advises a **low-level language like C** and notes that **open-source math libraries may be used**. This note answers: *what is the fastest practical architecture, and which language/library stack delivers it?*

**Headline conclusion (TL;DR).** For a lab-scale SH-WFS (≈10×10 to ≈30×30 sub-apertures) the per-frame compute is tiny. The canonical real-time workhorse — **precompute the reconstruction/command matrix offline, then do one dense matrix-vector multiply (MVM) per frame** — costs on the order of **10⁴–10⁶ floating-point operations per frame**, which a single CPU core with a tuned **BLAS `gemv`** executes in **single-digit-to-tens of microseconds**. That is **100×–1000× under a 10 ms budget**. The recommendation is a **C real-time core** (centroiding + MVM via BLAS, optional FFTW reconstructor) with **Python (NumPy/SciPy)** for the *offline* calibration, matrix building, turbulence characterization, and visualization. GPUs are unnecessary at PS9 scale and can even *lose* to an AVX CPU because of PCIe transfer latency.

---

## 1. The AO loop timing model — what "fast enough" means

The AO control-loop delay is **not** just the compute time. The total loop latency is the sum of four stages, and a minimum of **~1 frame of pure latency is unavoidable** regardless of how fast the math is:

| Stage | What happens | Typical contribution |
|---|---|---|
| **WFS integration** | Camera integrates the spot field for one frame | ≈ 0.5 frame (centroid of the integration window) |
| **Detector readout** | Pixels stream off the sensor | tens–hundreds of µs (sub-window ≈ 70 µs reported) |
| **Compute (RTC)** | Calibrate pixels → centroid → slopes → **MVM** → DM commands | the part we optimize; ~100–150 µs in tuned C pipelines, 600–800 µs if done in naive Python |
| **DM settle (sample & hold)** | Mirror moves to commanded shape | ≈ 0.5 frame; physical settle < 20 µs for fast DMs |

Sources for these stage numbers: the AO servolag/error-transfer-function literature notes the loop delay is *"at least twice the operating frame-time"* — **0.5 frame from WFS integration + 0.5 frame from DM sample-and-hold** — plus readout and compute (A&A reinforcement-learning AO control paper [13]; Durham DARC capability paper [10]). The **compute stage is the only one we control in software**, and it is also the *smallest* contributor once a fast language/library is used.

**Design rule:** keep **compute ≪ frame period**, and keep **jitter** (frame-to-frame variation in compute time) small and bounded, because the closed-loop error-transfer function is sensitive to *delay variance*, not just mean delay. This drives three software requirements:
1. **Deterministic, allocation-free hot loop** (no `malloc`, no garbage collection, no page faults inside the loop).
2. **CPU core pinning + real-time scheduling** (PREEMPT_RT Linux keeps worst-case latency < ~150 µs; see §10).
3. **Pre-sized, contiguous, aligned buffers** reused every frame.

---

## 2. Method 1 — Matrix-Vector-Multiply (MVM) reconstructor: the canonical workhorse

**Idea.** Wavefront reconstruction is *linear* in the measured slopes. Build, **once, offline**, a reconstruction matrix `R` (and fold in the DM influence/coupling to get the command matrix `C`). Then **every frame** the real-time step is a single dense MVM:

```
commands = C · slopes          (a = C s,  a = command vector, s = slope vector)
```

This is the method used by *"most AO RTCs"* and is described across the literature as the preferred way to compute the DM shape from WFS measurements: *"Matrix-vector multiplication (MVM) is the preferred method for computing the deformable mirror shape from wavefront sensor pixel inputs"*; *"most control laws... require one or several matrix-vector multiplies, at frequencies around 1 kHz"* (Xeon-Phi AO latency work [1]; HEART/ELT RTC pipeline description [3]).

**Why it is effectively "O(1) per actuator".** For an AO system with `N` degrees of freedom, the reconstruction/command matrix is `~N × N`, so one MVM is `2N²` FLOPs → **O(N²) total, but O(N) per actuator** (each actuator's command is one dot-product of length `N`). Because **all the expensive math — pseudo-inverse / SVD / least-squares solve — is done offline**, the per-frame cost is *just multiply-accumulate*, with **no solves, no iteration, no branching**. This is exactly the "O(1) per actuator, precompute everything" technique the user asked for. The cost of *building* `R`/`C` (SVD of the interaction matrix) is paid once at calibration time and is irrelevant to the real-time budget.

**Latency evidence.** The PALM-3000 RTC sustains the *"vector-matrix-multiply wavefront reconstruction method at frame rates up to 2 kHz with latency under 250 µs"* for a **64×64 sub-aperture SH-WFS driving 3368 actuators** (Xeon-Phi AO paper [1]; FLOPs survey [16]). PS9's lab system (≤30×30) is **~4×–40× smaller per dimension**, i.e. roughly **15×–400× fewer FLOPs**, so microsecond-class compute is expected.

**Pipelining bonus.** Slopes can be fed to the MVM as soon as each sub-aperture centroid is ready, overlapping centroiding with reconstruction (*"calculation of the centroid begins while pixel values are read... slope and partial wavefront reconstruction follows immediately"* — stream-based SH-WFS [4][11]).

---

## 3. FLOP / byte budget for the MVM at PS9 scale

Let `n_sub` = sub-apertures across (e.g. 10, 20, 30). A SH-WFS yields **2 slopes per sub-aperture** (x,y). In **Fried geometry** the DM actuator count ≈ `(n_sub+1)²`. Take roughly **#slopes ≈ #commands ≈ M** and a dense `M × M` matrix.

For one MVM: **FLOPs ≈ 2·M²**; **bytes touched ≈ M²·sizeof(elt)** (the matrix dominates the memory traffic — the vectors are negligible). GEMV is **memory-bandwidth bound**, so bytes, not FLOPs, set the wall-clock time.

| System | `n_sub` | ~slopes `M` | Matrix `M×M` | FLOPs/frame (`2M²`) | Matrix bytes (fp32) | At 1 kHz: throughput / bandwidth |
|---|---|---|---|---|---|---|
| Small lab | 10×10 | ~150 | 150×150 | ~4.5×10⁴ | ~90 KB | 0.045 MFLOP/s · 0.09 GB/s |
| Mid lab | 20×20 | ~600 | 600×600 | ~7.2×10⁵ | ~1.4 MB | 0.72 MFLOP/s · 1.4 GB/s |
| Large lab | 30×30 | ~1400 | 1400×1400 | ~3.9×10⁶ | ~7.8 MB | 3.9 MFLOP/s · 7.8 GB/s |
| (ref) PALM-3000 | 64×64 | ~7000 | 12k×5k class | ~10⁸ | tens of MB | sustains 2 kHz <250 µs [1][16] |
| (ref) ELT SCAO | — | — | 12k×5k | — | — | **136 GFLOP/s @ 1 kHz** [16] |
| (ref) TMT NFIRAOS | — | — | 35k×7k | — | — | **1.5 TFLOP/s, ~800 GB/s, <1 ms** [16] |

**Reading the table.** A commodity CPU core delivers tens of GFLOP/s and ~tens of GB/s of usable bandwidth; a multi-core node delivers >100 GB/s. The PS9 matrices (≤ ~8 MB fp32) **fit in L2/L3 cache** and require **< 0.01 GB/s** of bandwidth at 1 kHz — three to four orders of magnitude under what one core provides. **Wall-clock per MVM at PS9 scale is single-digit-to-tens of microseconds.** Even the worst lab case is ~1000× under a 10 ms budget. The contrast with the ELT/TMT rows (which genuinely need GPUs/many-core and hundreds of GB/s) shows why **PS9 does not need exotic hardware**.

**Single vs double precision.** GEMV is bandwidth-bound, so **fp32 roughly doubles throughput vs fp64** (half the bytes; on many cores 2–4× the SIMD lanes). Wavefront slopes from an 8-bit/12-bit camera have far less than fp32 precision, so **fp32 (`cblas_sgemv`) is the right default** for the real-time path — faster and accurate enough. Keep fp64 for the *offline* SVD/calibration where conditioning matters. (Throughput/bandwidth argument: FBLAS GEMV memory-bound result [8]; float-vs-double bandwidth [9].)

---

## 4. Method 2 — Sparse / fast reconstructors (CuReD, FEWHA, sparse Cholesky)

When `N` gets large (ELT class), the dense `O(N²)` MVM stops scaling. Two families reduce it; both are **overkill for PS9 but worth citing as the "scales beyond MVM" option**:

- **Sparse reconstruction matrices.** Exploit that an actuator is mainly driven by *nearby* sub-aperture slopes, zeroing distant entries → a **sparse `R`**, solved by **sparse Cholesky / sparse MVM** (NASA/Caltech sparse-matrix wavefront reconstruction [17][18]).
- **CuReD — Cumulative Reconstructor with domain decomposition.** Treats the SH operator as a discrete *gradient* and reconstructs by **integration** instead of matrix inversion. **Computational complexity is linear in the number of unknowns (O(N))**, and it is *"pipelinable and parallelizable, which makes the effective computation even faster"* than standard MVM (Linz CuReD page [19]; JOSA A 2011 [20]). P-CuReD extends it to pyramid WFS / XAO.
- **FEWHA / FrIM / Fractal Iterative Method.** Iterative Fourier-/wavelet-domain solvers, also sub-`O(N²)`, used at ELT scale.

**PS9 takeaway:** at ≤30×30 the dense MVM is already microsecond-class, so **sparse/CuReD give no practical benefit here** — mention them as the scaling story and as an *alternative zonal reconstructor* to validate against, not as the primary engine.

---

## 5. Method 3 — Fourier-Transform (FFT) reconstructor — O(N log N)

Poyneer, Gavel & Brase (2002) showed wavefront reconstruction can be done with the **FFT + spatial filtering**, which is *"computationally tractable and sufficiently accurate for use in large Shack-Hartmann-based AO systems (up to at least 10,000 actuators)"* and is *"significantly faster than traditional vector-matrix-multiply reconstructors"* (JOSA A 19(10):2100 [21]; advanced-techniques SPIE [23]). It solved the **circular-aperture boundary problem** for square-grid FTs and has **lower noise propagation, no waffle, and is the fastest to compute** at large scale. Complexity is **O(N log N)** vs `O(N²)` for MVM, and FLOPs drop from `n²` to `n·log(n)` (FLOPs survey [16]).

**Implementation:** use **FFTW** (`fftw_plan` once with `FFTW_MEASURE` / wisdom, then `fftw_execute` every frame). FFTW is **O(N log N) even for prime sizes**, supports **fp32/fp64**, has **SSE/AVX/NEON SIMD**, and runs on *"any platform with a C compiler"* (fftw.org [24]).

**PS9 takeaway:** the FFT reconstructor's `O(N log N)` advantage only *beats* MVM at thousands of actuators. At ≤30×30 the **MVM is simpler and already fast**; the FT reconstructor is best offered as an **optional second reconstructor** — it doubles as a turbulence/PSD analysis tool (the wavefront power spectrum is a natural by-product) and a cross-check. **License caveat:** FFTW is **GPL** (commercial license available); for a hackathon deliverable this is fine, but note it if the final code must be permissively licensed (alternatives: PocketFFT, KISS FFT, or pure-NumPy offline).

---

## 6. Method 4 — BLAS backends for the MVM (the practical core)

The MVM should call a tuned **BLAS Level-2 `gemv`** (`cblas_sgemv` / `cblas_dgemv`: computes `y ← α·A·x + β·y`). Do **not** hand-write the loop — BLAS already does the cache-blocking, SIMD, and prefetching. Candidate backends:

| Backend | License | Notes |
|---|---|---|
| **OpenBLAS** | **BSD** (GotoBLAS2 lineage) | Best default for PS9: permissive, fast, easy to link (`-lopenblas`), pthread or OpenMP threading, hand-tuned `sgemv/dgemv` kernels per microarch, runtime thread control (`openblas_set_num_threads`) [25][26]. |
| **Intel MKL** | proprietary (free to use) | Often fastest on Intel CPUs; can be slower than OpenBLAS for *small* vectors/matrices and has been seen "6× slower on small DGEMV" in one report — i.e. **library choice is size-dependent; benchmark on your data** [6][7]. |
| **BLIS** | BSD-3 | *"Remarkably competitive vs MKL"* on Haswell-class cores; clean, portable, good for `gemv` [8]. |
| **ATLAS** | BSD | Auto-tuned at build; older, generally behind OpenBLAS/MKL today. |
| **Apple Accelerate** | system | If targeting Apple Silicon (NEON). |

**Key subtlety for small problems:** at PS9 sizes the matrix is small and **threading overhead can exceed the work** — OpenBLAS explicitly re-tuned its *"SGEMV/DGEMV load thresholds to avoid activating multithreading for too small workloads"* [25]. So **run the PS9 MVM single-threaded** (set `OPENBLAS_NUM_THREADS=1`), pin to one isolated core, and let SIMD do the work. Threading is for the ELT-scale rows, not for us.

---

## 7. Method 5 — SIMD vectorization (SSE/AVX/AVX-512/NEON)

GEMV is a stream of multiply-accumulates → **ideal for SIMD**. One **AVX-512** register holds **16 fp32 or 8 fp64** values; AVX2 holds 8 fp32 (Intel AVX-512 guide [5]; AVX/NEON overview [12]). Practical guidance:

- **Prefer auto-vectorization + a good BLAS first.** OpenBLAS/MKL/BLIS kernels are already SIMD-tuned; you rarely beat them by hand for `gemv`.
- **Data alignment matters.** Align matrix/vector buffers to **64 bytes** for AVX-512 (32 for AVX2) to get aligned loads and avoid penalties/crashes (AVX-512 alignment requirement, 64-byte [5][12]). Use `posix_memalign` / `aligned_alloc`.
- **fp32 doubles SIMD throughput** vs fp64 (twice the lanes) — another reason to use `sgemv` in the loop.
- **AVX2 vs AVX-512 caveat:** AVX-512 can trigger frequency down-clocking and is sometimes disabled by admins; **AVX2 is the safe, ubiquitous baseline** and is enough at PS9 scale. Notably, in AO RTC studies *"CPU implementations using AVX2 achieved better performance than highly-optimized GPU libraries like cuBLAS"* once data-transfer was counted [2][14] — SIMD on the CPU is genuinely competitive.
- **NEON** is the ARM equivalent (Apple Silicon, Jetson, Raspberry Pi) — FFTW and OpenBLAS both support it.

---

## 8. Method 6 — GPU acceleration (CUDA / cuBLAS / cuFFT) — and when it does *not* pay

GPUs (`cublasSgemv`, `cuFFT`) dominate **ELT-scale** RTCs because of raw throughput: *"GPU-based implementations exhibit lower latency due to superior floating-point capability and supporting libraries"* and are the chosen technology for E-ELT/TMT RTC prototypes (GPU-for-AO surveys [27][30]; COMPASS uses CUBLAS/CUFFT/CURAND [31]).

**But for PS9 the GPU is the wrong tool**, for concrete reasons drawn directly from the AO-RTC literature:
- **PCIe transfer latency dominates small problems.** *"The latency of the CPU-GPU link constitutes a tight bottleneck... which allowed CPU implementations using AVX2 to achieve better performance than highly-optimized GPU libraries like cuBLAS"* [2][14]. Copying a tiny slope vector to the GPU and the command vector back can cost **more than the entire MVM** would on the CPU.
- **Kernel-launch + sync overhead** (~10–20 µs) is comparable to the *whole* PS9 compute budget.
- **Jitter:** added DMA/scheduling variance hurts the error-transfer function.

GPUs pay off only when **compute throughput dominates transfer** — i.e. matrices of order 10k×5k+ at kHz rates (the NFIRAOS/SCAO rows in §3). **PS9 (≤ ~8 MB matrix) is firmly in CPU territory.** *Recommendation: skip the GPU for the real-time path; optionally use it offline for batch processing the whole frame time-series.*

---

## 9. Method 7 — Lookup tables & precomputation (O(1) centroiding and modal eval)

The user wants "O(1)" techniques; precomputation is how you get there beyond the MVM:

- **Centroiding regions precomputed.** The sub-aperture pixel windows, reference spot positions, and per-pixel coordinate weights are **fixed by the optics** → compute them **once** and store as flat arrays. Then per frame the **thresholded center-of-gravity** is just `Σ(w·I)/ΣI` over a tiny known window — a handful of MACs per sub-aperture (centroiding comparison MNRAS [33]; AO4ELT centroiding study [34]). **Thresholded CoG (TCoG)** is the standard fast, robust choice; weighted CoG (matched-filter) is better at low SNR. The *x/y pixel-index vectors are the lookup table* — multiply-accumulate against the live intensities.
- **Slope→reference precomputation.** Reference centroids (flat-wavefront calibration) are subtracted with a precomputed offset vector.
- **Modal (Zernike) reconstruction via a precomputed basis matrix.** If using a *modal* reconstructor `W = Z·A` with coefficients `A` fit from slopes, **precompute the Zernike derivative basis and its pseudo-inverse offline**; the per-frame step collapses to *another MVM*. Preloading precomputed Zernike matrices gives a **7×–10× speedup** vs recomputing them (ZernikeViewer [37]; lateral-shearing modal reconstruction [35]). This is the same "do the solve offline, multiply online" pattern as §2.
- **Pixel calibration LUTs.** Dark/flat/bad-pixel correction per pixel is a precomputed table applied with one fused multiply-add per pixel.

**Net effect:** the entire frame pipeline (calibrate → centroid → slopes → reconstruct → commands) becomes **a sequence of table-driven multiply-accumulates with no per-frame solves**, i.e. genuinely O(1)-per-element work.

---

## 10. Methods 8–10 — Memory, threading, and real-time scheduling

**Method 8 — Memory layout & cache locality (often the single biggest lever).**
- Store the matrix **row-major and contiguous** and multiply so the inner loop strides by 1: *"row-major GEMV has excellent locality (stride one); column-major has very bad cache locality"* — the wrong layout can be **10× slower** (memory-layout/GEMV locality [40][41]).
- **Allocate all buffers once, outside the loop.** No `malloc`/`free`/resize in the hot path — dynamic allocation and reallocation kill determinism; optimized BLAS *"want predictable strides"* and contiguous storage [40].
- **Keep the matrix cache-resident.** PS9 matrices (≤ ~8 MB) fit in L2/L3; touch them in a cache-friendly order and they stay hot frame-to-frame.
- **Fixed-point / integer** centroiding is an option on FPGA/embedded, but on a CPU fp32 is simpler and fast enough.

**Method 9 — Multithreading & pipeline parallelism (OpenMP / pthreads).**
- For PS9 the MVM is **best single-threaded** (work < threading overhead, §6). Use threads instead for **pipeline/stage parallelism**: e.g. one thread does centroiding of frame *k* while another runs the MVM of frame *k−1* (stream-based overlap [4][11]).
- **OpenBLAS** offers both pthread and OpenMP builds; if you ever do thread, **pin threads to isolated cores** and set thread count explicitly to avoid oversubscription. DASP and Durham systems use **pthreads/MPI** for large-scale parallelism [10][/DASP].
- Avoid implicit thread pools and busy-wait contention that add jitter.

**Method 10 — Real-time OS, scheduling, jitter, latency budget.**
- Run the loop under **PREEMPT_RT Linux**, which keeps **worst-case latency < ~150 µs** vs unbounded stock-kernel spikes (RT-PREEMPT measurements [42][43]). The Julia-AO demonstration ran on **PREEMPT_RT (Rocky Linux 9)** to reach sub-ms latency with bounded jitter [15][2].
- **Pin the RT thread to an isolated CPU core** (`isolcpus`, `taskset`/`pthread_setaffinity`), give it `SCHED_FIFO` priority, **lock memory** (`mlockall` — no page faults), and **disable frequency scaling**.
- **Budget the loop:** `t_integrate + t_readout + t_compute + t_DM_settle < τ0`. At 1 kHz the frame is 1000 µs; readout ~70 µs, compute ~10–150 µs, DM settle <20 µs → **hundreds of µs of headroom**, and the whole thing is **~10× under a 10 ms `τ0`**. Minimize and *bound* `t_compute` jitter; the closed-loop stability margin depends on delay *variance*.

---

## 11. Methods comparison table (reconstruction engines)

| Method | Complexity (per frame) | PS9 (≤30×30) wall-clock | Best for | Library | PS9 verdict |
|---|---|---|---|---|---|
| **Dense MVM** (zonal/modal, precomputed `C`) | O(N²) FLOPs, **O(1)/actuator** | **~µs–tens of µs** | the canonical real-time engine; small/medium N | BLAS `sgemv` (OpenBLAS) | **PRIMARY** ✅ |
| **Modal MVM** (Zernike coeffs, precomputed pinv) | O(N²) (one MVM) | ~µs | turbulence chars + reconstruction in one step | BLAS `sgemv` | **PRIMARY** ✅ (gives `r0`, modes) |
| **Sparse / sparse-Cholesky** | < O(N²) | ~µs (no gain at this N) | very large N | sparse BLAS / CHOLMOD | optional / validation |
| **CuReD** (integration) | **O(N)**, pipelinable | ~µs | ELT-scale zonal | reference code [19] | optional cross-check |
| **FFT reconstructor (Poyneer)** | **O(N log N)** | ~µs–tens of µs | ≥10³ actuators; also gives wavefront PSD | **FFTW** | **OPTIONAL** ⚠️ (PSD/analysis + cross-check; GPL) |
| **Iterative (CG/FrIM/FEWHA)** | O(N) per iter × iters | slower (iterations) | XAO/ELT, matrix-free | custom + BLAS | not needed |

---

## 12. Survey of existing open-source AO RTC & simulation frameworks

These are useful as **references, validators, and source of calibrated reconstructors/centroiders**. We are not required to use them, but they let us (a) cross-check our wavefront/`r0`, (b) borrow algorithm implementations, and (c) generate synthetic SH-WFS frames.

| Framework | Type | Language | License | Use for PS9 |
|---|---|---|---|---|
| **CACAO** (Compute & Control for AO) | RTC | **C** (+ Python tools) | **GPLv3** | Reference real-time architecture: shared-memory `ImageStreamIO` data streams, CPU/GPU MVM, kHz loops. In use on Subaru AO188 / SCExAO / MagAO-X. *Architectural blueprint.* [G1][G2] |
| **DARC** (Durham AO RTC) | RTC | **C** (+ Python control) | **GPL** | CPU-based RTC with FPGA/GPU acceleration; CANARY on-sky. Reference for centroiding + MVM pipeline & latency. [10][D] |
| **HEART** (Herzberg Ext. AO RT Toolkit) | RTC framework | C/C++ | research | CPU-based, scalable RTC; the modern "pixel-calib → SH centroid → MVM" pipeline reference for TMT NFIRAOS. [3] |
| **Julia-AO RTC** (Thompson et al. 2024) | RTC | **Julia** | open-source | Proof that a high-level language hits **sub-ms latency ≈ C** with bounded jitter under PREEMPT_RT; pixel streaming, multi-camera. Useful design ideas. [15][2] |
| **COMPASS** | end-to-end sim (+RT core) | **C++/CUDA** + Python (SHESHA) | **GPL** | GPU end-to-end simulator; generate synthetic SH-WFS frames & ground-truth wavefronts to validate our reconstruction/`r0`. [31] |
| **DASP** (Durham AO Sim Platform) | end-to-end sim | **Python + C** (pthreads/MPI) | open-source (GitHub) | Monte-Carlo AO simulation; reference reconstructors & turbulence generation. [DASP] |
| **Soapy** | end-to-end sim | **pure Python** | open-source (AOtools org) | Rapid concept simulation; easy to script SH-WFS scenarios for validation. [S] |
| **aotools** | analysis toolkit | **Python** | **LGPL-3.0** | Zernikes, `r0`/turbulence stats, centroiding, circular-pupil helpers — directly reusable in our **offline** Python analysis. [aotools] |
| **HCIPy** | sim toolkit | **Python** | open-source (BSD-style) | Wavefront/coronagraph propagation; modal bases; cross-check phase maps. [HCIPy] |
| **OOMAO / OOPAO** | end-to-end sim | **MATLAB** / **Python** | **MIT** / open | Object-oriented AO modeling; OOPAO is the Python port — easy reference for influence functions / interaction matrices. [OOMAO] |
| **YAO** | end-to-end sim | **Yorick + C** | **GPLv3** | Fast classic AO simulator; "many core routines in C." Reference for SH-WFS modeling. [YAO] |
| **CEO** | sim (GMT) | CUDA/Python | open | GPU AO sim for GMT; not needed at PS9 scale. |

**How we use them:** primarily **`aotools` + `HCIPy` (Python) for offline analysis/validation**, **COMPASS or Soapy/DASP to synthesize labeled SH-WFS frames** for testing our reconstructor against known truth, and **CACAO/DARC/HEART as the architectural template** for the C real-time pipeline (pixel-calibration → centroid → slopes → MVM → DM command, over reused shared buffers).

---

## 13. Language recommendation — C real-time core + Python offline

**Recommendation: a two-tier stack.**

**Tier 1 — C real-time core (the hot path).** Centroiding + slope computation + **MVM** (and optional FFTW reconstructor). Justification:
1. **Determinism.** C has **no garbage collector, no hidden allocation**, predictable memory and timing — essential for bounded jitter inside a kHz loop. (Even the Julia-AO paper's whole contribution was *engineering away* runtime/GC jitter to match C [15][2]; the safe default for a hackathon is to start in C.)
2. **The PS explicitly advises C** for speed against the ~10 ms coherence time.
3. **Direct access to the fastest libraries:** `cblas_sgemv` (**OpenBLAS**, BSD) for the MVM and **FFTW** for the FT reconstructor — both are C libraries with C ABIs, SIMD kernels, and zero marshalling overhead.
4. **SIMD + alignment + cache control** are natural in C (`aligned_alloc`, restrict, contiguous arrays), and `pthread`/affinity/`mlockall`/`SCHED_FIFO` give full RT control.
5. **Portability:** the same C compiles on x86 (AVX2/AVX-512) and ARM/Jetson (NEON), matching whatever lab hardware is provided.

**Tier 2 — Python (NumPy / SciPy / aotools) for everything offline & non-real-time.** Justification:
1. **Calibration is a one-time cost**, so developer speed beats run speed: build the interaction matrix, do the **SVD/pseudo-inverse** (`numpy.linalg`/`scipy.linalg`), fold in **DM influence functions and inter-actuator coupling** to produce the command matrix `C`, and **serialize `C` to a flat binary** the C core memory-maps at startup.
2. **Turbulence characterization** (`r0`, `τ0`, Zernike spectra, structure functions) is naturally done in Python/`aotools` over the *recorded* frame series — not in the real-time loop.
3. **Visualization & analysis** (phase maps, PSDs, actuator maps) — Matplotlib/NumPy.
4. **Rapid prototyping & validation** against COMPASS/Soapy/HCIPy ground truth before freezing the C kernels.

**Interface:** Python writes the precomputed matrices/LUTs (command matrix `C`, reference centroids, sub-aperture window tables, optional Zernike pinv) as a binary blob; the C core `mmap`s it once and never allocates again. This cleanly separates **"slow, smart, offline" (Python)** from **"fast, dumb, online" (C)** — the standard and proven AO-RTC division of labor (CACAO/DARC both pair a C real-time core with Python control/config [G1][10]).

---

## 14. Recommended tech stack & performance plan for PS9

**Stack**
- **Real-time core:** **C (C11)**, compiled `-O3 -march=native` (or `-mavx2`), 64-byte-aligned contiguous buffers, no in-loop allocation.
- **MVM:** **OpenBLAS `cblas_sgemv`** (BSD), **single-threaded**, `OPENBLAS_NUM_THREADS=1`, matrix in row-major cache-resident fp32.
- **Centroiding:** **thresholded center-of-gravity** over **precomputed sub-aperture windows + reference offsets** (weighted-CoG fallback for low SNR); compute begins as pixels stream in.
- **Optional reconstructors:** **FFTW** FT reconstructor (O(N log N), also yields the wavefront PSD for turbulence stats) and/or a **modal Zernike MVM** (precomputed pseudo-inverse) — both as cross-checks/extensions, not the critical path.
- **Offline (Python):** **NumPy/SciPy** for SVD/pinv → command matrix `C` (with DM influence + inter-actuator coupling), **aotools/HCIPy** for `r0`/`τ0`/Zernike analysis and validation, **COMPASS/Soapy** to generate labeled test frames. Serialize `C` + LUTs to binary for the C core to `mmap`.
- **OS/RT:** **PREEMPT_RT Linux**, `isolcpus` + core pinning + `SCHED_FIFO` + `mlockall`, frequency scaling off.

**Performance plan / acceptance targets**
1. **Compute budget:** per-frame pipeline (calibrate→centroid→slopes→MVM→commands) **< 100 µs** at PS9 scale (expected ~µs–tens of µs) → **>100× margin under 10 ms `τ0`**.
2. **Determinism:** worst-case loop jitter **< ~150 µs** under PREEMPT_RT; **zero in-loop allocations** (verify with `valgrind --tool=massif` / counting `malloc`).
3. **Throughput sanity:** MVM bandwidth need (≤ ~0.01 GB/s @ 1 kHz) is **10³–10⁴× below** one core's bandwidth → headroom for higher frame rates if the camera allows.
4. **Precision:** fp32 in the loop (camera SNR-limited), fp64 offline for SVD conditioning.
5. **Validation:** reconstruct synthetic COMPASS/Soapy frames, confirm phase maps and recovered `r0` match injected turbulence; cross-check MVM vs FFT reconstructor agreement.
6. **Scalability note:** if the lab system were ever ELT-scale (10³–10⁴ actuators), switch to **CuReD/FFT reconstructor** and **GPU (cuBLAS)** — but **PS9 does not require this**.

---

## 15. Sources (URLs)

**Real-time control / MVM / latency**
1. Reducing AO latency using Xeon Phi many-core processors — https://www.academia.edu/18006297/Reducing_adaptive_optics_latency_using_Xeon_Phi_many_core_processors
2. The use of CPU, GPU and FPGA in real-time control of AO (eScholarship) — https://escholarship.org/uc/item/2vj6w3gm
3. HEART: real-time controller toolkit (Herzberg) — https://arxiv.org/pdf/2412.18006 ; ELT RTC MVM pipeline — http://research.iac.es/congreso/AO4ELT5/media/proceedings/proceeding-176.pdf
4. FPGA stream-based center-of-gravity SH-WFS centroiding (MDPI Electronics 12(7):1714) — https://www.mdpi.com/2079-9292/12/7/1714
6. OpenBLAS 6× slower than MKL on DGEMV (issue) — https://github.com/xianyi/OpenBLAS/issues/532
7. BLAS sgemv Skylake vs Broadwell (Intel community) — https://community.intel.com/t5/Software-Tuning-Performance/BLAS-sgemv-Skylake-only-half-as-fast-as-Broadwell-Cache/td-p/1178362
8. FBLAS / FT-BLAS GEMV memory-bound & DGEMV speedups — https://arxiv.org/pdf/1907.07929 ; https://arxiv.org/pdf/2104.00897 ; BLIS performance — https://github.com/flame/blis/blob/master/docs/Performance.md
9. Float vs Double (bandwidth/throughput) — https://meshlib.io/documentation/FloatVSDouble.html
10. Durham AO RTC (DARC) capability & ELT suitability — https://arxiv.org/pdf/1010.3209 ; https://arxiv.org/pdf/1205.4532
11. Centroid estimation based on stream processing — https://www.researchgate.net/publication/318884381
12. AVX/NEON intrinsics overview — https://www.emergentmind.com/topics/avx-neon-intrinsic-functions
13. Model-based RL for AO control (loop-delay model) — https://www.aanda.org/articles/aa/full_html/2022/08/aa43311-22/aa43311-22.html
14. GPUs for AO: simulations & real-time control (SPIE 8447) — https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8447/1/GPUs-for-adaptive-optics-simulations-and-real-time-control/10.1117/12.925723.short
15. Real-time AO control with a high-level language (Julia; arXiv 2407.07207) — https://arxiv.org/abs/2407.07207 ; case study — https://discourse.julialang.org/t/case-study-real-time-hardware-control-for-adaptive-optics-with-julia/117155
16. Fast wavefront reconstruction algorithms (FLOPs scaling; PALM-3000, NFIRAOS, SCAO figures) — https://www.researchgate.net/publication/310727954 ; SPARC FPGA — https://arxiv.org/pdf/1807.00715 ; Xeon Phi ELT-scale — https://academic.oup.com/mnras/article/478/3/3149/4999636

**Reconstructors**
17. Sparse-matrix wavefront reconstruction: simulations & experiments (NASA NTRS) — https://ntrs.nasa.gov/citations/20210001318
18. Sparse-matrix wavefront reconstruction (Caltech) — https://authors.library.caltech.edu/records/zbvxz-qab22
19. CuReD — Cumulative Reconstructor with domain decomposition (Linz) — http://eso-ao.indmath.uni-linz.ac.at/index.php/algorithms/cured.html ; P-CuReD — http://eso-ao.indmath.uni-linz.ac.at/index.php/algorithms/pcured.html
20. Cumulative Reconstructor (JOSA A 28(10):2132) — https://opg.optica.org/josaa/abstract.cfm?uri=josaa-28-10-2132 ; PubMed — https://pubmed.ncbi.nlm.nih.gov/21979519/
21. Poyneer, Gavel & Brase 2002 — Fast wave-front reconstruction with the FFT (JOSA A 19(10):2100) — https://opg.optica.org/josaa/abstract.cfm?uri=josaa-19-10-2100 ; OSTI — https://www.osti.gov/servlets/purl/15013348
22. Performance analysis of Fourier vs Vector-Matrix-Multiply for phase reconstruction — https://arxiv.org/pdf/0911.0813
23. Advanced techniques for FT wavefront reconstruction (Poyneer, SPIE 4839) — https://ui.adsabs.harvard.edu/abs/2003SPIE.4839.1023P/abstract

**Libraries**
24. FFTW (O(N log N), GPL, SIMD, planner/wisdom) — https://www.fftw.org/
25. OpenBLAS (BSD; sgemv/dgemv thresholds; threading) — https://github.com/OpenMathLib/OpenBLAS ; FAQ — https://github.com/OpenMathLib/OpenBLAS/wiki/Faq
26. OpenBLAS USAGE / CBLAS interface — https://fossies.org/linux/OpenBLAS/USAGE.md
27. Suitability of GPUs for real-time control of large AO (J. Real-Time Image Proc.) — https://link.springer.com/article/10.1007/s11554-017-0702-7
30. Enabling technologies for GPU-driven AO RTC — https://www.researchgate.net/publication/269320303
31. COMPASS (C++/CUDA + SHESHA Python; GPL; CUBLAS/CUFFT/CURAND) — https://github.com/ANR-COMPASS ; https://compass.lesia.obspm.fr/the-platform/ao-development-platform/

**Centroiding / modal**
33. Comparison of centroid computation algorithms in a SH sensor (MNRAS) — https://academic.oup.com/mnras/article/371/1/323/980402
34. Study of centroiding algorithms to optimize SH (AO4ELT) — https://ao4elt.edpsciences.org/articles/ao4elt/pdf/2010/01/ao4elt_05004.pdf
35. Modal wavefront reconstruction based on Zernike polynomials — https://opg.optica.org/ao/abstract.cfm?uri=ao-51-21-5028
37. ZernikeViewer (precomputed Zernike matrices, 7–10× speedup) — https://doi.org/10.3390/asi9030051

**Memory / threading / RTOS**
40. Vector & matrix storage in memory (row- vs column-major) — https://receiptroller.co/en/technotes/p/vector-and-matrix-storage-in-memory-explained
41. High-performance Level-1/Level-2 BLAS (GEMV locality) — https://arxiv.org/pdf/2108.02025 ; matrix-multiplication cache blocking — https://iitd-plos.github.io/col729/lec/matrix_multiplication.html
42. Real-Time Linux: latency, jitter, scheduling — https://thomasthelliez.com/blog/real-time-linux-for-robotics/
43. Real-time performance & latency of Linux kernels (RT_PREEMPT < 150 µs) — https://www.mdpi.com/2073-431X/10/5/64

**Frameworks (additional)**
- [G1] CACAO software framework (SPIE 11448) — https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11448/2562822/ ; [G2] CACAO proceedings PDF — https://www.naoj.org/staff/guyon/publications/2018/2018-06-11_SPIE/cacao/proceedings/cacao.pdf ; repo — https://github.com/cacao-org
- [D] DARC download — https://sourceforge.net/projects/darc2/
- [DASP] Durham AO Simulation Platform — https://github.com/agb32/dasp ; paper — https://arxiv.org/pdf/1802.08503
- [S] Soapy — https://github.com/AOtools/soapy ; paper — https://www.semanticscholar.org/paper/Soapy...
- [aotools] AOtools (LGPL-3.0) — https://github.com/AOtools/aotools ; paper — https://arxiv.org/pdf/1910.04414
- [HCIPy] High Contrast Imaging for Python — https://ehpor.github.io/assets/pdf/Por-2018-HCIPy.pdf
- [OOMAO] OOMAO (MIT, MATLAB) — https://github.com/rconan/OOMAO ; OOPAO (Python) — https://hal.science/hal-04402878/document
- [YAO] YAO (GPLv3, Yorick+C) — https://github.com/frigaut/yao

---

### One-paragraph executive summary for the proposal
PS9's real-time requirement is met with enormous margin by the standard AO-RTC approach: **do all the heavy linear algebra offline in Python (build and SVD-invert the interaction matrix, fold in DM influence + inter-actuator coupling, emit a command matrix and centroiding lookup tables)**, then run a **lean C real-time core** that, every frame, computes thresholded center-of-gravity centroids over precomputed sub-aperture windows and applies **one BLAS `cblas_sgemv` matrix-vector multiply** (OpenBLAS, single precision, single thread, cache-resident matrix) to turn slopes into DM commands — *O(1) work per actuator*. At ≤30×30 sub-apertures this is ~10⁴–10⁶ FLOPs and a ≤8 MB matrix, executing in **microseconds to tens of microseconds**, i.e. **100×–1000× under the ~10 ms atmospheric coherence time**. Determinism is secured with PREEMPT_RT, core pinning, memory locking, and zero in-loop allocation; an optional **FFTW Fourier reconstructor** (O(N log N)) provides a cross-check and the wavefront power spectrum for turbulence characterization. GPUs/CuReD/sparse methods are documented as the path to ELT scale but are unnecessary here — and a GPU would likely *lose* to the CPU at this size because of PCIe transfer latency.
