# PS9 — Data Formats, Ingestion, Synthetic SH-WFS Generation & Validation

**Problem Statement 9 (ISRO BAH 2026):** Shack–Hartmann Wavefront Sensor (SH-WFS) reconstruction, turbulence characterization (r0, τ0), and DM actuator-map generation from a **time-series of `.bmp` frames** captured at a few-ms cadence.

**Why this document matters:** The real lab dataset (organizer-provided `.bmp` time-series + metadata) is **not in the repo yet**. To build and *validate* the full pipeline now, we must be able to (a) ingest `.bmp` + metadata in both C and Python, and (b) **generate realistic synthetic SH-WFS frames with a KNOWN, injected ground truth** (known r0, known τ0, known Zernike inputs). Recovering those injected values back out of our own pipeline is the verification backbone — it lets us prove correctness *before* the real data arrives and lets us quantify reconstruction error afterward.

This report covers three domains:
- **(A)** BMP file format & SH-WFS ingestion (byte layout, C + Python parsing, metadata organization, time-series cadence).
- **(B)** Synthetic SH-WFS data generation — ≥6 phase-screen methods + 2 spot-field propagation models + noise + frozen-flow time-series + AO toolkit survey.
- **(C)** Validation & metrics — r0/τ0 recovery error, RMS WFE, Strehl (Maréchal), residual variance, phase correlation, DM-residual, and unit tests with known Zernike inputs.

---

## PART A — FILE FORMAT & INGESTION

### A.1 BMP file format — exact byte layout

A Windows BMP (`BM`) file is: **BITMAPFILEHEADER (14 bytes)** → **DIB/Info header (usually BITMAPINFOHEADER = 40 bytes)** → optional **color palette** (for ≤8-bit) → **pixel array**. All multi-byte integers are **little-endian**. ([Wikipedia: BMP file format](https://en.wikipedia.org/wiki/BMP_file_format), [Microsoft: BITMAPFILEHEADER](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapfileheader), [Microsoft: Bitmap Storage](https://learn.microsoft.com/en-us/windows/win32/gdi/bitmap-storage))

#### BITMAPFILEHEADER (14 bytes)

| Offset | Size | Field | Meaning |
|-------:|-----:|-------|---------|
| 0  | 2 | `bfType`      | Magic `0x42 0x4D` = ASCII `"BM"`. **Validate this first.** |
| 2  | 4 | `bfSize`      | Total file size in bytes |
| 6  | 2 | `bfReserved1` | 0 |
| 8  | 2 | `bfReserved2` | 0 |
| 10 | 4 | `bfOffBits`   | **Byte offset from file start to the first pixel byte.** Seek here to read pixels. (54 for a headerless 24-bit BMP; 54+1024 = 1078 for an 8-bit BMP with a 256-entry palette.) |

#### BITMAPINFOHEADER (40 bytes)

| Offset | Size | Field | Meaning |
|-------:|-----:|-------|---------|
| 14 | 4 | `biSize`          | Header size. **40** = BITMAPINFOHEADER. (Other values: 12 = BITMAPCOREHEADER; 108 = BITMAPV4HEADER; 124 = BITMAPV5HEADER — branch on this.) |
| 18 | 4 | `biWidth`         | Image width in pixels (signed) |
| 22 | 4 | `biHeight`        | Image height (signed). **>0 = bottom-up** (origin lower-left); **<0 = top-down** (origin upper-left). Use `abs()` for height; flip rows if positive. |
| 26 | 2 | `biPlanes`        | Must be 1 |
| 28 | 2 | `biBitCount`      | Bits per pixel: 1, 4, **8** (grayscale via palette), 16, **24** (BGR), 32 |
| 30 | 4 | `biCompression`   | **0 = BI_RGB (uncompressed)** — expected for scientific frames. (1=BI_RLE8, 2=BI_RLE4, 3=BI_BITFIELDS.) Reject ≠0 unless handled. |
| 34 | 4 | `biSizeImage`     | Raw pixel-array size in bytes. May be 0 when `BI_RGB`. |
| 38 | 4 | `biXPelsPerMeter` | Horizontal DPM. **Can encode physical pixel pitch** (pixel size in m = 1/biXPelsPerMeter) if the camera writes it. |
| 42 | 4 | `biYPelsPerMeter` | Vertical DPM |
| 46 | 4 | `biClrUsed`       | Palette entry count; 0 ⇒ 2^biBitCount |
| 50 | 4 | `biClrImportant`  | 0 = all important |

**Color palette (only for biBitCount ≤ 8):** immediately after the info header, `biClrUsed` (or 2^bitCount) entries of **4 bytes each = B, G, R, 0 (reserved)**. For an 8-bit **grayscale** SH-WFS image the palette is the identity ramp `R=G=B=i` for i=0..255, so the pixel byte *is* the intensity — but **read the palette anyway** in case it is non-linear, and map `gray = 0.299R+0.587G+0.114B` only if truly colored. For pure grayscale the index equals the gray level.

#### Pixel array, row padding, and stride

Each scanline is padded up to a **multiple of 4 bytes**. The row size in bytes:

```
RowSize = floor( (biBitCount * biWidth + 31) / 32 ) * 4
PixelArraySize = RowSize * abs(biHeight)
```

Examples: 24-bit, width 100 → 100×3 = 300 (already a multiple of 4, 0 padding). 8-bit, width 101 → 101 → padded to 104 (3 pad bytes). **Always compute stride with the formula; never assume `width*bytesPerPixel`.** ([Wikipedia: BMP file format](https://en.wikipedia.org/wiki/BMP_file_format), [ECE/UAlberta BMP note](http://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm))

Within a 24-bit row, pixels are stored **B, G, R** (blue first). Rows are stored bottom-to-top when `biHeight > 0`.

#### Quick reference card (offsets to remember)

```
0  : "BM"
2  : file size (u32 LE)
10 : pixel-data offset  (u32 LE)  <-- seek here
14 : DIB header size (u32 LE)  (==40 for BITMAPINFOHEADER)
18 : width  (i32 LE)
22 : height (i32 LE)   (sign => row order)
28 : bits-per-pixel (u16 LE)
30 : compression (u32 LE)  (0 => uncompressed)
54 : pixel data (if 24-bit, no palette)
```

### A.2 Parsing BMP in C (from scratch, no heavy deps)

Strategy that avoids struct-padding/endianness pitfalls (do **not** `fread` a packed struct blindly — compiler may pad `BITMAPFILEHEADER`; instead read fields explicitly or use `#pragma pack(push,1)`):

```c
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// Read a little-endian u16/u32 from a byte buffer (portable, endian-safe).
static uint32_t rd_u32(const uint8_t *p){ return (uint32_t)p[0] | (p[1]<<8) | (p[2]<<16) | ((uint32_t)p[3]<<24); }
static uint16_t rd_u16(const uint8_t *p){ return (uint16_t)(p[0] | (p[1]<<8)); }

// Returns malloc'd row-major (top-left origin) grayscale buffer [h*w], or NULL.
uint8_t* bmp_read_gray(const char* path, int* out_w, int* out_h){
    FILE* f = fopen(path, "rb"); if(!f) return NULL;
    uint8_t hdr[54];
    if(fread(hdr,1,54,f)!=54){ fclose(f); return NULL; }
    if(hdr[0]!='B' || hdr[1]!='M'){ fclose(f); return NULL; }   // magic check
    uint32_t off   = rd_u32(hdr+10);
    int32_t  w     = (int32_t)rd_u32(hdr+18);
    int32_t  h_raw = (int32_t)rd_u32(hdr+22);
    uint16_t bpp   = rd_u16(hdr+28);
    uint32_t comp  = rd_u32(hdr+30);
    if(comp!=0){ fclose(f); return NULL; }                      // uncompressed only
    int bottom_up = (h_raw > 0);
    int h = bottom_up ? h_raw : -h_raw;
    int Bpp = bpp/8;                                            // 1 (8-bit) or 3 (24-bit)
    int stride = ((bpp*w + 31)/32)*4;                           // padded row size
    uint8_t* row = (uint8_t*)malloc(stride);
    uint8_t* img = (uint8_t*)malloc((size_t)w*h);
    for(int r=0; r<h; ++r){
        fseek(f, off + (long)r*stride, SEEK_SET);
        if(fread(row,1,stride,f)!=(size_t)stride){ free(row); free(img); fclose(f); return NULL; }
        int dst = bottom_up ? (h-1-r) : r;                      // flip if bottom-up
        for(int c=0;c<w;++c){
            uint8_t g = (Bpp==1) ? row[c]
                      : (uint8_t)((row[c*3+2]*299 + row[c*3+1]*587 + row[c*3]*114)/1000); // R,G,B -> luma
            img[(size_t)dst*w + c] = g;
        }
    }
    free(row); fclose(f); *out_w=w; *out_h=h; return img;
}
```

Key correctness points the code enforces: magic check, **explicit little-endian reads** (works on any host), **stride formula**, **row flip** for bottom-up, palette-free grayscale assumption (extend with a palette read for indexed 8-bit if the camera writes a non-identity LUT). For the **real-time C path**, frames are tiny (a few-hundred px square), so this reader is microseconds; pre-allocate `img`/`row` once and reuse across the time-series.

### A.3 Parsing BMP in Python (Pillow / imageio / numpy)

Three equivalent options; all give a numpy array you can centroid directly:

```python
# Option 1: Pillow (most robust, handles palette / bit-depths / top-down)
from PIL import Image
import numpy as np
arr = np.asarray(Image.open("frame_0001.bmp").convert("L"), dtype=np.float64)  # HxW grayscale

# Option 2: imageio
import imageio.v3 as iio
arr = iio.imread("frame_0001.bmp").astype(np.float64)   # may be HxWx3; take a channel/luma

# Option 3: pure numpy (no image lib) — mirror the C logic, validate against Pillow
import numpy as np
def read_bmp_gray(path):
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw[0]==0x42 and raw[1]==0x4D, "not a BMP"
    off  = int.from_bytes(raw[10:14], "little")
    w    = int.from_bytes(raw[18:22], "little", signed=True)
    h    = int.from_bytes(raw[22:26], "little", signed=True)
    bpp  = int.from_bytes(raw[28:30], "little")
    comp = int.from_bytes(raw[30:34], "little")
    assert comp == 0, "compressed BMP not supported"
    bottom_up = h > 0; H = abs(h); Bpp = bpp//8
    stride = ((bpp*w + 31)//32)*4
    px = raw[off:off+stride*H].reshape(H, stride)
    px = px[:, :w*Bpp].reshape(H, w, Bpp)
    img = px[..., 0] if Bpp==1 else (0.114*px[...,0]+0.587*px[...,1]+0.299*px[...,2])  # BGR
    if bottom_up: img = img[::-1]
    return img.astype(np.float64)
```

**Recommendation:** use **Pillow** in Python for ingestion (battle-tested against weird headers); keep the pure-numpy reader as a *cross-check* that exercises the same byte logic as the C reader (so a C/Python parity test is trivial). Pillow loads top-down already, so do **not** double-flip.

### A.4 SH-WFS frame data & metadata organization

PS9 lists the metadata that "shall be provided"; we should design an **ingestion schema** that captures all of it so the same config drives both real and synthetic runs:

| Group | Fields | Typical units | Use in pipeline |
|-------|--------|---------------|-----------------|
| **Frame** | `pixel_size` (detector pitch), `frame_resolution` (Nx×Ny), `bit_depth`, `frame_rate`/`dt` | µm, px, bits, ms | px→angle scale; centroiding grid; cadence → τ0 |
| **MLA** | `lenslet_pitch`/size, `n_lenslets` (e.g. NxN), `focal_length` f | mm, count, mm | subaperture extent; spot displacement → slope (`Δ = f·θ`) |
| **Pupil** | `pupil_diameter` of turbulated beam, magnification beam→MLA | mm | maps to D for r0 in physical units; valid-subaperture mask |
| **DM** | actuator count/grid, **inter-actuator coupling**, stroke length, geometry (Fried wrt MLA) | count, %, µm | influence-function model; actuator map units |
| **Geometry** | **Fried geometry**: actuators at corners of lenslet subapertures (slopes between phase points) | — | reconstructor stencil (Fried/Hudgin/Southwell) |

**Fried geometry note (from PS9):** the DM actuator grid and MLA lenslet grid are in **Fried geometry** — actuators sit at the *corners* of the subapertures, and each subaperture's two slopes are differences of the four surrounding phase points. This dictates the reconstructor (see Part C and the reconstruction research doc). ([Hudgin/Fried/Southwell geometries](https://www.researchgate.net/figure/Wavefront-estimation-schemes-a-Hudgin-geometry-b-Southwell-geometry-c-Fried_fig1_6807968))

**Ingestion design:** store metadata in a sidecar `meta.json`/`meta.yaml` (and optionally read `biXPelsPerMeter` from the BMP as a fallback pixel-pitch). Frames named with zero-padded indices (`frame_00001.bmp`) so glob+sort gives chronological order; record `dt` (inter-frame interval) explicitly since **τ0 estimation needs the true cadence**. A small `meta`-driven loader yields `(frames[T,H,W], dt, mla, pupil, dm)`.

### A.5 Reading time-series at few-ms cadence

- **Order by frame index**, not filesystem mtime. Verify monotonic timestamps if present.
- **Cadence drives τ0**: the temporal structure function / autocorrelation of slopes (or Zernike coefficients) needs the real `dt`. Store it; never assume.
- **Throughput:** at ~10 ms/frame (100 Hz) the C reader + centroider must finish well under 10 ms. Memory-map or stream frames; reuse buffers; for Python prototyping, pre-load the whole series into one `float32` array `[T,H,W]`.
- **Reference frame:** the *reference spot positions* (flat/plane-wave calibration) are typically a separate BMP or the time-average of an unaberrated run; define where it comes from in `meta`.

---

## PART B — SYNTHETIC SH-WFS DATA GENERATION

Pipeline: **phase screen φ(x,y)** (with injected r0, L0) → **propagate through MLA** (geometric tilt or Fraunhofer/FFT) → **spot field** → **detector sampling + photon + read noise** → **write `.bmp`** → optionally **translate screen by wind·dt** for the next frame (frozen flow ⇒ known τ0).

### B.1 Atmospheric phase-screen generation — ≥6 methods

The phase structure function and the **von Kármán PSD** of phase are the foundation. The aotools modified-von-Kármán PSD (radians²·m²) is:

```
Φ_φ(f) = 0.023 · r0^(-5/3) · exp(-(f/fm)^2) / (f^2 + f0^2)^(11/6)
   with  f0 = 1/L0  (outer scale),  fm = 5.92/(2π·l0)  (inner scale cutoff)
```

(Coefficient `0.023`, exponent `-5/3` on r0, the `(f²+1/L0²)^(-11/6)` outer-scale roll-off, and the `exp(-(f·l0/5.92)²)` inner-scale term are exactly what `aotools.turbulence.phasescreen` implements.) Setting `l0=0`, `L0=∞` recovers pure **Kolmogorov**. ([aotools phasescreen source](https://aotools.readthedocs.io/en/v1.0/_modules/aotools/turbulence/phasescreen.html), [arte von Kármán PSD](https://arte.readthedocs.io/en/latest/notebook/atmo/von_karman_psd_examples.html))

**Method 1 — FFT / Fourier filtering of the PSD ("spectral method").**
Fill a complex grid with white Gaussian noise, multiply by `sqrt(Φ_φ(f))·Δf` (with frequency spacing `Δf = 1/(N·Δx)`), inverse-FFT, take the real (and/or imaginary) part as an independent screen. Fast (O(N²logN)), but **under-represents low spatial frequencies** because the lowest non-zero frequency is `1/(N·Δx)` — tip/tilt and large-scale modes are systematically too weak. ([aotools `ft_phase_screen`](https://aotools.readthedocs.io/en/v1.0/_modules/aotools/turbulence/phasescreen.html), [Implementation of FFT-based extra-large screens, Sedmak 2004](https://opg.optica.org/ao/abstract.cfm?uri=ao-43-23-4527))

**Method 2 — FFT + subharmonic augmentation (low-frequency correction).**
Add extra low-frequency content by summing several **subharmonic grids**: for `p = 1..Np` use a 3×3 grid with spacing `Δf_p = 1/(3^p·D)`, draw random complex coefficients weighted by `sqrt(Φ_φ)`, and add their (continuous) inverse transform to the FFT screen. This is the **standard fix** and is exactly `ft_sh_phase_screen` ("Von Kármán statistics with added sub-harmonics to augment tip-tilt modes"). Optimal subharmonic count/levels discussed by Charnotskii / Johansson & Gavel. ([aotools `ft_sh_phase_screen`](https://aotools.readthedocs.io/en/latest/turbulence.html), [Optimal subharmonics selection](https://www.researchgate.net/publication/332164706_Optimal_subharmonics_selection_for_atmosphere_turbulence_phase_screen_simulation_using_the_subharmonic_method))

**Method 3 — Zernike-based screen synthesis.**
Draw Zernike coefficients with the **Noll covariance matrix** (∝ r0^(-5/3), known per-mode variances, e.g. residual after Z1..Zj) and sum `φ = Σ aⱼ Zⱼ(ρ,θ)`. Excellent for **unit tests** (you inject *exactly* known modal content and check recovery), and good for the low-order part; combine with a spectral high-frequency tail for full fidelity ("spectral + Zernike" hybrid). ([Noll 1976 via aotools zernike](https://aotools.readthedocs.io/en/latest/turbulence.html), [Accurate/fast Kolmogorov screen = spectral + Zernike](https://www.researchgate.net/publication/249334814_Accurate_and_fast_simulation_of_Kolmogorov_phase_screen_by_combining_spectral_method_with_Zernike_polynomials_method))

**Method 4 — Covariance / Cholesky (direct) method.**
Build the phase covariance matrix `C` (von Kármán covariance between sample points, from the structure function `D_φ(r)`), Cholesky-factor `C = LLᵀ`, and generate `φ = L·g` for white Gaussian `g`. **Statistically exact** at the sample points (no low-frequency deficit), but O(N³) factorization and O(N²) memory — practical only for modest grids; ideal as a **gold-standard reference** to validate FFT/subharmonic screens. (Basis of the "infinite phase screen" A/B covariance update, below.) ([Modal decomposition of von Kármán covariance](https://arxiv.org/pdf/0911.4710))

**Method 5 — "Infinite"/streaming phase screen (Assémat & Wilson 2006; Fried/Kolmogorov).**
Keep only a small strip and **extrude one new column/row at a time** using precomputed covariance matrices A, B so the new column is correctly correlated with the existing screen: `new_col = A·(existing) + B·g`. Constant memory, ideal for **long time-series** with frozen flow. Implemented as `aotools.turbulence.infinitephasescreen.PhaseScreenKolmogorov` / `PhaseScreenVonKarman` (`.add_row()`, `.scrn`) and HCIPy `InfiniteAtmosphericLayer`. ([aotools infinitephasescreen](https://aotools.readthedocs.io/en/v1.0/_modules/aotools/turbulence/infinitephasescreen.html), [Assémat & Wilson 2006], [HCIPy](https://docs.hcipy.org/dev/tutorials/ShackHartmannWFS/ShackHartmannWFS.html))

**Method 6 — Autoregressive (AR) frozen-flow + boiling.**
Computationally efficient AR(1)-type update that **combines deterministic translation (frozen flow) with stochastic "boiling"** so turbulence both drifts and decorrelates — more realistic temporal statistics than pure translation, and lets you dial τ0 directly. ([Computationally efficient AR phase screens, Srinath et al. 2015](https://arxiv.org/pdf/1512.05424), [Time-dependent Karhunen–Loève screens](https://arxiv.org/pdf/2510.12861))

**Method 7 (bonus) — NUFFT / real-time generation.** Non-uniform FFT lets you sample arbitrary low frequencies densely → real-time large screens without the subharmonic kludge. ([Real-time generation with NUFFT, MNRAS 2015](https://academic.oup.com/mnras/article/450/1/38/1002502))

> For PS9 the **recommended generator is Method 2 (FFT + subharmonics) for single frames** and **Method 5 or 6 for the moving time-series**, with **Method 3 (Zernike)** for deterministic unit tests and **Method 4 (Cholesky)** as a statistical gold standard.

### B.2 Propagating a screen through the lenslet array → spot field

Crop the pupil-plane field `U = exp(i·φ)` into per-subaperture tiles (the MLA grid). For each subaperture two models:

**Model 1 — Geometric tilt-per-subaperture (fast, what real SH-WFS reduction assumes).**
The *average local slope* over a subaperture tilts its focal spot. Mean wavefront gradient `⟨∂φ/∂x⟩` over the subaperture maps to a spot displacement:

```
θx = (λ/2π)·⟨∂φ/∂x⟩         (local tilt angle, rad)
Δx = f · θx                  (spot shift on detector;  f = lenslet focal length)
```

Place a Gaussian/Airy spot of the diffraction-limited width at the reference position + (Δx, Δy). This is the **inverse of the centroiding model**, so it gives a clean, analytically-known ground truth for slope recovery — perfect for end-to-end tests. ([RP-Photonics: SH-WFS](https://www.rp-photonics.com/shack_hartmann_wavefront_sensors.html))

**Model 2 — Fraunhofer / FFT diffraction (physically faithful).**
Each lenslet performs an optical Fourier transform: the focal-plane complex amplitude of a subaperture is the **FFT of its (apodized) pupil field**, intensity `I = |FFT(U_sub · pupil_mask)|²`. This naturally reproduces spot **broadening, speckle, and shape distortion** from higher-order aberration inside the subaperture, not just a clean shift. Assemble the full detector image by tiling each subaperture's `|FFT|²` into its cell — this is precisely how HCIPy/soapy synthesize SH images. ([Numerical SH-WFS simulation, Optics&Lasers in Eng. 2006](https://www.sciencedirect.com/science/article/abs/pii/S0143816605000734), [SH lenslet-array simulation, Optik 2011](https://www.sciencedirect.com/science/article/pii/S1875389211004809), [HCIPy SH tutorial](https://docs.hcipy.org/dev/tutorials/ShackHartmannWFS/ShackHartmannWFS.html))

Both models then **sample to the detector grid** (sum/bin to detector pixels at the given `pixel_size`), apply quantum efficiency, and proceed to noise.

### B.3 Photon (shot) + detector read noise + quantization

Realistic SH frames need a detector model so centroiding is tested under the noise it will see ([Konnik & Welsh, CCD/CMOS noise tutorial, arXiv:1412.4031](https://arxiv.org/pdf/1412.4031); [Modeling noise for image simulations](http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/)):

```
1. electrons_ideal = I_normalized * flux_photons * QE          # scale to expected e-
2. electrons        = Poisson(electrons_ideal)                  # PHOTON SHOT NOISE (signal-dependent)
3. electrons       += Normal(dark_e, sqrt(dark_e))             # dark current (+ its shot noise), optional
4. ADU_float        = electrons / gain  +  Normal(0, read_noise/gain)   # READ NOISE (Gaussian)
5. ADU              = clip(round(ADU_float + bias), 0, 2^bitdepth - 1)  # quantize to 8-bit, add offset
```

- **Shot noise**: Poisson with variance = mean (dominant in spots). At high flux → Gaussian approx.
- **Read noise**: additive Gaussian, σ in e- RMS (constant per pixel).
- **Quantization**: round to the BMP bit depth (usually **8-bit**, so 0–255). Tune `flux`, `read_noise`, `gain`, and background so the spots' SNR matches the real camera. This lets us study **centroiding bias vs SNR** with known truth.

### B.4 Writing `.bmp` frames

Inverse of A.2/A.3: emit a 14-byte BITMAPFILEHEADER + 40-byte BITMAPINFOHEADER, **8-bit** with a 256-entry identity grayscale palette (offset 54, palette 1024 bytes ⇒ `bfOffBits = 1078`), pad each row to a 4-byte multiple, write rows bottom-up (or set negative `biHeight` for top-down). In Python: `Image.fromarray(adu.astype('uint8'), mode='L').save('frame_00001.bmp')` (Pillow writes a valid 8-bit grayscale BMP with palette). Name frames with zero-padded indices and emit the matching `meta.json` (pixel_size, MLA, pupil, DM, **dt**, plus the **injected r0/τ0/Zernike truth** in a `ground_truth` block) — this makes every synthetic dataset self-validating.

### B.5 Time-series generation (frozen flow → known τ0)

Under the **Taylor / frozen-flow hypothesis**, the screen translates rigidly at the projected wind velocity; the coherence time follows analytically:

```
τ0 = 0.314 · r0 / v_eff
```

So if we generate a big screen with **known r0** and slide it by `v·dt` pixels per frame with **known wind speed v**, we have **injected a known τ0** and can test our τ0 estimator. (e.g., `v=10 m/s, r0=0.15 m @ 500 nm → τ0 ≈ 4.5 ms`.) Two implementations: (i) shift-and-interpolate a large precomputed screen (simple, periodic); (ii) **extrude** new columns with the infinite-screen/AR methods (Methods 5/6) for unbounded, non-periodic series and optional "boiling" to test robustness of frozen-flow assumptions. ([Frozen flow / τ0 = 0.314 r0/v_eff, SPHERE wind-halo paper](https://arxiv.org/pdf/2003.05794), [AR frozen-flow+boiling](https://arxiv.org/pdf/1512.05424), [Coherence time vs wind speed](https://academic.oup.com/mnras/article/397/3/1633/1078196))

### B.6 AO toolkits / references for synthetic generation

| Tool | Lang | Key function / class | Role | Link |
|------|------|----------------------|------|------|
| **aotools** | Python | `turbulence.phasescreen.ft_phase_screen(r0,N,delta,L0,l0)`, `ft_sh_phase_screen(...)` (subharmonics); `infinitephasescreen.PhaseScreenKolmogorov/VonKarman(.add_row, .scrn)`; `turbulence.r0_to_cn2`, `cn2_to_r0`, `structure_function_vk/kolmogorov`; `zernike`, `phaseFromZernikes`; `image.centreOfGravity` | **Primary reference** for screens, structure fns, r0↔Cn², Zernike, centroids | [readthedocs](https://aotools.readthedocs.io/en/latest/turbulence.html) · [paper arXiv:1910.04414](https://ar5iv.labs.arxiv.org/html/1910.04414) |
| **HCIPy** | Python | `make_standard_atmospheric_layers`, `InfiniteAtmosphericLayer`, `FiniteAtmosphericLayer`, `Cn_squared_from_fried_parameter`; `SquareShackHartmannWavefrontSensorOptics(grid,f_number,n_lenslets,sh_diameter)`, `ShackHartmannWavefrontSensorEstimator(mla_grid, mla_index).estimate([img])`; `NoisyDetector`/`NoiselessDetector` | **End-to-end SH simulator** (screen→MLA→detector→slopes); reference implementation for our pipeline | [SH tutorial](https://docs.hcipy.org/dev/tutorials/ShackHartmannWFS/ShackHartmannWFS.html) · [GitHub](https://github.com/ehpor/hcipy) |
| **soapy** | Python | `soapy.atmosphere` (FFT + subharmonic & infinite screens), `soapy.wfs.ShackHartmann`, frozen-flow looping | Full Monte-Carlo AO sim incl. moving screens & SH WFS | [paper (Reeves 2016)](https://www.semanticscholar.org/paper/Soapy:-an-adaptive-optics-simulation-written-purely-Reeves/db12a7d056575b6872b81d9a497dcb31b9a1306f) |
| **POPPY** | Python | `poppy` Fresnel/Fraunhofer optical propagation (`FresnelWavefront`, optical elements) | Physical-optics spot/PSF propagation (Model 2 reference) | [astropy/poppy] |
| **prysm** | Python | physical-optics / interferometry modeling, Zernikes, PSF/MTF | Optics modeling, Zernike & diffraction utilities | [prysm docs] |
| **OOMAO** | MATLAB | `atmosphere`, `source`, `telescope`, `shackHartmann`, `deformableMirror`, `imager` classes | Class-based AO sim (screens, SH, DM) — algorithm reference | [Conan & Correia 2014] |
| **YAO** | Yorick | full end-to-end AO (`atm`, `wfs`, `dm` structures) | End-to-end AO Monte-Carlo reference | [Rigaut & Van Dam 2013](https://www.researchgate.net/publication/260938844_Simulating_Astronomical_Adaptive_Optics_Systems_Using_Yao) |
| **CEO** | CUDA/Python | GPU AO sim (atmosphere, SH, DM) | GPU-accelerated reference | [CEO (rconan)] |
| **OOPAO / COMPASS / MAOS / SPECULA** | Py / CUDA | end-to-end AO frameworks | Cross-check / further reference | [PASSATA→SPECULA arXiv:2602.06688](https://arxiv.org/html/2602.06688) |

> **Note for the C deliverable:** these toolkits are *references and oracles* (validate our outputs against them). PS9 wants the fast path in **C**; we re-implement the minimal screen-generator + spot-field + centroider + reconstructor in C, and use aotools/HCIPy in Python to **generate the synthetic `.bmp` datasets and cross-check** the C results. (We are *not* installing packages here — this table is the integration plan.)

---

## PART C — VALIDATION & METRICS

Because synthetic data carries the **injected ground truth**, every metric below can be computed as *recovered vs. true*. This is the verification backbone.

### C.1 Core equations

- **RMS wavefront error:** `σ_WFE = sqrt( mean( (W_recon − W_true)² ) )` over valid pupil pixels (subtract piston; usually subtract tip/tilt too). In radians: `σ_φ = (2π/λ)·σ_WFE`.
- **Residual phase variance:** `σ_res² = Var(W_recon − W_true)` (the quantity Strehl depends on).
- **Strehl ratio — Maréchal approximation:** `S ≈ exp(−σ_φ²)` where `σ_φ` is the **residual** RMS phase (radians). Valid for `σ_φ ≲ 0.5` rad (S ≳ 0.6 within ~10%). Equivalent forms: `S = exp[−(2π·σ_WFE/λ)²]`. ([Eikonal Optics: Strehl & Maréchal](https://eikonaloptics.com/blogs/tutorials/image-quality-metrics-the-strehl-ratio-and-the-marechal-approximation))
- **Reconstructed-vs-true phase correlation:** Pearson `ρ = cov(W_recon, W_true)/(σ_recon·σ_true)` over the pupil (target ρ→1).
- **r0 recovery error:** compare estimated `r0_est` (from slope/Zernike variance) to injected `r0_true`; report `|r0_est − r0_true|/r0_true`.
  - From **Zernike-coefficient variances**: per-mode variance ∝ (D/r0)^(5/3) with Noll coefficients ⇒ invert for r0. ([Noll 1976; r0 from Zernike variance, SHWFS coherence-length paper](https://www.mdpi.com/2304-6732/11/12/1184))
  - From the **measured slope/structure function** vs the von Kármán model.
- **τ0 recovery error:** estimate `τ0_est` from the **temporal autocorrelation/structure function** of slopes or Zernike coefficients (the 1/e or defined drop) and compare to injected `τ0_true = 0.314 r0/v_eff`. Cross-check via Greenwood frequency `f_G = 0.427 v/r0` (and `τ0 ≈ 0.314 r0/v`). ([Coherence time from SHWFS modal variance, FADE/defocus](https://opg.optica.org/ao/abstract.cfm?uri=ao-58-31-8673), [τ0 & wind speed, MNRAS 2009](https://academic.oup.com/mnras/article/397/3/1633/1078196))
- **DM-correction residual:** after applying the actuator map (with the modeled **inter-actuator coupling** influence functions `IF(ρ)=exp[ln(ω)(ρ/d0)^α]`, ω=coupling), residual `W_res = W_true − DM_shape`; report `σ(W_res)` and Strehl `exp(−σ_φ,res²)` to show closed-loop benefit. ([Influence function & coupling model](https://www.researchgate.net/publication/216852286_A_novel_model_of_influence_function_calibration_of_a_continuous_membrane_deformable_mirror))

### C.2 Validation-metrics table

| Metric | Formula | Ground-truth source | Pass criterion (suggested) |
|--------|---------|---------------------|----------------------------|
| Centroid/slope error | `Δslope = slope_meas − slope_true` | Geometric-tilt generator (Model 1) | sub-0.1 px centroid bias at design SNR |
| RMS wavefront error | `sqrt(mean((W_rec−W_true)²))` | injected screen / Zernike | small vs λ (e.g. < λ/14 for "diffraction-limited") |
| Residual variance | `Var(W_rec−W_true)` | injected screen | minimized; feeds Strehl |
| Strehl (Maréchal) | `exp(−σ_φ,res²)` | residual phase | matches FFT-PSF Strehl within ~10% (S>0.6) |
| Phase correlation ρ | Pearson over pupil | injected screen | ρ > 0.95 (low-noise) |
| r0 recovery | `|r0_est−r0_true|/r0_true` | injected r0 | within a few % over many frames |
| τ0 recovery | `|τ0_est−τ0_true|/τ0_true` | injected v, r0 (frozen flow) | within ~10–20% |
| DM residual | `σ(W_true − DM_shape)` | injected screen + DM model | large reduction vs uncorrected; coupling-aware |
| Reconstructor self-consistency | `slope(W_rec) ≈ slope_meas` | — | round-trip residual ≈ 0 |
| C/Python parity | per-pixel diff of outputs | same input frame | bit-near-identical centroids/slopes |

### C.3 Unit tests with known Zernike inputs (deterministic)

The cleanest correctness tests inject a **single known Zernike mode** and check exact recovery — this isolates bugs in slopes, reconstructor geometry, and units:

1. **Pure tip/tilt (Z2/Z3):** a known global tilt must shift *all* spots by `Δ = f·θ` and reconstruct to a flat-gradient plane. Checks px↔slope scale and sign conventions.
2. **Pure defocus (Z4):** spots shift **radially** (∝ distance from center); reconstruct the parabola. Checks radial geometry; also the mode used by FADE-style τ0. ([defocus velocity τ0](https://opg.optica.org/ao/abstract.cfm?uri=ao-58-31-8673))
3. **Astigmatism / coma / higher Zernikes (Z5…Z11):** inject coefficient `aⱼ`, reconstruct, fit Zernikes to `W_rec`, assert `â_j ≈ a_j` and cross-terms ≈ 0 (mode purity / no cross-talk).
4. **Linear-combination / superposition:** inject `Σ aⱼ Zⱼ`, check linearity of the reconstructor (least-squares recon is linear ⇒ recovered vector ≈ injected within reconstructor null-space).
5. **Statistical screen test:** generate many von Kármán screens at known r0, run reconstruction, confirm the **ensemble** Zernike variances follow the Noll `(D/r0)^(5/3)` law ⇒ validates r0 recovery end-to-end.
6. **Reconstructor null-space / waffle:** confirm unmeasurable modes (piston, waffle in Fried geometry) are handled (regularized/filtered), not blown up.

Use `aotools.zernike`/`phaseFromZernikes` (or our own Zernike basis) to synthesize inputs; assert recovered coefficients within tolerance. These tests run in CI without any real data and guard every refactor. ([aotools zernike & centroids](https://aotools.readthedocs.io/en/latest/turbulence.html))

### C.4 Validating turbulence-parameter recovery specifically

- **r0:** generate N≥1000 independent screens at a *known* r0; compute slope variance / Zernike-coefficient variances per frame; fit to the von Kármán/Noll model; the fitted r0 must converge to the injected value. Report bias and scatter vs N (Monte-Carlo convergence). ([SHWFS r0 from modal variance](https://www.mdpi.com/2304-6732/11/12/1184))
- **τ0:** generate a frozen-flow series at known `(r0, v)` ⇒ known `τ0=0.314 r0/v`; estimate from the temporal autocorrelation of slopes/Zernikes (and cross-check Greenwood `f_G=0.427 v/r0`); confirm recovered τ0 matches. Vary v to sweep τ0 and check the linear `τ0 ∝ r0/v` trend. ([τ0=0.314 r0/v_eff](https://arxiv.org/pdf/2003.05794))

---

## Recommended Data + Synthetic-Generation + Validation Plan for PS9

**1. Ingestion layer (works for real *and* synthetic).**
- Implement BMP reader in **C** (Section A.2: explicit little-endian reads, stride formula, row-flip, 8-bit grayscale + palette aware) and **Python** (Pillow primary + pure-numpy cross-check, Section A.3).
- Drive everything from a `meta.json` schema (Section A.4): `pixel_size, frame_resolution, bit_depth, dt, MLA{pitch,n_lenslets,focal_length}, pupil{diameter}, DM{n_act, coupling, stroke, geometry=Fried}`, plus a `ground_truth{r0, tau0, wind_v, zernike[]}` block for synthetic sets. Glob+sort frames by zero-padded index; record true `dt`.

**2. Synthetic generator (build now; validate the whole pipeline before real data).**
- **Phase screens:** Method 2 (**FFT + subharmonics**, the aotools `ft_sh_phase_screen` PSD `0.023 r0^(−5/3)·exp(−(f/fm)²)/(f²+f0²)^(11/6)`) for single frames; **Method 5/6 (infinite/AR screens)** for the moving time-series. Keep **Method 4 (Cholesky)** as a statistical gold standard and **Method 3 (Zernike/Noll)** for deterministic unit tests.
- **Spot field:** support **both** Model 1 (geometric tilt `Δ=f·θ`, the analytic oracle) and Model 2 (per-subaperture **FFT/Fraunhofer** `|FFT(U_sub·mask)|²`, physically faithful), tiled to the MLA grid and binned to the detector pitch.
- **Noise:** Poisson shot + Gaussian read + quantize to 8-bit (Section B.3); expose `flux, QE, read_noise, gain, bias` to match the real camera SNR.
- **Output:** write valid 8-bit grayscale `.bmp` (identity palette, `bfOffBits=1078`) + `meta.json` carrying injected r0/τ0/Zernike truth.
- **Time-series / τ0:** frozen-flow translation at known wind `v` ⇒ **injected τ0 = 0.314 r0/v**; optionally add AR "boiling" to stress-test the frozen-flow assumption.

**3. Validation harness (the verification backbone).**
- **Unit tests:** inject single Zernikes (tip/tilt, defocus, astig, coma) → assert exact spot shifts and recovered coefficients; superposition/linearity; null-space/waffle handling (Section C.3).
- **End-to-end metrics:** RMS WFE, residual variance, **Strehl = exp(−σ_φ,res²)** (Maréchal), phase correlation ρ, **r0-recovery error**, **τ0-recovery error**, DM-corrected residual with **inter-actuator coupling** influence functions (Section C.2 table).
- **Monte-Carlo convergence:** ensemble Zernike variances must follow Noll `(D/r0)^(5/3)` ⇒ r0 recovery proven; τ0 sweep vs wind speed proves τ0 recovery.
- **C/Python parity:** identical synthetic frame through both readers/centroiders must give matching slopes (guards the fast C path).
- **Cross-tool oracle:** spot-check our screens/SH images/slopes against **aotools/HCIPy/soapy** outputs (references only; no install required here).

**Bottom line:** Generate `.bmp` SH-WFS frames with a **known r0 and known τ0** (frozen-flow), run them through the *same* ingestion + reconstruction + characterization pipeline that will consume the real data, and confirm the pipeline **recovers the injected r0, τ0, and Zernike content** within tolerance. That closed loop is what makes the PS9 solution demonstrably correct and robust before — and after — the organizers' dataset arrives.

---

## Sources

**BMP format & ingestion**
- Wikipedia — BMP file format: https://en.wikipedia.org/wiki/BMP_file_format
- Microsoft Learn — BITMAPFILEHEADER (wingdi.h): https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapfileheader
- Microsoft Learn — Bitmap Storage: https://learn.microsoft.com/en-us/windows/win32/gdi/bitmap-storage
- ECE/UAlberta — The BMP File Format: http://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm
- DigicamSoft — BMP File Format: https://www.digicamsoft.com/bmp/bmp.html

**Phase-screen generation & PSD**
- aotools — Atmospheric Turbulence (API): https://aotools.readthedocs.io/en/latest/turbulence.html
- aotools — phasescreen source (ft_phase_screen / ft_sh_phase_screen): https://aotools.readthedocs.io/en/v1.0/_modules/aotools/turbulence/phasescreen.html
- aotools — infinitephasescreen source: https://aotools.readthedocs.io/en/v1.0/_modules/aotools/turbulence/infinitephasescreen.html
- AOtools paper (arXiv:1910.04414): https://ar5iv.labs.arxiv.org/html/1910.04414
- arte — von Kármán PSD examples: https://arte.readthedocs.io/en/latest/notebook/atmo/von_karman_psd_examples.html
- Sedmak — FFT-based extra-large screens (Appl. Opt. 2004): https://opg.optica.org/ao/abstract.cfm?uri=ao-43-23-4527
- Optimal subharmonics selection: https://www.researchgate.net/publication/332164706_Optimal_subharmonics_selection_for_atmosphere_turbulence_phase_screen_simulation_using_the_subharmonic_method
- Comparison of phase-screen generation methods: https://www.researchgate.net/publication/334320019_Comparison_of_Phase-Screen-Generation_Methods_for_Simulating_the_Effects_of_Atmospheric_Turbulence
- Spectral + Zernike Kolmogorov screen: https://www.researchgate.net/publication/249334814_Accurate_and_fast_simulation_of_Kolmogorov_phase_screen_by_combining_spectral_method_with_Zernike_polynomials_method
- Modal decomposition of von Kármán covariance (arXiv:0911.4710): https://arxiv.org/pdf/0911.4710
- Real-time screens with NUFFT (MNRAS 2015): https://academic.oup.com/mnras/article/450/1/38/1002502
- AR frozen-flow + boiling screens (arXiv:1512.05424): https://arxiv.org/pdf/1512.05424
- Time-dependent Karhunen–Loève screens (arXiv:2510.12861): https://arxiv.org/pdf/2510.12861

**SH-WFS simulation, spot field, detectors**
- HCIPy — Adaptive optics with a Shack-Hartmann WFS (tutorial): https://docs.hcipy.org/dev/tutorials/ShackHartmannWFS/ShackHartmannWFS.html
- HCIPy — GitHub: https://github.com/ehpor/hcipy
- HCIPy paper (Por et al. 2018): https://ehpor.github.io/assets/pdf/Por-2018-HCIPy.pdf
- Soapy (Reeves 2016): https://www.semanticscholar.org/paper/Soapy:-an-adaptive-optics-simulation-written-purely-Reeves/db12a7d056575b6872b81d9a497dcb31b9a1306f
- Numerical SH-WFS simulation (Opt. & Lasers in Eng. 2006): https://www.sciencedirect.com/science/article/abs/pii/S0143816605000734
- SH lenslet-array sensing simulation (Optik 2011): https://www.sciencedirect.com/science/article/pii/S1875389211004809
- RP-Photonics — Shack–Hartmann wavefront sensors: https://www.rp-photonics.com/shack_hartmann_wavefront_sensors.html
- YAO (Rigaut & Van Dam 2013): https://www.researchgate.net/publication/260938844_Simulating_Astronomical_Adaptive_Optics_Systems_Using_Yao
- PASSATA → SPECULA (arXiv:2602.06688): https://arxiv.org/html/2602.06688
- Konnik & Welsh — CCD/CMOS noise tutorial (arXiv:1412.4031): https://arxiv.org/pdf/1412.4031
- Modeling noise for image simulations (K. Douglass): http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/

**Reconstruction geometry, DM coupling**
- Hudgin/Southwell/Fried geometries (figure): https://www.researchgate.net/figure/Wavefront-estimation-schemes-a-Hudgin-geometry-b-Southwell-geometry-c-Fried_fig1_6807968
- DM influence-function / coupling model: https://www.researchgate.net/publication/216852286_A_novel_model_of_influence_function_calibration_of_a_continuous_membrane_deformable_mirror

**Validation, Strehl, r0/τ0**
- Eikonal Optics — Strehl ratio & Maréchal approximation: https://eikonaloptics.com/blogs/tutorials/image-quality-metrics-the-strehl-ratio-and-the-marechal-approximation
- r0 from SHWFS (extended sources, Photonics 2024): https://www.mdpi.com/2304-6732/11/12/1184
- Single-star turbulence profiling (SHIMM, arXiv:2603.02817): https://arxiv.org/pdf/2603.02817
- τ0 by 4-aperture DIMM defocus velocity (Appl. Opt. 2019): https://opg.optica.org/ao/abstract.cfm?uri=ao-58-31-8673
- AO parameters vs wind speed (MNRAS 2009): https://academic.oup.com/mnras/article/397/3/1633/1078196
- Wind-driven halo / frozen flow & τ0 (SPHERE, arXiv:2003.05794): https://arxiv.org/pdf/2003.05794
- Fried-parameter estimation from single WFS image (arXiv:2504.17029): https://arxiv.org/html/2504.17029v1
