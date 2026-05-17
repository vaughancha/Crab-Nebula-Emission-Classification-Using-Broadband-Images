# Crab Nebula Emission Classification

BVR photometric calibration and pixel-level emission classification of the Crab Nebula using ground-based imaging from Leuschner Observatory.

---

## What this does

The Crab Nebula has two physically distinct emission components: a smooth synchrotron-emitting pulsar wind nebula (PWN) in the core, and a network of ionized filaments scattered across the outer shell. These components have different spectral shapes, which means they leave different fingerprints in broadband BVR imaging.

This project takes raw B, V, and R frames of the Crab, runs them through a full photometric calibration pipeline, and then uses the resulting flux ratios to spatially classify each pixel as synchrotron-like or filament-like. Three classical methods are compared side by side, and a Neural Ratio Estimation (NRE) approach produces a continuous P(filament) probability map.

---

## Pipeline overview

**`Project_D.py`** — the main calibration and classification pipeline

1. **Flat correction** — divides each raw frame by a normalized master flat. Dark subtraction was already handled by MaxIm DL (CALSTAT=D in headers), so only the flat division is needed.

2. **Image alignment** — uses phase cross-correlation (Fourier-based) to align the B and R frames to V with sub-pixel precision (~0.01 px). V is the reference.

3. **Star detection** — runs DAOStarFinder on the V frame with the Crab core masked out (500 px radius), so the nebula doesn't pollute the star catalog. Edge sources are also cut.

4. **Aperture photometry** — measures instrumental flux in all three bands using a 10 px circular aperture and a 15–25 px sky annulus for background subtraction.

5. **APASS catalog cross-match** — queries the APASS DR9 catalog via Vizier for field stars. Solves for the telescope's pointing offset with a brute-force grid search (±300 px coarse, then ±15 px fine), then matches detected sources to catalog positions within 10 arcsec. Uses the match set to compute per-band photometric zeropoints via sigma-clipped median.

6. **Calibrated FITS output** — saves `Crab_B_cal.fits`, `Crab_V_cal.fits`, `Crab_R_cal.fits` with zeropoints embedded in the headers.

7. **Ratio maps** — computes log₁₀(R/B) and log₁₀(V/B) per pixel over a signal-detected, star-masked region of the nebula. R/B traces Hα+[NII] emission; V/B traces [OIII]+Hβ.

8. **Color-color diagram** — plots all valid pixels in ratio space, colored by distance from the Crab centroid. Inner pixels cluster in the synchrotron region; outer pixels shift toward the filament locus.

9. **Classification (3 methods)**
   - *Histogram minimum*: finds the valley in the 1D log(R/B) distribution
   - *K-means (k=2)*: clusters pixels in 2D color-color space
   - *GMM (k=2)*: fits elliptical Gaussians, allows non-spherical clusters

All three are displayed as spatial overlays on the V-band image.

---

**`Project_E_NRE.py`** — Neural Ratio Estimation

Trains a binary MLP classifier on forward-simulated (log R/B, log V/B) spectra for synchrotron and filament pixels, then applies it per-pixel to produce a continuous P(filament) probability map. Uses a custom numpy MLP with Adam optimizer — no PyTorch or TensorFlow required.

---

**`RGB_Stack.py`** — false-color composite

Builds a flat-corrected BVR RGB image of the Crab for visual inspection.

---

## Output files

| File | Description |
|------|-------------|
| `data/Crab_B_cal.fits` / `Crab_V_cal.fits` / `Crab_R_cal.fits` | Flat-corrected, aligned, APASS-calibrated images |
| `data/p_fil_img.npy` | Raw NRE P(filament) array |
| `output/calibration_check.png` | Instrumental vs catalog magnitude scatter plots for all three bands |
| `output/ratio_maps.png` | log(R/B) and log(V/B) spatial maps of the Crab |
| `output/color_color.png` | Color-color diagram with density and radial distance panels |
| `output/spatial_overlay.png` | Side-by-side comparison of the three classification methods |
| `output/rgb_crab.png` | False-color BVR composite |
| `output/nre_classification.png` | NRE P(filament) probability map + histogram |

---

## Dependencies

```
numpy
astropy
astroquery
photutils
scikit-image
scipy
scikit-learn
matplotlib
```

---

## Data

Observations taken 2025-03-17 at Leuschner Observatory. Three 60-second exposures in B, V, and R. Plate scale ~1.018 arcsec/px (7.52 µm pixels, 2×2 binning, 3047 mm focal length).

Raw FITS frames are not included in the repo. The calibrated outputs (`data/Crab_*_cal.fits`) are checked in and are sufficient to run `Project_E_NRE.py` and `RGB_Stack.py` directly.

---

## Notes

- The telescope pointing had a measurable offset from the header RA/Dec. The pipeline solves for this offset using matched star counts and corrects it before catalog cross-matching.
- The Crab core is masked during star detection to prevent nebular emission from being cataloged as stars.
- Smoothed-image noise (not raw sky noise) is used for the SNR mask in the ratio maps — raw sky noise is ~20× too tight after Gaussian smoothing and would reject most valid nebula pixels.
