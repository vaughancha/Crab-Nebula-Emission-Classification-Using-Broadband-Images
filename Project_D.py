import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# set paths
data_dir   = "/home/connor/ASTR_470/HW8"
target_dir = f"{data_dir}/Leuschner Data/Targets"

BANDS = {
    'B': f"{target_dir}/Crab_B_60s_03_17_26.fit",
    'V': f"{target_dir}/Crab_V_60s_3_17_26.fit",
    'R': f"{target_dir}/Crab_R_60s_03_17_60_NO_IR.fit",
}
FLATS = {
    'B': f"{data_dir}/norm_flat_B.fits",
    'V': f"{data_dir}/norm_flat_V.fits",
    'R': f"{data_dir}/norm_flat_R.fits",
}

# read from header: XPIXSZ=7.52µm, XBINNING=2, FOCALLEN=3047mm 
# using plate scale = 206265*pixel pitch (mm) /focal length (mm) ~1.018 arcsec/px
# from: https://clarkvision.com/articles/platescale/
PLATE_SCALE  = 7.52e-3 * 2 / 3047.0 * 206265.0
FIELD_CENTER = SkyCoord("05 34 32", "+22 01 06", unit=(u.hourangle, u.deg))

#0. flat correction
# CALSTAT=D in all headers, so MaxIm DL already subtracted darks only flat divide necessary
#from: http://spiff.rit.edu/classes/phys445/lectures/darkflat/darkflat.html , https://heasarc.gsfc.nasa.gov/docs/heasarc/fits/java/v1.0/javadoc/nom/tam/fits/header/extra/MaxImDLExt.html#CALSTAT

print("flat-correcting frames...")
def flat_correct(band):
	#[band] is B,V or R files respectively
    raw  = fits.getdata(BANDS[band]).astype(float)
    flat = fits.getdata(FLATS[band]).astype(float)
	#corrected=raw/flat, np.where to prevent division by 0
    img  = raw / np.where(flat > 0, flat, 1.0)
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

imgs = {b: flat_correct(b) for b in ('B', 'V', 'R')}
print("Done")

# 1. align B&R to V so pixel coordinates correspond to same patch of sky
# V is the reference frame. upsample_factor=100 gives 0.01-pixel precision
#phase_cross_correlation uses standard fourier based method for translational shift between images
#from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html , https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation
print("Aligning frames to V (reference)...")

#img is image needing movement, ref is fixd reference image
def align_to_ref(img, ref):
	#compares ref & img with phase cross corr. returning 3 values. shift vector stored in shift_yx
    shift_yx, _, _ = phase_cross_correlation(ref, img, upsample_factor=100)
	#moves img by shift_yx with spline interpolation, returning shifted image and shift vector
    return nd_shift(img, shift_yx), shift_yx
#shifted image stored imgs['N'], shift vector stored in shift_N
imgs['B'], shift_B = align_to_ref(imgs['B'], imgs['V'])
imgs['R'], shift_R = align_to_ref(imgs['R'], imgs['V'])
print(f"  B shift: dy={shift_B[0]:.2f}  dx={shift_B[1]:.2f} px")
print(f"  R shift: dy={shift_R[0]:.2f}  dx={shift_R[1]:.2f} px")

# 2a. detect stars in V frame
# mask Crab Nebula core so DAOStarFinder only picks up real field stars
print("Detecting stars in V frame...")
#declare reference image and dimensions
ref = imgs['V']
ny, nx = ref.shape
#creates coordinate arrays covering each pixel in image. yy is row index of each pixel. ogrid makes them broadcast compatible without creating full 2D grid(saves memory)
yy, xx = np.ogrid[:ny, :nx]
#mask true pixels to close to nebula center, to avoid analysis on distant stars
crab_mask = ((xx - nx/2)**2 + (yy - ny/2)**2) < 500**2   #500px ≈ 8.5 arcmin radius

#computes background statistics using pixels outside the crab mask. ~crab_mask means "not crab_mask" , sigma clipping rejects pixels iteratively, more than 3 sigma from median. This removes stars and hot pixels from the background estimate. returns mean(null), median, and std. From: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html
_, bg_med, bg_std = sigma_clipped_stats(ref[~crab_mask], sigma=3.0)
#replaces every pixel inside crab mask with background median value. FIlls nebula region with flat backrgound, so DAOStarFinder wont mistake nebula emission for stars. 
ref_masked = np.where(crab_mask, bg_med, ref)
#DAOStarFinder convolves image with Gaussian kernal of given fwhm, finding peaks above the threshold. From: https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html
finder  = DAOStarFinder(fwhm=5.0, threshold=8.0 * bg_std)
sources = finder(ref_masked - bg_med)
#sorts detected sources by peak pixel value, reverse to put bright stars first for later calibration
sources.sort('peak')
sources.reverse()

#removes any detected sources within 60 px of image edge.Unreliable flux measurments taken at edges, where partially cutoff aperatures reside. 
margin = 60
keep = (
    (sources['xcentroid'] > margin) &
    (sources['xcentroid'] < ref.shape[1] - margin) &
    (sources['ycentroid'] > margin) &
    (sources['ycentroid'] < ref.shape[0] - margin)
)
sources = sources[keep]
print(f"{len(sources)} sources after edge + nebula mask")

#2b. aperture photometry on all three aligned frames
#builds (N,2) array of [x,y] pixl coordinates for all detected stars, one row per source. From: https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html
position = np.column_stack([sources['xcentroid'], sources['ycentroid']])
#radii used for aperture photometry
AP_R, IN, OUT = 10, 15, 25   # pixels

#loop over B,V,R, img is corresponding calibrated image array
flux_inst = {}
for band, img in imgs.items():
	#puts circular aperture of radius 10px and sky anulus 15-25px radius at all star positions. From: https://photutils.readthedocs.io/en/stable/api/photutils.aperture.CircularAperture.html , https://photutils.readthedocs.io/en/stable/api/photutils.aperture.CircularAnnulus.html
    ap   = CircularAperture(positions, r=AP_R)
    sky  = CircularAnnulus(positions, r_in=IN, r_out=OUT)
	#sums pixel values inside both apertutes for every star. returns table with aperture_sum_0 is total xlux inside star aperture, sum_1 is total flux inside the sky anulus. From: https://photutils.readthedocs.io/en/stable/user_guide/aperture.html
    phot = aperture_photometry(img, [ap, sky])
	#divides total sky annulus flux by its area in px to get mean sky background/px
    sky_mean          = phot['aperture_sum_1'] / sky.area
	#subtracts sky contribution from star aperture, giving instrumental flux by subtracting the background/px back to area of star aperture from raw aperture sum
    flux_inst[band]   = np.array(phot['aperture_sum_0'] - sky_mean * ap.area)

#2c. APASS query
print("Querying APASS catalog...")
#vizier is python interface to the CDS Vizier database. columns=['*'] requests all available colums. From: https://astroquery.readthedocs.io/en/stable/vizier/vizier.html
v = Vizier(columns=['*'], row_limit=2000)
#query_region downloads all catalog stars within 40 arcmin of FIELD_CENTER. 
result = v.query_region(FIELD_CENTER, radius=40*u.arcmin, catalog='II/336/apass9')
if not result:
    raise RuntimeError("query fail")
cat = result[0]

#searches returned column  names for r band column dynamically. next() walks generator and returns first match or none. 
r_col = next((c for c in cat.colnames if c.lower().startswith("r") and "mag" in c.lower()), None)
if r_col is None:
    raise RuntimeError(f"No R band column found, only available are: {cat.colnames}")
#maps the internal band letters to APASS column names
BAND_COL = {'B': 'Bmag', 'V': 'Vmag', 'R': r_col}
print(f"  Using APASS columns: B={BAND_COL['B']}, V={BAND_COL['V']}, R={BAND_COL['R']}")

#accounts for missing band data across stars, overwrites table keeping only fully-measured stars
complete = np.isfinite(cat['Bmag']) & np.isfinite(cat['Vmag']) & np.isfinite(cat[r_col])
cat = cat[complete]
print(f"{len(cat)}- APASS stars with complete BVr photometry")

#2d. find pointing offset then match to catalog
# Orientation confirmed N-up E-left. Header RA/Dec has a pointing error —
# we solve for it by brute-force maximising source↔catalog position matches.
print("Solving pointing offset...")
#converts plate scale to degrees/px , declination correction factor
ps_deg  = PLATE_SCALE / 3600.0
cos_dec = np.cos(np.radians(FIELD_CENTER.dec.deg))

#pulls RA&Dec columns from APASS into plain numpy arrays
ra_cat_arr  = np.array(cat['RAJ2000'])
dec_cat_arr = np.array(cat['DEJ2000'])
#projects catalog stars RA/dec into predicted pixel position
x_cat_nom   = nx/2 - (ra_cat_arr  - FIELD_CENTER.ra.deg)  * cos_dec / ps_deg
y_cat_nom   = ny/2 + (dec_cat_arr - FIELD_CENTER.dec.deg) / ps_deg

#builds kernel density tree from detected source positions, letting count_matches find the nearest detected source to given point in O(logN) time instead of O(N) detected sources should never change.From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
src_xy = np.column_stack([sources['xcentroid'], sources['ycentroid']])
src_tree = KDTree(src_xy)

#shifts the entire set of catalog predicted positions by (dx, dy) pixels, filters out any that fall outside the image frame, then queries the KDTree for the nearest detected source to each shifted catalog position. Returns how many catalog stars landed within 15 pixels of a real detected source. This count is the score,the correct offset maximizes it. From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
def count_matches(dx, dy, tol=15):
    x_try = x_cat_nom + dx
    y_try = y_cat_nom + dy
    in_f  = (x_try > 0) & (x_try < nx) & (y_try > 0) & (y_try < ny)
    if in_f.sum() == 0:
        return 0
    dist, _ = src_tree.query(np.column_stack([x_try[in_f], y_try[in_f]]))
    return int((dist < tol).sum())

# coarse search ±300 px in 10-px steps
best_n, best_dx, best_dy = 0, 0, 0
for dx in range(-300, 301, 10):
    for dy in range(-300, 301, 10):
        n = count_matches(dx, dy)
        if n > best_n:
            best_n, best_dx, best_dy = n, dx, dy

# fine search ±15 px around coarse best
for dx in range(best_dx - 15, best_dx + 16):
    for dy in range(best_dy - 15, best_dy + 16):
        n = count_matches(dx, dy)
        if n > best_n:
            best_n, best_dx, best_dy = n, dx, dy

print(f"  Best offset: dx={best_dx}px ({best_dx*PLATE_SCALE:.1f}\")  "
      f"dy={best_dy}px ({best_dy*PLATE_SCALE:.1f}\")  matches={best_n}")

# apply offset and match with tight radius
#gives each catalog star a corrected sky coordinate that accounts for the telescope's actual pointing rather than its nominal header position.
x_cat_corr = x_cat_nom + best_dx
y_cat_corr = y_cat_nom + best_dy
ra_cat_c   = FIELD_CENTER.ra.deg  - (x_cat_corr - nx/2) * ps_deg / cos_dec
dec_cat_c  = FIELD_CENTER.dec.deg + (ny/2 - y_cat_corr) * ps_deg

#compute RA&Dec of each detected source from pixel centroids
ra_src  = FIELD_CENTER.ra.deg  - (sources['xcentroid'] - nx/2) * ps_deg / cos_dec
dec_src = FIELD_CENTER.dec.deg + (ny/2 - sources['ycentroid']) * ps_deg

#Wraps both sets of coordinates in astropy's SkyCoord object and calls match_to_catalog_sky, finding the nearest catalog star for each detected source using great circle angular separation. Returns idx and sep (catalog row and angular distance of each match).From: https://docs.astropy.org/en/stable/coordinates/matchsep.html
src_coords = SkyCoord(ra=ra_src   * u.deg, dec=dec_src  * u.deg)
cat_coords = SkyCoord(ra=ra_cat_c * u.deg, dec=dec_cat_c * u.deg)
idx, sep, _ = src_coords.match_to_catalog_sky(cat_coords)

#Accepts matches when source and catalog star are within 10 arcsec of eachother only. 
MATCH_ARCSEC = 10.0
match = sep.arcsec < MATCH_ARCSEC
print(f"  {match.sum()} sources matched within {MATCH_ARCSEC}\"")
if match.sum() > 0:
    print(f"  Sep stats: min={sep.arcsec.min():.1f}\"  median={np.median(sep.arcsec[match]):.1f}\" (matched only)")

#2e. zeropoints (median over matched stars)
zp = {}
print()
#iterates over each band to compute a photometric zeropoint from cross-matched APASS stars
for band in ('B', 'V', 'R'):
	#extracts instrumental flux only for stars successfully matched to the catalog
    fl   = flux_inst[band][match]
	#extracts corresponding APASS catalog magnitude for each matched star
    cmag = np.array(cat[BAND_COL[band]][idx[match]], dtype=float)
	#filters to positive flux, finite catalog mag, 8-18 mag range (avoids saturation and sky-noise floor)
    good = (fl > 0) & np.isfinite(cmag) & (cmag < 18.0) & (cmag > 8.0)
	#requires at least 3 calibration stars for a statistically meaningful zeropoint
    if good.sum() < 3:
        raise RuntimeError(f"Only {good.sum()} usable calibration stars in {band} — check orientation or match radius")
	#instrumental magnitude via Pogson formula: m_inst = -2.5 * log10(flux)
    inst    = -2.5 * np.log10(fl[good])
	#per-star zeropoint: difference between catalog and instrumental magnitude
    zp_vals = cmag[good] - inst
	#sigma_clip (astropy.stats) iteratively masks values more than N-sigma from the median, rejecting outlier zeropoints from bad cross-matches or variable stars
    clipped = sigma_clip(zp_vals, sigma=2.0, maxiters=5)
	#np.ma.median is NumPy's masked-array median, which ignores sigma-clipped (masked) entries so outliers don't bias the final zeropoint
    zp[band] = float(np.ma.median(clipped))
	#counts unclipped stars for diagnostic reporting
    n_used   = int((~clipped.mask).sum())
    print(f"  ZP_{band} = {zp[band]:.3f}  (N={n_used}/{good.sum()} stars after sigma-clip, scatter={float(clipped.std()):.3f} mag)")

# Step 2f: apply zeropoints and save calibrated frames
#scales each image to calibrated flux units: multiplying by 10^(-ZP/2.5) converts ADU onto the APASS magnitude system
imgs_cal = {b: imgs[b] * 10**(-zp[b] / 2.5) for b in ('B', 'V', 'R')}

print()
#iterates over calibrated images to write each band as a FITS file with updated header
for band, img in imgs_cal.items():
	#reads original header to preserve telescope and observing metadata
    hdr = fits.getheader(BANDS[band])
	#adds calibration keywords so downstream tools know the zeropoint, alignment reference, and pixel scale
    hdr['ZP']    = (round(zp[band], 4), 'APASS photometric zeropoint')
    hdr['ALIGN']  = ('V',      'reference band for alignment')
    hdr['PSCALE'] = (PLATE_SCALE, 'plate scale arcsec/px')
    out = f"{data_dir}/Crab_{band}_cal.fits"
	#writes 32-bit float FITS, overwriting any previous calibrated output for this band
    fits.writeto(out, img.astype(np.float32), hdr, overwrite=True)
    print(f"  Saved {out}")

# Diagnostic: instrumental vs catalog magnitude scatter
#creates 3-panel figure, one per band, to visually verify zeropoint calibration quality
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
#loops over bands to fill each subplot with a calibration scatter plot
for ax, band in zip(axes, ('B', 'V', 'R')):
	#retrieves instrumental flux for matched stars in this band
    fl   = flux_inst[band][match]
	#retrieves corresponding APASS catalog magnitudes for matched stars
    cmag = np.array(cat[BAND_COL[band]][idx[match]], dtype=float)
	#filters to positive flux, finite catalog magnitude, brighter than 18 mag
    good = (fl > 0) & np.isfinite(cmag) & (cmag < 18.0)
	#converts flux to magnitude and applies zeropoint to produce calibrated magnitude estimate
    inst_zp = -2.5 * np.log10(fl[good]) + zp[band]
	#determines catalog magnitude range for drawing the 1:1 reference line
    lo, hi  = cmag[good].min(), cmag[good].max()
	#plots calibrated instrumental vs APASS catalog magnitude to check agreement
    ax.scatter(cmag[good], inst_zp, s=12, alpha=0.7)
	#draws 1:1 line showing where perfect calibration would place all points
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_xlabel(f"APASS {band} (mag)")
    ax.set_ylabel("Instrumental + ZP (mag)")
    ax.set_title(f"{band}  ZP={zp[band]:.3f}  N={good.sum()}")
plt.suptitle("Flux calibration: instrumental vs APASS catalog")
plt.tight_layout()
plt.savefig(f"{data_dir}/calibration_check.png", dpi=120, bbox_inches='tight')
plt.show()
print(f"Saved calibration_check.png")
print("Done. Calibrated frames: Crab_B_cal.fits, Crab_V_cal.fits, Crab_R_cal.fits")

#3.Ratio Maps  
print("Step 3: Ratio maps")
#imports spatial filtering tools: uniform_filter for coarse centroid finding, gaussian_filter for SNR smoothing, binary_dilation for star mask expansion
from scipy.ndimage import uniform_filter, gaussian_filter, binary_dilation

#reads calibrated FITS images back from disk as float arrays for ratio analysis
B = fits.getdata(f"{data_dir}/Crab_B_cal.fits").astype(float)
V = fits.getdata(f"{data_dir}/Crab_V_cal.fits").astype(float)
R = fits.getdata(f"{data_dir}/Crab_R_cal.fits").astype(float)
#stores image dimensions for coordinate array construction and masking
ny_f, nx_f = V.shape

#find Crab centroid: smoothed peak near image center
#initial estimate of Crab center as image center, assuming pointing was approximately correct
cy_img, cx_img = ny_f // 2, nx_f // 2
#box-smooths 400x400px central region at size=20 to suppress noise before peak finding
V_coarse = uniform_filter(V[cy_img-200:cy_img+200, cx_img-200:cx_img+200], size=20)
#np.argmax returns the flat (1D) index of the peak flux pixel; np.unravel_index converts that to a (row,col) pair within the cutout shape
pk = np.unravel_index(np.argmax(V_coarse), V_coarse.shape)
#converts cutout-relative peak coords back to full-image pixel coordinates
crab_row = cy_img - 200 + pk[0]
crab_col = cx_img - 200 + pk[1]
print(f"Crab centroid: row={crab_row}  col={crab_col}")

#sky: sigma-clipped stats in an annulus well away from the Crab
#np.mgrid is NumPy's dense mesh-grid generator, returning fully-evaluated 2D arrays covering every pixel; needed here because both axes are used simultaneously in the distance formula
yy_f, xx_f = np.mgrid[:ny_f, :nx_f]
#Euclidean distance in pixels from every image pixel to the Crab centroid
r_from_crab = np.sqrt((xx_f - crab_col)**2 + (yy_f - crab_row)**2)
#selects annular sky region 600-1200px from Crab, far enough to avoid nebular emission
sky_ann = (r_from_crab > 600) & (r_from_crab < 1200)

#sigma-clipped background stats per band from sky annulus; std stored as per-band noise estimate for later masking
_, sky_B, std_B = sigma_clipped_stats(B[sky_ann], sigma=3.0)
_, sky_V, std_V = sigma_clipped_stats(V[sky_ann], sigma=3.0)
_, sky_R, std_R = sigma_clipped_stats(R[sky_ann], sigma=3.0)
print(f"  Sky — B:{sky_B:.3e}  V:{sky_V:.3e}  R:{sky_R:.3e}")

#subtracts median sky level from each band to isolate true nebular flux
B_s = B - sky_B
V_s = V - sky_V
R_s = R - sky_R

#Gaussian smooth,using σ=3 px for better SNR on faint Crab emission
SMOOTH = 3.0
print("Smoothing images...")
B_sm = gaussian_filter(B_s, sigma=SMOOTH)
V_sm = gaussian_filter(V_s, sigma=SMOOTH)
R_sm = gaussian_filter(R_s, sigma=SMOOTH)

# Measure noise *in the smoothed images* from the sky annulus.
# Raw std drops by ~sqrt(4π σ²) after Gaussian smoothing; using raw std as the
# threshold on smoothed images would require a ~20σ detection — far too tight.
_, _, std_B_sm = sigma_clipped_stats(B_sm[sky_ann], sigma=3.0)
_, _, std_V_sm = sigma_clipped_stats(V_sm[sky_ann], sigma=3.0)
_, _, std_R_sm = sigma_clipped_stats(R_sm[sky_ann], sigma=3.0)
print(f"  Smoothed noise — B:{std_B_sm:.3e}  V:{std_V_sm:.3e}  R:{std_R_sm:.3e}")

#mask point sources via sharpness filter (uses raw std_V on unsmoothed V)
#broad Gaussian captures diffuse large-scale structure (nebula + sky gradient)
V_broad = gaussian_filter(V, sigma=5.0)
#subtracts broad from raw to isolate point-source-like high-frequency residuals
V_sharp = V - V_broad
#flags pixels where sharpness spike exceeds 5σ of raw sky noise as point sources
star_pix = V_sharp > 5.0 * std_V # std_V is raw image sky noise
#builds a circular disk kernel of 20px radius for morphological dilation
R_STAR = 20
d = 2 * R_STAR + 1
#np.ogrid is NumPy's open mesh-grid generator, returning broadcast-compatible 1D arrays instead of full 2D grids, which is sufficient and memory-efficient for evaluating the disk equation
yy_s, xx_s = np.ogrid[:d, :d]
disk = (xx_s - R_STAR)**2 + (yy_s - R_STAR)**2 <= R_STAR**2
#expands each flagged star pixel by 20px radius to mask diffraction halos and flux bleeding
star_mask = binary_dilation(star_pix, structure=disk)
print(f"  Point source mask: {star_pix.sum()} hot px → {star_mask.sum()} px after dilation")

#SNR + Crab-region + star mask
#limits analysis to pixels within 700px of Crab centroid, excluding unrelated background field
cra_region = r_from_crab < 700
# Require 5σ in V (the deepest band) and 3σ in B and R independently.
# Triple-band coincidence drives spurious-pixel rate to (3e-5)^3 ≈ 0 per circle.
mask = (V_sm > 5.0 * std_V_sm) & (B_sm > 3.0 * std_B_sm) & (R_sm > 3.0 * std_R_sm) & crab_region & ~star_mask
print(f"  Valid pixels: {mask.sum()}")

#log ratio maps
#computes log10 flux ratio only where mask is valid; suppresses divide-by-zero warnings since invalid pixels return NaN
def log_ratio(num, den, mask):
    #np.errstate is a NumPy context manager that temporarily suppresses floating-point warnings; needed because dividing by near-zero pixels raises warnings that are intentionally handled by the np.where below
    with np.errstate(divide='ignore', invalid='ignore'):
        #np.where(condition, x, y) returns x where condition is True and y elsewhere, selecting the valid log ratio or NaN per pixel
        return np.where(mask, np.log10(num / den), np.nan)

#R/B ratio traces Hα+[NII] emission relative to continuum; V/B traces [OIII]+Hβ
RB = log_ratio(R_sm, B_sm, mask)
VB = log_ratio(V_sm,  B_sm, mask)
print(f"  Finite ratio pixels — R/B: {np.isfinite(RB).sum()}   V/B: {np.isfinite(VB).sum()}")

#plot full image coordinates
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, ratio, title in zip(axes,
                             [RB, VB],
                             ['log₁₀(R/B) — Hα+[NII] tracer',
                              'log₁₀(V/B) — [OIII]+Hβ tracer']):
    valid = ratio[np.isfinite(ratio)]
    lo, hi = np.percentile(valid, [2, 98])
    im = ax.imshow(ratio, origin='lower', cmap='inferno', vmin=lo, vmax=hi)
    plt.colorbar(im, ax=ax, label='log₁₀ flux ratio')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x (px)');  ax.set_ylabel('y (px)')

plt.suptitle('Crab Nebula — ratio maps  (sky-subtracted, 3σ+star mask, log scale)',
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{data_dir}/ratio_maps.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved ratio_maps.png")


#4.color-color diagram — log(R/B) vs log(V/B) per pixel
print("Step 4: Color-color diagram")

#selects pixels where both ratio maps have finite values (passed all masks)
valid_2d = np.isfinite(RB) & np.isfinite(VB)
#extracts 1D arrays of valid pixel ratio values for scatter plotting
rb_vals  = RB[valid_2d]
vb_vals  = VB[valid_2d]
#converts pixel distance from Crab centroid to arcmin for colorbar labeling
dist_am  = r_from_crab[valid_2d] * PLATE_SCALE / 60.0   #arcsec to arcmin
print(f"  {len(rb_vals)} pixels")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#left: 2D density (hexbin) — reveals clusters without overplotting
ax = axes[0]
#aggregates pixels into hexagonal bins to show density structure without overplotting
hb = ax.hexbin(rb_vals, vb_vals, gridsize=50, cmap='inferno', mincnt=1)
plt.colorbar(hb, ax=ax, label='pixel count')
ax.set_xlabel('log₁₀(R/B)')
ax.set_ylabel('log₁₀(V/B)')
ax.set_title('Pixel density')

#right: scatter colored by distance from Crab centroid
# Inner pixels (blue) = PWN-dominated; outer pixels (yellow) = filament shell
ax = axes[1]
#colors each pixel by its angular distance from Crab centroid to reveal radial emission structure
sc = ax.scatter(rb_vals, vb_vals, c=dist_am, cmap='plasma_r',
                s=1, alpha=0.4, vmin=0, vmax=dist_am.max())
plt.colorbar(sc, ax=ax, label="distance from centroid (arcmin)")
ax.set_xlabel('log₁₀(R/B)')
ax.set_ylabel('log₁₀(V/B)')
ax.set_title('Colored by radial distance')

plt.suptitle('Crab Nebula — color-color diagram  (each point = 1 smoothed pixel)',
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{data_dir}/color_color.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved color_color.png")

#5. Four classification methods compared on the same spatial image
print("Step 5: Classification methods")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

X = np.column_stack([rb_vals, vb_vals])   # shape (N_valid, 2) from Step 4

#Method 1: histogram minimum in 1D log(R/B)
#np.histogram bins the 1D log(R/B) values into 80 equal-width bins, returning counts per bin (hist) and the bin-edge values (edges), used to locate the valley between the two emission populations
hist, edges = np.histogram(rb_vals, bins=80)
#bin centers for indexing and plotting
centers = (edges[:-1] + edges[1:]) / 2
#gaussian_filter1d (scipy.ndimage) convolves a 1D array with a Gaussian kernel of given σ, smoothing shot noise while preserving the broad peak structure needed for reliable valley detection
hist_sm  = gaussian_filter1d(hist.astype(float), sigma=2)
#find_peaks (scipy.signal) returns indices of local maxima in a 1D array, used here to locate the two histogram peaks that bracket the synchrotron/filament classification valley
peaks, _ = find_peaks(hist_sm)
if len(peaks) >= 2:
	#np.argsort returns the indices that would sort the peak heights ascending; [-2:] selects the two tallest peaks
    top2    = peaks[np.argsort(hist_sm[peaks])[-2:]]
    p1, p2  = int(min(top2)), int(max(top2))
	#np.argmin returns the index of the minimum value in the slice between the two peaks, identifying the valley used as the classification threshold
    valley  = p1 + int(np.argmin(hist_sm[p1:p2+1]))
    split_hist = float(centers[valley])
else:
    split_hist = float(np.median(rb_vals))   # fallback if unimodal
print(f"  Method 1 (hist min):  log(R/B) = {split_hist:.3f}")

#Method 2: k-means (k=2) in 2D color-color space
#fits 2-cluster k-means to (log R/B, log V/B) feature space; n_init=10 for stable solution
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = km.fit_predict(X)
#normalizes label assignment so label 0 always corresponds to lower R/B (synchrotron-like)
if km.cluster_centers_[0, 0] > km.cluster_centers_[1, 0]:
    km_labels = 1 - km_labels          # label 0 = lower R/B = synchrotron-like
c0_km, c1_km = km.cluster_centers_[0], km.cluster_centers_[1]
print(f"  Method 2 (k-means):   synch center=({c0_km[0]:.2f},{c0_km[1]:.2f})  "
      f"fil center=({c1_km[0]:.2f},{c1_km[1]:.2f})")

#Method 3: GMM (2 components) in 2D from https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
#GMM fits elliptical Gaussians in 2D, allowing flexible cluster shapes unlike k-means spheres
gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
gmm.fit(X)
#assigns each pixel to its most probable Gaussian component
gmm_labels = gmm.predict(X)
#normalizes so label 0 = lower R/B (synchrotron-like), matching k-means convention
if gmm.means_[0, 0] > gmm.means_[1, 0]:
    gmm_labels = 1 - gmm_labels

#map cluster labels to full-image boolean masks
#returns (synchrotron_mask, filament_mask) from a 1D log(R/B) threshold applied within the valid mask
def threshold_masks(thr):
    return mask & (RB < thr), mask & (RB >= thr)

#maps 1D cluster label array back to 2D image boolean masks; fills invalid pixels with -1
def label_masks(labels):
    #np.full creates an array of a given shape filled with a constant value; used here to initialize every pixel as -1 (unclassified) before writing valid labels
    lbl = np.full((ny_f, nx_f), -1, dtype=int)
	#writes cluster labels into the 2D array at the valid pixel positions only
    lbl[valid_2d] = labels
    return (lbl == 0), (lbl == 1)

methods = [
    ('Histogram minimum',
     threshold_masks(split_hist),
     f'1D valley at log(R/B) = {split_hist:.2f}'),
    ('k-means  (2D)',
     label_masks(km_labels),
     '2D cluster assignment in color-color space'),
    ('GMM  (2D)',
     label_masks(gmm_labels),
     '2-component Gaussian mixture in color-color space'),
]

#crop centered on geometric mean of detected pixels
#np.where with a single argument returns a tuple of index arrays where the condition is True, giving separate row and col arrays for all valid pixels
ys_v, xs_v = np.where(mask)
#geometric center of valid pixel cloud used as crop center
center_row  = int(np.mean(ys_v));  center_col = int(np.mean(xs_v))
#half-width of 1800px crop centered on valid pixel cloud
PAD5 = 900
r0, r1 = center_row - PAD5, center_row + PAD5
c0, c1 = center_col - PAD5, center_col + PAD5
#np.s_ is NumPy's index expression builder that stores a slice object, letting the same 2D crop bounds be reused across every imshow call without recomputing
sl5    = np.s_[r0:r1, c0:c1]
V_bg   = V_s[sl5]
#1-99.5th percentile stretch for display contrast in the background V image
vmin_bg, vmax_bg = np.percentile(V_bg, [1, 99.5])
#extent preserves full-image pixel coordinate labels on cropped imshow subplots
ext    = [c0, c1, r0, r1]

#builds a custom colormap from transparent to a solid color at given alpha, so classification masks overlay without obscuring background
def solid_cmap(color, alpha=0.65):
    #converts named color to (R,G,B) tuple
    rc, gc, bc = mcolors.to_rgb(color)
    #linear ramp from fully transparent to solid color at given alpha
    cmap = mcolors.LinearSegmentedColormap.from_list(
        color, [(rc, gc, bc, 0.0), (rc, gc, bc, alpha)])
    #NaN pixels fully transparent so background image shows through
    cmap.set_bad(alpha=0.0)
    return cmap

#synchrotron-like regions shown in blue, filament-like in red
blue_cm = solid_cmap('royalblue')
red_cm  = solid_cmap('tomato')

#creates 3-panel figure for spatial classification overlay, one panel per method
fig, axes = plt.subplots(1, 3, figsize=(21, 8))
#iterates over each classification method to produce one spatial overlay subplot
for ax, (title, (pwn_m, fil_m), subtitle) in zip(axes.flat, methods):
    #plots sky-subtracted V image as grayscale background for spatial context
    ax.imshow(V_bg, origin='lower', cmap='gray',
              vmin=vmin_bg, vmax=vmax_bg, extent=ext)
    #np.where(condition, x, y) converts the boolean mask to 1.0 where True and NaN where False, making unclassified pixels transparent in imshow
    ax.imshow(np.where(pwn_m[sl5], 1.0, np.nan), origin='lower',
              cmap=blue_cm, vmin=0, vmax=1, extent=ext)
    #overlays filament-like pixels in red
    ax.imshow(np.where(fil_m[sl5], 1.0, np.nan), origin='lower',
              cmap=red_cm,  vmin=0, vmax=1, extent=ext)
    #mpatches.Patch (matplotlib.patches) creates a solid-color rectangle used purely as a legend proxy, associating each classification color with its label without needing a plotted artist
    legend = [mpatches.Patch(color='royalblue', alpha=0.65,
                              label=f'Synchrotron-like  ({pwn_m.sum()} px)'),
              mpatches.Patch(color='tomato',     alpha=0.65,
                              label=f'Filament-like  ({fil_m.sum()} px)')]
    ax.legend(handles=legend, loc='upper right', fontsize=7)
    ax.set_title(f'{title}\n{subtitle}', fontsize=9)
    ax.set_xlabel('x (px)', fontsize=8);  ax.set_ylabel('y (px)', fontsize=8)

plt.suptitle('Crab Nebula — emission classification: three methods', fontsize=13)
plt.tight_layout()
plt.savefig(f"{data_dir}/spatial_overlay.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Saved spatial_overlay.png")
