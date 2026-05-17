#Project_E_NRE.py - Neural Ratio Estimation for Crab Nebula emission classification.
#loads calibrated B/V/R frames from Project_D.py, builds a forward model that
#simulates log(R/V) for synchrotron PWN and ionized filament pixels,
#trains a binary MLP classifier, and produces a per-pixel P(filament) map.

import warnings
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter, uniform_filter, binary_dilation
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_dir = "/home/connor/git/Crab-Nebula-Emission-Classification-Using-Broadband-Images/data"
out_dir  = "/home/connor/git/Crab-Nebula-Emission-Classification-Using-Broadband-Images/output"

# load calibrated frames produced by Project_D
print("Loading calibrated frames...")
#fits.getdata reads pixel array out of a FITS file, .astype(float) converts to 64-bit floats for numerical stability
B = fits.getdata(f"{data_dir}/Crab_B_cal.fits").astype(float)
V = fits.getdata(f"{data_dir}/Crab_V_cal.fits").astype(float)
R = fits.getdata(f"{data_dir}/Crab_R_cal.fits").astype(float)
#stores image dimensions for coordinate array construction and masking
ny_f, nx_f = V.shape

#reads V header to get the plate scale that Project_D embedded during calibration
hdr = fits.getheader(f"{data_dir}/Crab_V_cal.fits")
PLATE_SCALE = float(hdr.get('PSCALE', 1.018))   # arcsec/px
print("Done")

#reproduce the exact same pixel mask as Project_D so both scripts analyze identical regions
print("Building pixel mask...")
#initial estimate of Crab center as image center, assuming pointing was approximately correct
cy_img, cx_img = ny_f // 2, nx_f // 2
#box-smooths 400x400px central region at size=20 to suppress noise before finding the Crab peak
V_coarse = uniform_filter(V[cy_img-200:cy_img+200, cx_img-200:cx_img+200], size=20)
#np.unravel_index converts flat argmax index to (row, col) within the cutout. From: https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
pk = np.unravel_index(np.argmax(V_coarse), V_coarse.shape)
#converts cutout-relative peak coords back to full image pixel coordinates
crab_row = cy_img - 200 + pk[0]
crab_col = cx_img - 200 + pk[1]

#np.mgrid is NumPy's dense mesh-grid generator, returning full 2D arrays covering every pixel. From: https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html
yy_f, xx_f = np.mgrid[:ny_f, :nx_f]
#Euclidean distance in pixels from every image pixel to the Crab centroid
r_from_crab = np.sqrt((xx_f - crab_col)**2 + (yy_f - crab_row)**2)
#sky annulus 600-1200px from Crab, far enough to avoid nebular emission
sky_ann = (r_from_crab > 600) & (r_from_crab < 1200)

#sigma-clipped background stats per band from sky annulus. std stored as per-band noise estimate for masking. From: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html
_, sky_B, std_B = sigma_clipped_stats(B[sky_ann], sigma=3.0)
_, sky_V, std_V = sigma_clipped_stats(V[sky_ann], sigma=3.0)
_, sky_R, std_R = sigma_clipped_stats(R[sky_ann], sigma=3.0)

#subtracts median sky level from each band to isolate true nebular flux
B_s = B - sky_B
V_s = V - sky_V
R_s = R - sky_R

#Gaussian smooth using σ=3px to improve SNR on faint Crab emission. From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
SMOOTH = 3.0
B_sm = gaussian_filter(B_s, sigma=SMOOTH)
V_sm = gaussian_filter(V_s, sigma=SMOOTH)
R_sm = gaussian_filter(R_s, sigma=SMOOTH)

#measure noise in the smoothed images from the sky annulus. Raw noise shrinks by ~sqrt(4πσ²) after smoothing so it can't be used as a threshold on smoothed images directly
_, _, std_B_sm = sigma_clipped_stats(B_sm[sky_ann], sigma=3.0)
_, _, std_V_sm = sigma_clipped_stats(V_sm[sky_ann], sigma=3.0)
_, _, std_R_sm = sigma_clipped_stats(R_sm[sky_ann], sigma=3.0)

#broad Gaussian captures large-scale structure. Subtracting from raw isolates sharp point-source residuals
V_broad = gaussian_filter(V, sigma=5.0)
V_sharp = V - V_broad
#flags pixels where sharpness spike exceeds 5σ of raw sky noise as point sources
star_pix = V_sharp > 5.0 * std_V
#35px dilation radius (increased from 20 in Project_D) to catch bright-star halos more aggressively
R_STAR = 35
d = 2 * R_STAR + 1
#np.ogrid is NumPy's open mesh-grid generator, returning broadcast-compatible 1D arrays instead of full 2D grids - sufficient and memory-efficient for evaluating the disk equation. From: https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html
yy_s, xx_s = np.ogrid[:d, :d]
disk = (xx_s - R_STAR)**2 + (yy_s - R_STAR)**2 <= R_STAR**2
#expands each flagged star pixel by R_STAR radius to mask diffraction halos and flux bleeding. From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html
star_mask = binary_dilation(star_pix, structure=disk)

#limits analysis to pixels within 550px of Crab centroid (tightened from 700 in Project_D to reduce outer noise pixels)
crab_region = r_from_crab < 550
#requires 5σ in V and 3σ in B and R. triple-band coincidence drives spurious-pixel rate to near zero
mask = (
    (V_sm > 5.0 * std_V_sm) &
    (B_sm > 3.0 * std_B_sm) &
    (R_sm > 3.0 * std_R_sm) &
    crab_region & ~star_mask
)
print(f"  Valid pixels: {mask.sum()}")

#computes log10 flux ratios only where mask is valid. np.errstate suppresses divide-by-zero warnings since invalid pixels return NaN intentionally
with np.errstate(divide='ignore', invalid='ignore'):
    RB = np.where(mask, np.log10(R_sm / B_sm), np.nan)
    VB = np.where(mask, np.log10(V_sm / B_sm), np.nan)

#selects pixels where both ratio maps have finite values and extracts 1D arrays for the classifier
valid_2d = np.isfinite(RB) & np.isfinite(VB)
rb_vals  = RB[valid_2d]
vb_vals  = VB[valid_2d]
print(f"  {len(rb_vals)} pixels in color-color space")
print("Done")

#crop centered on geometric mean of detected pixels, matching Project_D Step 5 framing
ys_v, xs_v = np.where(mask)
center_row  = int(np.mean(ys_v));  center_col = int(np.mean(xs_v))
PAD = 900
r0, r1 = center_row - PAD, center_row + PAD
c0, c1 = center_col - PAD, center_col + PAD
#np.s_ is NumPy's index expression builder that stores a slice object, letting the same 2D crop bounds be reused across every imshow call without recomputing. From: https://numpy.org/doc/stable/reference/generated/numpy.s_.html
sl     = np.s_[r0:r1, c0:c1]
V_bg   = V_s[sl]
#1-99.5th percentile stretch for display contrast in the background V image
vmin_bg, vmax_bg = np.percentile(V_bg, [1, 99.5])
ext    = [c0, c1, r0, r1]

#Step 1: forward model
print("Step 1: Setting up forward model...")
#effective frequencies for B, V, R bands in Hz (c/λ)
nu_B = 3e8 / 440e-9
nu_V = 3e8 / 550e-9
nu_R = 3e8 / 640e-9

#Cardelli+1989 extinction coefficients A_λ/A_V: B=1.324, V=1.000, R=0.748. From: https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C
#we use log(R/V) as the single feature because ZP_B cancels out (shifts both log(R/B) and log(V/B) equally). combined ZP_R+ZP_V uncertainty is only ~0.034 mag
R_V = 3.1
#change in log(R/V) per unit E(B-V) derived from the Cardelli coefficients
dRV_per_EBV = R_V * (1.000 - 0.748) / 2.5   # Δlog(R/V) per E(B-V) ≈ +0.312

#simulate_class generates N synthetic log(R/V) measurements for one emission class, modelling real physical scatter so the classifier learns to separate them
def simulate_class(N, filament_dominated):
    #synchrotron spectral index drawn from Uniform[0.3, 0.8], covering the known PWN range
    alpha = np.random.uniform(0.3, 0.8, N)
    #synchrotron R/V color follows power law Fν ∝ ν^-α
    log_RV_synch = np.log10((nu_R / nu_V) ** (-alpha))

    #mixing fraction controls how much line emission is added. gap at 0.2-0.7 between classes avoids label overlap at the boundary
    f_mix = (np.random.uniform(0.7, 1.0, N) if filament_dominated
             else np.random.uniform(0.0, 0.2, N))

    #line emission: exponential prior for R (Hα+[NII] dominated), V-band lines ([OIII]+Hβ) correlated at 10-40% of R. np.random.exponential draws from Exp(scale). From: https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
    line_R = np.random.exponential(0.5, N)
    line_V = np.random.uniform(0.1, 0.4, N) * line_R

    #adds line contribution on top of synchrotron continuum color. Hα >> [OIII] so R/V is boosted for filament pixels
    log_RV = log_RV_synch + np.log10((1.0 + f_mix * line_R) / (1.0 + f_mix * line_V))

    #Crab Nebula E(B-V) ≈ 0.52 mag (Miller 1985) with ~0.10 mag spatial variation. np.clip prevents unphysical negative extinction
    E_BV    = np.clip(np.random.normal(0.52, 0.10, N), 0, None)
    log_RV += dRV_per_EBV * E_BV

    #ZP nuisance: global calibration offset negligible per-pixel after smoothing, modeled as 0.010 mag scatter
    log_RV += np.random.normal(0, 0.010, N)

    #photon + sky noise appropriate for Gaussian-smoothed (σ=3px) bright nebula pixels with SNR >> 5
    log_RV += np.random.normal(0, 0.020, N)

    #reshape to (N,1) column vector for sklearn/MLP compatibility
    return log_RV.reshape(-1, 1)

print("Done")

#Step 2: simulate training data
print("Step 2: Simulating training data...")
#seed for reproducibility
np.random.seed(42)
N_sim = 200_000

#generate N_sim synthetic observations for each class
X_synch = simulate_class(N_sim, filament_dominated=False)   # shape (N_sim, 1)
X_fil   = simulate_class(N_sim, filament_dominated=True)
print(f"  Synchrotron log(R/V): mean={X_synch.mean():.3f}  std={X_synch.std():.3f}")
print(f"  Filament    log(R/V): mean={X_fil.mean():.3f}  std={X_fil.std():.3f}")
print(f"  Class separation: {X_fil.mean() - X_synch.mean():.3f} mag  "
      f"(noise std ≈ 0.022 mag)")

#stack classes vertically into one training array and assign labels 0=synchrotron, 1=filament. np.vstack joins along axis 0, np.hstack along axis 1 (for 1D)
X_tr    = np.vstack([X_synch, X_fil])
y_tr    = np.hstack([np.zeros(N_sim), np.ones(N_sim)])

#shuffle training data so mini-batches see both classes equally
perm = np.random.permutation(len(y_tr))
X_tr, y_tr = X_tr[perm], y_tr[perm]
print("Done")

#Step 3: train MLP classifier
#custom numpy MLP with Adam optimizer - avoids sklearn warm_start optimizer-reset bug where sklearn resets Adam state between partial_fit calls, causing unstable convergence on small batches
class _MLP:

    def __init__(self, sizes, lr=3e-4, seed=42):
        #np.random.default_rng creates a Generator with the given seed for reproducible weight init. From: https://numpy.org/doc/stable/reference/random/generator.html
        rng = np.random.default_rng(seed)
        #He initialization: weights drawn from N(0, sqrt(2/n_in)) stabilizes ReLU gradients at start of training
        self.W  = [rng.normal(0, np.sqrt(2/ni), (ni, no))
                   for ni, no in zip(sizes[:-1], sizes[1:])]
        #biases initialized to zero, standard practice
        self.b  = [np.zeros(no) for no in sizes[1:]]
        self.lr = lr
        #Adam hyperparameters β1=0.9, β2=0.999, ε=1e-8 (standard defaults). From: https://arxiv.org/abs/1412.6980
        self.b1, self.b2, self.eps_adam = 0.9, 0.999, 1e-8
        #first and second moment estimates for weights and biases, initialized to zero
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        #global step counter for Adam bias correction
        self.t  = 0

    def _fwd(self, X):
        #stores activations at each layer for use during backprop
        self._a = [X]
        #ReLU activation: np.maximum(0, z) clips negative pre-activations to zero
        for W, b in zip(self.W[:-1], self.b[:-1]):
            self._a.append(np.maximum(0, self._a[-1] @ W + b))
        #sigmoid output: clips z to [-500,500] to prevent overflow in exp
        z = self._a[-1] @ self.W[-1] + self.b[-1]
        return 1.0 / (1.0 + np.exp(-np.clip(z.ravel(), -500, 500)))

    @staticmethod
    def _bce(p, y, eps=1e-12):
        #binary cross-entropy loss. eps prevents log(0) from blowing up
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    def _adam(self, p, i, g):
        #updates first and second moment estimates for weight gradient p at layer i
        self.mW[i][:] = self.b1*self.mW[i] + (1-self.b1)*p
        self.vW[i][:] = self.b2*self.vW[i] + (1-self.b2)*p*p
        #bias correction terms account for the zero initialization at step 0
        c1 = 1 - self.b1**self.t;  c2 = 1 - self.b2**self.t
        #applies Adam weight update in-place to the weight matrix slice g
        g -= self.lr * (self.mW[i]/c1) / (np.sqrt(self.vW[i]/c2) + self.eps_adam)
        #same Adam update for biases using the sum of the gradient over the batch
        self.mb[i][:] = self.b1*self.mb[i] + (1-self.b1)*p.sum(0)
        self.vb[i][:] = self.b2*self.vb[i] + (1-self.b2)*p.sum(0)**2
        self.b[i] -= self.lr*(self.mb[i]/c1)/(np.sqrt(self.vb[i]/c2)+self.eps_adam)

    def _back(self, p, y):
        self.t += 1
        n  = len(y)
        #gradient of BCE w.r.t. sigmoid output is (p - y)/n for the output layer
        d  = ((p - y) / n).reshape(-1, 1)
        #compute output layer weight gradient first, then backprop through hidden layers
        gW = [self._a[-1].T @ d]
        #chain rule through ReLU: d @ W.T zeroes gradient where activation was ≤ 0 (dead neurons)
        d  = d @ self.W[-1].T * (self._a[-1] > 0)
        for i in range(len(self.W) - 2, -1, -1):
            gW.insert(0, self._a[i].T @ d)
            if i: d = d @ self.W[i].T * (self._a[i] > 0)
        #apply Adam update to each layer using the computed gradients
        for i, g in enumerate(gW):
            self._adam(g, i, self.W[i])

    def fit_epoch(self, X, y, bs=4096):
        #shuffle at start of each epoch for unbiased mini-batch gradient estimates
        idx = np.random.permutation(len(y))
        epoch_loss = []
        for s in range(0, len(y), bs):
            b  = idx[s:s+bs];  Xb, yb = X[b], y[b]
            p  = self._fwd(Xb)
            epoch_loss.append(self._bce(p, yb))
            self._back(p, yb)
        #returns mean loss over all mini-batches this epoch
        return float(np.mean(epoch_loss))

    def predict_proba(self, X):
        #returns (N, 2) array: column 0 is P(synchrotron), column 1 is P(filament)
        p = self._fwd(X)
        return np.column_stack([1 - p, p])

    def score(self, X, y):
        #fraction of correctly classified samples at the 0.5 decision boundary
        return float(np.mean((self._fwd(X) > 0.5) == y))

print("Step 3: Training MLP classifier...")
#StandardScaler subtracts mean and divides by std so the input feature lives on unit scale - important for stable Adam convergence. From: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler().fit(X_tr)
X_sc   = scaler.transform(X_tr)

#architecture: 1 input (log R/V) → 32 → 16 → 1 output (P(filament))
clf    = _MLP([1, 32, 16, 1], lr=3e-4)
losses = []

#early stopping: halt if loss improves by less than TOL over N_NO_CHANGE consecutive epochs
MAX_EPOCHS  = 300
N_NO_CHANGE = 30
TOL         = 1e-4

#live loss plot updates every epoch so you can watch training converge in real time
plt.ion()
fig_live, ax_live = plt.subplots(figsize=(9, 4))
fig_live.canvas.manager.set_window_title('MLP Training — live loss')

for ep in range(MAX_EPOCHS):
    loss = clf.fit_epoch(X_sc, y_tr)
    losses.append(loss)
    print(f"  Epoch {ep+1:3d},  loss = {loss:.6f}")

    #redraws the loss curve live after each epoch
    ax_live.clear()
    ax_live.plot(losses, color='darkorange', lw=1.5)
    ax_live.set_xlabel('Epoch')
    ax_live.set_ylabel('Log-loss')
    ax_live.set_title(f'Epoch {ep+1}   loss = {loss:.5f}')
    ax_live.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.pause(0.001)

    #checks if best loss in the last N_NO_CHANGE epochs improved by less than TOL vs the epoch before that window
    if ep >= N_NO_CHANGE and losses[-N_NO_CHANGE-1] - min(losses[-N_NO_CHANGE:]) < TOL:
        print(f"  Converged at epoch {ep+1}")
        break

plt.ioff()
plt.close(fig_live)
print(f"  Training accuracy: {clf.score(X_sc, y_tr):.3f}  ({len(losses)} epochs)")
print("Done")

#Step 4: apply classifier to real Crab pixels
print("Step 4: Classifying real Crab pixels...")
#log(R/V) = log(R/B) - log(V/B). ZP_B appears in both ratios and cancels in the difference
rv_vals = rb_vals - vb_vals
print(f"  Real data log(R/V): mean={rv_vals.mean():.3f}  std={rv_vals.std():.3f}  "
      f"range=[{rv_vals.min():.3f}, {rv_vals.max():.3f}]")
#reshape to (N,1) column vector matching the training data format
X_real    = rv_vals.reshape(-1, 1)
#classifier returns P(filament) for each valid pixel using scaled real data
p_fil     = clf.predict_proba(scaler.transform(X_real))[:, 1]
#np.full initializes the full-image probability map to NaN, then valid pixels are filled in
p_fil_img = np.full((ny_f, nx_f), np.nan)
p_fil_img[valid_2d] = p_fil

#Gaussian smooth the probability map to suppress pixel-to-pixel noise before display.
#replaces NaN with 0.5 (neutral) for the smoothing step so the filter doesn't bleed NaN across the border, then restores the mask
p_smooth = p_fil_img.copy()
p_smooth[~valid_2d] = 0.5
p_smooth = gaussian_filter(p_smooth, sigma=3.0)
p_smooth[~valid_2d] = np.nan

#counts pixels on each side of the 0.5 decision boundary
n_fil   = (p_fil > 0.5).sum()
n_synch = (p_fil <= 0.5).sum()
print(f"  P(filament) > 0.5: {n_fil:,} px   ≤ 0.5: {n_synch:,} px")
#saves raw probability map as float32 — P(filament) in [0,1] doesn't need float64 precision
np.save(f"{data_dir}/p_fil_img.npy", p_fil_img.astype(np.float32))
print("Done")

#Step 5: plot results
print("Step 5: Plotting...")
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

#left panel: P(filament) spatial map overlaid on V-band background
ax = axes[0]
ax.imshow(V_bg, origin='lower', cmap='gray', vmin=vmin_bg, vmax=vmax_bg, extent=ext)
im = ax.imshow(p_smooth[sl], origin='lower', cmap='RdBu_r',
               vmin=0, vmax=1, alpha=0.75, extent=ext)
plt.colorbar(im, ax=ax, label='P(filament)')
ax.set_title('NRE: P(filament) per pixel\nBlue = synchrotron PWN   Red = ionized filaments',
             fontsize=10)
ax.set_xlabel('x (px)');  ax.set_ylabel('y (px)')

#center panel: histogram of P(filament) values across all valid pixels
ax = axes[1]
ax.hist(p_fil, bins=60, color='steelblue', edgecolor='k', linewidth=0.3)
ax.axvline(0.5, color='red', linestyle='--', lw=1.5, label='P = 0.5 boundary')
ax.set_xlabel('P(filament)')
ax.set_ylabel('pixel count')
ax.set_title(f'Filament probability distribution\n'
             f'synch-like: {n_synch:,} px   filament-like: {n_fil:,} px')
ax.legend()

#right panel: training loss curve showing convergence history
ax = axes[2]
ax.plot(losses, color='darkorange', lw=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Training loss (log-loss)')
ax.set_title(f'MLP training loss curve\n{len(losses)} iterations, converged')
ax.grid(True, alpha=0.3)

plt.suptitle('Crab Nebula — NRE emission classification\n'
             'MLP trained on forward-simulated spectra (200 k per class)',
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{out_dir}/nre_classification.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved nre_classification.png")
