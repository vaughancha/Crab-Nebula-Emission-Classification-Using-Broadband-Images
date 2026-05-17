import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats

raw_dir = "/home/connor/ASTR_470/HW8"
out_dir = "/home/connor/git/Crab-Nebula-Emission-Classification-Using-Broadband-Images/output"

#Takes all flat frames under the "pattern" and stacks them to find median flat, accounting for random noise 
#Divide by mean so flat is centered on  1.0                                                                            
def normalized_flat(filter_letter, data_dir):                               
    pattern = f"{data_dir}/Calibration/Flats_*{filter_letter}.fit"
    #glob.glob searches filesystem for the filenames matching pattern string, returning a list               
    filenames = sorted(glob.glob(pattern))                                       
    images = [fits.getdata(fn).astype(float) for fn in filenames]    
    #np.array(images) converts list of 2D arrays into single 3D array (cube)            
    cube = np.array(images)
    #np.median(cube, axis=0) takes the median at each pixel postion across all flat frames in the stack
    median_flat = np.median(cube, axis=0)  
    #np.mean(median_flat) give single number, the average of the whole median flat, to be used for normalization                                     
    norm_flat = median_flat / np.mean(median_flat)                               
    hdr = fits.getheader(filenames[0])                                           
    out_name = f"norm_flat_{filter_letter}.fits" 
    #fits.writeto saves an array plus header to a new header                                 
    fits.writeto(out_name, norm_flat, hdr, overwrite=True)                       
    print(f"Saved {out_name}  (from {len(filenames)} flats)")                    
    return norm_flat, hdr

#Filters record different total counts, to stack them stretch is needed to remove sky background, then find the faint and bright end of signal (5% and 98% percentile here)
#rescaled to 0 to 1, three bands are in same array format
def stretch(arr):
    #sigma_clipped_stats(arr,sigma=3.0) computes mean, median, and std of array while rejecting pixels more than 3sigma away from median. All three are returned, here we only use median(bg)
    _,bg,_ = sigma_clipped_stats(arr,sigma=3.0)
    arr = arr - bg
    #np.percentile(arr, [5,98]) finds pixel value at 5th and 98th biightness percentile
    p1,p99 = np.percentile(arr,[5, 98])
    #np.clip(...,0,1) forces the values outside 0-1 to clamp ti 0 or 1
    return np.clip((arr - p1)/(p99 - p1), 0, 1)

#fits.getdata(filename) reads pixel array out of a fits file
r=fits.getdata(f"{raw_dir}/Leuschner Data/Targets/Crab_R_60s_03_17_60_NO_IR.fit").astype(float)
g=fits.getdata(f"{raw_dir}/Leuschner Data/Targets/Crab_V_60s_3_17_26.fit").astype(float)
b=fits.getdata(f"{raw_dir}/Leuschner Data/Targets/Crab_B_60s_03_17_26.fit").astype(float)

#apply flat correction
r= r / fits.getdata(f"{raw_dir}/norm_flat_R.fits")
g= g / fits.getdata(f"{raw_dir}/norm_flat_V.fits")
b= b / fits.getdata(f"{raw_dir}/norm_flat_B.fits")

#each stretched band is a 2D array (ny,nx), stack adds 3rd dimension in (ny,nx,3), to color array
#.85 tuning because bloue was overexposed
#nantonum makes nan values 0 so stack can process them
rgb = np.stack([stretch(np.nan_to_num(r)), stretch(np.nan_to_num(g)), stretch(np.nan_to_num(b))*.85],axis=-1)
fig, ax = plt.subplots(figsize=(10, 8))
#ax.imshow(rgb,origin='lower') displays (ny,nx,3) array as color image, origin at bottom left
#imshow  assumes rgb in that order because of array shape. 
ax.imshow(rgb, origin='lower')
ax.set_title("Crab Nebula — RGB")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
plt.tight_layout()
plt.savefig(f"{out_dir}/rgb_crab.png")
plt.show()
