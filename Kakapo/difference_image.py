import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import fftconvolve, convolve2d
from scipy.ndimage import center_of_mass, zoom, shift
from scipy.optimize import minimize

from tqdm import tqdm

def create_diff_image(tpf, num_cores = 1, plot = False):
    
    flux = tpf.flux.value
    flux_err = tpf.flux_err.value
    flux[np.isnan(flux)] = 0
    flux_err[np.isnan(flux_err)] = 1

    flux_cor = np.zeros_like(flux)
    
    # print(flux.shape)
    
    flux_err_cor = np.zeros_like(flux_err)
    # for i in tqdm(range(len(flux)), total=len(flux), desc='Shifting'):
    for i in range(len(flux)):
        
        # flux_cor[i] = register(flux[i], tpf.pos_corr1[i], tpf.pos_corr2[i])
        flux_cor[i] = shift(flux[i], (-tpf.pos_corr2[i], -tpf.pos_corr1[i]))
        # flux_err_cor[i] = shift(flux_err[i], (-tpf.pos_corr2[i], -tpf.pos_corr1[i]))
    
    # results = Parallel(n_jobs=num_cores)(delayed(shift_flux)(flux[i], tpf.pos_corr1[i], tpf.pos_corr2[i]) for i in tqdm(range(len(flux)), total=len(flux), desc='Shifting'))
    # for i, result in enumerate(results):
    #     flux_cor[i] = result

    ref = np.nanmedian(flux_cor, axis=0)
    diff = flux_cor - ref
    
    fx = np.nansum(flux, axis = (1,2))
    
    if plot:
        print(diff.shape)
        print(ref.shape)
        print(fx.shape)

        plt.figure()
        plt.imshow(ref, origin='lower')
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(diff[200], origin='lower')
        plt.colorbar()
        plt.show()
        
    return ref, diff, fx

def shift_flux(flux, dx, dy):
    """
    Shift the flux by dx and dy using a simple 2D shift.
    dx, dy are the number of pixels to shift.
    """
    return shift(flux, shift=(dy, dx))

def cost_function(shift_params, flux_cor, ref, masking_array):
    """
    The cost function to minimize. It computes the difference between the
    shifted flux and the reference frame, returning the sum of squared differences.
    
    Parameters:
    - shift_params: a list/array containing [dx, dy] (the shift in x and y direction).
    - flux_cor: the flux data to be shifted.
    - ref: the reference frame (median flux image).
    
    Returns:
    - The sum of squared differences between the shifted flux and the reference frame.
    """
    dx, dy = shift_params

    flux_shifted = shift_flux(flux_cor, dx, dy)
    
    diff = (flux_shifted - ref)*masking_array
    return np.nansum(diff**2)

def create_diff_image_de(tpf_info, plot=False, tol=0.01, mask_value=1000):
    flux = tpf_info.flux
    flux_err = tpf_info.flux_err
    quality = tpf_info.quality
    pos_corr1 = tpf_info.pos_corr1
    pos_corr2 = tpf_info.pos_corr2
    
    bad_frames = []
    
    for i in range(len(flux)):
        if np.isnan(flux[i]).sum() >= flux[i].shape[0] * flux[i].shape[1] *  0.5:
            bad_frames.append(i)
        elif quality[i] != 0:
            bad_frames.append(i)
    
    flux[np.isnan(flux)] = 0
    flux_err[np.isnan(flux_err)] = 1

    flux_cor = np.zeros_like(flux)
    # flux_err_cor = np.zeros_like(flux_err)
    
    r = np.sqrt((pos_corr1**2 + pos_corr2**2))
    
    mask = np.where((r < tol) & (quality == 0))[0]
    if len(mask) == 0:
        mask = np.where((quality == 0))[0]
    arg = np.nanargmin(np.nansum(flux[mask], axis = (1,2)))
    ref = flux[mask[arg]]
    
    ref = shift(ref, (-pos_corr1[mask[arg]], -pos_corr2[mask[arg]]))
    
    ref[np.isnan(ref)] = 0
    
    poisson_noise = np.sqrt(np.abs(ref))
    poisson_noise[poisson_noise < 1] = 1
    
    bright_mask = ref >= mask_value
    
    masking_array = np.ones_like(ref)
    reverse_mask = np.ones_like(ref)
    
    if bright_mask.any() == True:
        masking_array[~bright_mask] = 0
        reverse_mask[bright_mask] = 0
    
    dxs = []
    dys = []
    
    for i in range(len(flux)):
        result = minimize(cost_function, 
                        x0=(-pos_corr1[i], -pos_corr2[i]), 
                        args=(flux[i], ref, masking_array),
                        method='L-BFGS-B', 
                        bounds=[(-0.6, 0.6), (-0.6, 0.6)],  # Setting bounds for both parameters
                        tol=1e-8)
        
        dxs.append(result.x[0])
        dys.append(result.x[1])

    flux_shifted = np.zeros_like(flux_cor)
    for i in range(flux_cor.shape[0]):
        if i in bad_frames:
            flux_shifted[i] = np.nan*np.ones_like(flux[i])
        else:
            flux_shifted[i] = shift_flux(flux[i], dxs[i], dys[i]) * reverse_mask
    
    ref *= reverse_mask
    diff_shifted = flux_shifted - ref
    
    diff_shifted[bad_frames] = np.nan*np.ones_like(flux[0])
    flx = np.nansum(flux_shifted, axis = (1,2))
    
    if plot:
        
        plt.figure()
        plt.imshow(ref, origin='lower')
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(poisson_noise, origin='lower')
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(flux_shifted[200], origin='lower')
        plt.colorbar()
        plt.title('Shifted flux')
        plt.show()

        plt.figure()
        plt.imshow(diff_shifted[200], origin='lower')
        plt.colorbar()
        plt.title('Difference after shift')
        plt.show()

    return ref, diff_shifted, poisson_noise, flx, mask[arg]