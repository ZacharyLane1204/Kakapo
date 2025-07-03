import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from time import time
import operator

import ruptures as rpt

from scipy.signal import fftconvolve, convolve2d, find_peaks
from scipy.ndimage import center_of_mass, zoom, shift, convolve
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft2, ifft2, fftshift
from scipy.interpolate import PchipInterpolator, UnivariateSpline, interp1d
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import binary_dilation
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter

from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy.modeling import models, fitting

# from Kakapo.photometry import forced_photometry

from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from skimage.registration import phase_cross_correlation
from statsmodels.nonparametric.smoothers_lowess import lowess

from copy import deepcopy
from tqdm import tqdm

import psutil
import os

def thrusters_firing(pos1, pos2):

    mask = (pos1 > -2) & (pos1 < 2)
    pos1[~mask] = 0
    pos1[np.isnan(pos1)] = 0

    mask = (pos2 > -2) & (pos2 < 2)
    pos2[~mask] = 0
    pos2[np.isnan(pos2)] = 0

    r = np.sqrt(pos1**2 + pos2**2)

    peak_indices = find_peaks(r, distance = 8)[0]
    
    return peak_indices

def shift_flux(flux, dx, dy):
    """
    Shift the flux by dx and dy using a simple 2D shift.
    dx, dy are the number of pixels to shift.
    """
    # return shift(flux, shift=, order = 3, mode='nearest', prefilter=False)
    return shift(flux, shift=(dy, dx), order=3, mode='reflect')
    
def cost_function(shift_params, flux_cor, ref, noise):
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
    
    diff = (flux_shifted - ref)
    return np.nansum(diff**2 / (noise + 1e-8)**2)

def build_reference(flux, quality, r, pos_corr1, pos_corr2, tol=0.05, frac=0.2):
    # mask = np.where((r < tol) & (quality == 0))[0]
    # if len(mask) == 0:
    #     mask = np.where(quality == 0)[0]

    # flux_sums = np.nansum(flux[mask], axis=(1,2))
    # N = max([int(frac * len(mask)), 1])  # bottom X% flux frames
    # idx = np.argsort(flux_sums)[:N]
    # ref = np.nanmedian(flux[mask[idx]], axis=0)
    # ref[np.isnan(ref)] = 0
    
    # return ref

    mask = np.where((r < tol) & (quality == 0))[0]
    if len(mask) == 0:
        mask = np.where((quality == 0))[0]
    arg = np.nanargmin(np.nansum(flux[mask], axis = (1,2)))
    ref = flux[mask[arg]]
    
    ref = shift(ref, (-pos_corr1[mask[arg]], -pos_corr2[mask[arg]]))
    
    ref[np.isnan(ref)] = 0
    return ref, mask[arg]

def create_diff_image(tpf_info, psf, plot=False, tol=0.05, mask_value=1e6):
    start_time = time()
    flux = tpf_info.flux
    time_kp = tpf_info.time
    flux_err = tpf_info.flux_err
    quality = tpf_info.quality
    pos_corr1 = tpf_info.pos_corr1
    pos_corr2 = tpf_info.pos_corr2
    
    pos_corr1[np.isnan(pos_corr1)] = 0
    pos_corr2[np.isnan(pos_corr2)] = 0
    
    r = np.sqrt((pos_corr1**2 + pos_corr2**2))
    
    
    bad_frames, thrusters = _bad_frames(flux, quality, pos_corr1, pos_corr2)
    
    ref, ref_frame = build_reference(flux, quality, r, pos_corr1, pos_corr2, tol=tol, frac=0.2)
    
    flux[np.isnan(flux)] = 0
    flux_err[np.isnan(flux_err)] = 1
    flux[bad_frames] = np.nan
    
    poisson_variance = np.clip(np.abs(ref), a_min=1.0, a_max=None)
    
    noise = np.sqrt(flux_err**2 + poisson_variance)
    
    dxs, dys = _minimisation_routine(flux, ref, noise)
    
    print(f'Minimised: {(time() - start_time)/60:.2f}')
    
    # diffs, sig_map = compute_zogy_differences(flux, ref, dxs, dys, psf, poisson_variance)
    diffs, jump_metric = compute_difference_images_with_psf(flux, ref, dxs, dys)
    print(f'Computed Diff. Images: {(time() - start_time)/60:.2f} min')
    # dxs, dys = compute_all_shifts_with_temporal_correction(flux, noise, dxs, dys, jump_metric)
    # print(f'Computed Temporal Outliers: {(time() - start_time)/60:.2f} min')
    # diffs, _ = compute_difference_images_with_psf(flux, ref, dxs, dys)
    # print(f'Compute Diff. Images: {(time() - start_time)/60:.2f} min')
    
    dist_mask = np.sqrt(dxs**2 + dys**2) > 1.5
    
    bad_jumps = detect_jump_discontinuities(jump_metric, sigma_thresh=7)
    
    diffs[bad_frames] = np.nan*np.ones_like(flux[0])
    diffs[dist_mask] = np.nan*np.ones_like(flux[0])
    diffs[bad_jumps] = np.nan*np.ones_like(flux[0])
    
    return ref, diffs, np.sqrt(poisson_variance), thrusters, np.sqrt(dxs**2 + dys**2)
    
def detect_jump_discontinuities(jump_metric, sigma_thresh=5):
    median = np.nanmedian(jump_metric)
    mad = np.nanmedian(np.abs(jump_metric - median))
    threshold = median + sigma_thresh * mad
    flagged = np.where(jump_metric > threshold)[0] + 1  # +1 for diff offset
    return flagged  

def _bad_frames(flux, quality, pos_corr1, pos_corr2):
    
    r = np.sqrt((pos_corr1**2 + pos_corr2**2))
    bad_frames = []
    
    for i in range(len(flux)):
        if np.isnan(flux[i]).sum() >= flux[i].shape[0] * flux[i].shape[1] *  0.7:
            bad_frames.append(i)
        elif r[i] > 2:
            bad_frames.append(i)

    thrusters = thrusters_firing(pos_corr1, pos_corr2)

    offsets = np.array([-1, 0, 1, 2])
    bad_thrusters = (thrusters[:, None] + offsets).ravel()
    bad_thrusters = np.sort(np.unique(bad_thrusters))

    bad_frames += thrusters.tolist()
    
    bad_frames = np.array(bad_frames)
    bad_frames = bad_frames[(bad_frames >= 0) & (bad_frames <= (len(flux) -1))]
    bad_frames = np.sort(np.unique(bad_frames))
    
    if len(bad_frames) < 1:
        mask = quality > 0
        if np.nansum(mask) < 1:
            bad_frames = np.array([0, len(flux) - 1])
            thrusters = np.array([0, len(flux) - 1])
        else:
            bad_frames = np.arange(len(flux))[mask]
    
    return bad_frames, thrusters

def _minimisation_routine(flux, ref, noise):
    dxs, dys = [], []
 
    # for i in tqdm(range(len(flux)), desc='Frames'):
    for i in range(len(flux)):
        
        if np.sum(np.isnan(flux[i])) >= flux[i].shape[0] * flux[i].shape[1] * 0.7:
            dxs.append(np.nan)
            dys.append(np.nan)
            continue
        
        dx, dy = compute_shift(flux[i], ref, noise[i])
        dxs.append(dx)
        dys.append(dy)

    return np.array(dxs), np.array(dys)

def compute_shift(frame, ref, noise_frame):

    def cost_fn(shift_params):
        return cost_function_safe(shift_params, frame, ref, noise_frame)

    result = differential_evolution(cost_fn, bounds=[(-3, 3), (-3, 3)],
                                    tol=1e-6, strategy='best1bin', 
                                    mutation=0.8, recombination=0.6, polish=True)

    return result.x

def cost_function_safe(shift_params, *args):
    try:
        cost = cost_function(shift_params, *args)
        if not np.isfinite(cost):
            return 1e10
        return cost
    except Exception as e:
        print(f"Exception in cost function at shift={shift}: {e}")
        return 1e10
    
def temporal_cost_fn(shift_params, i, flux, dxs, dys, noise):
    shifted_frame = shift_flux(flux[i], *shift_params)
    if i == 0 or i == len(flux)-1:
        return 1e6  # skip edges
    prev = shift_flux(flux[i-1], dxs[i-1], dys[i-1])
    next = shift_flux(flux[i+1], dxs[i+1], dys[i+1])
    median_neighbor = np.nanmean([prev, next], axis = 0)
    diff = shifted_frame - median_neighbor
    
    return np.nansum(diff**2 / (noise[i]+ 1e-6)**2)

def compute_all_shifts_with_temporal_correction(flux, noise, dxs, dys, jump_metric):
    
    bad_frames = detect_jump_discontinuities(jump_metric, sigma_thresh=7)
    
    for i in bad_frames:
        
        def cost_fn(shift_params):
            return temporal_cost_fn(shift_params, i, flux, dxs, dys, noise)
        
        result = differential_evolution(cost_fn, bounds=[(-3, 3), (-3, 3)], 
                                        tol=1e-6, strategy='best1bin', 
                                        mutation=0.8, recombination=0.6,
                                        polish=True)
        dxs[i] = result.x[0]
        dys[i] = result.x[1]
    
    return dxs, dys

def compute_difference_images_with_psf(flux, ref, dxs, dys):
    diff_images = []
    residuals_t = []
    for i in range(len(flux)):
        
        if np.isnan(flux[i]).sum() >= flux[i].shape[0] * flux[i].shape[1] *  0.7:
            diff_images.append(np.nan*np.ones_like(flux[0]))
            continue
        
        shifted = shift_flux(flux[i], dxs[i], dys[i])
        bkg = gaussian_filter(shifted, sigma=3)
        diff_clean = shifted - ref  - bkg
        residuals_t.append(diff_clean)

        diff_images.append(diff_clean)
    return np.array(diff_images), np.nansum(np.abs(np.diff(np.array(residuals_t), axis=0)), axis=(1,2))

# def estimate_initial_shift(frame, ref):
#     # cross-correlation in Fourier space
#     f_frame = np.fft.fft2(frame)
#     f_ref = np.fft.fft2(ref)
#     cross_power = f_frame * f_ref.conj()
#     cross_power /= np.abs(cross_power)
#     shift = np.fft.ifft2(cross_power)
#     maxima = np.unravel_index(np.argmax(np.abs(shift)), shift.shape)
#     shifts = np.array(maxima, dtype=np.float64)
#     shifts[shifts > np.array(frame.shape) // 2] -= np.array(frame.shape)[shifts > np.array(frame.shape) // 2]
#     return shifts[::-1]  # (dy, dx) -> (x, y)

# def generate_mask(frame, ref, sigma=5.0):
#     diff = frame - ref
#     med = np.median(diff)
#     std =(np.nanpercentile(diff, 84) - np.nanpercentile(diff, 16))/2
#     mask = np.abs(diff - med) > (sigma * std)
#     # Morphological filtering or smoothing can help reduce overmasking
#     return binary_dilation(mask, iterations=1)

# def create_weight_mask(shape, alpha=0.3):
#     ywin = tukey(shape[0], alpha)
#     xwin = tukey(shape[1], alpha)
#     return np.outer(ywin, xwin)

# def correct_motion_lightcurve(flux, distance, thruster_peaks, pen=8, degree=3, window=5):
#     flux = np.array(flux)
#     distance = np.array(distance)
#     T = len(flux)

#     # Convert thruster peaks to intervals and merge close ones
#     thruster_intervals = thrusters_to_intervals(thruster_peaks, T, window=window)

#     # Detect change points from flux
#     change_points = detect_change_points(flux, pen=pen, model="rbf")

#     # Combine intervals and change points into segment boundaries
#     boundaries = set()
#     boundaries.add(0)
#     boundaries.add(T)
#     # Add thruster interval boundaries
#     for start, end in thruster_intervals:
#         boundaries.add(start)
#         boundaries.add(end)
#     # Add change point boundaries
#     for cp in change_points:
#         boundaries.add(cp)
#     boundaries = sorted(boundaries)

#     # Build segments (start, end) pairs
#     segments = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

#     fit_curve = np.full_like(flux, np.nan, dtype=np.float64)
#     global_x = []
#     global_y = []
#     global_weights = []

#     for start, end in segments:
#         if end - start < 7:
#             # Too small to fit, skip
#             continue

#         section = flux[start:end]
#         d = distance[start:end]
#         valid = np.isfinite(section) & np.isfinite(d)

#         x = np.arange(start, end)[valid]
#         y = section[valid]
#         d = d[valid]

#         if len(x) < 7:
#             continue

#         # Mask outliers robustly
#         outlier_mask = median_clip(y, sigma=3)
#         x_clean = x[~outlier_mask]
#         y_clean = y[~outlier_mask]
#         d_clean = d[~outlier_mask]

#         if len(x_clean) < 5:
#             continue

#         try:
#             poly = weighted_robust_poly_fit(x_clean, y_clean, 1 - d_clean, degree=degree)
#             fit_vals = poly(x_clean)
#             fit_curve[x_clean] = fit_vals

#             low_motion = d_clean < 0.2
#             if np.any(low_motion):
#                 global_x.extend(x_clean[low_motion])
#                 global_y.extend(y_clean[low_motion])
#                 global_weights.extend(1 - d_clean[low_motion])
#         except Exception as e:
#             # Fall back to simple polyfit
#             try:
#                 coeffs = np.polyfit(x_clean, y_clean, degree)
#                 y_fit_simple = np.polyval(coeffs, x_clean)
#                 fit_curve[x_clean] = y_fit_simple
#             except:
#                 continue

#     # Global smooth spline from low motion points
#     global_x = np.array(global_x)
#     global_y = np.array(global_y)
#     global_weights = np.array(global_weights)

#     valid_global = np.isfinite(global_x) & np.isfinite(global_y)
#     global_x = global_x[valid_global]
#     global_y = global_y[valid_global]
#     global_weights = global_weights[valid_global]

#     spline_curve = np.full_like(flux, np.nan, dtype=np.float64)

#     if len(global_x) >= 4:
#         sort_idx = np.argsort(global_x)
#         global_x = global_x[sort_idx]
#         global_y = global_y[sort_idx]
#         global_weights = global_weights[sort_idx]

#         try:
#             pchip = PchipInterpolator(global_x, global_y, extrapolate=False)
#             t = np.arange(T)
#             spline_curve = pchip(t)
#             outside_mask = (t < global_x[0]) | (t > global_x[-1])
#             spline_curve[outside_mask] = np.nan
#         except:
#             pass

#     final = flux - fit_curve + spline_curve
#     return final

# def correct_motion_lightcurve(flux, distance, thrust_indices):
#     flux = np.array(flux)
#     distance = np.array(distance)
#     T = len(flux)

#     fit_curve = np.zeros_like(flux)  # Changed from NaN-filled to zero-filled

#     global_x = []
#     global_y = []
#     global_weights = []

#     for i in range(len(thrust_indices) - 1):
#         start, end = thrust_indices[i], thrust_indices[i + 1]
#         section = flux[start:end]
#         d = distance[start:end]
#         valid = np.isfinite(section) & np.isfinite(d)

#         x = np.arange(start, end)[valid]
#         y = section[valid]
#         d = d[valid]
        
#         if np.sum(valid) < 7:
#             fit_curve[x] = np.nan
#             continue
        
#         fit_curve[x[0]] = np.nan
#         fit_curve[x[-1]] = np.nan
#         outlier_mask = median_clip(y, sigma=3)
#         x_clean = x[~outlier_mask]
#         y_clean = y[~outlier_mask]
#         d_clean = d[~outlier_mask]
        
#         if len(x_clean) < 5:
#             fit_curve[x_clean] = np.nan
#             continue
        
#         x_outliers = x[outlier_mask]  # indices within the full fit_curve
#         fit_curve[x_outliers] = np.nan

#         try:
#             poly = weighted_robust_poly_fit(x_clean, y_clean, 1 - d_clean)
#             fit_curve[x_clean] = poly(x_clean)

#             low_motion = d_clean < 0.2
#             if np.any(low_motion):
#                 global_x.extend(x_clean[low_motion])
#                 global_y.extend(y_clean[low_motion])
#                 global_weights.extend(1 - d_clean[low_motion])
#         except Exception as e:
#             try:
#                 coeffs = np.polyfit(x_clean, y_clean, 3)
#                 y_fit_simple = np.polyval(coeffs, x_clean)
#                 fit_curve[x_clean] = y_fit_simple
#             except:
#                 continue

#     global_x = np.array(global_x)
#     global_y = np.array(global_y)
#     global_weights = np.array(global_weights)

#     spline_curve = np.zeros_like(flux)

#     valid = np.isfinite(global_x) & np.isfinite(global_y)
#     global_x = global_x[valid]
#     global_y = global_y[valid]

#     spline_curve = np.zeros_like(flux)
#     if len(global_x) >= 4:
#         sort_idx = np.argsort(global_x)
#         global_x = global_x[sort_idx]
#         global_y = global_y[sort_idx]

#         try:
#             pchip = PchipInterpolator(global_x, global_y, extrapolate=False)
#             t = np.arange(T)
#             spline_curve = pchip(t)
#             outside_mask = (t < global_x[0]) | (t > global_x[-1])
#             spline_curve[outside_mask] = np.nan

#         except:
#             spline_curve = np.zeros_like(flux)

#     final = flux - fit_curve + spline_curve

#     return final