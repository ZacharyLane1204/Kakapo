import numpy as np
import pywt

from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import median_filter
from scipy.ndimage import label

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.robust.scale import mad

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

import george
from george import kernels

from astropy.stats import sigma_clipped_stats
from astropy.timeseries import LombScargle

# import celerite2
# from celerite2 import GaussianProcess
# from celerite2.terms import xpSquaredTerm, RealTerm
# from celerite2.terms import SHOTerm, RealTerm

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def _twoD_gaussian(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coords
    xo = float(x0)
    yo = float(y0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a = (cos_t**2)/(2*sigma_x**2) + (sin_t**2)/(2*sigma_y**2)
    b = -(sin_t*cos_t)/(2*sigma_x**2) + (sin_t*cos_t)/(2*sigma_y**2)
    c = (sin_t**2)/(2*sigma_x**2) + (cos_t**2)/(2*sigma_y**2)
    g = offset + amp * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def fit_psf_fwhm(psf):
    y, x = np.indices(psf.shape)
    x0, y0 = np.array(psf.shape) / 2
    amp0 = psf.max() - psf.min()
    offset0 = psf.min()
    p0 = (amp0, x0, y0, 1.0, 1.0, 0.0, offset0)  # initial guess
    popt, _ = curve_fit(_twoD_gaussian, (x, y), psf.ravel(), p0=p0)
    _, _, _, sigma_x, sigma_y, _, _ = popt
    fwhm_x = 2*np.sqrt(2*np.log(2)) * sigma_x
    fwhm_y = 2*np.sqrt(2*np.log(2)) * sigma_y
    return fwhm_x, fwhm_y, np.nanmean([fwhm_x, fwhm_y])

def check_periodicity(times, flux=None, flux_err=None, fap_level=0.025):
    """
    Quick periodicity check. Returns (bool, period, confidence).
    - If flux provided, uses LC.
    - Otherwise, treats times as delta events.
    """
    if len(times) < 3:
        return False, np.nan, 0.0

    y = flux if flux is not None else np.ones_like(times)
    dy = flux_err if flux_err is not None else None

    ls = LombScargle(times, y, dy)
    freq, power = ls.autopower()
    best_freq = freq[np.argmax(power)]
    best_period = 1.0 / best_freq

    # Confidence = 1 - FAP (False Alarm Probability)
    fap = ls.false_alarm_probability(power.max())
    confidence = 1 - fap
    is_periodic = fap < fap_level

    return is_periodic, best_period, confidence

def median_clip(data, sigma=3):
    med = np.nanmedian(data)
    std = (np.nanpercentile(data, 84) - np.nanpercentile(data, 16)) / 2
    return np.abs(data - med) > sigma * std

def weighted_robust_poly_fit(x, y, weights, degree=3):
    X = PolynomialFeatures(degree=degree).fit_transform(x[:, None])
    model = HuberRegressor()
    model.fit(X, y, sample_weight=weights)

    def poly(x_eval):
        X_eval = PolynomialFeatures(degree=degree).fit_transform(x_eval[:, None])
        return model.predict(X_eval)
    
    return poly

def wavelet_denoise(flux, wavelet='coif5', level=2, keep='low', mode='symmetric'):
    flux = np.asarray(flux)
    nan_mask = ~np.isfinite(flux)
    valid_flux = flux[~nan_mask]

    if len(valid_flux) < 10:
        return flux.copy()

    coeffs = pywt.wavedec(valid_flux, wavelet, level=level, mode=mode)
    if keep == 'low':
        for i in range(1, len(coeffs)):
            coeffs[i] = np.zeros_like(coeffs[i])
    elif keep == 'high':
        coeffs[0] = np.zeros_like(coeffs[0])

    filtered = pywt.waverec(coeffs, wavelet)[:len(valid_flux)]
    output = np.full_like(flux, np.nan, dtype=float)
    output[~nan_mask] = filtered
    
    return output

def _lowess_smoothing(flux):
    T = len(flux)
    smoothed = lowess(flux, np.arange(T), frac=0.05, return_sorted=False)
    corrected = flux - (smoothed - np.nanmedian(smoothed))
    return corrected 

def mask_nan_edges(flux, pad=3):
    nan_idx = np.flatnonzero(~np.isfinite(flux))
    mask = np.zeros_like(flux, dtype=bool)
    for i in nan_idx:
        mask[max(0, i - pad):min(len(flux), i + pad + 1)] = True
    return mask

def correct_motion_lightcurve(flux, distance, thrust_indices, wavelet_params=None, mask_edges=True):
    flux = np.array(flux)
    distance = np.array(distance)
    T = len(flux)

    fit_curve = np.full_like(flux, np.nan)
    global_x, global_y, global_weights = [], [], []

    for i in range(len(thrust_indices) - 1):
        start, end = thrust_indices[i], thrust_indices[i + 1]
        section = flux[start:end]
        d = distance[start:end]
        valid = np.isfinite(section) & np.isfinite(d)

        x = np.arange(start, end)[valid]
        y = section[valid]
        d = d[valid]

        if len(x) < 7:
            continue

        outlier_mask = median_clip(y, sigma=3)
        x_clean = x[~outlier_mask]
        y_clean = y[~outlier_mask]
        d_clean = d[~outlier_mask]

        if len(x_clean) < 5:
            continue

        # fit_curve[x[outlier_mask]] = np.nan
        # fit_curve[x_clean[0]] = np.nan
        # fit_curve[x_clean[-1]] = np.nan
        
        for idx in x[outlier_mask]:
            baseline_left = flux[max(0, idx - 22):max(0, idx - 22)]
            baseline_right = flux[min(T, idx + 22):min(T, idx + 22)]

            # Combine and filter baseline values
            baseline_vals = np.concatenate([baseline_left, baseline_right])
            baseline_vals = baseline_vals[np.isfinite(baseline_vals)]

            if len(baseline_vals) >= 11:
                baseline_median = np.median(baseline_vals)
                baseline_mad = (np.percentile(baseline_vals, 84) - np.percentile(baseline_vals, 16)) / 2
                if np.abs(flux[idx] - baseline_median) >= 15 * baseline_mad:
                    fit_curve[idx] = np.nan  # mask extreme outlier

        success = False
        try:
            weights = np.clip(1 - d_clean, 1e-2, 1.0)
            poly = weighted_robust_poly_fit(x_clean, y_clean, weights, degree=2)
            fit_curve[x_clean] = poly(x_clean)
            success = True
        except:
            try:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                fit_curve[x_clean] = np.polyval(coeffs, x_clean)
                success = True
            except:
                pass

        if not success:
            med = np.nanmedian(y_clean)
            fit_curve[x_clean] = med

        low_motion = (d_clean < 0.2)
        if np.any(low_motion):
            global_x.extend(x_clean[low_motion])
            global_y.extend(y_clean[low_motion])
            global_weights.extend((1 - d_clean[low_motion])**2)

    spline_curve = np.full_like(flux, np.nan)
    global_x = np.array(global_x)
    global_y = np.array(global_y)
    global_weights = np.array(global_weights)

    valid = np.isfinite(global_x) & np.isfinite(global_y)
    global_x = global_x[valid]
    global_y = global_y[valid]

    if len(global_x) >= 4:
        sort_idx = np.argsort(global_x)
        global_x = global_x[sort_idx]
        global_y = global_y[sort_idx]
        global_y = median_filter(global_y, size=5)

        try:
            pchip = PchipInterpolator(global_x, global_y, extrapolate=False)
            t = np.arange(T)
            inside = (t >= global_x[0]) & (t <= global_x[-1])
            spline_curve[inside] = pchip(t[inside])
        except:
            pass

    if mask_edges:
        edge_mask = mask_nan_edges(flux, pad=1)
        fit_curve[edge_mask] = np.nan
        spline_curve[edge_mask] = np.nan

    combined = np.full_like(flux, np.nan)
    valid_local = np.isfinite(fit_curve)
    valid_global = np.isfinite(spline_curve)
    both = valid_local & valid_global
    only_local = valid_local & ~valid_global
    only_global = ~valid_local & valid_global

    combined[both] = 0.5 * fit_curve[both] + 0.5 * spline_curve[both]
    combined[only_local] = fit_curve[only_local]
    combined[only_global] = spline_curve[only_global]

    corrected = flux - combined

    # Final NaN-safety and outlier capping
    median_flux = np.nanmedian(corrected)
    mad = (np.nanpercentile(corrected, 84) - np.nanpercentile(corrected, 16)) / 2
    too_high = np.abs(corrected - median_flux) > 20 * mad
    corrected[too_high] = np.nan

    # Optional wavelet smoothing
    if wavelet_params:
        corrected = wavelet_denoise(corrected, **wavelet_params)

    return corrected

def correction_smoothing_lightcurve(flux, low_motion, window=35, sigma=2):
    """
    Smooth and detrend a lightcurve using rolling cubic splines (local) and
    global PCHIP spline interpolation within contiguous valid segments.
    Gaps where local fits fail remain NaN.

    Parameters:
    -----------
    flux : array-like
        1D lightcurve array with possible NaNs.
    low_motion : array-like
        Boolean mask indicating low-motion frames to prioritize in fitting.
    window : int
        Rolling window size for local cubic spline fitting.
    sigma : float
        Threshold multiplier for clipping outliers based on local percentile width.

    Returns:
    --------
    detrended_flux : np.ndarray
        Smoothed/detrended flux array. Gaps where local fits failed remain NaN.
    """
    flux = np.asarray(flux)
    low_motion = np.asarray(low_motion)
    x = np.arange(len(flux))
    detrended = np.full_like(flux, np.nan, dtype=float)

    for i in range(len(flux)):
        lo = max(0, i - window // 2)
        hi = min(len(flux), i + window // 2 + 1)

        x_win = x[lo:hi]
        y_win = flux[lo:hi]
        weights = low_motion[lo:hi].astype(float)

        mask = ~np.isnan(y_win)
        if np.nansum(mask) < 4:
            continue  # Not enough points for cubic spline

        x_fit = x_win[mask]
        y_fit = y_win[mask]
        w_fit = weights[mask]

        p84 = np.percentile(y_fit, 84)
        p16 = np.percentile(y_fit, 16)
        iqr_like = (p84 - p16) /2
        threshold = sigma * iqr_like

        try:
            cs = CubicSpline(x_fit, y_fit, bc_type='natural')
            local_pred = cs(x[i])

            if np.abs(flux[i] - local_pred) < threshold:
                detrended[i] = local_pred
            else:
                detrended[i] = np.nan  # Clip as outlier
        except Exception:
            continue  # Leave as NaN if fit fails

    mask_valid = ~np.isnan(detrended)
    labels, num = label(mask_valid)  # Label contiguous valid segments

    result = np.full_like(detrended, np.nan, dtype=float)

    for i in range(1, num + 1):
        segment_mask = labels == i
        if np.sum(segment_mask) < 4:
            continue  # Skip tiny segments

        x_seg = x[segment_mask]
        y_seg = detrended[segment_mask]

        try:
            spline = PchipInterpolator(x_seg, y_seg)
            result[segment_mask] = spline(x_seg)
        except Exception:
            result[segment_mask] = y_seg  # Fall back to local fit if PCHIP fails

    return result

def flatten_and_clip_outliers(flux, frac=0.05, sigma=5.0):
    """
    Flatten a lightcurve using LOWESS and mask extreme outliers using sigma-clipping on residuals.

    Parameters
    ----------
    flux : array_like
        1D lightcurve flux array.
    time : array_like or None
        Time array. If None, uses np.arange(len(flux)).
    frac : float
        Fraction of data used for each LOWESS fit (smoothing span).
    sigma : float
        Sigma threshold for outlier clipping.

    Returns
    -------
    flat_flux : np.ndarray
        Flattened lightcurve (flux / smoothed trend).
    mask : np.ndarray
        Boolean array where True indicates an inlier (not clipped).
    """
    flux = np.asarray(flux)
    time = np.arange(len(flux))

    # Mask out NaNs before fitting
    valid = np.isfinite(flux) & np.isfinite(time)
    t_valid = time[valid]
    f_valid = flux[valid]

    # LOWESS smoothing
    smooth = lowess(f_valid, t_valid, frac=frac, return_sorted=False)

    # Interpolate smoothed trend to full time array
    trend = np.full_like(flux, np.nan)
    trend[valid] = smooth

    # Flatten
    flat_flux = flux / trend

    # Sigma-clipping on residuals
    residuals = f_valid - smooth
    _, median, std = sigma_clipped_stats(residuals, sigma=3)
    
    print(std)
    keep = np.abs(residuals - median) < sigma * std

    # Build full mask and replace outliers with NaN
    mask = np.full_like(flux, False, dtype=bool)
    mask[valid] = keep
    flat_flux[~mask] = np.nan  # mask outliers
    flux[~mask] = np.nan  # mask outliers

    return flux, flat_flux, mask

def flatten_and_mask_outliers(flux, mask=None, gp_kernel=None, gp_scale=11, outlier_sigma=5.0):
    """
    Flatten a lightcurve using a long-timescale Gaussian Process and mask >8sigma outliers.

    Parameters
    ----------
    time : array_like
        Time array (should be sorted and ~uniformly sampled if possible).
    flux : array_like
        Flux array with possible NaNs.
    mask : array_like, optional
        Boolean mask to exclude bad data from the fit (True = use, False = ignore).
    gp_kernel : george kernel, optional
        If provided, use this GP kernel; otherwise, use a default RBF kernel with `gp_scale`.
    gp_scale : float
        Characteristic timescale for GP in the same units as `time`.
    outlier_sigma : float
        Threshold for outlier clipping (e.g., 8sigma).

    Returns
    -------
    flux_flat : ndarray
        Flattened flux.
    flux_masked : ndarray
        Same as `flux_flat` but with >8sigma outliers masked as NaN.
    """

    time = np.arange(0, len(flux))
    flux = np.asarray(flux)

    if mask is None:
        mask = np.isfinite(time) & np.isfinite(flux)
    else:
        mask &= np.isfinite(time) & np.isfinite(flux)

    
    if gp_kernel is None: # Use a simple long-term RBF kernel if not provided
        kernel = gp_scale**2 * kernels.ExpSquaredKernel(metric=gp_scale**2)
    else:
        kernel = gp_kernel

    gp = george.GP(kernel)
    
    flux_filled = flux.copy()
    flux_filled[~np.isfinite(flux_filled)] = np.nanmedian(flux_filled)

    local_std = uniform_filter1d(np.abs(flux_filled - np.nanmedian(flux_filled)), size=101)
    yerr = 0.01 * local_std + 1e-8
    yerr[~np.isfinite(yerr)] = np.nanmedian(yerr[np.isfinite(yerr)])
    
    gp.compute(time[mask], yerr = yerr[mask])

    flux_gp = gp.predict(flux[mask], time, return_cov=False)

    flux_flat = flux - flux_gp # Flatten the flux

    # Estimate robust σ using inter-percentile range
    _, _, iqr = sigma_clipped_stats(flux_flat, sigma = 4)
    sigma_clip = outlier_sigma * iqr

    outliers = (np.abs(flux_flat) > sigma_clip) & np.isfinite(flux_flat)

    flux_masked = flux_flat.copy()
    flux_masked[outliers] = np.nan
    flux_flat[outliers] = np.nan  # Optional: remove outliers from flat version too
    flux[outliers] = np.nan  
    flux_masked[outliers] = np.nan
    flux[outliers] = np.nan
    
    print(f"Sigma clip threshold: {sigma_clip}")
    print(f"Max(flat): {np.nanmax(flux_flat)}")
    print(f"Number of outliers: {np.sum(outliers)}")

    return flux, flux_flat, outliers

def gauss_smooth(time, flux, flux_error=None, n_samples=11):
    kernel = (C(0.1) * RBF(length_scale=3.0) +
              C(1e-4, (5e-5, 5e-3)) * RBF(length_scale=0.25, length_scale_bounds=(0.05, 0.5)) +
              WhiteKernel(noise_level=3e-3))

    base_alpha_values = [0.05, 0.15, 0.25]
    weights_norm = 1

    for base_alpha in base_alpha_values:
        try:
            
            alpha_per_point = (base_alpha / weights_norm) ** 2
            if flux_error is not None:
                # max_var = np.nanpercentile(flux_error**2, 95)  # or a fixed ceiling
                # alpha = np.minimum(flux_error**2, max_var) + alpha_per_point
                alpha = (flux_error**2) + alpha_per_point
            else:
                alpha = alpha_per_point

            gp = GaussianProcessRegressor(kernel=kernel,
                                          alpha=alpha,
                                          normalize_y=True,
                                          optimizer=None)
            gp.fit(time[:, None], flux)

            samples = gp.sample_y(time[:, None], n_samples=n_samples) # GP draws
            sampled_mean = samples.mean(axis=1)
            sampled_std  = samples.std(axis=1)

           
            _, gp_std = gp.predict(time[:, None], return_std=True) # GP predictive std

            residuals = flux - sampled_mean # Residual scatter (MAD-based, robust)
            local_scatter = mad(residuals)

            total_std = np.sqrt(gp_std**2 + sampled_std**2 + local_scatter**2) # Final error = quadrature of GP std, sample scatter, and residual scatter

            return sampled_mean, 2 * total_std  # mean + error band
        except Exception:
            continue

    mean_pred = PchipInterpolator(time, flux)(time)
    std_pred = np.full_like(mean_pred, np.std(flux))
    return mean_pred, std_pred

def weighted_value_and_uncertainty(data, uncertainties):

    # Calculate weights as the inverse of the uncertainties squared
    weights = 1 / uncertainties**2
    weighted_mean = np.nansum(weights * data) / np.nansum(weights)
    weighted_uncertainty = np.sqrt(1 / np.nansum(weights))

    return weighted_mean, weighted_uncertainty

def binned_averages(time, flux, flux_err, bin_size = 3):

    num_data_points = len(time)
    
    points_per_bin = bin_size * 2
    
    num_bins = num_data_points // points_per_bin
    remainder = num_data_points % points_per_bin
    
    if remainder > 0:
        num_bins += 1

    blc, bins_median, bins_std = errors(time, flux, flux_err, num_bins, points_per_bin)

    return blc, bins_median, bins_std

def errors(time, flux, flux_err, num_bins, points_per_bin):
    bins_median = []
    bins_std = []
    blc = []
    # print('Number of bins:', num_bins)
    for i in range(num_bins):
        start = i * points_per_bin
        end = (i + 1) * points_per_bin
        
        bin_time = time[start:end]
        bin_flux = flux[start:end]
        bin_flux_err = flux_err[start:end]
        
        blc.append(np.nanmean(bin_time))
        med, std = weighted_value_and_uncertainty(bin_flux, bin_flux_err)
        bins_median.append(med)
        bins_std.append(std)

    blc = np.array(blc)
    bins_median = np.array(bins_median)
    bins_std = np.array(bins_std)
    
    return blc, bins_median, bins_std