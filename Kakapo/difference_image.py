import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from time import time
# import operator

# import ruptures as rpt

from scipy.signal import fftconvolve, convolve2d, find_peaks
from scipy.ndimage import center_of_mass, zoom, shift, convolve, gaussian_filter
from scipy.optimize import minimize, differential_evolution

# from Kakapo.photometry import forced_photometry

from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

# from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor
# from sklearn.preprocessing import PolynomialFeatures
from skimage.registration import phase_cross_correlation
# from statsmodels.nonparametric.smoothers_lowess import lowess

from copy import deepcopy
from tqdm import tqdm


class Difference_Imaging():
    def __init__(self, tpf_info, psf, tol=0.003):
        
        self.flux = tpf_info.flux
        self.time_kp = tpf_info.time
        self.flux_err = tpf_info.flux_err
        self.quality = tpf_info.quality
        self.pos_corr1 = tpf_info.pos_corr1
        self.pos_corr2 = tpf_info.pos_corr2
        self.psf = psf
        self.tol = tol

        self.create_diff_image()

    def create_diff_image(self):
        # start_time = time
        self.pos_corr1[np.isnan(self.pos_corr1)] = 0
        self.pos_corr2[np.isnan(self.pos_corr2)] = 0
        
        self.r = np.sqrt((self.pos_corr1**2 + self.pos_corr2**2))
        self._bad_frames()
        
        self.build_reference()
        
        self.flux[np.isnan(self.flux)] = 0
        self.flux_err[np.isnan(self.flux_err)] = 1
        self.flux[self.bad_frames] = np.nan
        
        poisson_variance = np.clip(np.abs(self.ref), a_min=1.0, a_max=None)
        
        self.noise = np.sqrt(self.flux_err**2 + poisson_variance)
        self.ref_shape = self.ref.shape
        
        self._minimisation_routine()
        self.compute_difference_images_with_psf()
        
        dist_mask = np.sqrt(self.dxs**2 + self.dys**2) > 2.5
        
        self._detect_jump_discontinuities(sigma_thresh=8)
        
        self.diffs[self.bad_frames] = np.nan*np.ones_like(self.flux[0])
        self.diffs[dist_mask] = np.nan*np.ones_like(self.flux[0])
        self.diffs[self.bad_jumps] = np.nan*np.ones_like(self.flux[0])
        
        self.difference_images = deepcopy(self.diffs)
        self.poisson_noise = np.sqrt(poisson_variance)
        self.distance = np.sqrt(self.dxs**2 + self.dys**2)

    def _detect_jump_discontinuities(self, sigma_thresh=8):
        median = np.nanmedian(self.jump_metrics)
        mad = np.nanmedian(np.abs(self.jump_metrics - median))
        threshold = median + sigma_thresh * mad
        flagged = np.where(self.jump_metrics > threshold)[0] + 1  # +1 for diff offset
        self.bad_jumps = flagged

    def _bad_frames(self):
    
        bad_frames = []
        
        for i in range(len(self.flux)):
            if np.isnan(self.flux[i]).sum() >= self.flux[i].shape[0] * self.flux[i].shape[1] *  0.7:
                bad_frames.append(i)
            elif self.r[i] > 2:
                bad_frames.append(i)

        thrusters = self._thrusters_firing()

        offsets = np.array([-1, 0, 1, 2])
        bad_thrusters = (thrusters[:, None] + offsets).ravel()
        bad_thrusters = np.sort(np.unique(bad_thrusters))

        bad_frames += thrusters.tolist()
        
        bad_frames = np.array(bad_frames)
        bad_frames = bad_frames[(bad_frames >= 0) & (bad_frames <= (len(self.flux) -1))]
        bad_frames = np.sort(np.unique(bad_frames))
        
        if len(bad_frames) < 1:
            mask = self.quality > 0
            if np.nansum(mask) < 1:
                bad_frames = np.array([0, len(self.flux) - 1])
                thrusters = np.array([0, len(self.flux) - 1])
            else:
                bad_frames = np.arange(len(self.flux))[mask]
        
        self.bad_frames = bad_frames
        self.thrusters = thrusters

    def _thrusters_firing(self):
        
        pos1 = self.pos_corr1
        pos2 = self.pos_corr2

        mask = (pos1 > -2) & (pos1 < 2)
        pos1[~mask] = 0
        pos1[np.isnan(pos1)] = 0

        mask = (pos2 > -2) & (pos2 < 2)
        pos2[~mask] = 0
        pos2[np.isnan(pos2)] = 0

        r = np.sqrt(pos1**2 + pos2**2)

        peak_indices = find_peaks(r, distance = 8)[0]
        
        return peak_indices
    
    def build_reference(self):
        mask = np.where((self.r < self.tol) & (self.quality == 0))[0]
        if len(mask) == 0:
            mask = np.where(self.quality == 0)[0]

        arg = np.nanargmin(np.nansum(self.flux[mask], axis = (1,2)))
        ref = self.flux[mask[arg]]
        ref[np.isnan(ref)] = 0
        
        ref = shift(ref, (-self.pos_corr1[mask[arg]], -self.pos_corr2[mask[arg]]))
        
        # bkg = self._fit_background(ref, mask=None)
        bkg = self._fit_background_plane(ref)
        # bkg = gaussian_filter(ref, sigma=2)
        
        self.ref = ref - bkg
        self.ref_frame = mask[arg]
        
    def _minimisation_routine(self):
        dxs, dys = [], []
    
        # for i in tqdm(range(len(self.flux)), desc='Frames'):
        for i in range(len(self.flux)):
            
            if np.sum(np.isnan(self.flux[i])) >= self.flux[i].shape[0] * self.flux[i].shape[1] * 0.7:
                dxs.append(np.nan)
                dys.append(np.nan)
                continue
            
            # bkg = self._fit_background(self.flux[i], mask=None)
            # bkg = gaussian_filter(self.flux[i], sigma = 2)
            
            dx, dy = self.compute_shift(self.flux[i], self.noise, self.noise[i])
            dxs.append(dx)
            dys.append(dy)

        self.dxs = np.array(dxs)
        self.dys = np.array(dys)
    
    def compute_shift(self, frame, noise, noise_frame):

        def cost_fn(shift_params):
            return self._cost_function_safe(shift_params, frame, noise, noise_frame)

        result = differential_evolution(cost_fn, bounds=[(-3.5, 3.5), (-3.5, 3.5)],
                                        tol=1e-6, strategy='best1bin', 
                                        mutation=0.8, recombination=0.6, polish=True)

        return result.x
    
    def _cost_function_safe(self, shift_params, *args):
        try:
            cost = self._cost_function(shift_params, *args)
            if not np.isfinite(cost):
                return 1e10
            return cost
        except Exception as e:
            print(f"Exception in cost function at shift={shift}: {e}")
            return 1e10
        
    def _cost_function(self, shift_params, flux_cor, mass_noise, noise):
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
        
        # bkg = self._fit_background_plane(flux_cor)

        flux_shifted = self._shift_flux(flux_cor, dx, dy)
        
        diff = (flux_shifted - self.ref)
        
        # alpha = 0.7
        # effective_var = 1.0 / (alpha / (noise + 1e-8)**2 + (1 - alpha) / (mass_noise + 1e-8)**2)

        # return np.nansum(diff**2 / effective_var) #/ (noise + 1e-8)**2
        return np.nansum(diff**2 / (noise + 1e-8)**2) #/ (noise + 1e-8)**2

    def _estimate_initial_shift(self, frame):
        try:
            shift_guess, _, _ = phase_cross_correlation(self.ref, frame, upsample_factor=10)
        except Exception:
            shift_guess = (0.0, 0.0)  # fallback in case of failure
        return shift_guess[1], shift_guess[0]  # return as (dx, dy)
        
    def compute_difference_images_with_psf(self):
        diff_images = []
        # for i in tqdm(range(len(self.flux)), desc = 'Offsetting'):
        for i in range(len(self.flux)):
            if np.isnan(self.flux[i]).sum() >= self.flux[i].shape[0] * self.flux[i].shape[1] *  0.7:
                diff_images.append(np.nan*np.ones_like(self.flux[0]))
                continue
            
            shifted = self._shift_flux(self.flux[i], self.dxs[i], self.dys[i])
            bkg = self._fit_background_plane(shifted - self.ref)
            diff_clean = shifted - self.ref - bkg
            diff_images.append(diff_clean)
        self.diffs = np.array(diff_images)
        self.jump_metrics = np.nansum(np.abs(np.diff(np.array(diff_images), axis=0)), axis=(1,2))

    def _shift_flux(self, flux, dx, dy):
        """
        Shift the flux by dx and dy using a simple 2D shift.
        dx, dy are the number of pixels to shift.
        """
        return shift(flux, shift=(dy, dx), order=3, mode='reflect')

    def _match_psf_and_subtract(self, frame, method='both'):
        """
        Match PSFs by convolving frame or ref, then subtract.
        
        method: 'ref_only', 'both', or 'science_only'
        """
        if method == 'ref_only':
            ref_conv = convolve(self.ref, self.psf, mode='mirror')
            sub = frame - ref_conv
        elif method == 'science_only':
            frame_conv = convolve(frame, self.psf, mode='mirror')
            sub = frame_conv - self.ref
        elif method == 'both':
            frame_conv = convolve(frame, self.psf, mode='mirror')
            ref_conv = convolve(self.ref, self.psf, mode='mirror')
            sub = frame_conv - ref_conv
        else:
            raise ValueError("Invalid method")
        
        return sub

    def _fit_background(self, image, mask=None, degree=2):
        y, x = np.indices(image.shape)
        if mask is None:
            clipped = sigma_clip(image, sigma=2, maxiters=5)
            mask = clipped.mask == False

        p_init = models.Polynomial2D(degree=degree)
        fit_p = LinearLSQFitter()
        p = fit_p(p_init, x[mask], y[mask], image[mask])
        return p(x, y)
    
    def _fit_background_plane(self, image, mask=None):
        y, x = np.indices(image.shape)
        if mask is None:
            mask = np.isfinite(image)
        clipped = np.copy(image)
        clipped[~mask] = np.nan

        # Simple sigma-clipping mask to avoid transient bias
        median = np.nanmedian(clipped)
        std = np.nanstd(clipped)
        bg_mask = (np.abs(clipped - median) < 3 * std)

        model_init = models.Polynomial2D(degree=1)
        fit_p = fitting.LinearLSQFitter()
        model = fit_p(model_init, x[bg_mask], y[bg_mask], image[bg_mask])

        return model(x, y)

