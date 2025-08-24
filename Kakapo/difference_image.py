# import lightkurve as lk
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# from scipy.ndimage import binary_dilation
# from scipy.signal import fftconvolve, find_peaks
# from scipy.ndimage import shift
# from scipy.optimize import minimize
# from scipy.ndimage import gaussian_filter
# from scipy.ndimage import fourier_shift
# from numpy.fft import fftn, ifftn

# from skimage.registration import phase_cross_correlation

# from copy import deepcopy
# from tqdm import tqdm

# READOUT = 95

# class Difference_Imaging():
#     def __init__(self, tpf_info, psf, tol=0.2):
        
#         self.flux = tpf_info.flux
#         self.time_kp = tpf_info.time
#         self.flux_err = tpf_info.flux_err
#         self.quality = tpf_info.quality
#         self.pos_corr1 = tpf_info.pos_corr1
#         self.pos_corr2 = tpf_info.pos_corr2
#         self.psf = psf
#         self.tol = tol

#         self.exptime = np.nanmedian(np.diff(tpf_info.time.value)*24*60*60)
        
#         self.create_diff_image()

#     def create_diff_image(self):
#         # start_time = time
#         self.pos_corr1[np.isnan(self.pos_corr1)] = 0
#         self.pos_corr2[np.isnan(self.pos_corr2)] = 0
        
#         self.r = np.sqrt((self.pos_corr1**2 + self.pos_corr2**2))
#         self._bad_frames()
        
#         self.build_reference()
        
#         self.flux[np.isnan(self.flux)] = 0
#         self.flux_err[np.isnan(self.flux_err)] = 1
#         self.flux[self.bad_frames] = np.nan
        
#         poisson_variance_stack = np.clip(np.abs(self.flux), a_min=0.0, a_max=None) / self.exptime
        
#         # self.noise = np.sqrt(self.flux_err**2 + poisson_variance_stack + (READOUT/self.exptime)**2)
        
#         var_model = np.clip(np.abs(self.flux),0,None) / self.exptime + (READOUT**2)/(self.exptime**2)
#         sigma_model = np.sqrt(var_model)

#         sigma_pipe = np.nan_to_num(self.flux_err, nan=1)  # pipeline-provided σ (e-/s)
#         sigma = np.maximum(sigma_model, sigma_pipe)
        
#         self.noise = sigma
        
#         var_model = np.clip(np.abs(self.flux),0,None) + (READOUT**2)/(self.exptime**2)
#         sigma_model = np.sqrt(var_model)

#         sigma_pipe = np.nan_to_num(self.flux_err, nan=1)  # pipeline-provided σ (e-/s)
#         sigma = np.maximum(sigma_model, sigma_pipe)
        
#         self.modelling_noise = sigma
        
#         self.ref_shape = self.ref.shape
        
#         self._minimisation_routine()
#         self.compute_difference_images_with_psf()
        
#         dist_mask = np.sqrt(self.dxs**2 + self.dys**2) > 2.5
        
#         self._detect_jump_discontinuities(sigma_thresh=8)
        
#         self.diffs[self.bad_frames] = np.nan*np.ones_like(self.flux[0])
#         self.diffs[dist_mask] = np.nan*np.ones_like(self.flux[0])
#         self.diffs[self.bad_jumps] = np.nan*np.ones_like(self.flux[0])
        
#         self.difference_images = deepcopy(self.diffs)
#         self.poisson_noise = np.sqrt(poisson_variance_stack)
        
#         # self.diff_noise = np.sqrt(self.noise**2 + self.ref_noise**2)# + self.diff_noise_model**2)
#         self.diff_noise = np.sqrt(self.diff_noise_model**2)
        
#         self.distance = np.sqrt(self.dxs**2 + self.dys**2)

#     def _detect_jump_discontinuities(self, sigma_thresh=8):
#         median = np.nanmedian(self.jump_metrics)
#         mad = np.nanmedian(np.abs(self.jump_metrics - median))
#         threshold = median + sigma_thresh * mad
#         flagged = np.where(self.jump_metrics > threshold)[0] + 1  # +1 for diff offset
#         self.bad_jumps = flagged

#     def _bad_frames(self):
    
#         bad_frames = []
        
#         for i in range(len(self.flux)):
#             if np.isnan(self.flux[i]).sum() >= self.flux[i].shape[0] * self.flux[i].shape[1] *  0.7:
#                 bad_frames.append(i)
#             elif self.r[i] > 2:
#                 bad_frames.append(i)

#         thrusters = self._thrusters_firing()

#         offsets = np.array([-1, 0, 1, 2])
#         bad_thrusters = (thrusters[:, None] + offsets).ravel()
#         bad_thrusters = np.sort(np.unique(bad_thrusters))

#         bad_frames += thrusters.tolist()
        
#         bad_frames = np.array(bad_frames)
#         bad_frames = bad_frames[(bad_frames >= 0) & (bad_frames <= (len(self.flux) -1))]
#         bad_frames = np.sort(np.unique(bad_frames))
        
#         if len(bad_frames) < 1:
#             mask = self.quality > 0
#             if np.nansum(mask) < 1:
#                 bad_frames = np.array([0, len(self.flux) - 1])
#                 thrusters = np.array([0, len(self.flux) - 1])
#             else:
#                 bad_frames = np.arange(len(self.flux))[mask]
        
#         self.bad_frames = bad_frames
#         self.thrusters = thrusters

#     def _thrusters_firing(self):
        
#         pos1 = self.pos_corr1
#         pos2 = self.pos_corr2

#         mask = (pos1 > -2) & (pos1 < 2)
#         pos1[~mask] = 0
#         pos1[np.isnan(pos1)] = 0

#         mask = (pos2 > -2) & (pos2 < 2)
#         pos2[~mask] = 0
#         pos2[np.isnan(pos2)] = 0

#         r = np.sqrt(pos1**2 + pos2**2)

#         peak_indices = find_peaks(r, distance = 8)[0]
        
#         return peak_indices
    
#     def build_reference(self):
#         mask = np.where((self.r < self.tol) & (self.quality == 0))[0]
#         if len(mask) == 0:
#             mask = np.where(self.quality == 0)[0]
            
#         shifted_stack = []
#         noise_stack = []
        
#         for idx in mask:
#             f = np.copy(self.flux[idx])
#             d_f = np.copy(self.flux_err[idx])
#             dx, dy = self.pos_corr1[idx], self.pos_corr2[idx]
#             f_shifted = self._shift_fourier(f, -dx, -dy)  # shift to reference frame
#             df_shifted = self._shift_fourier(d_f, -dx, -dy)  # shift to reference frame
#             # f_shifted = self._shift_flux(f, -dx, -dy)
#             # df_shifted = self._shift_flux(d_f, -dx, -dy)
#             shifted_stack.append(f_shifted)
#             noise_stack.append(df_shifted)

#         shifted_stack = np.array(shifted_stack)
#         noise_stack = np.array(noise_stack)

#         # arg = np.nanargmin(np.nansum(self.flux[mask], axis = (1,2)))
#         # ref = self.flux[mask[arg]]
#         # ref[np.isnan(ref)] = 0
        
#         # ref = shift(ref, (-self.pos_corr1[mask[arg]], -self.pos_corr2[mask[arg]]))
#         # ref_noise = self.flux_err[mask[arg]]
#         # ref_noise[np.isnan(ref)] = 0
        
#         self.persistent_mask = self.persistent_mask_from_stack(shifted_stack, self.psf, 
#                                                                noise_map_stack = noise_stack, 
#                                                                min_fraction=0.6)
        
#         # ref_transient_mask = self.psf_matched_mask(ref, self.psf, self.flux_err[mask[arg]], 
#         #                                            snr_thresh=4.0, dilate_radius=1)
        
#         # self.persistent_mask = ref_transient_mask
        
#         ref = np.nanmedian(shifted_stack, axis=0)
#         ref_noise = np.nanmedian(noise_stack, axis=0)/np.sqrt(len(noise_stack)) * 1.253
#         bkg_result = self._fit_background_plane_fast(ref, mask = self.persistent_mask)
        
#         bkg = bkg_result['background']
#         # bkg_err = bkg_result['bg_rms']
        
#         # poisson_variance_ref = np.clip(np.abs(ref), a_min=0.0, a_max=None) / self.exptime
#         # bkg_variance_ref = np.clip(np.abs(bkg), a_min=0.0, a_max=None) / self.exptime
#         # ref_noise = np.sqrt(ref_noise**2 + poisson_variance_ref + bkg_variance_ref + (READOUT/self.exptime)**2)# + (bkg_err)**2)
        
#         var_model = (np.clip(np.abs(ref),0,None) + np.clip(np.abs(bkg), a_min=0.0, a_max=None)) / self.exptime + (READOUT**2)/(self.exptime**2)
#         sigma_model = np.sqrt(var_model)

#         sigma_pipe = np.nan_to_num(ref_noise, nan=1)  # pipeline-provided σ (e-/s)
#         sigma = np.maximum(sigma_model, sigma_pipe)
        
#         self.ref_noise = sigma
        
#         var_model = (np.clip(np.abs(ref),0,None) + np.clip(np.abs(bkg), a_min=0.0, a_max=None)) + (READOUT**2)/(self.exptime**2)
#         sigma_model = np.sqrt(var_model)

#         sigma_pipe = np.nan_to_num(ref_noise, nan=1)  # pipeline-provided σ (e-/s)
#         sigma = np.maximum(sigma_model, sigma_pipe)
        
#         ref -= bkg
        
#         self.ref = ref
        
#         # self.ref = self.convolve_with_gaussian(ref, 1.35, factor=1.1)
        
#         self.ref_frame = 'Median' # mask[arg]
        
#         self.modelling_ref_noise = var_model
        
#     def psf_matched_mask(self, img, psf, noise_map, snr_thresh=4.0, dilate_radius=1):
#         """
#         img: 2D science or reference frame (flux)
#         psf: 2D normalized PSF (sum=1)
#         noise_map: 2D sigma estimate per pixel (same shape)
#         """
        
#         mf = self.safe_fftconvolve(img, psf)
#         mf_var = self.safe_fftconvolve(noise_map**2, psf**2)
#         mf_sigma = np.sqrt(mf_var + 1e-12)
#         snr = mf / mf_sigma
#         mask = snr >= snr_thresh
#         if dilate_radius > 0:
#             se = np.ones((2*dilate_radius+1, 2*dilate_radius+1), dtype=bool)
#             mask = binary_dilation(mask, structure=se)
#         return mask
        
#     def persistent_mask_from_stack(self, flux_stack, psf, noise_map_stack=None,
#                                 snr_thresh=4, min_fraction=0.5, dilate_radius=1):
#         """
#         flux_stack: (N_frames, H, W)
#         psf: 2D psf
#         noise_map_stack: (N, H, W) or None
#         min_fraction: fraction of frames where a PSF-detection must occur to mark persistent
#         """
#         N = flux_stack.shape[0]
#         if noise_map_stack is None:
#             noise_map_stack = np.sqrt(np.abs(flux_stack) + 1.0)[:,:, :] # crude global noise per frame

#         masks = np.zeros_like(flux_stack, dtype=bool)
#         for i in range(N):
#             masks[i] = self.psf_matched_mask(flux_stack[i], psf, noise_map_stack[i], 
#                                              snr_thresh=snr_thresh, dilate_radius=0)
        
#         counts = masks.sum(axis=0) # count frames where detection occurs at each pixel
#         persistent = counts >= (min_fraction * N)
#         if dilate_radius > 0:
#             se = np.ones((2*dilate_radius+1, 2*dilate_radius+1), bool)
#             persistent = binary_dilation(persistent, structure=se)
#         return persistent
      
#     def _minimisation_routine(self):
#         dxs, dys, chi2s = [], [], []
    
#         for i in range(len(self.flux)):
            
#             if np.sum(np.isnan(self.flux[i])) >= self.flux[i].shape[0] * self.flux[i].shape[1] * 0.7:
#                 dxs.append(np.nan)
#                 dys.append(np.nan)
#                 continue
            
#             cand_mask = self.psf_matched_mask(self.flux[i], self.psf, 
#                                                    self.modelling_noise[i], snr_thresh=4.0, dilate_radius=1)
            
#             # keep the persistent host; mask only "new" stuff
#             # transient_mask = cand_mask & (~self.persistent_mask)
            
#             bkg_result = self._fit_background_plane_fast(self.flux[i], mask = cand_mask)
#             bkg = bkg_result['background']
#             # bkg_err = bkg_result['bg_rms']

#             self.flux[i] -= bkg
            
#             bkg_var = np.clip(np.abs(bkg), a_min=0.0, a_max=None) / self.exptime**2
            
#             self.modelling_noise[i] = np.sqrt(self.modelling_noise[i]**2 + bkg_var)
            
#             # precompute weight map: 1/σ, but zero where transient
#             # base_var = self.modelling_noise[i]**2 + self.modelling_ref_noise**2 + 1e-12
#             # W = np.ones_like(base_var) / np.sqrt(base_var)
#             # W[transient_mask] = 0.0   # no influence from transients

#             # tukeying = self.tukey2d(*self.ref.shape, alpha=0.5)
#             # W *= tukeying
#             W = 1
            
#             (dx, dy), chi2 = self.compute_shift(self.flux[i], self.modelling_noise[i], W)

#             dxs.append(dx)
#             dys.append(dy)
#             chi2s.append(chi2)

#         self.dxs = np.array(dxs)
#         self.dys = np.array(dys)
#         self.chi2s = np.array(chi2s)
        
#     def tukey2d(self, h, w, alpha=0.5):
#         def tukey(n, a): # separable 1D tukey
#             # simple symmetric Tukey window
#             x = np.linspace(0, 1, n)
#             w = np.ones(n)
#             m = a/2
#             left = x < m
#             right = x > 1 - m
#             w[left] = 0.5*(1 + np.cos(np.pi*(2*x[left]/a - 1)))
#             w[right]= 0.5*(1 + np.cos(np.pi*(2*(1-x[right])/a - 1)))
#             return w
#         tx = tukey(w, alpha)
#         ty = tukey(h, alpha)
#         return np.outer(ty, tx)

#     def compute_shift(self, frame, noise_frame, W):

#         def cost_fn(shift_params):
#             return self._cost_function_safe(shift_params, frame, noise_frame, W)
        
        
#         tukeying = self.tukey2d(*self.ref.shape, alpha=0.5)
#         initial_shift_guess, _, _ = phase_cross_correlation(np.nan_to_num(self.ref * tukeying), 
#                                                             np.nan_to_num(frame * tukeying), upsample_factor=50)
        
#         # x0 = np.array([initial_shift_guess[1], initial_shift_guess[0], 0.0, 0.0])
#         x0 = (initial_shift_guess[1], initial_shift_guess[0])
#         # bounds = [(-3.5, 3.5), (-3.5, 3.5), (np.deg2rad(-0.5), np.deg2rad(0.5)), (np.log(0.995), np.log(1.005))]
#         bounds = [(-3.5, 3.5), (-3.5, 3.5)]
        
#         result = minimize(cost_fn, x0 = x0, method = 'L-BFGS-B', bounds=bounds, tol = 1e-8)
        
#         return result.x, result.fun
    
#     def _cost_function_safe(self, shift_params, *args):
#         try:
#             cost = self._cost_function_zogy(shift_params, *args)
#             if not np.isfinite(cost):
#                 return 1e10
#             return cost
#         except Exception as e:
#             print(f"Exception in cost function at shift={shift_params}: {e}")
#             return 1e10
    
#     def _cost_function_zogy(self, shift_params, flux_frame, flux_noise_frame, W):
#         """
#         ZOGY-style cost function for alignment.

#         Parameters
#         ----------
#         shift_params : (dx, dy)
#             Subpixel shifts to apply to flux_frame.
#         flux_frame : 2D ndarray
#             Science frame to be aligned.
#         flux_noise_frame : 2D ndarray
#             Per-pixel noise of the science frame.
#         ref_noise : 2D ndarray, optional
#             Per-pixel noise of the reference frame. If None, uses self.poisson_noise.

#         Returns
#         -------
#         cost : float
#             Variance-weighted sum of squared differences.
#         """

#         dx, dy = shift_params
        
#         frame_shifted = self._shift_fourier(flux_frame, dx, dy) # Shift science frame and its noise
#         noise_shifted = self._shift_fourier(flux_noise_frame**2, dx, dy)

#         ref = self.ref # Reference frame and noise
#         ref_noise = self.modelling_ref_noise

#         denom = np.sqrt(noise_shifted + ref_noise**2 + 1e-12) # Variance-weighted difference
#         D = (frame_shifted - ref) / denom * W

#         return np.nansum(D**2) # Return sum of squared differences

#     def _estimate_initial_shift(self, frame):
#         try:
#             shift_guess, _, _ = phase_cross_correlation(self.ref, frame, upsample_factor=10)
#         except Exception:
#             shift_guess = (0.0, 0.0)  # fallback in case of failure
#         return shift_guess[1], shift_guess[0]  # return as (dx, dy)
        
#     def compute_difference_images_with_psf(self):
#         diff_images = []
#         diff_noises = []
#         # alphas = []
#         # betas = []
#         # for i in tqdm(range(len(self.flux)), desc = 'Offsetting'):
#         for i in range(len(self.flux)):
#             if np.isnan(self.flux[i]).sum() >= self.flux[i].shape[0] * self.flux[i].shape[1] *  0.7:
#                 diff_images.append(np.nan*np.ones_like(self.flux[0]))
#                 diff_noises.append(np.nan*np.ones_like(self.flux[0]))
#                 # alphas.append(np.nan)
#                 # betas.append(np.nan)
#                 continue
            
#             shifted = self._shift_fourier(self.flux[i], self.dxs[i], self.dys[i])
            
#             ref_c     = self.safe_fftconvolve(self.ref, self.psf)
#             shifted_c = self.safe_fftconvolve(shifted, self.psf)

#             ref_var_c = self.safe_fftconvolve(self.ref_noise**2, self.psf**2)
#             frm_var_c = self.safe_fftconvolve(self.noise[i]**2, self.psf**2)

#             diff_img = shifted_c - ref_c
#             diff_var = frm_var_c + ref_var_c
#             diff_sig = np.sqrt(np.maximum(diff_var, 1e-18))

#             diff_images.append(diff_img)
#             diff_noises.append(diff_sig)
        
#         self.diffs = np.array(diff_images)
#         # self.alphas = np.array(alphas)
#         self.diff_noise_model = np.array(diff_noises)
#         # self.final_noise = np.array(diff_noises)
#         self.jump_metrics = np.nansum(np.abs(np.diff(np.array(diff_images), axis=0)), axis=(1,2))
    
#     def _shift_fourier(self, img, dx, dy, pad=20):
#         """
#         Shift an image using Fourier shift theorem with zero padding.

#         Parameters
#         ----------
#         img : 2D numpy array
#             Input image to shift.
#         dx : float
#             Shift along x-axis (columns).
#         dy : float
#             Shift along y-axis (rows).
#         pad : int
#             Number of pixels to pad with zeros on each side before shifting.

#         Returns
#         -------
#         shifted : 2D numpy array
#             Shifted image, cropped back to original shape.
#         """
#         img[~np.isfinite(img)] = 0
#         img_padded = np.pad(img, pad_width=pad, mode="constant", constant_values=0) # Pad with zeros to avoid wrap-around

#         f = fftn(img_padded)
#         fshift = fourier_shift(f, shift=(dy, dx)) # Fourier shift expects (dy, dx) ordering
#         shifted_padded = np.real(ifftn(fshift))

#         ny, nx = img.shape # Crop back to original size
#         shifted = shifted_padded[pad:pad+ny, pad:pad+nx]

#         return shifted
    
#     def safe_fftconvolve(self, img, psf):
#         mask = np.isfinite(img).astype(float)
#         img_filled = np.nan_to_num(img, nan=0.0)
#         conv_img = fftconvolve(img_filled, psf[::-1, ::-1], mode='same')
#         conv_mask = fftconvolve(mask, psf[::-1, ::-1], mode='same')
#         with np.errstate(invalid='ignore'):
#             conv_img = conv_img / np.maximum(conv_mask, 1e-6)
#         conv_img[conv_mask < 1e-6] = np.nan
#         return conv_img
    
#     def _xy_grid(self, shape):
#         h, w = shape
#         return np.indices((h, w)).astype(np.float32)   # returns (2, h, w)

#     def _fit_background_plane_fast(self, image, mask=None):
#         """
#         Fit a 2-D linear background plane  z = a·x + b·y + c
#         using NumPy least squares (fast) instead of Astropy.

#         Parameters
#         ----------
#         image : 2-D ndarray
#             Difference frame.
#         mask : 2-D bool ndarray or None
#             True for valid pixels.  If None, all finite pixels are valid.

#         Returns
#         -------
#         background : 2-D ndarray
#             Evaluated plane over the entire image.
#         """
        
#         if mask is None: # 1. initial mask
#             mask = np.isfinite(image)
#         else:
#             mask = mask & np.isfinite(image)

#         if not np.any(mask):
#             fallback = np.full_like(image, np.nanmedian(image, overwrite_input=True))
#             return {'background': fallback, 'bg_rms': 0.0}  # or np.nan

#         img_vals = image[mask] # 2. sigma‑clip (one pass, 5σ)
#         med, std = np.nanmedian(img_vals), np.nanstd(img_vals)
#         bg_mask  = mask & (np.abs(image - med) < 3.0 * std)

#         y_grid, x_grid = self._xy_grid(image.shape)
#         x = x_grid[bg_mask].ravel()
#         y = y_grid[bg_mask].ravel()
#         z = image[bg_mask].ravel()

#         A = np.stack((x, y, np.ones_like(x)), axis=1) # (N, 3)
#         coeffs, *_ = np.linalg.lstsq(A, z, rcond=None) # (a, b, c)

#         background = coeffs[0] * x_grid + coeffs[1] * y_grid + coeffs[2]
        
#         residuals = image[bg_mask] - z  # residuals at valid pixels
#         bg_rms = np.nanstd(residuals, ddof = 1)

#         return {'background': background, 'bg_rms': bg_rms}

import numpy as np

from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve, find_peaks
from scipy.optimize import minimize
from scipy.ndimage import fourier_shift
from numpy.fft import fftn, ifftn

from skimage.registration import phase_cross_correlation

from copy import deepcopy
from tqdm import tqdm

READOUT = 95

class Difference_Imaging():
    def __init__(self, tpf_info, psf, tol=0.2):
        
        self.flux = tpf_info.flux
        self.time_kp = tpf_info.time
        self.flux_err = tpf_info.flux_err
        self.quality = tpf_info.quality
        self.pos_corr1 = tpf_info.pos_corr1
        self.pos_corr2 = tpf_info.pos_corr2
        self.psf = psf
        self.tol = tol

        self.exptime = np.nanmedian(np.diff(tpf_info.time.value)*24*60*60)
        
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
        
        poisson_variance_stack = np.clip(np.abs(self.flux), a_min=1.0, a_max=None) #/ self.exptime
        self.noise = np.sqrt(self.flux_err**2 + poisson_variance_stack + (READOUT/self.exptime)**2)
        
        self.ref_shape = self.ref.shape
        
        self._minimisation_routine()
        self.compute_difference_images_with_psf()
        
        dist_mask = np.sqrt(self.dxs**2 + self.dys**2) > 2.5
        
        self._detect_jump_discontinuities(sigma_thresh=8)
        
        self.diffs[self.bad_frames] = np.nan*np.ones_like(self.flux[0])
        self.diffs[dist_mask] = np.nan*np.ones_like(self.flux[0])
        self.diffs[self.bad_jumps] = np.nan*np.ones_like(self.flux[0])
        
        self.difference_images = deepcopy(self.diffs)
        self.poisson_noise = np.sqrt(poisson_variance_stack)
        
        
        true_poisson_variance_stack = np.clip(np.abs(self.flux), a_min=1.0, a_max=None) / self.exptime
        self.true_noise = np.sqrt(self.flux_err**2 + true_poisson_variance_stack + (READOUT/self.exptime)**2)
        
        self.diff_noise = 2*np.sqrt(self.true_noise**2 + self.true_ref_noise**2)# + self.diff_noise_model**2)
        
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
            
        shifted_stack = []
        noise_stack = []
        
        for idx in mask:
            f = np.copy(self.flux[idx])
            d_f = np.copy(self.flux_err[idx])
            dx, dy = self.pos_corr1[idx], self.pos_corr2[idx]
            f_shifted = self._shift_fourier(f, -dx, -dy)  # shift to reference frame
            df_shifted = self._shift_fourier(d_f, -dx, -dy)  # shift to reference frame
            shifted_stack.append(f_shifted)
            noise_stack.append(df_shifted)

        shifted_stack = np.array(shifted_stack)
        noise_stack = np.array(noise_stack)
        
        self.persistent_mask = self.persistent_mask_from_stack(shifted_stack, self.psf, 
                                                               noise_map_stack = noise_stack, 
                                                               min_fraction=0.6)
        
        ref = np.nanmedian(shifted_stack, axis=0)
        ref_noise = np.nanmedian(noise_stack, axis=0)/np.sqrt(len(noise_stack)) * 1.253
        bkg_result = self._fit_background_plane_fast(ref, mask = self.persistent_mask)
        
        bkg = bkg_result['background']
        
        poisson_variance_ref = np.clip(np.abs(ref + bkg), a_min=1.0, a_max=None) #/ self.exptime
        true_poisson_variance_ref = np.clip(np.abs(ref + bkg), a_min=1.0, a_max=None) / self.exptime
        bkg_variance_ref = np.clip(np.abs(bkg), a_min=1.0, a_max=None) #/ self.exptime
        true_bkg_variance_ref = np.clip(np.abs(bkg), a_min=1.0, a_max=None) / self.exptime
        ref_noise = np.sqrt(ref_noise**2 + poisson_variance_ref + bkg_variance_ref + (READOUT/self.exptime)**2)# + (bkg_err)**2)
        true_ref_noise = np.sqrt(ref_noise**2 + true_poisson_variance_ref + true_bkg_variance_ref + (READOUT/self.exptime)**2)# + (bkg_err)**2)
        
        ref -= bkg
        
        self.ref = ref
        
        self.ref_frame = 'Median' # mask[arg]
        self.ref_noise = ref_noise
        self.true_ref_noise = true_ref_noise
        
    def psf_matched_mask(self, img, psf, noise_map, snr_thresh=4.0, dilate_radius=1):
        """
        img: 2D science or reference frame (flux)
        psf: 2D normalized PSF (sum=1)
        noise_map: 2D sigma estimate per pixel (same shape)
        """
        
        mf = self.safe_fftconvolve(img, psf)
        mf_var = self.safe_fftconvolve(noise_map**2, psf**2)
        mf_sigma = np.sqrt(mf_var + 1e-12)
        snr = mf / mf_sigma
        mask = snr >= snr_thresh
        if dilate_radius > 0:
            se = np.ones((2*dilate_radius+1, 2*dilate_radius+1), dtype=bool)
            mask = binary_dilation(mask, structure=se)
        return mask
        
    def persistent_mask_from_stack(self, flux_stack, psf, noise_map_stack=None,
                                snr_thresh=4, min_fraction=0.5, dilate_radius=1):
        """
        flux_stack: (N_frames, H, W)
        psf: 2D psf
        noise_map_stack: (N, H, W) or None
        min_fraction: fraction of frames where a PSF-detection must occur to mark persistent
        """
        N = flux_stack.shape[0]
        if noise_map_stack is None:
            noise_map_stack = np.sqrt(np.abs(flux_stack) + 1.0)[:,:, :] # crude global noise per frame

        masks = np.zeros_like(flux_stack, dtype=bool)
        for i in range(N):
            masks[i] = self.psf_matched_mask(flux_stack[i], psf, noise_map_stack[i], 
                                             snr_thresh=snr_thresh, dilate_radius=0)
        
        counts = masks.sum(axis=0) # count frames where detection occurs at each pixel
        persistent = counts >= (min_fraction * N)
        if dilate_radius > 0:
            se = np.ones((2*dilate_radius+1, 2*dilate_radius+1), bool)
            persistent = binary_dilation(persistent, structure=se)
        return persistent
      
    def _minimisation_routine(self):
        dxs, dys, chi2s = [], [], []
    
        # for i in tqdm(range(len(self.flux)), desc='Frames'):
        for i in range(len(self.flux)):
            
            if np.sum(np.isnan(self.flux[i])) >= self.flux[i].shape[0] * self.flux[i].shape[1] * 0.7:
                dxs.append(np.nan)
                dys.append(np.nan)
                continue
            
            transient_mask = self.psf_matched_mask(self.flux[i], self.psf, 
                                                   self.noise[i], snr_thresh=4.0, dilate_radius=1)
            
            bkg_result = self._fit_background_plane_fast(self.flux[i], mask = transient_mask)
            bkg = bkg_result['background']
            # bkg_err = bkg_result['bg_rms']

            self.flux[i] -= bkg
            
            # bkg_var = np.clip(np.abs(bkg), a_min=0.0, a_max=None)# / self.exptime**2
            
            # self.noise[i] = np.sqrt(self.noise[i]**2 + bkg_var)
            
            (dx, dy), chi2 = self.compute_shift(self.flux[i], self.noise[i])

            dxs.append(dx)
            dys.append(dy)
            chi2s.append(chi2)

        self.dxs = np.array(dxs)
        self.dys = np.array(dys)
        self.chi2s = np.array(chi2s)
        
    def tukey2d(self, h, w, alpha=0.5):
        def tukey(n, a): # separable 1D tukey
            # simple symmetric Tukey window
            x = np.linspace(0, 1, n)
            w = np.ones(n)
            m = a/2
            left = x < m
            right = x > 1 - m
            w[left] = 0.5*(1 + np.cos(np.pi*(2*x[left]/a - 1)))
            w[right]= 0.5*(1 + np.cos(np.pi*(2*(1-x[right])/a - 1)))
            return w
        tx = tukey(w, alpha)
        ty = tukey(h, alpha)
        return np.outer(ty, tx)
    
    def huber(self, r, delta=3.0):
        a = np.abs(r)
        quad = a <= delta
        out = np.empty_like(a)
        out[quad] = 0.5 * (a[quad]**2)
        out[~quad] = delta * (a[~quad] - 0.5*delta)
        return out

    def compute_shift(self, frame, noise_frame):

        def cost_fn(shift_params):
            return self._cost_function_safe(shift_params, frame, noise_frame)
        
        
        tukeying = self.tukey2d(*self.ref.shape, alpha=0.5)
        initial_shift_guess, _, _ = phase_cross_correlation(np.nan_to_num(self.ref * tukeying), 
                                                            np.nan_to_num(frame * tukeying), upsample_factor=50)
        
        # x0 = np.array([initial_shift_guess[1], initial_shift_guess[0], 0.0, 0.0])
        x0 = (initial_shift_guess[1], initial_shift_guess[0])
        # bounds = [(-3.5, 3.5), (-3.5, 3.5), (np.deg2rad(-0.5), np.deg2rad(0.5)), (np.log(0.995), np.log(1.005))]
        bounds = [(-3.5, 3.5), (-3.5, 3.5)]
        
        result = minimize(cost_fn, x0 = x0, method = 'L-BFGS-B', bounds=bounds, tol = 1e-8)
        
        return result.x, result.fun
    
    def _cost_function_safe(self, shift_params, *args):
        try:
            cost = self._cost_function_zogy(shift_params, *args)
            if not np.isfinite(cost):
                return 1e10
            return cost
        except Exception as e:
            print(f"Exception in cost function at shift={shift_params}: {e}")
            return 1e10
    
    def _cost_function_zogy(self, shift_params, flux_frame, flux_noise_frame):
        """
        ZOGY-style cost function for alignment.

        Parameters
        ----------
        shift_params : (dx, dy)
            Subpixel shifts to apply to flux_frame.
        flux_frame : 2D ndarray
            Science frame to be aligned.
        flux_noise_frame : 2D ndarray
            Per-pixel noise of the science frame.
        ref_noise : 2D ndarray, optional
            Per-pixel noise of the reference frame. If None, uses self.poisson_noise.

        Returns
        -------
        cost : float
            Variance-weighted sum of squared differences.
        """

        dx, dy = shift_params
        
        frame_shifted = self._shift_fourier(flux_frame, dx, dy) # Shift science frame and its noise
        noise_shifted = self._shift_fourier(flux_noise_frame**2, dx, dy)

        ref = self.ref # Reference frame and noise
        ref_noise = self.ref_noise
        
        tukeying = self.tukey2d(*ref.shape, alpha=0.5)

        denom = np.sqrt(noise_shifted + ref_noise**2 + 1e-12) # Variance-weighted difference
        D = (frame_shifted - ref) / denom * tukeying

        return np.nansum(D**2) # Return sum of squared differences
        
    def compute_difference_images_with_psf(self):
        diff_images = []
        diff_noises = []
        alphas = []
        # for i in tqdm(range(len(self.flux)), desc = 'Offsetting'):
        for i in range(len(self.flux)):
            if np.isnan(self.flux[i]).sum() >= self.flux[i].shape[0] * self.flux[i].shape[1] *  0.7:
                diff_images.append(np.nan*np.ones_like(self.flux[0]))
                diff_noises.append(np.nan*np.ones_like(self.flux[0]))
                alphas.append(np.nan)
                continue
            
            shifted = self._shift_fourier(self.flux[i], self.dxs[i], self.dys[i])

            ref = self.safe_fftconvolve(self.ref, self.psf)
            
            shifted = self.safe_fftconvolve(shifted, self.psf)

            ref_var_c = self.safe_fftconvolve(self.ref_noise**2, self.psf**2)
            frm_var_c = self.safe_fftconvolve(self.noise[i]**2, self.psf**2)

            diff_sig  = np.sqrt(ref_var_c + frm_var_c) 
            
            diff_clean = shifted - ref #- bkg_final
            diff_images.append(diff_clean)
            diff_noises.append(diff_sig)
        
        self.diffs = np.array(diff_images)
        self.alphas = np.array(alphas)
        self.diff_noise_model = np.array(diff_noises)
        # self.final_noise = np.array(diff_noises)
        self.jump_metrics = np.nansum(np.abs(np.diff(np.array(diff_images), axis=0)), axis=(1,2))
    
    def _shift_fourier(self, img, dx, dy, pad=20):
        """
        Shift an image using Fourier shift theorem with zero padding.

        Parameters
        ----------
        img : 2D numpy array
            Input image to shift.
        dx : float
            Shift along x-axis (columns).
        dy : float
            Shift along y-axis (rows).
        pad : int
            Number of pixels to pad with zeros on each side before shifting.

        Returns
        -------
        shifted : 2D numpy array
            Shifted image, cropped back to original shape.
        """
        img[~np.isfinite(img)] = 0
        img_padded = np.pad(img, pad_width=pad, mode="constant", constant_values=0) # Pad with zeros to avoid wrap-around

        f = fftn(img_padded)
        fshift = fourier_shift(f, shift=(dy, dx)) # Fourier shift expects (dy, dx) ordering
        shifted_padded = np.real(ifftn(fshift))

        ny, nx = img.shape # Crop back to original size
        shifted = shifted_padded[pad:pad+ny, pad:pad+nx]

        return shifted
    
    def safe_fftconvolve(self, img, psf):
        mask = np.isfinite(img).astype(float)
        img_filled = np.nan_to_num(img, nan=0.0)
        conv_img = fftconvolve(img_filled, psf[::-1, ::-1], mode='same')
        conv_mask = fftconvolve(mask, psf[::-1, ::-1], mode='same')
        with np.errstate(invalid='ignore'):
            conv_img = conv_img / np.maximum(conv_mask, 1e-6)
        conv_img[conv_mask < 1e-6] = np.nan
        return conv_img
    
    def _xy_grid(self, shape):
        h, w = shape
        return np.indices((h, w)).astype(np.float32)   # returns (2, h, w)

    def _fit_background_plane_fast(self, image, mask=None):
        """
        Fit a 2-D linear background plane  z = a·x + b·y + c
        using NumPy least squares (fast) instead of Astropy.

        Parameters
        ----------
        image : 2-D ndarray
            Difference frame.
        mask : 2-D bool ndarray or None
            True for valid pixels.  If None, all finite pixels are valid.

        Returns
        -------
        background : 2-D ndarray
            Evaluated plane over the entire image.
        """
        
        if mask is None: # 1. initial mask
            mask = np.isfinite(image)
        else:
            mask = mask & np.isfinite(image)

        if not np.any(mask):
            fallback = np.full_like(image, np.nanmedian(image, overwrite_input=True))
            return {'background': fallback, 'bg_rms': 0.0}  # or np.nan

        img_vals = image[mask] # 2. sigma‑clip (one pass, 5σ)
        med, std = np.nanmedian(img_vals), np.nanstd(img_vals)
        bg_mask  = mask & (np.abs(image - med) < 3.0 * std)

        y_grid, x_grid = self._xy_grid(image.shape)
        x = x_grid[bg_mask].ravel()
        y = y_grid[bg_mask].ravel()
        z = image[bg_mask].ravel()

        A = np.stack((x, y, np.ones_like(x)), axis=1) # (N, 3)
        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None) # (a, b, c)

        background = coeffs[0] * x_grid + coeffs[1] * y_grid + coeffs[2]
        
        residuals = image[bg_mask] - z  # residuals at valid pixels
        bg_rms = np.nanstd(residuals, ddof = 1)

        return {'background': background, 'bg_rms': bg_rms}