import numpy as np

from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve, find_peaks
from scipy.optimize import minimize
from scipy.ndimage import fourier_shift
from numpy.fft import fftn, ifftn
from scipy.ndimage import gaussian_filter

from skimage.registration import phase_cross_correlation

from astropy.stats import sigma_clipped_stats

from Kakapo.cleaning_curve import fit_psf_fwhm
from Kakapo.robust_pca import remove_thermal_rpca_trends, clean_thermal_cadence

from copy import deepcopy
from tqdm import tqdm

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
        
        _, _, self.fwhm = fit_psf_fwhm(psf)

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
        self.noise = np.sqrt(self.flux_err**2 + poisson_variance_stack)
        self.true_noise = np.sqrt(self.flux_err**2)
        
        self.beta_max = np.nanmedian(self.true_noise)
        # self.beta_max = np.nanmedian(self.noise)
        
        self.ref_shape = self.ref.shape
        
        # self._original_background_factor(self.flux, self.flux_err)
        self._background_mapping()
        self._minimisation_routine()
        
        self.T_max = 2*np.nanpercentile(self.bkg_scalar, 95)
        
        self.distance = np.sqrt(self.dxs**2 + self.dys**2)
        
        # motion_factors = self._motion_inflation(self.distance)
        # self.motion_factors = motion_factors
        
        self.compute_difference_images_with_psf()
        
        dist_mask = np.sqrt(self.dxs**2 + self.dys**2) > 2.5
        self._detect_jump_discontinuities(sigma_thresh=8)
        
        self.diffs[self.bad_frames] = np.nan*np.ones_like(self.flux[0])
        self.diffs[dist_mask] = np.nan*np.ones_like(self.flux[0])
        self.diffs[self.bad_jumps] = np.nan*np.ones_like(self.flux[0])
        
        # persistent_mask = self._persistent_mask_from_stack(self.diffs, self.psf, self.diff_noise_model, 
        #                                                    snr_thresh=4, min_fraction=0.6, dilate_radius=2)
        # mask_stack = np.broadcast_to(persistent_mask, self.diffs.shape)
        
        self.difference_images = deepcopy(self.diffs)
        self.poisson_noise = np.sqrt(poisson_variance_stack)
        
        # true_poisson_variance_stack = np.clip(np.abs(self.flux), a_min=1.0, a_max=None) / self.exptime
        
        self.diff_noise = self.diff_noise_model.copy()# np.sqrt(self.true_noise**2 + self.true_ref_noise**2)# + self.diff_noise_model**2)

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
        
        self.persistent_mask = self._persistent_mask_from_stack(shifted_stack, self.psf, 
                                                               noise_map_stack = noise_stack, 
                                                               min_fraction=0.6)
        
        ref = np.nanmedian(shifted_stack, axis=0)
        init_ref_noise = np.nanmedian(noise_stack, axis=0)/np.sqrt(len(noise_stack)) * 1.253
        bkg_result = self._fit_background_plane_fast(ref, mask = self.persistent_mask)
        
        bkg = bkg_result['background']
        
        poisson_variance_ref = np.clip(np.abs(ref), a_min=1.0, a_max=None) #/ self.exptime
        # true_poisson_variance_ref = np.clip(np.abs(ref + bkg), a_min=1.0, a_max=None) / self.exptime
        bkg_variance_ref = np.clip(np.abs(bkg), a_min=1.0, a_max=None) #/ self.exptime
        true_bkg_variance_ref = np.clip(np.abs(bkg), a_min=1.0, a_max=None) / self.exptime
        ref_noise = np.sqrt(init_ref_noise**2 + poisson_variance_ref + bkg_variance_ref)# + (bkg_err)**2)
        true_ref_noise = np.sqrt(init_ref_noise**2 + true_bkg_variance_ref)# + (bkg_err)**2)
        
        ref -= bkg
        
        self.ref = ref
        
        self.ref_frame = 'Median' # mask[arg]
        self.ref_noise = ref_noise
        self.true_ref_noise = true_ref_noise
        
    def _psf_matched_mask(self, img, psf, noise_map, snr_thresh=4.0, dilate_radius=1):
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
        
    def _persistent_mask_from_stack(self, flux_stack, psf, noise_map_stack=None,
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
            masks[i] = self._psf_matched_mask(flux_stack[i], psf, noise_map_stack[i], 
                                             snr_thresh=snr_thresh, dilate_radius=0)
        
        counts = masks.sum(axis=0) # count frames where detection occurs at each pixel
        persistent = counts >= (min_fraction * N)
        if dilate_radius > 0:
            se = np.ones((2*dilate_radius+1, 2*dilate_radius+1), bool)
            persistent = binary_dilation(persistent, structure=se)
        return persistent
      
    def _background_mapping(self):
        bkg_var_list = []
        bkg_list = []
        bkg_scalar = []
    
        for i in range(len(self.flux)):
            
            if np.sum(np.isnan(self.flux[i])) >= self.flux[i].shape[0] * self.flux[i].shape[1] * 0.7:
                bkg_var_list.append(np.nan*np.ones_like(self.flux[i]))
                bkg_list.append(np.nan*np.ones_like(self.flux[i]))
                bkg_scalar.append(np.nan)
                continue
            
            transient_mask = self._psf_matched_mask(self.flux[i], self.psf, 
                                                   self.noise[i], snr_thresh=4.0, dilate_radius=1)
            
            bkg_result = self._fit_background_plane_fast(self.flux[i], mask = transient_mask)
            bkg = bkg_result['background']
            
            bkg_var = np.clip(np.abs(bkg), a_min=1.0, a_max=None) / self.exptime
            
            bkg_list.append(bkg)
            bkg_scalar.append(np.nanmedian(bkg))
            bkg_var_list.append(bkg_var)

        self.bkg_var_arr = np.array(bkg_var_list)
        self.bkg_arr = np.array(bkg_list)
        self.bkg_scalar = np.array(bkg_scalar)
        np.save('bkg_evol.npy', self.bkg_arr)
    
    def _minimisation_routine(self):
        dxs, dys, chi2s = [], [], []
        ds, ps, cs, betas, alpha_Bs = [], [], [], [], []
    
        # for i in tqdm(range(len(self.flux)), desc='Frames'):
        for i in range(len(self.flux)):
            
            if np.nansum(np.isnan(self.flux[i])) >= self.flux[i].shape[0] * self.flux[i].shape[1] * 0.7:
                dxs.append(np.nan)
                dys.append(np.nan)
                chi2s.append(np.nan)
                ds.append(np.nan)
                ps.append(np.nan)
                cs.append(np.nan)
                betas.append(np.nan)
                alpha_Bs.append(np.nan)
                continue
            
            self.flux[i] -= self.bkg_arr[i]
            
            # (dx, dy, log_p, dlog_sigma, c, beta, alpha_B), chi2 = self.compute_shift(self.flux[i], self.noise[i], self.bkg_arr[i])
            (dx, dy, log_p, dlog_sigma, c, beta, alpha_B), chi2 = self.compute_shift(self.flux[i], self.noise[i], self.bkg_arr[i])

            dxs.append(dx)
            dys.append(dy)
            chi2s.append(chi2)
            ps.append(np.exp(log_p))
            ps.append(np.exp(log_p))
            ds.append(np.exp(dlog_sigma))
            cs.append(c)
            betas.append(beta)
            alpha_Bs.append(alpha_B)

        self.dxs = np.array(dxs)
        self.dys = np.array(dys)
        self.chi2s = np.array(chi2s)
        self.ds = np.array(ds)
        self.ps = np.array(ps)
        self.cs = np.array(cs)
        self.betas = np.array(betas)
        self.alpha_Bs = np.array(alpha_Bs)
        
    def _tukey2d(self, h, w, alpha=0.5):
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
    
    def _huber(self, r, delta=3.0):
        a = np.abs(r)
        quad = a <= delta
        out = np.empty_like(a)
        out[quad] = 0.5 * (a[quad]**2)
        out[~quad] = delta * (a[~quad] - 0.5*delta)
        return out

    def compute_shift(self, frame, noise_frame, bkg_frame):

        def cost_fn(shift_params):
            return self._cost_function_safe(shift_params, frame, noise_frame, bkg_frame)
        
        tukeying = self._tukey2d(*self.ref.shape, alpha=0.5)
        initial_shift_guess, _, _ = phase_cross_correlation(np.nan_to_num(self.ref * tukeying), 
                                                            np.nan_to_num(frame * tukeying), upsample_factor=100)
        
        _, _, stddev = sigma_clipped_stats(bkg_frame)
        
        # x0 = np.array([initial_shift_guess[1], initial_shift_guess[0], 0.0, 0.0])
        # x0 = (initial_shift_guess[1], initial_shift_guess[0])
        x0 = (initial_shift_guess[1], initial_shift_guess[0], 0, 0, 0, 0.1, 0.1)
        # x0 = (initial_shift_guess[1], initial_shift_guess[0], 0, 0, 0, 0.1)
        # bounds = [(-3.5, 3.5), (-3.5, 3.5), (np.deg2rad(-0.5), np.deg2rad(0.5)), (np.log(0.995), np.log(1.005))]
        # bounds = [(-3.5, 3.5), (-3.5, 3.5), (-0.25, 0.25), (-0.25, 0.25), 
        #           (-10*stddev, 10*stddev), (0, self.beta_max/2.5), (0, 3)]
        
        # log_p, dlog_sigma, c, beta, alpha_B
        bounds = [(-3.5, 3.5), (-3.5, 3.5), (-0.25, 0.25), (-0.25, 0.25),
                  (-10*stddev, 10*stddev), (0, self.beta_max), (0, 3)]
        
        # result = minimize(cost_fn, x0 = x0, method = 'L-BFGS-B', bounds=bounds, tol = 1e-8)
        result = minimize(cost_fn, x0 = x0, method = 'Powell', bounds=bounds, tol = 1e-8)
        
        # D, sigma_D, S_N = self.zogy_difference(result.x, flux_frame=frame, flux_noise_frame=noise_frame,
        #                                        bkg_poisson=bkg_frame, t_factor=1, ref_noise=self.ref_noise)

        # # Build PSF-core mask from S/N map
        # mask = self._psf_matched_mask(img=D, psf=self.psf, noise_map=sigma_D)
        
        # def cost_fn_2(shift_params):
        #     return self._cost_function_chi2(shift_params, frame, noise_frame, bkg_frame, mask)
        
        # # result_tt = minimize(cost_fn_2, x0 = result.x, method = 'L-BFGS-B', bounds=bounds, tol = 1e-8)
        # result_tt = minimize(cost_fn_2, x0 = result.x, method = 'Powell', bounds=bounds, tol = 1e-8)
        
        return result.x, result.fun
    
    def _cost_function_safe(self, shift_params, *args):
        try:
            # cost = self._cost_function_zogy(shift_params, *args)
            cost = self._cost_function_zogy_scale_focus(shift_params, *args)
            if not np.isfinite(cost):
                return 1e10
            return cost
        except Exception as e:
            print(f"Exception in cost function at shift={shift_params}: {e}")
            return 1e10
    
    def _cost_function_zogy(self, shift_params, flux_frame, flux_noise_frame, bkg_frame):
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
        noise_shifted = self._shift_fourier((flux_noise_frame)**2, dx, dy)

        ref = self.ref # Reference frame and noise
        ref_noise = self.ref_noise
        
        r = np.sqrt(dx**2 + dy**2)
        # alpha_motion = self._motion_inflation(r)
        
        tukeying = self._tukey2d(*ref.shape, alpha=0.5)

        denom = np.sqrt(noise_shifted + ref_noise**2 + 1e-12) # Variance-weighted difference
        D = (frame_shifted - ref) / denom * tukeying
        
        # w = self._tukey_biweight(D)
        w = 1

        return np.nansum(w*D**2) # Return sum of squared differences
        
    def compute_difference_images_with_psf(self):
        diff_images = []
        diff_noises = []
        # for i in tqdm(range(len(self.flux)), desc = 'Offsetting'):
        for i in range(len(self.flux)):
            if np.isnan(self.flux[i]).sum() >= self.flux[i].shape[0] * self.flux[i].shape[1] *  0.7:
                diff_images.append(np.nan*np.ones_like(self.flux[0]))
                diff_noises.append(np.nan*np.ones_like(self.flux[0]))
                continue
            
            # D, sigma_D, S_N = self.zogy_difference([self.dxs[i], self.dys[i], self.ds[i], self.ps[i],
            #                                         self.cs[i], self.betas[i], self.alpha_Bs[i]], 
            #                                        self.flux[i], self.true_noise[i], self.bkg_var_arr[i], 
            #                                        t_factor = self.exptime, ref_noise = self.true_ref_noise)
            
            D, sigma_D, S_N = self.zogy_difference([self.dxs[i], self.dys[i], self.ds[i], self.ps[i],
                                                    self.cs[i], self.betas[i], self.alpha_Bs[i]], 
                                                   self.flux[i], self.true_noise[i], self.bkg_var_arr[i], 
                                                   t_factor = self.exptime, ref_noise = self.true_ref_noise)
            
            # diff_clean = model - ref #- bkg_final
            diff_images.append(D)
            diff_noises.append(sigma_D)
        
        self.diffs = np.array(diff_images)
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
    
    def _original_background_factor(self, flux, flux_err):
        bkgs = []

        for i in range(len(flux)):

            temp_data = flux[i]
            temp_data_err = flux_err[i]
            temp_data[temp_data < 0] = 0.05
            
            bkg = np.nanmedian(np.sqrt(temp_data_err**2 - (np.sqrt(temp_data*self.exptime)/self.exptime)**2 - (95/self.exptime)**2))
            
            if np.isfinite(bkg) & np.isfinite(self.time_kp.value[i]):
                bkgs.append(bkg)
            else:
                bkgs.append(np.nan)
            
        bkgs = np.array(bkgs)
        bkgs /= np.nanpercentile(bkgs, 2)
        bkgs = np.clip(bkgs, 1, 5)
        self.bkg_factors = bkgs
        
    def _motion_inflation(self, r):
        fmax=5.0
        p=1.2
        r = np.asarray(r, float)
        f = 1.0 + (r / self.tol/3)**p
        return np.sqrt(np.clip(f, 1.0, fmax))

    def _cost_function_zogy_scale_focus(self, params, flux_frame, flux_noise_frame, bkg_poisson):
        """
        Faithful ZOGY-style cost function with:
        - subpixel shift (dx, dy)
        - photometric scale (p)
        - PSF width perturbation (ds)
        - background offset (c)
        - variance inflation (α, β)

        Parameters
        ----------
        params : tuple
            (dx, dy, log_p, dlog_sigma, c, beta, alpha_B)
        flux_frame : 2D ndarray
            Science frame (S).
        flux_noise_frame : 2D ndarray
            Per-pixel noise estimate for science.
        bkg_poisson : 2D ndarray
            Background variance estimate for science.
        """

        # --- Generate difference image, noise map, significance image ---
        D, sigma_D, S_N = self.zogy_difference(params, flux_frame=flux_frame, 
                                               flux_noise_frame=flux_noise_frame,
                                               bkg_poisson=bkg_poisson, t_factor=1,
                                               ref_noise=self.ref_noise)
        
        R_clean, W = self.student_t_clean_whitened(D, sigma_D, nu=6.0, iters=2, use_mad_scale=True)

        # --- Chi² in real space (hybrid form) ---
        # chi2_map = (D / (sigma_D + 1e-12))**2
        
        nu = 6.0

        loss = 0.5*(nu+1) * np.nansum(np.log1p(R_clean**2 / nu))
        
        # chi2 = np.nansum(chi2_map)

        return loss

    # def _cost_function_zogy(self, shift_params, flux_frame, flux_noise_frame):
    #     """
    #     ZOGY-style cost function for alignment.

    #     Parameters
    #     ----------
    #     shift_params : (dx, dy)
    #         Subpixel shifts to apply to flux_frame.
    #     flux_frame : 2D ndarray
    #         Science frame to be aligned.
    #     flux_noise_frame : 2D ndarray
    #         Per-pixel noise of the science frame.
    #     ref_noise : 2D ndarray, optional
    #         Per-pixel noise of the reference frame. If None, uses self.poisson_noise.

    #     Returns
    #     -------
    #     cost : float
    #         Variance-weighted sum of squared differences.
    #     """

    #     dx, dy = shift_params
        
    #     frame_shifted = self._shift_fourier(flux_frame, dx, dy) # Shift science frame and its noise
    #     noise_shifted = self._shift_fourier((flux_noise_frame)**2, dx, dy)

    #     ref = self.ref # Reference frame and noise
    #     ref_noise = self.ref_noise
        
    #     r = np.sqrt(dx**2 + dy**2)
    #     # alpha_motion = self._motion_inflation(r)
        
    #     tukeying = self._tukey2d(*ref.shape, alpha=0.5)
    #     denom = np.sqrt(noise_shifted + ref_noise**2 + 1e-12) # Variance-weighted difference
    #     D = (frame_shifted - ref) / denom * tukeying

    #     return np.nansum(D**2) # Return sum of squared differences

    def _broaden_epsf(self, epsf, delta_sigma):
        """
        Broaden a small empirical PSF by convolving with a Gaussian kernel.

        Parameters
        ----------
        epsf : 2D ndarray
            Empirical PSF stamp (e.g. 5x5).
        delta_sigma : float
            Extra Gaussian sigma to add in quadrature.

        Returns
        -------
        psf_broadened : 2D ndarray
            Broadened PSF, normalized to sum=1.
        """
        if delta_sigma <= 0:
            return epsf / np.nansum(epsf)
        
        pad = int(3*delta_sigma)
        epsf_padded = np.pad(epsf, pad, mode="constant")
        broadened = gaussian_filter(epsf_padded, delta_sigma, mode="constant")
        broadened = broadened / broadened.sum()
        # Crop back
        broadened = broadened[pad:-pad, pad:-pad]

        return broadened / np.nansum(broadened)

    def zogy_difference(self, params, flux_frame, flux_noise_frame, bkg_poisson, t_factor = 1, ref_noise = None):
        """
        Construct ZOGY difference image + noise map.
        
        Parameters
        ----------
        params : tuple
            (dx, dy, log_p, dlog_sigma, c, beta, alpha_B)
        flux_frame : 2D ndarray
            Science frame (S).
        flux_noise_frame : 2D ndarray
            Per-pixel noise estimate for science.
        bkg_poisson : 2D ndarray
            Background variance estimate for science.
        
        Returns
        -------
        D : 2D ndarray
            ZOGY difference image.
        sigma_D : 2D ndarray
            Standard deviation (noise) map of D.
        S_N : 2D ndarray
            Significance image (S/N per pixel).
        """

        # dx, dy, log_p, dlog_sigma, c, beta, alpha_B = params
        dx, dy, log_p, dlog_sigma, c, beta, alpha_B = params
        ds = np.exp(dlog_sigma)
        p = np.exp(log_p)

        # --- Shift + scale science frame ---
        S = self._shift_fourier(flux_frame, dx, dy)
        noise_shifted = self._shift_fourier((flux_noise_frame)**2, dx, dy)
        S = p*S + c

        # --- Variance maps ---
        # var_S = (noise_shifted +
        #         (1 + alpha_B) * np.clip(bkg_poisson, 0, None)/t_factor +
        #         beta * (dx**2 + dy**2))

        var_S = (noise_shifted +
                (1 + alpha_B) * np.clip(bkg_poisson, 0, None)/t_factor +
                beta * (dx**2 + dy**2))

        R = self.ref
        if ref_noise is None:
            var_R = self.ref_noise**2
        else:
            var_R = ref_noise**2

        # --- Fourier transforms ---
        # S_hat = fftn(S)
        # R_hat = fftn(R)

        # --- PSFs in Fourier space ---
        # P_S = fftn(self._broaden_epsf(self.psf, ds), S.shape)
        # P_R = fftn(self.psf, R.shape)

        # --- Numerator & denominator (ZOGY Eq. 13) ---
        # numerator   = R_hat * P_S - S_hat * P_R
        numerator   = S - R
        # denom_noise = var_R * np.abs(P_S)**2 + var_S * np.abs(P_R)**2
        denom_noise = var_R + var_S
        denom_noise = np.maximum(denom_noise, 1e-12)

        # --- Difference image in Fourier space ---
        D_hat = numerator / np.sqrt(denom_noise)
        # D     = np.real(ifftn(D_hat))

        # # --- Noise image ---
        # sigma_D = np.sqrt(np.real(ifftn(1.0 / denom_noise)))  # careful: not pixel-by-pixel, but an approx

        # # --- Significance image ---
        # S_N = np.divide(D, sigma_D, where=(sigma_D > 0))

        # return D, sigma_D, S_N
        return numerator, np.sqrt(denom_noise), D_hat

    def student_t_clean_whitened(self, D, sigma, nu=6.0, iters=2, use_mad_scale=True):
        """
        Input:
            D      : difference image (H, W) or stack (T, H, W)
            sigma  : per-pixel noise map, same shape as D
            nu     : Student-t degrees of freedom (~4..8 good; 6 default)
            iters  : small number of IRLS rounds (1-3 is fine)
            use_mad_scale : keep thresholds meaningful under non-Gaussian tails
        Returns:
            R_clean : whitened, Student-t weighted residuals (same shape as D)
            W       : final per-pixel weight map in quadratic form (same shape)
        """
        eps = 1e-12
        R = np.divide(D, sigma + eps)  # whitened
        valid = np.isfinite(R)
        R = np.where(valid, R, 0.0)

        def mad(x):
            x = x[np.isfinite(x)]
            if x.size == 0: return 1.0
            return np.nanmedian(np.abs(x - np.nanmedian(x))) + eps

        for _ in range(iters):
            if use_mad_scale:
                s = mad(R)
                Rn = R / s
            else:
                Rn = R

            # Student-t IRLS weight (quadratic weight)
            W = (nu + 1.0) / (nu + Rn**2)

            # Form the “cleaned whitened residuals” used for L2 detection:
            # Equivalent to pre-multiplying by sqrt(weight).
            R = np.sqrt(W) * R

        # Final weights corresponding to the last iteration’s residuals
        if use_mad_scale:
            s = mad(R)
            Rn = R / s
        else:
            Rn = R
        W = (nu + 1.0) / (nu + Rn**2)
        R_clean = np.sqrt(W) * R
        return R_clean, W
    
    def _cost_function_studentt(self, params, flux_frame, flux_noise_frame, bkg_poisson):
        """
        Robust cost function using Student-t likelihood on the difference image.

        Parameters
        ----------
        params : tuple
            (dx, dy, log_p, dlog_sigma, c, beta, alpha_B) or similar
        flux_frame : 2D ndarray
            Science frame (S).
        flux_noise_frame : 2D ndarray
            Per-pixel noise estimate for science.
        bkg_poisson : 2D ndarray
            Background variance estimate for science.
        nu : float
            Degrees of freedom of Student-t. Smaller = heavier tails.
        iters : int
            Iterations of reweighting to downweight outliers.
        use_mad_scale : bool
            If True, rescales residuals using MAD instead of σ.

        Returns
        -------
        cost : float
            Negative log Student-t likelihood (to minimize).
        """
        
        nu = 6

        # --- Generate difference image, noise map ---
        D, sigma_D, S_N = self.zogy_difference(params, flux_frame=flux_frame,
                                               flux_noise_frame=flux_noise_frame,
                                               bkg_poisson=bkg_poisson,
                                               t_factor=1,
                                               ref_noise=self.ref_noise)

        # --- Clean residuals with Student-t weighting ---
        R_clean, W = self.student_t_clean_whitened(D, sigma_D, nu=nu, iters=2, use_mad_scale=True)

        # --- Student-t negative log-likelihood ---
        # (ignoring constants that don't affect optimization)
        cost = np.nansum((nu + 1) / 2.0 * np.log1p(R_clean**2 / nu))

        return cost
    
    def _cost_function_chi2(self, params, flux_frame, flux_noise_frame, bkg_poisson, mask):
        """
        Standard Chi² cost function with masking of strong transients.

        Parameters
        ----------
        params : tuple
            (dx, dy, log_p, dlog_sigma, c, beta, alpha_B) or similar
        flux_frame : 2D ndarray
            Science frame (S).
        flux_noise_frame : 2D ndarray
            Per-pixel noise estimate for science.
        bkg_poisson : 2D ndarray
            Background variance estimate for science.
        snr_thresh : float
            SNR threshold for masking transients.
        dilate_radius : int
            Dilation radius for mask growth around detected peaks.

        Returns
        -------
        cost : float
            Chi² cost (to minimize).
        """

        # --- Generate difference image and noise map ---
        D, sigma_D, S_N = self.zogy_difference(
            params,
            flux_frame=flux_frame,
            flux_noise_frame=flux_noise_frame,
            bkg_poisson=bkg_poisson,
            t_factor=1,
            ref_noise=self.ref_noise
        )

        valid = ~mask

        chi2_map = (D[valid] / (sigma_D[valid] + 1e-12))**2
        chi2 = np.nansum(chi2_map)

        return chi2

    def _dewhiten_spatial_match_filter(self, D, sigma_D):
        """
        Full robust ZOGY pipeline:
        1. Robust (Student-t) alignment/scale/PSF fit
        2. Masked L2 refinement
        3. Final difference image & matched filter detection
        """

        # Whitened residuals
        Dw = D / (sigma_D + 1e-12)

        # ---------------------------
        # Stage 4: Spatial matched filter
        # ---------------------------
        # matched filter: PSF flipped
        spatial_mf = self.safe_fftconvolve(Dw, self.psf[::-1, ::-1])

        # ---------------------------
        # Stage 5: Temporal matched filter (if multiple frames)
        # ---------------------------
        # e.g. stack [spatial_mf(t)] into a cube, then convolve along time
        # temporal_kernel = np.array([0.25, 0.5, 0.25])  # simple example
        # SNR_t = convolve_along_time(spatial_mf_cube, temporal_kernel)

        return spatial_mf
    
    def _broaden_epsf(self, epsf, delta_sigma):
        if delta_sigma <= 1e-3:  # treat as negligible broadening
            return epsf / np.nansum(epsf)

        pad = max(1, int(3*delta_sigma))  # ensure at least 1 pixel padding
        epsf_padded = np.pad(epsf, pad, mode="constant")
        broadened = gaussian_filter(epsf_padded, delta_sigma, mode="constant")
        broadened = broadened / broadened.sum()

        # only crop if pad < array size
        if pad < broadened.shape[0]//2 and pad < broadened.shape[1]//2:
            broadened = broadened[pad:-pad, pad:-pad]

        return broadened / np.nansum(broadened)
    
    def _broaden_frame_with_epsf(self, frame, epsf, delta_sigma):
        """Broaden entire frame by convolving with broadened ePSF."""
        broadened_epsf = self._broaden_epsf(epsf, delta_sigma)
        broadened_frame = self.safe_fftconvolve(frame, broadened_epsf)
        return broadened_frame
    
    # def _psf_matched_mask(self, img, psf, noise_map, snr_thresh=4.0, dilate_radius=1):
    #     """
    #     Create a source mask using matched filtering with the PSF.
    #     """
    #     # Matched filter response
    #     mf = self.safe_fftconvolve(img, psf)
    #     mf_var = self.safe_fftconvolve(noise_map**2, psf**2)
    #     mf_sigma = np.sqrt(mf_var + 1e-12)

    #     # Stabilize denominator with local RMS
    #     local_rms = median_filter(mf_sigma, size=5)
    #     snr = mf / np.maximum(mf_sigma, local_rms)

    #     # Threshold + morphology
    #     mask = snr >= snr_thresh
    #     if dilate_radius > 0:
    #         se = np.ones((2*dilate_radius+1, 2*dilate_radius+1), dtype=bool)
    #         mask = binary_dilation(mask, structure=se)
    #         # Optional erosion to shrink back toward cores
    #         mask = binary_erosion(mask, structure=se)

    #     return mask
    
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

        # 5. full background plane (2D image, not just at mask points)
        background = coeffs[0]*x_grid + coeffs[1]*y_grid + coeffs[2]

        # 6. residuals only at the good pixels
        model_vals = coeffs[0]*x + coeffs[1]*y + coeffs[2]
        residuals  = z - model_vals
        bg_rms = np.nanstd(residuals, ddof=1)

        return {'background': background, 'bg_rms': bg_rms}