import numpy as np

from scipy.ndimage import fourier_shift
from numpy.fft import fftn, ifftn

from copy import deepcopy

class Crappy_Difference_Imaging():
    def __init__(self, tpf_info, psf, tol=0.2):
        
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
        self.pos_corr1[np.isnan(self.pos_corr1)] = 0
        self.pos_corr2[np.isnan(self.pos_corr2)] = 0
        
        self.r = np.sqrt((self.pos_corr1**2 + self.pos_corr2**2))
        
        self.build_reference()
        
        self.flux[np.isnan(self.flux)] = 0
        self.flux_err[np.isnan(self.flux_err)] = 1
        
        poisson_variance_stack = np.clip(np.abs(self.flux), a_min=1.0, a_max=None)
        
        self.noise = np.sqrt(self.flux_err**2 + poisson_variance_stack)
        self.ref_shape = self.ref.shape
        
        self.compute_difference_images_with_psf()
        
        self.difference_images = deepcopy(self.diffs)
        self.poisson_noise = np.sqrt(poisson_variance_stack)
    
    def build_reference(self):
        
        mask = np.where((self.r < self.tol) & (self.quality == 0))[0]
        if len(mask) == 0:
            mask = np.where(self.quality == 0)[0]
            
        shifted_stack = []
        noise_stack = []
        
        for idx in mask:
            f = np.copy(self.flux[idx])
            d_f = np.copy(self.flux_err[idx])
            # dx, dy = self.pos_corr1[idx], self.pos_corr2[idx]
            # f_shifted = self._shift_fourier(f, -dx, -dy)  # shift to reference frame
            # df_shifted = self._shift_fourier(d_f, -dx, -dy)  # shift to reference frame
            shifted_stack.append(f)
            noise_stack.append(d_f)

        shifted_stack = np.array(shifted_stack)
        noise_stack = np.array(noise_stack)
        
        ref = np.nanmedian(shifted_stack, axis=0)
        ref_noise = np.nanmedian(noise_stack, axis=0)/np.sqrt(len(noise_stack))
        
        poisson_variance_ref = np.clip(np.abs(ref), a_min=1.0, a_max=None)
        ref_noise = np.sqrt(ref_noise**2 + poisson_variance_ref)
        
        self.ref = ref
        
        self.ref_frame = 'Median' # mask[arg]
        self.ref_noise = ref_noise
        
    def compute_difference_images_with_psf(self):
        diff_images = []
        diff_noises = []
        # alphas = []
        for i in range(len(self.flux)):
            if np.isnan(self.flux[i]).sum() >= self.flux[i].shape[0] * self.flux[i].shape[1] *  0.7:
                diff_images.append(np.nan*np.ones_like(self.flux[0]))
                diff_noises.append(np.nan*np.ones_like(self.flux[0]))
                # alphas.append(np.nan)
                continue
            
            shifted = self.flux[i].copy() #self._shift_fourier(self.flux[i], -self.pos_corr1[i], -self.pos_corr2[i])
        
            alpha = 1
            
            diff_clean = shifted - alpha*self.ref
            diff_images.append(diff_clean)
        
        self.diffs = np.array(diff_images)

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