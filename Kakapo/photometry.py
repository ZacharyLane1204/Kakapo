from photutils.psf import extract_stars, EPSFStars, EPSFBuilder, EPSFModel
from photutils.detection import DAOStarFinder, StarFinder
from photutils.psf import PSFPhotometry, IterativePSFPhotometry
from photutils.background import LocalBackground, MMMBackground
from photutils.aperture import RectangularAperture, RectangularAnnulus,CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats, aperture_photometry

from astropy.nddata import NDData
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy.modeling import models, fitting

from scipy.signal import fftconvolve
from scipy.ndimage import shift, center_of_mass
from scipy.interpolate import PchipInterpolator
from statsmodels.nonparametric.smoothers_lowess import lowess

from Kakapo.cleaning_curve import correction_smoothing_lightcurve, wavelet_denoise

import numpy as np

import pywt

import warnings

warnings.filterwarnings('ignore')

def _local_centroid(image, x, y, box_size=5):
    x_int, y_int = int(round(x)), int(round(y))
    half = box_size // 2
    y1, y2 = max(0, y_int - half), min(image.shape[0], y_int + half + 1)
    x1, x2 = max(0, x_int - half), min(image.shape[1], x_int + half + 1)
    subimg = image[y1:y2, x1:x2]

    subimg = np.nan_to_num(subimg, nan=0.0)
    subimg[subimg < 0] = 0.0

    if subimg.sum() == 0:
        return x + 0.5, y + 0.5  # fallback to rough center

    cy, cx = center_of_mass(subimg)
    return x1 + cx, y1 + cy

def forced_photometry(diff, x, y, epsf, bkg = True, method = 'aperture'):
    
    if method.lower() == 'aperture':
        fluxes = _forced_aperture(diff, x, y, bkg)
    elif method.lower() == 'psf':
       fluxes = _forced_psf(diff, x, y, epsf, bkg)
    else:
        raise ValueError('Must be "psf" or "aperture" ')
    
    return fluxes

def pad_psf_to_image(epsf, shape):
    """Center-pad a small PSF array to match the image shape."""
    padded = np.zeros(shape)
    psf_h, psf_w = epsf.shape
    img_h, img_w = shape
    start_y = (img_h - psf_h) // 2
    start_x = (img_w - psf_w) // 2
    padded[start_y:start_y+psf_h, start_x:start_x+psf_w] = epsf
    return padded

def _forced_aperture(diff, x, y, bkg = True):
    fluxes = []
    
    # epsf_norm = epsf / np.sum(epsf)
    
    for i in range(len(diff)):
        if np.isnan(diff[i]).sum() > diff[i].shape[0] * diff[i].shape[1] * 0.7:
            fluxes.append(np.nan)
            continue
        
        x_fit, y_fit = _local_centroid(diff[i], x, y, box_size=5)
        aperture = CircularAperture([x_fit, y_fit], 1.91)
        
        image = np.nan_to_num(diff[i], nan=0.0)
        
        if bkg:
            background = fit_background(image, x_fit, y_fit)
            image -= background
        
        phot_table = aperture_photometry(image, aperture)
        phot_table = phot_table.to_pandas()
        flux = phot_table['aperture_sum'].values[0]
        fluxes.append(flux)
    
    # fluxes = lowess_smooth_with_nans(np.array(fluxes))
    return np.array(fluxes)

def fit_background(image, psf_x, psf_y, r_exclude=1.91):
    y, x = np.indices(image.shape)

    r = np.sqrt((x - psf_x)**2 + (y - psf_y)**2)
    mask = (r > r_exclude) & np.isfinite(image)

    if np.sum(mask) < image.shape[0]*image.shape[1]*0.4:  # arbitrary threshold
        return np.full_like(image, np.nanmedian(image))

    p_init = models.Polynomial2D(degree=3) # Fit 2D polynomial to background-only pixels
    fit_p = fitting.LinearLSQFitter()
    p = fit_p(p_init, x[mask], y[mask], image[mask])

    background = p(x, y)
    return background

# def fit_background(image, mask=None):
#     y, x = np.indices(image.shape)
#     p_init = models.Polynomial2D(degree=3)
#     fit_p = LinearLSQFitter()

#     if mask is None:
#         mask = np.isfinite(image)

#     p = fit_p(p_init, x[mask], y[mask], image[mask])
#     background = p(x, y)
#     return background

def _forced_psf(diff, x, y, epsf, bkg = True):
    fluxes = []

    for i in range(len(diff)):
        image = diff[i]

        if np.isnan(image).sum() > image.size * 0.7:
            fluxes.append(np.nan)
            continue

        image = np.nan_to_num(image, nan=0.0)
        
        x_fit, y_fit = _local_centroid(image, x, y, box_size=5)
        
        if bkg:
            background = fit_background(image, x_fit, y_fit)
            image -= background

        x_frac = (x_fit % 1) - 0.5
        y_frac = (y_fit % 1) - 0.5
        shifted_psf = shift(epsf, shift=[y_frac, x_frac], order=3)
        
        # shifted_psf = shifted_psf[1:-1,1:-1]
        
        shifted_psf /= np.sum(shifted_psf)

        epsf_model = EPSFModel(shifted_psf)
        nddata = NDData(image)

        positions = Table()
        positions['x_0'] = [x_fit]
        positions['y_0'] = [y_fit]

        psf_photometry = PSFPhotometry(psf_model=epsf_model, fitter=LevMarLSQFitter(), 
                                       finder=None, fit_shape=(3, 3), aperture_radius=1.91, 
                                       fitter_maxiters=10)

        result = psf_photometry(nddata, init_params=positions)
        fitted_flux = result['flux_fit'][0]
        fluxes.append(fitted_flux)

    return np.array(fluxes)
