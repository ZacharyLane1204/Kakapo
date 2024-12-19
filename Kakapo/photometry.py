from photutils.psf import extract_stars, EPSFStars, EPSFBuilder
from photutils.detection import DAOStarFinder, StarFinder
from photutils.psf import PSFPhotometry, IterativePSFPhotometry
from photutils.background import LocalBackground, MMMBackground
from photutils.aperture import RectangularAperture, RectangularAnnulus,CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats, aperture_photometry

import numpy as np

from copy import deepcopy

# Forced photometry at a certain x and y position

def forced_photometry(diff, x, y):

    fluxes = []
    
    for i in range(len(diff)):
        if np.isnan(diff[i]).sum() > diff[i].shape[0] * diff[i].shape[1] * 0.5:
            fluxes.append(np.nan)
            continue

        aperture = CircularAperture([x, y], 1.91)
        
        phot_table = aperture_photometry(diff[i], aperture)
        phot_table = phot_table.to_pandas()
        flux = phot_table['aperture_sum'].values[0]
        fluxes.append(flux)
        
    return np.array(fluxes)

        