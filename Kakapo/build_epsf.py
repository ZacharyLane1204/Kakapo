import numpy as np
import matplotlib.pyplot as plt

from astropy.nddata import NDData
from astropy.table import Table

from photutils.psf import extract_stars, EPSFStars, EPSFBuilder
from photutils.detection import DAOStarFinder, StarFinder

import os
from tqdm import tqdm

from Kakapo.gaia_matching import get_gaia_region

def angular_distance(ra1, dec1, ra2, dec2):
    """
    Angular distance between two points on the sky
    """

    d1 = np.sin(np.radians(dec1)) * np.sin(np.radians(dec2))
    d2 = np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.cos(np.radians(ra1 - ra2))
    return np.degrees(np.arccos(d1 + d2))

def stars_add(star, all_stars, fwhm=1.5):
    
    for i in range(len(star)):
        data = star[i]
        mask = np.isnan(data)
        data[mask] = 0
        
        nddata = NDData(data=data) 
        daofind = DAOStarFinder(fwhm=fwhm, threshold=100)
        sources = daofind(data)
        
        if sources is None:
            continue
        
        positions = np.zeros((len(sources), 2))
        positions[:,0] = sources['xcentroid'].value
        positions[:,1] = sources['ycentroid'].value
        
        stars_tbl = Table()
        stars_tbl['x'] = positions[:,0]
        stars_tbl['y'] = positions[:,1]

        stars = extract_stars(nddata, stars_tbl, size=7)
        
        all_stars.append(stars)
        
    return all_stars

def epsf_data_creation(tpfs_res, path = '/Users/zgl12/Python_Scripts/K2/epsf.npy'):
    """
    Create the effective PSF data for the TPFs
    """

    stars_for_epsf = []
    all_stars = []
    fwhm = 2

    if os.path.exists(path):
        pass
    else:

        for i in tqdm(range(len(tpfs_res)), desc = 'Finding Gaia stars'):
            try:
                cat = get_gaia_region(tpfs_res[i].ra, tpfs_res[i].dec, 6)
                if len(cat) == 1:
                    distances = angular_distance(tpfs_res[i].ra, tpfs_res[i].dec, cat['RA_ICRS'].values, cat['DE_ICRS'].values)
                    stars_for_epsf.append(i)
                    all_stars = stars_add(tpfs_res[i].flux.value, all_stars, fwhm=fwhm)
            except:
                continue
            
    if os.path.exists(path):
        epsf_data = np.load(path)
    else:
        all_stars_combined = EPSFStars(all_stars[0].all_stars)
        for stars in all_stars[1:]:
            all_stars_combined.all_stars.extend(stars.all_stars)

        epsf_builder = EPSFBuilder(oversampling=1, maxiters=30, progress_bar=True, smoothing_kernel='quartic', recentering_maxiters=50)
        epsf, fitted_stars = epsf_builder(all_stars_combined)

        plt.figure()
        plt.imshow(epsf.data, origin='lower', cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Intensity')  # Add your desired label here
        # plt.title('Effective PSF')
        plt.xlabel(r'$x$ [pixel]')
        plt.ylabel(r'$y$ [pixel]')
        plt.savefig('epsf.png', dpi = 900, bbox_inches='tight')
        plt.show()
        
        np.save('epsf.npy', epsf.data)
        
        epsf_data = epsf.data
    return epsf_data