import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lightkurve.correctors import CBVCorrector

from astropy.coordinates import SkyCoord

from multiprocessing import Pool

from tqdm import tqdm
import os
import glob
import shutil
import time
from copy import deepcopy

from Kakapo.build_epsf import epsf_data_creation
from Kakapo.kakapo import Kakapo
from Kakapo.send_myself_email import send_mail
# from Kakapo.difference_image import create_diff_image
from Kakapo.photometry import forced_photometry

from photutils.aperture import RectangularAperture, RectangularAnnulus,CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats, aperture_photometry
from photutils.psf import extract_stars, EPSFStars, EPSFBuilder

from scipy.ndimage import shift
from scipy.signal import find_peaks

import warnings

warnings.filterwarnings('ignore')

def _tpf_addition(tpf_info, tpf_input):
    
    if tpf_input.campaign is None:
        campaign = tpf_input.quarter
        mission = 'Kepler'
        print(f"Adding TPF {tpf_input.targetid} from {mission} quarter {campaign}")
    else:
        campaign = tpf_input.campaign
        mission = 'K2'
    
    tpf_info.loc[len(tpf_info)] = [mission, campaign, tpf_input.targetid, tpf_input.ra, tpf_input.dec, 
                                   tpf_input.flux.value, tpf_input.flux_err.value, tpf_input.quality, 
                                   tpf_input.pos_corr1, tpf_input.pos_corr2, tpf_input.time]
    
    return tpf_info

def _check_tpf_type(tpf_input):
    
    tpf_info = pd.DataFrame(columns=['mission', 'campaign', 'targetid', 'ra', 'dec', 'flux', 'flux_err', 
                                     'quality', 'pos_corr1', 'pos_corr2', 'time'])
    
    if isinstance(tpf_input, lk.targetpixelfile.KeplerTargetPixelFile):
        tpf_info = _tpf_addition(tpf_info, tpf_input)
        
    return tpf_info

def access_tpfs():
    """
    """

    test_case = []

    lightkurve_file_folder = '/home/phys/astronomy/zgl12/kepler/Temp_TPFs/'

    files = sorted(glob.glob(lightkurve_file_folder + '*/*.fits.gz'))

    for file in tqdm(files, desc='Reading TPFs'):
        tpf = lk.read(file, quality_bitmask = 'none')
        test_case.append(tpf)
        
    return test_case

test_case = access_tpfs()

test_case = test_case[4:]

epsf_data = np.genfromtxt('epsf_data.txt')
epsf_data = epsf_data[2:-2, 2:-2]

kea = Kakapo(test_case, epsf_data, num_cores = 50, 
             filtered = True, overwrite = True, savepath = './Data/',
             tol = 0.2, std1 = 2, std2 = 2, detect = True, 
             corrlim = 0.05, difflim = 1.1, fwhmlim = 5, maxlim = 0, snrlim = 4,
             roundness = 0.9, poiss_val = 2, siglim = 2, dist_cut = 0.9,
             break_point = None)