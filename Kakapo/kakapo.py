import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import pandas as pd

from scipy.ndimage import center_of_mass, zoom, shift

from sklearn.cluster import DBSCAN

from astropy.stats import sigma_clipped_stats
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils.detection import StarFinder
from photutils.aperture import RectangularAperture, RectangularAnnulus,CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats, aperture_photometry

from tqdm import tqdm
from copy import deepcopy
from functools import partial
import os
from time import time

from multiprocessing import Pool, cpu_count

# from Kakapo.difference_image import create_diff_image
from Kakapo.difference_image import Difference_Imaging
from Kakapo.selection_criteria import Implement_reductions
from Kakapo.photometry import forced_photometry
from Kakapo.cleaning_curve import correction_smoothing_lightcurve, wavelet_denoise

import warnings

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# plt.rc('font', size=13)
# plt.rc('xtick', labelsize=13)
# plt.rc('ytick', labelsize=13)

fig_width_pt = 244.0  # Get this from LaTeX using \the\columnwidth
text_width_pt = 508.0 # Get this from LaTeX using \the\textwidth

inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt*1.5 # width in inches
fig_width_full = text_width_pt*inches_per_pt  # 17
fig_height =fig_width*golden_mean # height in inches
fig_size = [fig_width,fig_height] #(9,5.5) #(9, 4.5)

Simbad.ROW_LIMIT = 10  # Set a limit on the number of rows returned
simbad = Simbad()

warnings.filterwarnings('ignore')

def get_object_type(ra, dec):
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs') # Create a SkyCoord object
    
    result = simbad.query_region(coord, radius=6 * u.arcsec) # Query Simbad with the RA and Dec
    
    if result is None:
        return 0, 0, 0
    
    objects = result['MAIN_ID'] # Check classifications of the objects returned
    types = result['SP_TYPE']  # Spectral type or classification
    
    star_count = 0
    galaxy_count = 0
    other_count = 0

    for obj, obj_type in zip(objects, types):
        print(f"Object: {obj}, Type: {obj_type}")
        if 'star' in obj_type.lower():
            star_count += 1
        elif 'galaxy' in obj_type.lower():
            galaxy_count += 1
        else:
            other_count += 1
    
    return star_count, galaxy_count, other_count

def _number_cores(num_cores):
    if num_cores is None:
        num_cores = max(1, cpu_count()//2)
    elif isinstance(num_cores, int):
        if num_cores >= cpu_count():
            num_cores = cpu_count()
        else:
            print(f"Using {num_cores} cores")
    else:
        raise ValueError("num_cores must be an integer or NoneType")
    
    return num_cores

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
    
    elif isinstance(tpf_input, str):
        tpf_input = lk.open(tpf_input, quality_bitmask = 'none')
        tpf_info = _tpf_addition(tpf_info, tpf_input)
    
    elif isinstance(tpf_input, lk.collections.TargetPixelFileCollection):
        tpf_input_list = deepcopy(tpf_input)
        
        for i in range(len(tpf_input)):
            tpf_info = _tpf_addition(tpf_info, tpf_input_list[i])
            
    elif isinstance(tpf_input, list):
        tpf_input_list = deepcopy(tpf_input)
        
        for i in range(len(tpf_input)):
            if isinstance(tpf_input_list[i], lk.targetpixelfile.KeplerTargetPixelFile):
                tpf_info = _tpf_addition(tpf_info, tpf_input_list[i])
            elif isinstance(tpf_input_list[i], str):
                tpf_input_list[i] = lk.open(tpf_input_list[i], quality_bitmask = 'none')
                tpf_info = _tpf_addition(tpf_info, tpf_input_list[i])
            else:
                raise ValueError("tpf_input must be a string, a KeplerTargetPixelFile object, or a list of KeplerTargetPixelFile objects")
    else:
        raise ValueError("tpf_input must be a string, a KeplerTargetPixelFile object, or a list of KeplerTargetPixelFile objects")
    
    return tpf_info

def correlating(difference, found, downsampled_array):
    x = int(np.round(found['xcentroid']))
    y = int(np.round(found['ycentroid']))
    
    y_start, y_end = y-1, y+2
    x_start, x_end = x-1, x+2
    
    padded_difference = np.pad(deepcopy(difference), pad_width=((max(0, -y_start), 
                                                                 max(0, y_end - difference.shape[0])),
                                                                (max(0, -x_start), 
                                                                 max(0, x_end - difference.shape[1]))), 
                               mode='constant', constant_values=0)

    cut = padded_difference[y_start:y_end, x_start:x_end]
    
    cut[cut<0] = 0
    
    if np.nansum(cut) > 0.95:
        cut /= np.nansum(cut)
        
        cm_test = center_of_mass(cut)
        
        cm = [cm_test[0] + x - 1, cm_test[1] + y - 1]
        
        # print('Centroids Check')
        # print(found['xcentroid'].values[0], found['ycentroid'].values[0])
        # print(cm[0], cm[1])
        
        dx = cm_test[0] - cut.shape[0]/2
        dy = cm_test[1] - cut.shape[1]/2
                    
        shifted_array = shift(downsampled_array, (-dy,-dx))[1:-1, 1:-1]
        
        if shifted_array.shape != cut.shape:
            raise ValueError("Arrays must have the same shape")
                
        shifted_array_flat = shifted_array.flatten()
        cut_flat = cut.flatten()
        
        diff = np.nansum(abs(cut_flat-shifted_array_flat))

        correlation_matrix = np.corrcoef(cut_flat, shifted_array_flat)
        correlation_coefficient = correlation_matrix[0, 1]
        
    else:
        correlation_coefficient = 0
        diff = 10001
        cm = [found['xcentroid'].values[0], found['ycentroid'].values[0]]
    
    return correlation_coefficient, diff, cm[0], cm[1]

def main_correlation(diff, found, downsampled_array):
    
    stars = []
    if len(found) == 1:
        correlation_coefficient, diffn, com_x, com_y = correlating(diff, found, downsampled_array)
        
        found['correlation'] = correlation_coefficient
        found['psfdiff'] = diffn
        found['xcentroid'] = com_x
        found['ycentroid'] = com_y
        stars = [found]
    elif len(found) > 1:
        raise ValueError(f"More than one star found in frame {found['frame']}")
    else:
        found['correlation'] = 0
        found['psfdiff'] = 10001
        stars = [found]
    return stars

def star_finding_procedure(data, downsampled_array, std1 = 3.0, std2 = 3.0):

    prf = downsampled_array

    tables = []
    for shift_y in np.arange(-0.5, 0.75, 0.25):
        for shift_x in np.arange(-0.5, 0.75, 0.25):
            tables += [_finding_the_stars(data, prf, std1, std2, (shift_y, shift_x))]
    
    good_tables = [table for table in tables if table is not None]
    if len(good_tables)>0:
        total = pd.concat(good_tables)
        total = total[~pd.isna(total['xcentroid'])]
        if len(total) > 0:
            grouped = spatial_group(total,distance=1.5)
            res = grouped.groupby('objid').head(1)
            res = res.reset_index(drop=True)
            res = res.drop(['id','objid'],axis=1)
        else:
            res=None
    else:
        res = None

    return res

def _finding_the_stars(data, prf, std1, std2, shifts):
    
    shift_y, shift_x = shifts
    mean, med, std = sigma_clipped_stats(data, sigma=std1)
    psf = shift(prf, (shift_y, shift_x))
    finder = StarFinder(med + std2*std,kernel=psf)
    res = finder(data)
    
    if res is not None:
        res = res.to_pandas()
        res = find_stars(res, data)
    
    return res

def count_detections(result):

    ids = result['objid'].values
    unique = np.unique(ids, return_counts=True)
    unique = list(zip(unique[0],unique[1]))

    array = np.zeros_like(ids)

    for id,count in unique:
        index = (result['objid'] == id).values
        array[index] = count

    result['n_detections'] = array

    return result

def spatial_group(result,distance=1.5,njobs=-1):
    """
    Groups events based on proximity.
    """

    pos = np.array([result.xcentroid,result.ycentroid]).T
    cluster = DBSCAN(eps=distance,min_samples=1,n_jobs=njobs).fit(pos)
    labels = cluster.labels_
    unique_labels = set(labels)
    for label in unique_labels:
        result.loc[label == labels,'objid'] = label + 1
    result['objid'] = result['objid'].astype(int)
    return result

def find_stars(star, data, negative=False):
    
    data[np.isnan(data)] = 0
    
    if negative:
        data = data * -1
    
    if star is None:
        return None
    
    pos_ind = ((star.xcentroid.values >=1) & (star.xcentroid.values < data.shape[1]-1) & 
                (star.ycentroid.values >=1) & (star.ycentroid.values < data.shape[0]-1))
    star = star.iloc[pos_ind]
    
    if len(star) == 0:
        return None

    # x = np.round(star.xcentroid.values).astype(int)
    # y = np.round(star.ycentroid.values).astype(int)
    
    x = star.xcentroid.values
    y = star.ycentroid.values
    pos = list(zip(x, y))
    
    aperture = CircularAperture(pos, 1.91)
    phot_table = aperture_photometry(data, aperture)
    phot_table = phot_table.to_pandas()
    _, _, bkg_std = sigma_clipped_stats(data, sigma= 2)
    star['snr'] = phot_table['aperture_sum'].values / (aperture.area * bkg_std)
    star['snr'] = np.clip(star['snr'], None, 10001)
    star['flux_err'] = aperture.area * bkg_std
    star['bkg_std'] = bkg_std
    star['flux'] = phot_table['aperture_sum'].values
    star['mag'] = -2.5*np.log10(phot_table['aperture_sum'].values)
    
    if negative:
        star['flux_sign'] = -1
        star['flux'] = star['flux'].values * -1
        star['max_value'] = star['max_value'].values * -1
    else:
        star['flux_sign'] = 1
    return star

def poisson_threshold(star, diff, poisson_noise):
    x = star['xcentroid'].iloc[0]
    y = star['ycentroid'].iloc[0]
    
    radius_mask = check_within_radius(diff, x, y)
    
    try:
    # print(diff.shape, np.abs(diff[radius_mask]).shape)
    # print(poisson_noise.shape)
        poisson_value = np.nanmax(np.abs(diff[radius_mask]) / poisson_noise[radius_mask])
        star['poisson_thresh'] = poisson_value
    except: 
        return None
    
    return star

def check_within_radius(array, x, y, radius=0.6):
    rows, cols = array.shape
    
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))

    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    within_radius = distances <= radius
    
    corners_x = [xx - 0.5, xx + 0.5, xx - 0.5, xx + 0.5]
    corners_y = [yy - 0.5, yy - 0.5, yy + 0.5, yy + 0.5]
    
    corner_distances = [np.sqrt((cx - x) ** 2 + (cy - y) ** 2) <= radius 
                        for cx, cy in zip(corners_x, corners_y)]
    
    corners_within_radius = np.any(corner_distances, axis=0)
    
    overlapping_pixels = within_radius | corners_within_radius
    
    return overlapping_pixels

def process_frame(i, diff, downsampled_array, poisson_noise, std1, std2):
    m, med, std = sigma_clipped_stats(diff[i], sigma=std1)
    found = star_finding_procedure(diff[i], downsampled_array[1:-1,1:-1], std1=std1, std2=std2)
    stars = []
    if found is not None:
        found['frame'] = int(i)
        for j in range(len(found)):
            foundj = pd.DataFrame(found.iloc[j]).T
            temp_star = main_correlation(diff[i], foundj, downsampled_array)
            temp_star = spatial_group(temp_star[0],distance=1.5)
            temp_star = count_detections(temp_star)
            temp_star = poisson_threshold(temp_star, diff[i], poisson_noise[i])
            if temp_star is not None:
                stars += [temp_star]
    # else:
    #     print(f'ZZZ None found in {i} frame')
    return stars

class Kakapo():
    def __init__(self, tpf_input, epsf_data, num_cores = None, detect = True, 
                 filtered = True, plot_diff = False, overwrite = False,
                 savepath = './', break_point = None,
                 mask_value=1000, tol=0.003, std1 = 3.0, std2 = 3.0, dist_cut = 0.2, 
                 corrlim = 0.6, difflim = 0.8, fwhmlim = 2.5, maxlim = 0, snrlim = 4, 
                 roundness = 0.35, poiss_val = 3, siglim = 2):
        
        tpf_info = _check_tpf_type(tpf_input)
        
        if (len(tpf_info) > 0) & (isinstance(break_point, int)):
            tpf_info = tpf_info.head(break_point)
        
        if savepath[-1] != '/':
            savepath += '/'
        
        self.epsf = epsf_data
        self.count = 0
        
        num_cores = _number_cores(num_cores)
        
        if len(tpf_info) > 0:
            jobs = [(tpf_info.iloc[i], epsf_data, std1, std2, mask_value, tol, detect, 
                     filtered, overwrite, corrlim, difflim, fwhmlim, maxlim, snrlim, 
                     roundness, poiss_val, siglim, plot_diff, dist_cut, savepath) 
                    for i in range(len(tpf_info))]
            
            with Pool(processes=num_cores) as pool: # Use multiprocessing pool to parallelise the process
                results = list(tqdm(pool.imap(self._process_tpf, jobs), total=len(jobs), desc='TPFs'))
                
        else:
            if tpf_info['mission'].iloc[0] == 'Kepler':
                name = 'diff_q{}_t{}'.format(tpf_info['campaign'].iloc[0], tpf_info['targetid'].iloc[0])
                mission = 'Kepler'
            else:
                name = 'diff_c{}_t{}'.format(tpf_info['campaign'].iloc[0], tpf_info['targetid'].iloc[0])
                mission = 'K2'
            
            os.makedirs(savepath + f'difference_arrays/c{tpf_info["campaign"].iloc[0]}/', exist_ok = True)
            
            full_file_name = savepath + f'difference_arrays/c{tpf_info["campaign"].iloc[0]}/' + name
            
            if os.path.exists(full_file_name) & (overwrite == False):
                pass
            else:
                
                diff, stars, reset_indices, distances = self.run(tpf_info.iloc[0], self.epsf, 
                                                                mask_value = mask_value, tol = tol, 
                                                                detect = detect, overwrite = overwrite, 
                                                                plot_diff = plot_diff)
                
                

                np.save(full_file_name, diff)
                
                if filtered & (stars is not None):
                    self._filter_and_save_stars(stars, tpf_info.iloc[0], diff, corrlim, difflim, fwhmlim,
                                                maxlim, snrlim, roundness, poiss_val, siglim, dist_cut, 
                                                savepath, reset_indices, epsf_data, distances)
            
    def _run_object(self, tpf_info, epsf, std1=3.0, std2=3.0, plot=False, mask_value=1000, tol=0.003, 
                   detect = True, overwrite = False, full_file_name = 'test.csv', savepath = './'):
        
        # args = create_diff_image(tpf_info, epsf, plot = plot, tol=tol)
        # ref, diff, poisson_noise, ref_frame_idx, reset_indices, distances = args
        
        chunky_bird = Difference_Imaging(tpf_info, epsf, tol=0.2)

        ref = chunky_bird.ref
        poisson_noise = chunky_bird.noise
        diff = chunky_bird.diffs
        reset_indices = chunky_bird.thrusters
        distances = chunky_bird.distance
        ref_frame_idx = chunky_bird.ref_frame
        
        # print(np.sum(np.isnan(diff)), np.count_nonzero(diff))
        
        
        # print('ZZZ DIFF IM')
        
        if ref is None:
            # print('Na ah!!!')
            return None, None
        
        ref_value = np.nansum(ref)
        
        campaign = tpf_info['campaign']
        targetid = tpf_info['targetid']
        ra = tpf_info['ra']
        dec = tpf_info['dec']
        
        if detect:
            # print('ZZZ Detection')
            stars = self.detection(diff, epsf, poisson_noise, ref_value, 
                                   std1 = std1, std2 = std2)
            if stars is not None:
                stars['campaign'] = campaign
                stars['target_id'] = targetid
                stars['ra'] = ra
                stars['dec'] = dec
                stars['filename'] = full_file_name
                stars['ref_frame'] = ref_frame_idx
                # print('Saving Func')
                self.saving_func(tpf_info, stars, overwrite, full_file_name, savepath)
            else:
                stars = None
        else:
            stars = None
        
        return diff, stars, reset_indices, distances
    
    def _process_tpf(self, args):
        tpf_info_row, epsf_data, std1, std2, mask_value, tol, detect, filtered, overwrite, \
        corrlim, difflim, fwhmlim, maxlim, snrlim, roundness, poiss_val, siglim, plot_diff, dist_cut, savepath = args

        # start = time()
        # print(f'ZZZ Pre-run: c{tpf_info_row.campaign} t{tpf_info_row.targetid}')
        diff, stars, reset_indices, distances = self.run(tpf_info_row, epsf_data, std1=std1, std2=std2, 
                                                         mask_value=mask_value, tol=tol, detect=detect, 
                                                         overwrite=overwrite, plot_diff = plot_diff)
        # print(f'ZZZ Post-run: c{tpf_info_row.campaign} t{tpf_info_row.targetid} {(time() - start)/60:.2f}')
        if diff is None:
            print('Fail')
            pass
        else:
            name = 'diff_c{}_t{}'.format(tpf_info_row['campaign'], tpf_info_row['targetid'])
            
            os.makedirs(savepath + f'difference_arrays/c{tpf_info_row["campaign"]}/', exist_ok = True)
            
            full_file_name = savepath + f'difference_arrays/c{tpf_info_row["campaign"]}/' + name
            np.save(full_file_name, diff)
            
            if filtered & (stars is not None):
                self._filter_and_save_stars(stars, tpf_info_row, diff, corrlim, difflim, fwhmlim,
                                            maxlim, snrlim, roundness, poiss_val, siglim, dist_cut, 
                                            savepath, reset_indices, epsf_data, distances)
               
    def detection(self, diff, epsf, poisson_noise, ref_value, std1=3.0, std2=3.0):

        stars = []  # Initialise an empty list to store all non-empty DataFrames

        for i in range(len(diff)):
            temp_stars = process_frame(i, diff, epsf, poisson_noise, std1, std2)
            # print('ZZZ temp_stars', temp_stars)
            
            if temp_stars: # Check if temp_stars is not empty (i.e., contains DataFrames)
                stars.extend(temp_stars)  # Append all elements of temp_stars to stars
        
        if stars:
            stars = pd.concat(stars, ignore_index=True)
            stars['ref_flux'] = ref_value
            # stars['sum_flux'] = sum_fluxes[stars['frame'].to_numpy().astype(int)]
            
            return stars
        else:
            return None
        
    def saving_func(self, tpf_info, stars, overwrite, full_file_name, savepath = './'):
        
        columns = ['campaign', 'target_id', 'ra', 'dec', 'filename']
        
        processed_file_name = savepath + 'object_ids/object_process.csv'
        
        os.makedirs(savepath + 'object_ids/', exist_ok = True)
        
        save_list = [tpf_info.campaign, tpf_info.targetid, 
                     tpf_info.ra, tpf_info.dec, full_file_name]
        
        if os.path.exists(processed_file_name):
            processed = pd.read_csv(processed_file_name)
            if 'campaign' in processed.columns and 'target_id' in processed.columns:
                mask = (processed['campaign'] == tpf_info.campaign) & (processed['target_id'] == tpf_info.targetid)
                
                if mask.any():
                    if overwrite:
                        processed.loc[mask] = save_list
                    else:
                        pass
                else:
                    processed = pd.concat([processed, pd.DataFrame([save_list], columns = columns)])
                
            processed.to_csv(processed_file_name, index=False)
        else:
            new_row = pd.DataFrame([save_list], columns=columns)
            new_row.to_csv(processed_file_name, index=False)
        
        stars.to_csv(full_file_name, index=False)
        
    def run(self, tpf_info, epsf, std1 = 3, std2 = 3, mask_value = 1000, tol = 0.003, 
            detect = True, overwrite = False, plot_diff = False, savepath = './'):
        
        campaign = tpf_info['campaign']
        targetid = tpf_info['targetid']
        
        name = 'c{}_t{}.csv'.format(campaign, targetid)
        full_file_name = savepath + f'Data/csv_files/c{campaign}/' + name
        
        os.makedirs(savepath + f'Data/csv_files/c{campaign}/', exist_ok = True)

        
        if os.path.exists(full_file_name) & (overwrite == False):
            self.stars = pd.read_csv(full_file_name)
            
            args = self._run_object(tpf_info, epsf, plot=plot_diff, mask_value=mask_value, 
                                    tol=tol, detect = False, full_file_name = full_file_name, 
                                    savepath=savepath)
            
        else:
        
            args = self._run_object(tpf_info, epsf, std1 = std1, std2 = std2, plot=plot_diff, 
                                    mask_value=mask_value, tol=tol, detect = detect,
                                    overwrite = overwrite, full_file_name = full_file_name, savepath=savepath)
        
        diff, stars, reset_indices, distances = args
        return diff, stars, reset_indices, distances

    def _filter_and_save_stars(self, stars, tpf_info, diff, corrlim, difflim, fwhmlim, 
                                maxlim, snrlim, roundness, poiss_val, siglim, dist_cut, 
                                savepath, reset_indices, epsf_data, distances):
        """
        Handle filtering and saving of star data to CSV after processing.
        """
        
        campaign = tpf_info['campaign']
        targetid = tpf_info['targetid']
        mission = tpf_info['mission']
        ra = tpf_info['ra']
        dec = tpf_info['dec']
        bjds = tpf_info['time'].value + 54832.5
        
        time_mask = tpf_info['time'].value <= 40
        
        # pos_corr1 = tpf_info['pos_corr1']
        # pos_corr2 = tpf_info['pos_corr2']
        start = time()
        # print(f'ZZZ Pre-reductions: c{tpf_info.campaign} t{tpf_info.targetid}')
        
        impred = Implement_reductions(stars, tpf_info, diff, epsf_data, reset_indices, distances, 
                                      corrlim=corrlim, difflim=difflim, 
                                      fwhmlim=fwhmlim, maxlim=maxlim, snrlim=snrlim, 
                                      roundness=roundness, poiss_val=poiss_val, 
                                      siglim=siglim, dist_cut=dist_cut)

        # print(f'ZZZ Post-reductions: c{tpf_info.campaign} t{tpf_info.targetid} {(time() - start)/60:.2f}')

        filtered_stars = impred.filtered_stars
        full_events = impred.full_events

        if (filtered_stars is not None) & (full_events is not None):
                
            if mission == 'Kepler':
                stars_name = f'filtered_q{campaign}_t{targetid}.csv'
                events_name = f'events_q{campaign}_t{targetid}.csv'
            elif mission == 'K2':
                stars_name = f'filtered_c{campaign}_t{targetid}.csv'
                events_name = f'events_c{campaign}_t{targetid}.csv'
            else:
                raise ValueError("Mission must be either 'Kepler' or 'K2'")
            
            # full_events['star_count'] = star_count
            # full_events['galaxy_count'] = galaxy_count
            # full_events['other_count'] = other_count
            
            os.makedirs(f'{savepath}filtered_stars/c{campaign}/', exist_ok=True)
            os.makedirs(f'{savepath}detected_events/c{campaign}/', exist_ok=True)
            os.makedirs(f'{savepath}figures/c{campaign}/', exist_ok=True)
            
            full_star_file_name = f'{savepath}filtered_stars/c{campaign}/{stars_name}'
            full_events_file_name = f'{savepath}detected_events/c{campaign}/{events_name}'
            filtered_stars.to_csv(full_star_file_name, index=False)
            full_events.to_csv(full_events_file_name, index=False)
            
            for i in range(len(full_events)):
                cluster_number = int(full_events['cluster'].iloc[i])
                x = full_events['x'].iloc[i]
                y = full_events['y'].iloc[i]
                frame_min = int(full_events['frame_min'].iloc[i])
                frame_max = int(full_events['frame_max'].iloc[i])
                if mission == 'Kepler':
                    figures_name = f'figures_q{campaign}_t{targetid}_e{cluster_number}.png'
                elif mission == 'K2':
                    figures_name = f'figures_c{campaign}_t{targetid}_e{cluster_number}.png'
                figures_file_name = os.path.abspath(f'{savepath}figures/c{campaign}/{figures_name}')
                
                fluxes = forced_photometry(diff, x, y, epsf_data)
                fluxes = correction_smoothing_lightcurve(fluxes, distances < 0.25, window=35, sigma=3)
                fluxes = wavelet_denoise(fluxes, wavelet='coif5', level=3, keep='low', mode = 'smooth')
                
                len_frames = (frame_max - frame_min)/2
                if len_frames < 50:
                    len_frames = 50
                
                plot_frame_min = max(0, frame_min - len_frames)
                plot_frame_max = min(len(fluxes) - 1, frame_max + len_frames)
                
                if np.nanmax(fluxes) > 0:
                    max_lim = np.nanpercentile(fluxes, 99.5)*1.1
                else:
                    max_lim = np.nanpercentile(fluxes, 99.5)* 0.9
                
                if np.nanmin(fluxes) < 0:
                    min_lim = np.nanpercentile(fluxes, 0.5)* 1.1
                else:
                    min_lim = np.nanpercentile(fluxes, 0.5) * 0.9
                
                fig, axs = plt.subplots(2, 1, figsize = [fig_width, fig_height*2])
                plt.subplots_adjust(hspace=0.13)

                for i in range(2):
                    axs[i].fill_betweenx([min_lim, max_lim], bjds[frame_min], bjds[frame_max], color='C1', alpha = 0.5)
                    axs[i].set_ylabel('Counts')
                    
                axs[0].plot(bjds[np.isfinite(fluxes)], fluxes[np.isfinite(fluxes)], color = 'k')
                axs[1].plot(bjds[np.isfinite(fluxes)], fluxes[np.isfinite(fluxes)], color = 'k')
                
                axs[0].set_xlim(bjds[int(np.floor(plot_frame_min))], bjds[int(np.floor(plot_frame_max))])
                axs[0].set_ylim(min_lim, max_lim)
                axs[0].set_title('Forced Photometry')
                
                axs[1].set_xlabel('Time (BMJD)')
                
                # axs[0].set_ylim(p_min_lim, p_max_lim)
                # axs[0].set_xlim(plot_frame_min, plot_frame_max)
                
                unique, counts = np.unique(bjds, return_counts=True)
                bjds_temp = unique[counts == 1]
                
                axs[1].set_xlim(min(bjds_temp), max(bjds_temp))
                axs[1].set_ylim(min_lim, max_lim)
                
                if mission == 'Kepler':
                    axs[0].set_title(f'q{campaign} t{targetid} e{cluster_number}')
                elif mission == 'K2':
                    axs[0].set_title(f'c{campaign} t{targetid} e{cluster_number}')

                
                axs[1].text((max(bjds_temp) - min(bjds_temp))/10 + min(bjds_temp), 
                            max_lim / 1.1, f'Frame: {frame_min}--{frame_max}, x: {x:.2f}, y: {y:.2f}')
                # axs[1].text(bjds[int(np.floor(plot_frame_min))] + 100, np.nanmax(fluxes), f'Frame: {frame_min}--{frame_max}, x: {x:.2f}, y: {y:.2f}')
                
                plt.savefig(figures_file_name, dpi = 800, bbox_inches='tight')
                plt.close()
        # else:
        #     print(campaign, targetid, filtered_stars, full_events)

# downsampled_data = downsample_bin(shifted_data, factor)