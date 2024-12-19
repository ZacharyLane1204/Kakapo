import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from scipy.signal import fftconvolve, convolve2d
from scipy.ndimage import center_of_mass, zoom, shift
# from scipy.spatial.distance import pdist, squareform
# from scipy.ndimage import map_coordinates
# from scipy.optimize import minimize

from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# from astropy.coordinates import SkyCoord
# from astropy.nddata import NDData
# from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip, sigma_clip

from photutils.detection import StarFinder
from photutils.aperture import RectangularAperture, RectangularAnnulus,CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats, aperture_photometry

from tqdm import tqdm
from copy import deepcopy
from functools import partial
import os

# from billiard.pool import Pool
from multiprocessing import Pool, cpu_count
# from joblib import Parallel, delayed

from Kakapo.difference_image import create_diff_image_de
from Kakapo.build_epsf import epsf_data_creation
from diagnostic_plots import diagnostic_plotting
from Kakapo.selection_criteria import Implement_reductions

import warnings
# from astropy.utils.exceptions import AstropyUserWarning

# Ignore warnings from the photutils package
# warnings.filterwarnings('ignore', category=AstropyUserWarning, module='photutils')
warnings.filterwarnings('ignore')

def correlating(difference, found, downsampled_array, plot = False):
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
        
        dx = cm_test[0] - cut.shape[0]/2
        dy = cm_test[1] - cut.shape[1]/2
                    
        shifted_array = shift(downsampled_array, ( -(dy), -(dx) ))[1:-1, 1:-1]
    
        if plot:
                    
            plt.figure()
            plt.imshow(difference, origin='lower')
            plt.colorbar()
            plt.scatter(cm[0], cm[1], c='m')
            # plt.scatter(cm_test3[0], cm_test3[1], c='r')
            # plt.scatter(cm_test2[0], cm_test2[1], c='limegreen')
            plt.scatter(found['xcentroid'], found['ycentroid'])
            plt.title("Difference")
            plt.show()
            
            plt.figure()
            plt.imshow(downsampled_array, origin='lower')
            plt.colorbar()
            plt.title("Downsampled EPSF")
            plt.show()
            
            plt.figure()
            plt.imshow(cut, origin='lower')
            plt.scatter(cm_test[0], cm_test[1], c='m')
            plt.colorbar()
            plt.title("Cut")
            plt.show()
                            
            plt.figure()
            plt.imshow(shifted_array, origin='lower')
            plt.colorbar()
            # plt.scatter(cm[0], cm[1], c='m')
            plt.title("Shifted EPSF")
            plt.show()
                    
        if shifted_array.shape != cut.shape:
            print()
            print(shifted_array.shape)
            print(cut.shape)
            print(downsampled_array.shape)
            print(difference.shape)
            print()
                    
            raise ValueError("Arrays must have the same shape")
                
        shifted_array_flat = shifted_array.flatten()
        cut_flat = cut.flatten()
        
        diff = np.nansum(abs(cut_flat-shifted_array_flat))

        correlation_matrix = np.corrcoef(cut_flat, shifted_array_flat)
        correlation_coefficient = correlation_matrix[0, 1]
        
    else:
        correlation_coefficient = 0
        diff = 10001
    
    return correlation_coefficient, diff

def main_correlation(diff, found, downsampled_array, plot = False):

    # found = found.to_frame().T
    
    stars = []
    if len(found) == 1:
        correlation_coefficient, diffn = correlating(diff, found, downsampled_array, plot)
        
        found['correlation'] = correlation_coefficient
        found['psfdiff'] = diffn
        # found['']
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
    mean, med, std = sigma_clipped_stats(data, sigma=std1)

    psfCentre = deepcopy(prf)
    finder = StarFinder(med + std2*std,kernel=psfCentre)
    res1 = finder(data)
    
    if res1 is not None:
        res1 = res1.to_pandas()
        res1 = find_stars(res1, data)
        
    psfUR = shift(prf, (-0.25, 0))
    finder = StarFinder(med + std2*std,kernel=psfUR)
    res2 = finder(data)
    
    if res2 is not None:
        res2 = res2.to_pandas()
        res2 = find_stars(res2, data)

    psfUL = shift(prf, (0.25, 0))
    finder = StarFinder(med + std2*std,kernel=psfUL)
    res3 = finder(data)
    
    if res3 is not None:
        res3 = res3.to_pandas()
        res3 = find_stars(res3, data)

    psfDR = shift(prf, (0, -0.25))
    finder = StarFinder(med + std2*std,kernel=psfDR)
    res4 = finder(data)
        
    if res4 is not None:
        res4 = res4.to_pandas()
        res4 = find_stars(res4, data)

    psfDL = shift(prf, (0, 0.25))
    finder = StarFinder(med + std2*std,kernel=psfDL)
    res5 = finder(data)

    if res5 is not None:
        res5 = res5.to_pandas()
        res5 = find_stars(res5, data)
    
    tables = [res1, res2, res3, res4, res5]
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

def spatial_group(result,distance=0.5,njobs=-1):
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

    x = np.round(star.xcentroid.values).astype(int)
    y = np.round(star.ycentroid.values).astype(int)
    
    x = star.xcentroid.values
    y = star.ycentroid.values
    pos = list(zip(x, y))
    
    aperture = CircularAperture(pos, 2)
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

def process_frame(i, diff, downsampled_array, std1, std2, plot):
    m, med, std = sigma_clipped_stats(diff[i], sigma=std1)
    found = star_finding_procedure(diff[i], downsampled_array[1:-1,1:-1], std1=std1, std2=std2)
    stars = []
    if found is not None:
        found['frame'] = int(i)
        for j in range(len(found)):
            foundj = pd.DataFrame(found.iloc[j]).T
            temp_star = main_correlation(diff[i], foundj, downsampled_array, plot=plot)
            temp_star = spatial_group(temp_star[0],distance=1.5)
            temp_star = count_detections(temp_star)
            stars += [temp_star]
    return stars

class Kakapo():
    def __init__(self, tpf_input, epsf_data, plot = False, num_cores = None, 
                 mask_value=1000, tol=0.003, std1 = 3.0, std2 = 3.0,
                 detect = True, diagnostic_plots = True, save = True, overwrite = False):
        
        self.tpf = tpf_input
        self.plot = plot
        self.epsf = epsf_data
        self.count = 0
        
        self.num_cores = self._number_cores(num_cores)
        
        
        name = 'c{}_t{}.csv'.format(tpf_input.campaign, tpf_input.targetid)
        
        full_file_name = './csv_files/' + name
        self.full_file_name = os.path.abspath(full_file_name)
        
        if os.path.exists(full_file_name) & (overwrite == False):
            self.stars = pd.read_csv(self.full_file_name)
            
            self.run(plot=plot, mask_value=mask_value, tol=tol, 
                     detect = False)
        
        else:
            self.run(std1=std1, std2=std2, plot=plot, mask_value=mask_value, tol=tol, 
                     detect = detect, diagnostic_plots = diagnostic_plots, 
                     save = save, overwrite = overwrite)
        
    def run(self, std1=3.0, std2=3.0, plot=False, mask_value=1000, tol=0.003, detect = True, diagnostic_plots = True, save = True, overwrite = False):
        
        ref, diff, poisson_noise, fx = create_diff_image_de(self.tpf, num_cores = self.num_cores, plot = plot, mask_value=mask_value, tol=tol)
        
        self.ref = ref
        self.diff = diff
        self.poisson_noise = poisson_noise
        
        ref_value = np.nansum(ref)
        self.ref_value = ref_value
        self.sum_fluxes = fx
        
        if detect:
            stars = self.detection(std1 = 3.0, std2 = 3.0, plot = False)
            if diagnostic_plots:
                diagnostic_plotting(stars, processed = False)
            self.stars = stars
            
            if save:
                self.saving_func(self.tpf, stars, overwrite)
               
    def detection(self, std1=3.0, std2=3.0, plot=False):
        diff = deepcopy(self.diff)
        epsf = deepcopy(self.epsf)
        ref_value = deepcopy(self.ref_value)
        sum_fluxes = deepcopy(self.sum_fluxes)
        
        with Pool(processes=self.num_cores) as pool:
            process_frame_partial = partial(process_frame, diff = diff, downsampled_array = epsf, std1 = std1, std2 = std2, plot = plot)
            results = list(tqdm(pool.imap(process_frame_partial, range(len(diff))), total=len(diff), desc='Finding stars'))

        stars = [star for result in results for star in result]

        if len(stars) > 0:
            stars = pd.concat(stars, ignore_index=True)
            stars['ref_flux'] = ref_value
            stars['sum_flux'] = sum_fluxes[stars['frame'].to_numpy().astype(int)]
            
            return stars
        else:
            return None
        
    def saving_func(self, tpf_input, stars, overwrite):
        
        columns = ['campaign', 'target_id', 'ra', 'dec', 'filename']
        
        processed_file_name = './object_ids/object_process.csv'
        
        save_list = [tpf_input.campaign, tpf_input.targetid, 
                    tpf_input.ra, tpf_input.dec, self.full_file_name]
        
        if os.path.exists(processed_file_name):
            processed = pd.read_csv(processed_file_name)
            if 'campaign' in processed.columns and 'target_id' in processed.columns:
                mask = (processed['campaign'] == tpf_input.campaign) & (processed['target_id'] == tpf_input.targetid)
                
                if mask.any():
                    if overwrite:
                        processed.loc[mask] = save_list
                    else:
                        pass
                else:
                    processed = processed.append(pd.DataFrame([save_list], columns = columns), ignore_index=True)
                
            processed.to_csv(processed_file_name, index=False)
        else:
            new_row = pd.DataFrame([save_list], columns=columns)
            new_row.to_csv(processed_file_name, index=False)
        
        stars.to_csv(self.full_file_name, index=False)
        
    def _number_cores(self, num_cores):
        if num_cores is None:
            self.num_cores = max(1, cpu_count()//2)
        elif isinstance(num_cores, int):
            if num_cores >= cpu_count():
                self.num_cores = cpu_count()
            else:
                self.num_cores = num_cores
                print(f"Using {self.num_cores} cores")
        else:
            raise ValueError("num_cores must be an integer or NoneType")
 