
from Kakapo.photometry import forced_photometry
# from Kakapo.difference_image im
from Kakapo.cleaning_curve import correction_smoothing_lightcurve, wavelet_denoise

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.stats import sigma_clipped_stats

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from tqdm import tqdm
import os
from copy import deepcopy

minimum_number = 5

def get_valid_baseline_indices(lc, frame_start, frame_end, base_range, min_points=42):
    """
    Expands the baseline region until at least `min_points` non-NaN values are found.
    
    Parameters:
        lc : array-like
            The lightcurve.
        frame_start : int
            Start of the event window.
        frame_end : int
            End of the event window.
        base_range : int
            Initial extension from event to define baseline.
        min_points : int
            Minimum number of non-NaN values required in the baseline.
    
    Returns:
        baseline_start : int
            Final start index of baseline region.
        baseline_end : int
            Final end index of baseline region.
        valid_inds : np.ndarray (bool)
            Boolean array marking valid baseline indices (non-NaN).
    """
    baseline_start = frame_start - base_range
    baseline_end = frame_end + base_range

    max_len = len(lc)
    frames = np.arange(max_len)

    while True:
        baseline_start = max(0, baseline_start)
        baseline_end = min(max_len, baseline_end)

        baseline_mask = ((frames > baseline_start) & (frames < frame_start)) | \
                        ((frames < baseline_end) & (frames > frame_end))
        valid_inds = baseline_mask & ~np.isnan(lc)

        if np.sum(valid_inds) >= min_points or (baseline_start == 0 and baseline_end == max_len):
            break

        baseline_start -= 1
        baseline_end += 1

    return baseline_start, baseline_end, baseline_mask

class Implement_reductions:
    def __init__(self, stars, tpf_info, diff, epsf_data, 
                 thrusters, distance, corrlim = 0, difflim = 100, 
                 fwhmlim = 5, maxlim = 10, snrlim = 1, roundness = 0.8, 
                 poiss_val= 1, siglim = 2, dist_cut = 0.2):
        
        self.stars = stars
        self.diff = diff
        self.corrlim = corrlim
        self.difflim = difflim
        self.fwhmlim = fwhmlim
        self.maxlim = maxlim
        self.snrlim = snrlim
        self.roundness = roundness
        self.poiss_val = poiss_val
        self.epsf = epsf_data
        self.thrusters = thrusters
        self.distance = distance
        self.dist_cut = dist_cut
        
        self.time = tpf_info.time
            
        corr = self.filter_detections(self.stars, initial = True)
        corr = corr.reset_index(drop=True)
        
        # print('ZZZ Length detections:', len(corr))
        
        # events = self.events_discover(corr)
        events = self._grouping(corr, f_dist = 48)
        # print('ZZZ Length events:', len(events))
        # events = self._merge_close_clusters_fast(corr, dist_tolerance=1, f_dist=30)
        
        if events is not None:
            if len(events) > 0:
                
                full_events, filtered_stars = self.detected_events(events, siglim = siglim)
                self.filtered_stars = filtered_stars
                self.full_events = full_events
                
            else:
                self.filtered_stars = None
                self.full_events = None
        else:
            self.filtered_stars = None
            self.full_events = None
        
    def filter_detections(self, stars, initial = False):
        
        if initial:
            corr = stars[(stars.correlation >= 0.01) & 
                         (stars.psfdiff <= 2) & 
                         (stars.fwhm <= 8) &  (stars.fwhm >= 0.85) & 
                         (stars.max_value >= 0) & 
                         (stars.snr >= 4) & (stars.snr < 10000) & 
                         (abs(stars.roundness) <= 0.99) & 
                         (stars.poisson_thresh >= 1)]
        else:
            corr = stars[(stars.correlation > self.corrlim) & 
                         (stars.psfdiff < self.difflim) & 
                         (stars.fwhm < self.fwhmlim) &  (stars.fwhm >= 0.9) & 
                         (stars.max_value > self.maxlim) & 
                         (stars.snr > self.snrlim) & (stars.snr < 10000) & 
                         (abs(stars.roundness) <= self.roundness) & 
                         (stars.poisson_thresh >= self.poiss_val)]
        
        return corr
    
    def mask_detections(self, correlation, psfdiff, fwhm, snr, 
                        roundness, poisson_thresh, xstd, ystd):
        
        mask =  (correlation >= self.corrlim) & (psfdiff <= self.difflim) & \
                (fwhm <= self.fwhmlim) & (fwhm >= 0.9) & \
                (snr >= self.snrlim) & (snr < 10000) & (abs(roundness) <= self.roundness) & \
                (poisson_thresh >= self.poiss_val) & \
                (xstd <= self.dist_cut) & (ystd <= self.dist_cut)
        
        
        return mask
    
    def _grouping(self, corr, f_dist = 15):
        
        def custom_distance(p1, p2):
            xy_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) # Euclidean distance for xcentroid and ycentroid
            frame_dist = np.abs(p1[2] - p2[2]) # Absolute difference for frame
            
            if xy_dist <= 0.9 and frame_dist <= f_dist: # Combine both distances with their respective thresholds
                return 0  # In the same cluster (distance 0 means they are close enough)
            else:
                return 5  # Distance larger than threshold, separate clusters

        try:
            data = corr[['xcentroid', 'ycentroid', 'frame']].values

            db = DBSCAN(eps=1, min_samples=minimum_number, metric=custom_distance)  # 'metric' could be adjusted for your use case
            corr['cluster'] = db.fit_predict(data)
            
            corr = corr[corr['cluster'] != -1]
            
        except:
            return None
        
        return corr
    
    def _merge_close_clusters_fast(self, corr, dist_tolerance=0.6, f_dist=12):
        if 'cluster' not in corr.columns or len(corr) == 0:
            return corr

        unique_clusters = np.unique(corr['cluster'])
        n_clusters = len(unique_clusters)

        cluster_idx_map = {cid: i for i, cid in enumerate(unique_clusters)}
        idx_to_cluster = {i: cid for cid, i in cluster_idx_map.items()}

        cluster_props = np.zeros((n_clusters, 5))  # x, y, fmin, fmax, cluster_id
        for i, cid in enumerate(unique_clusters):
            sub = corr[corr['cluster'] == cid]
            cluster_props[i, 0] = sub['xcentroid'].mean()
            cluster_props[i, 1] = sub['ycentroid'].mean()
            cluster_props[i, 2] = sub['frame'].min()
            cluster_props[i, 3] = sub['frame'].max()
            cluster_props[i, 4] = cid

        tree = cKDTree(cluster_props[:, :2])
        pairs = tree.query_pairs(r=dist_tolerance)

        rows, cols = [], []
        for i, j in pairs:
            fmin_i, fmax_i = cluster_props[i, 2], cluster_props[i, 3]
            fmin_j, fmax_j = cluster_props[j, 2], cluster_props[j, 3]

            frame_gap = max(0, min(abs(fmin_i - fmax_j), abs(fmin_j - fmax_i)))
            if frame_gap <= f_dist:
                rows.extend([i, j])
                cols.extend([j, i])  # Make the graph undirected

        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_clusters, n_clusters))
        n_components, labels = connected_components(csgraph=graph, directed=False)

        new_cluster_ids = {idx_to_cluster[i]: new_id for i, new_id in enumerate(labels)}
        corr['cluster'] = corr['cluster'].map(new_cluster_ids)

        return corr

    def _filter_cluster_by_frame_gradient(self, cluster, max_gap=48):
        cluster = cluster.sort_values('frame').reset_index(drop=True)
        
        while len(cluster) > minimum_number:
            frame_diffs = np.diff(cluster['frame'].values)

            if np.all(frame_diffs <= max_gap):
                return cluster

            if frame_diffs[0] > frame_diffs[-1]:
                cluster = cluster.iloc[1:].reset_index(drop=True)
            else:
                cluster = cluster.iloc[:-1].reset_index(drop=True)

        return None  # Cluster too short after trimming
    
    def detected_events(self, events, siglim = 2):
        
        cluster_ids = np.unique(events['cluster'])
        
        full_events = pd.DataFrame(columns=['cluster', 'sig_max', 'sig_84', 'frame_min', 'frame_max'])
        
        new_stars = pd.DataFrame(columns=events.columns)
        
        events['lc_sig'] = -1*np.ones(len(events))
        
        # print('ZZZ Length of Clusters:', len(cluster_ids))
        # print(events)
        
        # for cluster_id in tqdm(cluster_ids, desc='Clusters'):
        for cluster_id in cluster_ids:
            cluster = events[events['cluster'] == cluster_id]
            
            if len(cluster) < minimum_number:
                continue
            
            frame_min = cluster['frame'].min()
            frame_max = cluster['frame'].max()
            
            x, _, xstd = sigma_clipped_stats(cluster['xcentroid'].values, sigma = 3)
            y, _, ystd = sigma_clipped_stats(cluster['ycentroid'].values, sigma = 3)
            
            if (xstd >= self.dist_cut) | (ystd >= self.dist_cut):
                continue
            
            sig_max, sig_med, lc_sig, indices = self._check_lc_significance(frame_min, frame_max, x, y, 1, 
                                                                            buffer = 1.2, base_range=2.6, grad_val = -60)
            if sig_med < siglim:
                # print('ZZZ Sig. Med', sig_med)
                continue
            
            filtered_cluster = cluster[cluster['frame'].isin(indices)]
            filtered_cluster = filtered_cluster.reset_index(drop=True)
            frame_inds = filtered_cluster['frame'].values.astype(int)
            
            filtered_lc_sig = lc_sig[frame_inds]
            filtered_lc_sig_indices = filtered_lc_sig > siglim
            
            filtered_frame_values = filtered_cluster['frame'].values[filtered_lc_sig_indices]
            filtered_frame_values = np.sort(filtered_frame_values, axis=None) 
            
            filtered_lc_sig = filtered_lc_sig[filtered_lc_sig_indices]
            filtered_final_cluster = filtered_cluster[filtered_cluster['frame'].isin(filtered_frame_values)]
            filtered_final_cluster['lc_sig'] = filtered_lc_sig
            
            if len(filtered_lc_sig) < minimum_number:
                continue
            else:
                # print(len(filtered_lc_sig))
                filtered_cluster['cluster'] = len(full_events)
                new_stars = pd.concat([new_stars, filtered_cluster])
                full_events.loc[len(full_events)] = [len(full_events), sig_max, sig_med, 
                                                     filtered_final_cluster['frame'].min(), 
                                                     filtered_final_cluster['frame'].max()]
        
        new_stars = new_stars.reset_index(drop=True)
        
        if len(full_events) == 0:
            # print('ZZZ We have failure!')
            return None, None
        else:
            full_events, events_filtered = self._lightcurve_event_checker(new_stars, full_events)

            return full_events, events_filtered 
   
    def _check_lc_significance(self, start, end, x, y, flux_sign, buffer = 1.1, base_range=2.85, grad_val = -40):
        cadence = self.time[1] - self.time[0]
        cadence = cadence.value
        
        buffer = int(buffer/cadence)
        base_range = int(base_range/cadence)
        
        lc = forced_photometry(self.diff, x, y, self.epsf)
        lc = correction_smoothing_lightcurve(lc, self.distance < 0.25, window=35, sigma=3)
        lc = wavelet_denoise(lc, wavelet='coif5', level=3, keep='low', mode = 'smooth')
        
        # if len(self.thrusters) > 10:
        #     lc = correct_motion_lightcurve(lc, self.distance, self.thrusters)
        
        gradients = np.gradient(lc)
        
        # Setting up the light curve
        frame_start = start - buffer
        frame_end = end + buffer
        if frame_start < 0:
            frame_start = 0
            frame_end += buffer
        if frame_end > len(lc):
            frame_end = len(lc) - 1 
            frame_start -= buffer
        
        if (frame_start < 0):
            frame_start = 0
        if (frame_end > len(lc)):
            frame_end = len(lc) - 1 
        
        baseline_start = frame_start - base_range
        baseline_end = frame_end + base_range
        if baseline_start < 0:
            baseline_start = 0
        if baseline_end > len(lc):
            baseline_end = len(lc) - 1
        
        frames = np.arange(len(lc))
        
        baseline_start, baseline_end, ind = get_valid_baseline_indices(lc, frame_start, frame_end, base_range, min_points=48)
        med = np.nanmedian(lc[ind])
        gradmed = np.nanmedian(gradients[ind])
        std = np.nanstd(lc[ind], ddof = 1)
        gradstd = np.nanstd(gradients[ind], ddof = 1)
        lcevent = lc[int(start):int(end)]
        gradevent = gradients[int(start):int(end)]
        
        lc_sig = (lcevent - med) / std
        grad_sig = (gradients - gradmed) / gradstd
        
        indices = np.where((np.abs(grad_sig) < 8) & (gradients > grad_val))[0]
        
        try:
            sig_max = abs(np.nanmax(lc_sig))
            sig_med = abs(np.nanpercentile(lc_sig, 84))
        except:
            sig_med = -1
            sig_max = -1
        
        # print('Frame Start:', frame_start, frame_end, type(frame_start), type(frame_end))
        
        if np.nansum(np.isfinite(lc[int(frame_start):int(frame_end)])) < minimum_number:
            sig_med = -1
        
        lc_sig = (lc - med) / std
        return sig_max, sig_med, lc_sig * flux_sign, indices

    def _lightcurve_event_checker(self, stars, events):
        
        cluster_ids = np.unique(stars['cluster'])
        
        # print('ZZZ Length of Events:', len(cluster_ids))
        # print(events)
        
        full_events = pd.DataFrame(columns=['cluster', 'frame_min', 'frame_max', 'time_min', 'time_max', 
                                            'x', 'y', 'xstd', 'ystd', 'sig_max', 'sig_84', 
                                            'roundness', 'fwhm', 'snr', 'psfdiff', 'correlation', 'poisson_thresh', 
                                            'e_roundness', 'e_fwhm', 'e_snr', 'e_psfdiff', 'e_correlation', 'e_poisson_thresh'])
        
        stars_new_df = pd.DataFrame(columns=stars.columns)
        
        # for cluster_id in tqdm(cluster_ids, desc='Events'):
        for cluster_id in cluster_ids:
            cluster = stars[stars['cluster'] == cluster_id]
            
            if len(cluster) < minimum_number:
                continue
            
            cluster_events = events[events['cluster'] == cluster_id]
            cluster_events = cluster_events.reset_index(drop=True)
            
            frame_min = int(cluster['frame'].min()) # Get frame range for the cluster
            frame_max = int(cluster['frame'].max())
            
            if frame_min >= frame_max:
                continue
            
            time_min = self.time[int(frame_min)]
            time_max = self.time[int(frame_max)]
            
            x, _, xstd = sigma_clipped_stats(cluster['xcentroid'].values, sigma = 3)
            y, _, ystd = sigma_clipped_stats(cluster['ycentroid'].values, sigma = 3)
            roundness, _, e_roundness = sigma_clipped_stats(cluster['roundness'].values, sigma = 3)
            fwhm, _, e_fwhm = sigma_clipped_stats(cluster['fwhm'].values, sigma = 3)
            snr, _, e_snr = sigma_clipped_stats(cluster['snr'].values, sigma = 3)
            psfdiff, _, e_psfdiff = sigma_clipped_stats(cluster['psfdiff'].values, sigma = 3)
            correlation, _, e_correlation = sigma_clipped_stats(cluster['correlation'].values, sigma = 3)
            poisson_thresh, _, e_poisson_thresh = sigma_clipped_stats(cluster['poisson_thresh'].values, sigma = 3)
            
            cluster = cluster.reset_index(drop=True)
            cluster['cluster'] = len(full_events) + 1
            
            temp_cluster = self.filter_detections(cluster, initial = False)
            temp_cluster = self._filter_cluster_by_frame_gradient(temp_cluster, max_gap=45)
            
            if temp_cluster is None:
                continue
            
            mask = self.mask_detections(correlation, psfdiff, fwhm, snr, roundness, poisson_thresh, xstd, ystd)
            
            # print('ZZZ:', correlation, psfdiff, fwhm, snr, roundness, poisson_thresh, xstd, ystd)
            
            if mask != True:
                continue
            
            # if len(cluster) < minimum_number:
            #     continue

            stars_new_df = pd.concat([stars_new_df, cluster])
            
            full_events.loc[len(full_events)] = [len(full_events) + 1, frame_min, frame_max, time_min, time_max,
                                                x, y, xstd, ystd, cluster_events['sig_max'].iloc[0], 
                                                cluster_events['sig_84'].iloc[0], roundness, fwhm, snr, psfdiff,
                                                correlation, poisson_thresh, 
                                                e_roundness, e_fwhm, e_snr, e_psfdiff, e_correlation, e_poisson_thresh]
        
        if len(full_events) == 0:
            return None, None
        else:
            stars_new_df = stars_new_df.reset_index(drop=True)
            
            # print('ZZZ New Length of Events:', len(full_events))
            # print(full_events)
            
            return full_events, stars_new_df 