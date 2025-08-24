
from Kakapo.photometry import forced_photometry
# from Kakapo.difference_image im
from Kakapo.cleaning_curve import correction_smoothing_lightcurve, wavelet_denoise, gauss_smooth, binned_averages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.stats import sigma_clipped_stats

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from astropy.time import Time

from tqdm import tqdm
import os
from copy import deepcopy

minimum_number = 5

def causal_baseline_mask(lc, end_idx, min_pts=48, max_lookback=500):
    n = lc.size
    start = max(0, end_idx - min_pts)
    mask = np.zeros(n, bool)
    # start with a small causal window, grow left until we have min_pts finite
    L = 32
    left = max(0, end_idx - L)
    mask[left:end_idx] = True
    while np.sum(np.isfinite(lc) & mask) < min_pts and left > 0 and (end_idx-left) < max_lookback:
        step = min(32, left)       # grow in chunks
        left -= step
        mask[left:end_idx] = True
    # strictly causal (exclude end_idx and after)
    mask[end_idx:] = False
    return mask

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

class Implement_reductions:
    def __init__(self, stars, tpf_info, diff, epsf_data, 
                 noise, corrlim = 0, difflim = 100, 
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
        self.dist_cut = dist_cut
        
        self.time = tpf_info.time.value
        self.campaign = tpf_info.campaign
        self.targetid = tpf_info.targetid
        
        self.cadence = self.time[1] - self.time[0]
        
        self.noise = noise
            
        corr = self.filter_detections(self.stars, initial = True)
        corr = corr.reset_index(drop=True)
        
        # events = self.events_discover(corr)
        events = self._grouping(corr, f_dist = 50)
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
        
    def _grouping(self, corr: pd.DataFrame, f_dist: int = 50) -> pd.DataFrame | None:
        """
        Group detections with DBSCAN in O(N log N) time, using only C/Fortran
        code paths from scikit-learn (no Python callback per point pair).
        """
        if corr.empty:
            return None

        # Scale the frame axis so that `eps` of 1.25 encloses ±f_dist frames.
        data = corr[['xcentroid', 'ycentroid', 'frame']].values.astype(np.float32)
        data[:, 2] *= 1.5 / f_dist

        db = DBSCAN(eps=1.5,
                    min_samples=minimum_number,
                    metric='euclidean',           # now fully compiled
                    algorithm='auto',        # fastest for 3‑D Euclidean
                    n_jobs=1)                     # keep it serial – you already parallelise at a higher level
        labels = db.fit_predict(data)

        corr = corr.assign(cluster=labels)
        corr = corr[corr.cluster != -1]           # drop noise points

        return corr if not corr.empty else None

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

    def _filter_cluster_by_frame_gradient(self, cluster, max_gap=50):
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
        
        full_events = pd.DataFrame(columns=['cluster', 'sig_max', 'sig_84', 
                                            'frame_min', 'frame_max', 
                                            'remapped_frame_min', 'remapped_frame_max', 
                                            'bjds', 'flux', 'flux_err'])
        
        new_stars = pd.DataFrame(columns=events.columns)
        
        events['lc_sig'] = -1*np.ones(len(events))
        
        # for cluster_id in tqdm(cluster_ids, desc='Clusters'):
        for cluster_id in cluster_ids:
            # print(cluster_id, self.campaign, self.targetid)
            cluster = events[events['cluster'] == cluster_id]
            
            if len(cluster) < minimum_number:
                # print(f'ZZZ det len: {len(cluster)}')
                continue
            
            frame_min = int(cluster['frame'].min())
            frame_max = int(cluster['frame'].max())
            
            # time_min = self.time[frame_min]
            # time_max = self.time[frame_max]
            
            x, _, xstd = sigma_clipped_stats(cluster['xcentroid'].values, sigma = 3)
            y, _, ystd = sigma_clipped_stats(cluster['ycentroid'].values, sigma = 3)
            
            if (xstd >= self.dist_cut) | (ystd >= self.dist_cut):
                # print(f'ZZZ Dist cut fail; x = {xstd:.3f}, y = {ystd:.3f}')
                continue
            
            args = self._check_lc_significance(frame_min, frame_max, x, y, 1, grad_val = -60)
            
            sig_max, sig_med, lc_sig, indices, bjds, flux, flux_err, og_ind, rmap_ind = args
            
            if sig_med < siglim:
                # print(f'ZZZ Sig cut fail: {sig_med:.3f}')
                continue
            
            filtered_cluster = cluster[cluster['frame'].isin(indices)]
            filtered_cluster = filtered_cluster.reset_index(drop=True)
            frame_inds = filtered_cluster['frame'].values.astype(int)
            
            # ZZZ DO THIS MANUALLY
            
            missing_vals = np.setdiff1d(frame_inds, og_ind)
            mask_inds = np.array([np.where(og_ind == val)[0][0] for val in frame_inds])
            filtered_lc_sig = lc_sig[mask_inds]
            
            filtered_final_cluster = deepcopy(filtered_cluster)
            try:
                filtered_final_cluster['lc_sig'] = filtered_lc_sig
            except:
                raise ValueError('Yeet')
            
            rmap_ind_min = rmap_ind[og_ind == filtered_final_cluster['frame'].min()][0]
            rmap_ind_max = rmap_ind[og_ind == filtered_final_cluster['frame'].max()][0]
            
            if len(filtered_lc_sig) < minimum_number:
                continue
            else:
                filtered_cluster['cluster'] = len(full_events)
                new_stars = pd.concat([new_stars, filtered_cluster])
                full_events.loc[len(full_events)] = [len(full_events), sig_max, sig_med, 
                                                     filtered_final_cluster['frame'].min(), 
                                                     filtered_final_cluster['frame'].max(), 
                                                     rmap_ind_min, rmap_ind_max,
                                                     bjds, flux, flux_err]
        
        new_stars = new_stars.reset_index(drop=True)
        
        if len(full_events) == 0:
            return None, None
        else:
            full_events, events_filtered = self._lightcurve_event_checker(new_stars, full_events)

            return full_events, events_filtered 
   
    def _check_lc_significance(self, start, end, x, y, flux_sign, grad_val = -60):
        cadence = self.cadence
        
        bjds = deepcopy(self.time)
        
        lc = forced_photometry(self.diff, x, y, self.epsf)
        flux_err = forced_photometry(self.noise, x, y, self.epsf)
        
        og_ind = np.arange(0, len(lc))
        
        mask = np.isfinite(lc) & np.isfinite(bjds) & np.isfinite(flux_err)
        
        ind_mask = np.where(mask)[0]
        
        og_ind = og_ind[mask]
        
        lc = lc[mask]
        bjds = bjds[mask]
        flux_err = flux_err[mask]
        
        rmap_ind = np.arange(0, len(lc))
        
        lc = gauss_smooth(bjds, lc)
        
        sig_max, sig_med, lc_sig = self._significance_runner(lc, start, end, min_points = 60, 
                                                             flux_sign = flux_sign, grad_val = grad_val)
        
        return sig_max, sig_med, lc_sig, ind_mask, bjds, lc, flux_err, og_ind, rmap_ind
    
    def _significance_runner(self, lc, start, end, min_points = 60, 
                             flux_sign = 1, grad_val = -60):
        
        gradients = np.gradient(lc)
        buffer = 10
        base_range = 40
        
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
        
        ind = causal_baseline_mask(lc, frame_start, frame_end, base_range, min_points=min_points)
        med = np.nanmedian(lc[ind])
        gradmed = np.nanmedian(gradients[ind])
        std = np.nanstd(lc[ind], ddof = 1)
        gradstd = np.nanstd(gradients[ind], ddof = 1)
        lcevent = lc[int(start):int(end)]
        gradevent = gradients[int(start):int(end)]
        
        lc_sig = (lcevent - med) / std
        # grad_sig = (gradients - gradmed) / gradstd
        # indices = np.where((np.abs(grad_sig) < 10) & (gradients > grad_val))[0]
        
        try:
            sig_max = abs(np.nanmax(lc_sig))
            sig_med = abs(np.nanpercentile(lc_sig, 84))
        except:
            sig_med = -1
            sig_max = -1
        
        if np.nansum(np.isfinite(lc[int(frame_start):int(frame_end)])) < minimum_number:
            sig_med = -1
        
        lc_sig = (lc - med) / std
        
        # indices_mask = start to end
        
        return sig_max, sig_med, lc_sig * flux_sign

    def _lightcurve_event_checker(self, stars, events):
        
        cluster_ids = np.unique(stars['cluster'])
        
        full_events = pd.DataFrame(columns=['cluster', 'frame_min', 'frame_max', 'time_min', 'time_max', 
                                            'remapped_frame_min', 'remapped_frame_max', 'x', 'y', 'xstd', 'ystd', 
                                            'sig_max', 'sig_84', 'mjds', 'flux', 'flux_err',
                                            'roundness', 'fwhm', 'snr', 'psfdiff', 'correlation', 
                                            'poisson_thresh', 'e_roundness', 'e_fwhm', 'e_snr', 'e_psfdiff', 
                                            'e_correlation', 'e_poisson_thresh'])
        
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
            
            rmap_frame_min = cluster_events['remapped_frame_min'].iloc[0]
            rmap_frame_max = cluster_events['remapped_frame_max'].iloc[0]
            
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
            # print(f"ZZZ length filt. cluster: {len(temp_cluster)}")
            
            temp_cluster = self._filter_cluster_by_frame_gradient(temp_cluster, max_gap=50)
            
            if temp_cluster is None:
                # print(f"ZZZ frame grad.")
                continue
            
            mask = self.mask_detections(correlation, psfdiff, fwhm, snr, roundness, poisson_thresh, xstd, ystd)
            
            if not mask: 
                continue
            
            if len(cluster) < minimum_number:
                continue
            
            bjds = np.array(cluster_events['bjds'].iloc[0] + 54832.5)
            
            t_bary = Time(bjds, format="mjd", scale="tdb")
            t_mjd_utc = t_bary.utc.mjd   # MJD in UTC
            
            flux = np.array(cluster_events['flux'].iloc[0])
            flux_err = np.array(cluster_events['flux_err'].iloc[0])

            stars_new_df = pd.concat([stars_new_df, cluster])
            
            full_events.loc[len(full_events)] = [len(full_events) + 1, frame_min, frame_max, time_min, time_max,
                                                 rmap_frame_min, rmap_frame_max, x, y, xstd, ystd, 
                                                 cluster_events['sig_max'].iloc[0], cluster_events['sig_84'].iloc[0], 
                                                 t_mjd_utc, flux, flux_err, roundness, fwhm, snr, psfdiff, 
                                                 correlation, poisson_thresh, 
                                                 e_roundness, e_fwhm, e_snr, e_psfdiff, e_correlation, e_poisson_thresh]
        
        if len(full_events) == 0:
            return None, None
        else:
            stars_new_df = stars_new_df.reset_index(drop=True)
            
            return full_events, stars_new_df 