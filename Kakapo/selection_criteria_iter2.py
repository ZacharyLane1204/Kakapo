
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

def detect_onset(lc, w=97, k=4.5, gmin=0.0, consec=3):
    med, mad = rolling_median_mad(lc, w) # rolling robust baseline (causal)
    if np.isnan(med).any():  
        # Try a "declining baseline" from future points
        med_rev, mad_rev = rolling_median_mad(lc[::-1], w)
        med_rev, mad_rev = med_rev[::-1], mad_rev[::-1]  # reverse back to align
        # fill NaNs from forward baseline with reversed baseline values
        med = np.where(np.isnan(med), med_rev, med)
        mad = np.where(np.isnan(mad), mad_rev, mad)
    
    mad = np.clip(mad, np.nanmedian(mad)*0.05, None)  # floor MAD to avoid division spikes
    z = (lc - med) / mad
    g = np.gradient(lc)

    hits = (z > k) #& (g > gmin) # causal condition
    run = 0 # persistence (consec points)
    for i, h in enumerate(hits):
        run = run + 1 if h else 0
        if run >= consec:
            return max(0, i - consec + 1)   # onset index
    return None

def rolling_median_mad(x, w):
    # Utility: rolling median + MAD (replace sigma clipping)
    s = pd.Series(x)
    med = s.rolling(w, min_periods=w//2).median()
    mad = (s - med).abs().rolling(w, min_periods=w//2).median()
    mad = 1.4826 * mad
    return med.to_numpy(), mad.to_numpy()

def detrend_with_median(x, w=48):
    # Stage 1: Baseline detrending
    """Remove sawtooth-like baseline using rolling median"""
    med, _ = rolling_median_mad(x, w)
    return x - med

def detect_anomalies(x, w=48, z_thresh=1.5, grad_thresh=2, persistence=2):
    # Stage 2: Anomaly detectors
    """
    Combined Causal Z + Gradient + Persistence detector
    - Causal Z-score (rolling median, MAD-based)
    - Gradient threshold
    - Persistence check
    """
    med, mad = rolling_median_mad(x, w)
    
    if np.isnan(med).any():
        med_rev, mad_rev = rolling_median_mad(x[::-1], w) # Try a "declining baseline" from future points
        med_rev, mad_rev = med_rev[::-1], mad_rev[::-1]  # reverse back to align
        med = np.where(np.isnan(med), med_rev, med) # fill NaNs from forward baseline with reversed baseline values
        mad = np.where(np.isnan(mad), mad_rev, mad)

    z = (x - med) / mad # Causal z-score
    causal_mask = z > z_thresh

    # grad = np.diff(x, prepend=x[0]) # Gradient check
    # grad_mask = np.abs(grad) > grad_thresh * np.nanmedian(np.abs(grad))

    persistent_mask = np.convolve(causal_mask.astype(int), np.ones(persistence, dtype=int), 'same') >= persistence # Persistence: flag if condition holds >= N frames

    return causal_mask | persistent_mask

def cusum_detector(x, drift=0.01, threshold=2):
    # Stage 3: CUSUM/Page–Hinkley detector (optional)
    """CUSUM style change-point detection"""
    g_plus, g_minus = 0, 0
    alarms = np.zeros_like(x, dtype=bool)
    mean_x = np.nanmean(x)

    for i in range(len(x)):
        g_plus = max(0, g_plus + (x[i] - mean_x - drift))
        g_minus = max(0, g_minus - (x[i] - mean_x + drift))
        if g_plus > threshold or g_minus > threshold:
            alarms[i] = True
            g_plus, g_minus = 0, 0
    return alarms

def causal_baseline_mask(lc, end_idx, min_pts=48, max_lookback=500):
    n = lc.size
    start = max(0, end_idx - min_pts)
    mask = np.zeros(n, bool)
    L = 32  # start with a small causal window, grow left until we have min_pts finite
    left = max(0, end_idx - L)
    mask[left:end_idx] = True
    while np.sum(np.isfinite(lc) & mask) < min_pts and left > 0 and (end_idx-left) < max_lookback:
        step = min(32, left)       # grow in chunks
        left -= step
        mask[left:end_idx] = True
    mask[end_idx:] = False # strictly causal (exclude end_idx and after)
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
        
        self.det_min_pts = 3   # Stage-1 (early) clustering len threshold
        self.val_min_pts = 5   # Stage-2 (validation) threshold
        
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
                         (stars.fwhm <= 8) &  (stars.fwhm >= 0.8) & 
                         (stars.snr >= 4) & (stars.snr < 10000) & 
                         (abs(stars.roundness) <= 0.99) & 
                         (stars.poisson_thresh >= 1)]
        else:
            corr = stars[(stars.correlation > self.corrlim) & 
                         (stars.psfdiff < self.difflim) & 
                         (stars.fwhm < self.fwhmlim) &  (stars.fwhm >= 0.9) & 
                         (stars.snr > self.snrlim) & (stars.snr < 10000) & 
                         (abs(stars.roundness) <= self.roundness) & 
                         (stars.poisson_thresh >= self.poiss_val)]
        
        return corr
    
    def mask_detections(self, correlation, psfdiff, fwhm, snr, 
                        roundness, poisson_thresh, xstd, ystd):
        
        mask =  (correlation >= self.corrlim) & (psfdiff <= self.difflim) & \
                (fwhm <= self.fwhmlim) & (fwhm >= 0.8) & \
                (snr >= self.snrlim) & (snr < 10000) & (abs(roundness) <= self.roundness) & \
                (poisson_thresh >= self.poiss_val) & \
                (xstd <= self.dist_cut) & (ystd <= self.dist_cut)
        
        
        return mask
        
    def _grouping(self, corr: pd.DataFrame, f_dist: int = 15) -> pd.DataFrame | None:
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
                    min_samples=self.det_min_pts,   # Stage-1 len threshold
                    metric='euclidean',
                    algorithm='auto',
                    n_jobs=1)
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

    def _filter_cluster_by_frame_gradient(self, cluster, max_gap=20):
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
        print(f'ZZZ length {len(cluster_ids)}')
        for cluster_id in cluster_ids:
            # print(cluster_id, self.campaign, self.targetid)
            cluster = events[events['cluster'] == cluster_id]
            
            if len(cluster) < self.det_min_pts: 
                continue
            
            frame_min = int(cluster['frame'].min())
            frame_max = int(cluster['frame'].max())
            
            x, _, xstd = sigma_clipped_stats(cluster['xcentroid'].values, sigma = 3)
            y, _, ystd = sigma_clipped_stats(cluster['ycentroid'].values, sigma = 3)
            
            if (xstd >= self.dist_cut) | (ystd >= self.dist_cut):
                continue
            
            args = self._check_lc_significance(frame_min, frame_max, x, y, 1, grad_val = -60)
            
            sig_max, sig_med, lc_sig, indices, bjds, flux, flux_err, og_ind, rmap_ind = args
            
            print(f"ZZZ cluster:{cluster_id}, xstd:{xstd}, ystd:{ystd}, sig_med:{sig_med}, sig_max:{sig_max}")
            
            if sig_med < siglim:
                continue
            
            filtered_cluster = cluster[cluster['frame'].isin(indices)]
            filtered_cluster = filtered_cluster.reset_index(drop=True)
            frame_inds = filtered_cluster['frame'].values.astype(int)
            
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
            
            if len(filtered_lc_sig) < self.det_min_pts: 
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
            print(f"Light-curve checking: {len(full_events)}")
            full_events, events_filtered = self._lightcurve_event_checker(new_stars, full_events)

            return full_events, events_filtered 
   
    def _check_lc_significance(self, start, end, x, y, flux_sign, grad_val=-60):
        bjds = deepcopy(self.time) # Raw photometry (no smoothing for detection)
        lc_raw   = forced_photometry(self.diff,  x, y, self.epsf)
        flux_err = forced_photometry(self.noise, x, y, self.epsf)

        mask = np.isfinite(lc_raw) & np.isfinite(bjds) & np.isfinite(flux_err) # finite mask
        lc_raw, bjds, flux_err = lc_raw[mask], bjds[mask], flux_err[mask]
        lc_raw = gauss_smooth(bjds, lc_raw)
        og_ind = np.where(mask)[0]
        rmap_ind = np.arange(lc_raw.size)

        # --- Detrend K2 6h sawtooth before change-point detection ---
        # cadence is in days; 6h = 0.25 day
        six_hr = 0.25
        w_jitter = max(25, int(round(0.8 * six_hr / self.cadence)))  # causal-ish, smaller than period
        lc_det = detrend_with_median(lc_raw, w=w_jitter)

        # -------------------- Stage 1: fast screen --------------------
        s1_mask = detect_anomalies(lc_det, w=w_jitter, z_thresh=1.5, grad_thresh=1, persistence=2)
        onset_idx = np.argmax(s1_mask) if s1_mask.any() else None

        # Fallback to onset finder if stage1 didn’t localize
        if onset_idx is None:
            onset_idx = detect_onset(lc_raw, w=97, k=4.5, gmin=0.0, consec=3)

        # -------------------- Stage 2: confirmation --------------------
        s2_mask = cusum_detector(lc_det, drift=0.01, threshold=5)

        # Choose combination strategy:
        # Strict (fewer FPs): require both → s1 & s2
        # Lenient (catch faint): either → s1 | s2
        combo_mask = s1_mask | s2_mask
        onset_idx = None
        if s1_mask.any():
            onset_idx = np.argmax(s1_mask)
        elif s2_mask.any():
            onset_idx = np.argmax(s2_mask)
        elif combo_mask.any():
            onset_idx = np.argmax(combo_mask)

        if onset_idx is None: # If no onset at all, keep legacy path but with robust stats
            end_idx = max(0, int(start))
        else:
            end_idx = int(onset_idx)

        # --- robust, CAUSAL baseline up to end_idx ---
        base_mask = causal_baseline_mask(lc_raw, end_idx=end_idx, min_pts=60, max_lookback=800)
        med = np.nanmedian(lc_raw[base_mask])
        mad = 1.4826 * np.nanmedian(np.abs(lc_raw[base_mask] - med))
        if not np.isfinite(mad) or mad == 0:
            mad = np.nanstd(lc_raw[base_mask], ddof=1)

        lc_sig = (lc_raw - med) / mad * flux_sign

        # score in a short horizon after onset to avoid waiting for peak
        if onset_idx is None:
            win_lo, win_hi = int(start), int(end)
        else:
            win_lo, win_hi = end_idx, min(end_idx + 150, lc_raw.size)

        lcevent = lc_sig[win_lo:win_hi]
        try:
            sig_max = float(np.nanmax(np.abs(lcevent))) if lcevent.size else -1
            sig_med = float(np.nanpercentile(np.abs(lcevent), 84)) if lcevent.size else -1
        except Exception:
            sig_max = sig_med = -1

        # keep a smoothed copy only for plots
        lc_viz = lc_raw.copy()

        ind_mask = np.where(mask)[0]
        return sig_max, sig_med, lc_sig, ind_mask, bjds, lc_viz, flux_err, og_ind, rmap_ind

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
            
            if len(cluster) < self.val_min_pts: 
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
            
            if len(temp_cluster) < self.val_min_pts: 
                continue
            
            # temp_cluster = self._filter_cluster_by_frame_gradient(temp_cluster, max_gap=50)
            
            # if temp_cluster is None:
            #     continue
            
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