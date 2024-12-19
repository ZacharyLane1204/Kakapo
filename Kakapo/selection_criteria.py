
from Kakapo.photometry import forced_photometry

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import os
from copy import deepcopy

minimum_number = 5

class Implement_reductions:
    def __init__(self, stars, tpf_info, diff, corrlim = 0, difflim = 100, 
                 fwhmlim = 5, maxlim = 10, snrlim = 1, roundness = 0.8, 
                 poiss_val= 1, siglim = 2):
        
        self.stars = stars
        self.diff = diff
        self.corrlim = corrlim
        self.difflim = difflim
        self.fwhmlim = fwhmlim
        self.maxlim = maxlim
        self.snrlim = snrlim
        self.roundness = roundness
        self.poiss_val = poiss_val
        
        self.time = tpf_info.time
            
        corr = self.filter_detections()
        corr = corr.reset_index(drop=True)
        
        # events = self.events_discover(corr)
        events = self._grouping(corr)
        
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
        
    def filter_detections(self):
        
        corr = self.stars[(self.stars.correlation > self.corrlim) & (self.stars.psfdiff < self.difflim) & 
                    (self.stars.fwhm < self.fwhmlim) & ((self.stars.fwhm > 0.8)) & (self.stars.max_value > self.maxlim) & 
                    (self.stars.snr > self.snrlim) & (self.stars.snr < 10000) & 
                    (abs(self.stars.roundness) < self.roundness) &
                    (self.stars.poisson_thresh >= self.poiss_val)]
        
        return corr        
    
    def _grouping(self, corr):
        
        def custom_distance(p1, p2):
            xy_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) # Euclidean distance for xcentroid and ycentroid
            frame_dist = np.abs(p1[2] - p2[2]) # Absolute difference for frame
            
            if xy_dist <= 0.85 and frame_dist <= 30: # Combine both distances with their respective thresholds
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
    
    def detected_events(self, events, siglim = 2):
        
        cluster_ids = np.unique(events['cluster'])
        
        full_events = pd.DataFrame(columns=['cluster', 'sig_max', 'sig_med', 'frame_min', 'frame_max'])
        
        new_stars = pd.DataFrame(columns=events.columns)
        
        events['lc_sig'] = -1*np.ones(len(events))
        
        for cluster_id in cluster_ids:
            cluster = events[events['cluster'] == cluster_id]
            
            frame_min = cluster['frame'].min()
            frame_max = cluster['frame'].max()
            
            x = cluster['xcentroid'].mean()
            y = cluster['ycentroid'].mean()
            
            sig_max, sig_med, lc_sig, indices = self._check_lc_significance(frame_min, frame_max, x, y, 1, 
                                                                            buffer = 1.3, base_range=1.8, grad_val = -100)
            if sig_med < siglim:
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
                filtered_cluster['cluster'] = len(full_events)
                new_stars = pd.concat([new_stars, filtered_cluster])
                full_events.loc[len(full_events)] = [len(full_events), sig_max, sig_med, 
                                                     filtered_final_cluster['frame'].min(), 
                                                     filtered_final_cluster['frame'].max()]
        
        new_stars = new_stars.reset_index(drop=True)
        
        if len(full_events) == 0:
            return None, None
        else:
            full_events, events_filtered = self._lightcurve_event_checker(new_stars, full_events)

            return full_events, events_filtered 
   
    def _check_lc_significance(self, start, end, x, y, flux_sign, buffer = 1.3, base_range=1.8, grad_val = -100):
        cadence = self.time[1] - self.time[0]
        cadence = cadence.value
        
        buffer = int(buffer/cadence)
        base_range = int(base_range/cadence)
        
        lc = forced_photometry(self.diff, x, y)
        
        lc[np.isnan(lc)] = 0
        
        gradients = np.gradient(lc)
        
        indices = np.where(gradients > grad_val)[0]
        
        
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
            
            
        # plt.figure()
        # plt.plot(lc)
        # for i in indices:
        #     plt.axvline(i, color = 'r')
        # plt.axvline(start, color = 'k')
        # plt.axvline(end, color = 'k')
        # plt.axvline(baseline_start, color = 'C1')
        # plt.axvline(baseline_end, color = 'C1')
        # plt.show()
        
        frames = np.arange(len(lc))
        ind = ((frames > baseline_start) & (frames < frame_start)) | ((frames < baseline_end) & (frames > frame_end))
        med = np.nanmedian(lc[ind])
        std = np.nanstd(lc[ind], ddof = 1)
        lcevent = lc[int(start):int(end)]
        
        # Light curve significance
        lc_sig = (lcevent - med) / std

        if flux_sign >= 0:
            sig_max = np.nanmax(lc_sig)
            sig_med = np.nanmean(lc_sig)
            
        else:
            sig_max = abs(np.nanmin(lc_sig))
            sig_med = abs(np.nanmean(lc_sig))
        
        lc_sig = (lc - med) / std
        return sig_max, sig_med, lc_sig * flux_sign, indices

    def _lightcurve_event_checker(self, stars, events):
        
        cluster_ids = np.unique(stars['cluster'])
        
        full_events = pd.DataFrame(columns=['cluster', 'frame_min', 'frame_max', 'time_min', 'time_max', 
                                            'x', 'y', 'xstd', 'ystd', 'sig_max', 'sig_med', 
                                            'roundness', 'fwhm', 'snr', 'psfdiff', 'correlation', 'poisson_thresh', 
                                            'e_roundness', 'e_fwhm', 'e_snr', 'e_psfdiff', 'e_correlation', 'e_poisson_thresh'])
        
        stars_new_df = pd.DataFrame(columns=stars.columns)
        
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

            x = cluster['xcentroid'].mean() # Calculate central positions and other statistics
            y = cluster['ycentroid'].mean()
            
            xstd = cluster['xcentroid'].std()
            ystd = cluster['ycentroid'].std()
            
            time_min = self.time[int(frame_min)]
            time_max = self.time[int(frame_max)]
            
            roundness = cluster['roundness'].mean()
            e_roundness = cluster['roundness'].std()
            fwhm = cluster['fwhm'].mean()
            e_fwhm = cluster['fwhm'].std()
            snr = cluster['snr'].mean()
            e_snr = cluster['snr'].std()
            psfdiff = cluster['psfdiff'].mean()
            e_psfdiff = cluster['psfdiff'].std()
            correlation = cluster['correlation'].mean()
            e_correlation = cluster['correlation'].std()
            poisson_thresh = cluster['poisson_thresh'].mean()
            e_poisson_thresh = cluster['poisson_thresh'].std()
            
            cluster = cluster.reset_index(drop=True)
            cluster['cluster'] = len(full_events) + 1

            stars_new_df = pd.concat([stars_new_df, cluster])
            
            full_events.loc[len(full_events)] = [len(full_events) + 1, frame_min, frame_max, time_min, time_max,
                                                x, y, xstd, ystd, cluster_events['sig_max'].iloc[0], 
                                                cluster_events['sig_med'].iloc[0], roundness, fwhm, snr, psfdiff,
                                                correlation, poisson_thresh, 
                                                e_roundness, e_fwhm, e_snr, e_psfdiff, e_correlation, e_poisson_thresh]
        
        if len(full_events) == 0:
            return None, None
        else:
            stars_new_df = stars_new_df.reset_index(drop=True)
            return full_events, stars_new_df 