
from photutils.aperture import RectangularAperture, RectangularAnnulus,CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats, aperture_photometry

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import os
from copy import deepcopy

minimum_number = 5

def check_within_radius(array, x, y, radius=0.6):
    array = array.astype(bool)
    
    rows, cols = array.shape
    
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    
    within_radius = distances <= radius
    return np.any(within_radius & array)

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

def consecutive_points(data, stepsize=5, min_group_size=minimum_number):
    data = np.sort(data) # Sort the data to ensure correct ordering
    
    split_indices = np.where(np.diff(data) > stepsize)[0] + 1 # Find the indices where the difference between consecutive points is larger than the stepsize
    
    groups = np.split(data, split_indices) # Split the data at these indices
    
    valid_groups = [group for group in groups if len(group) >= min_group_size] # Filter out groups with fewer than min_group_size points
    
    if valid_groups: # If no valid groups, return None
        return valid_groups
    else:
        return None

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
        
        # gradient_removal = self._remove_gradient(corr)
        
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
        
        print(len(corr))
        
        return corr        
    
    def _grouping(self, corr):
        
        def custom_distance(row1, row2):
            # Euclidean distance for x and y centroids
            spatial_distance = np.sqrt((row1['xcentroid'] - row2['xcentroid'])**2 + (row1['ycentroid'] - row2['ycentroid'])**2)
            # Absolute distance for frames
            temporal_distance = abs(row1['frame'] - row2['frame'])
            # Return the combined distance (adjust weights if necessary)
            return spatial_distance, temporal_distance
        
        def combined_distance(x, y):
            # Extract spatial distance and temporal distance
            spatial_distance, temporal_distance = custom_distance(x, y)
            # You could tune this based on the relevance of the spatial/temporal distances
            return spatial_distance <= 0.85 and temporal_distance <= 30
        
        try:
            data = corr[['xcentroid', 'ycentroid', 'frame']]

            db = DBSCAN(eps=0.85, min_samples=minimum_number, metric='euclidean')  # 'metric' could be adjusted for your use case
            corr['cluster'] = db.fit_predict(data)
        except:
            return None
        
        return corr

    # def events_discover(self, corr, frame_buffer=30, spatial_buffer=1):
    #     new_df = pd.DataFrame(columns=['xcentroid', 'ycentroid', 'frame', 'cluster_number'])

    #     for i in range(len(corr)):
            
    #         x = corr['xcentroid'].iloc[i]
    #         y = corr['ycentroid'].iloc[i]
    #         frame = corr['frame'].iloc[i]
    
    #         if i == 0:
    #             new_df.loc[len(new_df)] = [x, y, frame, 0]
    #         else:
    #             restricted = new_df[(abs(new_df['frame'] - frame) < frame_buffer) & 
    #                                 (abs(new_df['xcentroid'] - x) < spatial_buffer) & 
    #                                 (abs(new_df['ycentroid'] - y) < spatial_buffer)]
                
    #             restricted = restricted.reset_index(drop=True)
                
    #             if len(restricted) > 0:
    #                 new_df.loc[len(new_df)] = [x, y, frame, restricted.iloc[0]['cluster_number']]
    #             else:
    #                 new_df.loc[len(new_df)] = [x, y, frame, max(new_df['cluster_number']) + 1]


    #     cluster_counts = new_df['cluster_number'].value_counts()
    #     valid_clusters = cluster_counts[cluster_counts >= minimum_number].index
    #     filtered_df = new_df[new_df['cluster_number'].isin(valid_clusters)].reset_index(drop=True)

    #     filtered_df['cluster_number'] = pd.factorize(filtered_df['cluster_number'])[0]
        
    #     if len(filtered_df) == 0:
    #         return None
    #     else:
    #         merged_df = corr.merge(filtered_df[['xcentroid', 'ycentroid', 'frame', 'cluster_number']], 
    #                         on=['xcentroid', 'ycentroid', 'frame'], 
    #                         how='inner')
    #         merged_df = merged_df.rename(columns={'cluster_number': 'cluster'})
    #         return merged_df
    
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
                                                                            buffer = 1, base_range=1.5, grad_val = -100)
            if sig_med < siglim:
                continue
            
            # f_indices = np.where(grads)[0]
            
            # new_f_indices = set()

            # for idx in f_indices: # Loop through the original indices and add +/- 2 ranges
            #     new_f_indices.update(range(max(0, idx-1), min(len(grads), idx+2)))
            
            # new_f_indices = np.sort(list(new_f_indices)) # Convert the set back to a sorted list (if needed)
            
            # # print(new_f_indices)
            
            # if len(new_f_indices) == 0:
            #     indices = np.arange(len(lc_sig))
            # else:
            #     indices = np.arange(len(lc_sig))[~new_f_indices]
            
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
        
        full_events, events_filtered = self._lightcurve_event_checker(new_stars, full_events, siglim = siglim)

        return full_events, events_filtered 
   
    def _check_lc_significance(self, start, end, x, y, flux_sign, buffer = 1, base_range=1.5, grad_val = -300):
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

    def _lightcurve_event_checker(self, stars, events, siglim = 2, spatial_tol = 1, frame_tol = 30):
        
        cluster_ids = np.unique(stars['cluster'])
        
        full_events = pd.DataFrame(columns=['cluster', 'frame_min', 'frame_max', 'time_min', 'time_max', 
                                            'x', 'y', 'xstd', 'ystd', 'sig_max', 'sig_med', 
                                            'roundness', 'fwhm', 'snr', 'psfdiff', 'correlation', 'poisson_thresh', 
                                            'e_roundness', 'e_fwhm', 'e_snr', 'e_psfdiff', 'e_correlation', 'e_poisson_thresh'])
        
        stars_new_df = pd.DataFrame(columns=stars.columns)
        
        def is_within_tolerance(cluster1, cluster2):
            # Check if the two clusters are within the specified distance tolerance
            x_dist = np.abs(cluster1['xcentroid'].mean() - cluster2['x'])
            y_dist = np.abs(cluster1['ycentroid'].mean() - cluster2['y'])
            
            frame1_min = int(cluster1['frame'].min())
            frame1_max = int(cluster1['frame'].max())
            frame2_min = int(cluster2['frame_min'])
            frame2_max = int(cluster2['frame_max'])
            
            if (frame1_min >= frame1_max) | (frame2_min >= frame2_max):
                return False
            
            frame1_array = np.arange(frame1_min, frame1_max + 1)
            frame2_array = np.arange(frame2_min, frame2_max + 1)
            
            frame_dist = np.nanmin(np.abs(np.subtract.outer(frame1_array, frame2_array)))
            return x_dist <= spatial_tol and y_dist <= spatial_tol and frame_dist <= frame_tol
        
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

            merge_cluster = None
            for i in range(len(full_events)): # Check if the cluster should be merged with an existing one
                if is_within_tolerance(cluster, full_events.iloc[i]):
                    merge_cluster = full_events.iloc[i].cluster
                    break
        
            if merge_cluster is not None: # If merging is necessary, update the cluster and full events
                existing_cluster_df = stars[stars['cluster'] == merge_cluster] # Merge the stars
                cluster = pd.concat([cluster, existing_cluster_df])
                
                frame_min = int(cluster['frame'].min()) # Get frame range for the cluster
                frame_max = int(cluster['frame'].max())
                
                x = cluster['xcentroid'].mean() # Recompute the statistics after merging
                y = cluster['ycentroid'].mean()
                xstd = cluster['xcentroid'].std()
                ystd = cluster['ycentroid'].std()
                
                time_min = self.time[frame_min]
                time_max = self.time[frame_max]
                
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

                
                cluster['cluster'] = len(full_events) + 1  # Assign new cluster ID
                stars_new_df = pd.concat([stars_new_df, cluster]) # Update the stars dataframe with the merged cluster
        
                full_events.loc[len(full_events)] = [len(full_events) + 1, frame_min, frame_max, time_min, time_max,
                                                    x, y, xstd, ystd, cluster_events['sig_max'].iloc[0], 
                                                    cluster_events['sig_med'].iloc[0], roundness, fwhm, snr, psfdiff,
                                                    correlation, poisson_thresh, 
                                                    e_roundness, e_fwhm, e_snr, e_psfdiff, e_correlation, e_poisson_thresh]
            else:
                cluster['cluster'] = len(full_events) + 1 # If no merging is needed, just add the new cluster
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # stars = stars.reset_index(drop=True)
        # filtered_stars = stars[stars['lc_sig'] >= siglim]
        
        # cluster_ids = np.unique(filtered_stars['cluster'])
        
        # full_events = pd.DataFrame(columns=['cluster', 'frame_min', 'frame_max', 'time_min', 'time_max', 
        #                                     'x', 'y', 'xstd', 'ystd', 'sig_max', 'sig_med', 
        #                                     'roundness', 'fwhm', 'snr', 'psfdiff', 'correlation', 'poisson_thresh'])
        
        # stars_new_df = pd.DataFrame(columns=stars.columns)
        
        # for cluster_id in cluster_ids:
        #     cluster = filtered_stars[filtered_stars['cluster'] == cluster_id]
            
        #     if len(cluster) < minimum_number:
        #         continue
            
        #     cluster_events = events[events['cluster'] == cluster_id]
        #     cluster_events = cluster_events.reset_index(drop=True)
            
        #     # frame_inds = cluster['frame'].values.astype(int)
            
        #     frame_min = cluster['frame'].min()
        #     frame_max = cluster['frame'].max()
            
        #     x = cluster['xcentroid'].mean()
        #     y = cluster['ycentroid'].mean()
            
        #     xstd = cluster['xcentroid'].std()
        #     ystd = cluster['ycentroid'].std()
            
        #     time_min = self.time[int(frame_min)]
        #     time_max = self.time[int(frame_max)]
            
        #     roundness = cluster['roundness'].mean()
        #     fwhm = cluster['fwhm'].mean()
        #     snr = cluster['snr'].mean()
        #     psfdiff = cluster['psfdiff'].mean()
        #     correlation = cluster['correlation'].mean()
        #     poisson_thresh = cluster['poisson_thresh'].mean()
            
        #     if cluster_events['sig_med'].iloc[0] >= siglim:
                
        #         cluster = cluster.reset_index(drop=True)
        #         cluster['cluster'] = len(full_events) + 1
                
        #         stars_new_df = pd.concat([stars_new_df, cluster])
            
        #         full_events.loc[len(full_events)] = [len(full_events) + 1, frame_min, frame_max, time_min, time_max, 
        #                                             x, y, xstd, ystd, cluster_events['sig_max'].iloc[0], 
        #                                             cluster_events['sig_med'].iloc[0], roundness, fwhm, snr, psfdiff, 
        #                                             correlation, poisson_thresh]
                
        #         # Add min roundness, max fwhm, max snr, max psfdiff, 
        #         # max correlation, max poisson_thresh, max significance, star or galaxy
        
        # if len(full_events) == 0:
        #     return None, None
        # else:
        #     stars_new_df = stars_new_df.reset_index(drop=True)
        #     return full_events, stars_new_df
        