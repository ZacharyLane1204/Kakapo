from Kakapo.photometry import forced_photometry, forced_photometry_psf
# from Kakapo.difference_image im
from Kakapo.cleaning_curve import check_periodicity, gauss_smooth

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.stats import sigma_clipped_stats
from astropy.stats import bayesian_blocks
from astropy.stats import sigma_clip

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.signal import butter, filtfilt

from astropy.time import Time

from tqdm import tqdm
import os
from copy import deepcopy

# ---------- helpers new/updated ----------

def rolling_median_mad(x, w):
    s = pd.Series(x)
    med = s.rolling(w, min_periods=max(1, w//2)).median().to_numpy()
    mad = (s - med).abs().rolling(w, min_periods=max(1, w//2)).median().to_numpy()
    mad = 1.4826 * mad
    return med, mad

def make_err_model_from_local(raw_flux, smooth_flux, flux_err, baseline_mask, w_mad=21):
    """
    Build conservative per-sample error estimate without running GP.
    Uses local rolling MAD from raw_flux when available.
    """
    if raw_flux is not None:
        _, mad_local = rolling_median_mad(raw_flux, w=w_mad)
    else:
        _, mad_local = rolling_median_mad(smooth_flux, w=w_mad)
    
    global_mad = np.nanmedian(np.abs((raw_flux if raw_flux is not None else smooth_flux) - 
                                     np.nanmedian(raw_flux if raw_flux is not None else smooth_flux))) # fallback
    mad_local = np.where((~np.isfinite(mad_local)) | (mad_local == 0), global_mad, mad_local)
    err = np.empty_like(smooth_flux, dtype=float)
    # baseline: prefer empirical mad but don't go below measurement error if available
    err[baseline_mask] = np.maximum(mad_local[baseline_mask], flux_err[baseline_mask] if flux_err is not None else mad_local[baseline_mask])
    tmask = ~baseline_mask # transient: keep measured error or local mad
    if flux_err is not None:
        err[tmask] = np.maximum(flux_err[tmask], mad_local[tmask])
    else:
        err[tmask] = mad_local[tmask] 
    err[~np.isfinite(err)] = float(global_mad if np.isfinite(global_mad) and global_mad > 0 else 1.0) # fix NaNs
    return err

def adapt_z_enter_from_local(flux, center_idx, half_window=100, base_z=2.5, clamp=(0.6, 3.0)):
    N = flux.size
    left_lo = max(0, center_idx - half_window*2)
    left_hi = max(0, center_idx - max(4, half_window//4))
    right_lo = min(N, center_idx + max(4, half_window//4))
    right_hi = min(N, center_idx + half_window*2)
    cand = []
    for lo, hi in ((left_lo, left_hi), (right_lo, right_hi)):
        seg = flux[lo:hi]
        if seg.size > 8 and np.any(np.isfinite(seg)):
            m = np.nanmedian(seg)
            mad = 1.4826 * np.nanmedian(np.abs(seg - m))
            if np.isfinite(mad) and mad > 0:
                cand.append(mad)
    if len(cand) == 0:
        return float(base_z)
    local_mad = float(np.median(cand))
    global_mad = float(np.nanmedian(np.abs(flux - np.nanmedian(flux))))
    if global_mad <= 0 or not np.isfinite(global_mad):
        return float(base_z)
    ratio = local_mad / global_mad
    ratio = float(np.clip(ratio, clamp[0], clamp[1]))
    return float(base_z * ratio)

def iterative_baseline_zscore_fast(time, 
                                  flux_smooth,       # smoothed LC (e.g. your gauss_smooth mean)
                                  raw_flux=None,     # optional: original unsmoothed LC (recommended)
                                  flux_err=None,     # optional: measurement errors (from forced photometry)
                                  frame_min=None, frame_max=None,
                                  base_w=21,
                                  z_enter_base=2.5, z_exit=1.0, persist=2,
                                  want_baseline_pts=60,
                                  pad_frac=0.5,
                                  max_iter=3,
                                  rolling_w_for_adapt=200):
    """
    Iterative (fast) baseline finder that:
      - works on a pre-smoothed LC (no iterative GP)
      - masks the transient, estimates baseline med/MAD, recomputes Z,
      - adaptively scales z_enter from local rolling MAD
    Returns:
       istart, iend, z_array, baseline_med, baseline_mad, baseline_mask, err_model
    """
    N = flux_smooth.size
    finite = np.isfinite(flux_smooth)
    if not np.any(finite):
        return None, None, None, None, None, None, None

    try: # initial stable baseline using your stable_mask (or simple rolling mad fallback)
        stab = stable_mask(flux_smooth, w=base_w, z_tol=1.5)
    except Exception:
        stab = finite.copy() # fallback: any finite points

    if np.sum(stab & finite) < want_baseline_pts:
        if frame_min is not None: # fallback to a causal mask or all finite
            stab = causal_baseline_mask(flux_smooth, end_idx=max(0, frame_min), min_pts=want_baseline_pts, max_lookback=2000)
        else:
            stab = finite.copy()

    med = float(np.nanmedian(flux_smooth[stab & finite])) # initial med/mad on smoothed baseline candidate
    mad = 1.4826 * float(np.nanmedian(np.abs(flux_smooth[stab & finite] - med)))
    if not np.isfinite(mad) or mad == 0:
        mad = float(np.nanstd(flux_smooth[stab & finite])) if np.any(np.isfinite(flux_smooth[stab & finite])) else 1.0

    z = (flux_smooth - med) / mad

    if frame_min is not None and frame_max is not None: # initial detection window: use frame_min/frame_max as seed if provided
        win_lo = max(0, int(frame_min) - 1) # restrict to region around cluster first
        win_hi = min(N, int(frame_max) + 2)
        center_idx = int((frame_min + frame_max) // 2) # compute local adaptive z_enter using center of cluster
        z_enter = adapt_z_enter_from_local(flux_smooth, center_idx, half_window=rolling_w_for_adapt//2, base_z=z_enter_base)
        istart, iend = find_event_window_from_z(z, z_enter=z_enter, z_exit=z_exit, persist=persist)
        if istart is None: # fallback: if no event found from z, use the provided frame_min/frame_max region
            istart, iend = max(0, int(frame_min)), min(N-1, int(frame_max))
    else:
        center_idx = N//2 # try global
        z_enter = adapt_z_enter_from_local(flux_smooth, center_idx, half_window=rolling_w_for_adapt//2, base_z=z_enter_base)
        istart, iend = find_event_window_from_z(z, z_enter=z_enter, z_exit=z_exit, persist=persist)
        if istart is None:
            return None, None, z, med, mad, stab, make_err_model_from_local(raw_flux, flux_smooth, flux_err, stab, w_mad=base_w)

    prev_istart, prev_iend = istart, iend

    for it in range(max_iter):
        evt_len = max(1, prev_iend - prev_istart + 1)
        
        pad = int(max(20, min(int(0.1 * N), pad_frac * evt_len)))
        mask_transient = np.zeros(N, bool)
        mask_transient[max(0, prev_istart - pad): min(N, prev_iend + pad + 1)] = True
        baseline_mask = (~mask_transient) & finite

        if np.sum(baseline_mask) < want_baseline_pts:
            baseline_mask = stab & finite # fall back to stab / causal growth
            if np.sum(baseline_mask) < want_baseline_pts and (frame_min is not None):
                baseline_mask = causal_baseline_mask(flux_smooth, end_idx=max(0, frame_min), min_pts=want_baseline_pts, max_lookback=2000)

        
        if np.any(baseline_mask): # recompute med/mad on baseline (smoothed flux)
            baseline_med = float(np.nanmedian(flux_smooth[baseline_mask]))
            baseline_mad = 1.4826 * float(np.nanmedian(np.abs(flux_smooth[baseline_mask] - baseline_med)))
            if not np.isfinite(baseline_mad) or baseline_mad == 0:
                baseline_mad = float(np.nanstd(flux_smooth[baseline_mask])) if np.any(np.isfinite(flux_smooth[baseline_mask])) else mad
        else:
            baseline_med, baseline_mad = med, mad

        z = (flux_smooth - baseline_med) / baseline_mad

        center_idx = int(max(0, min(N - 1, (prev_istart + prev_iend) // 2))) # adapt z_enter from local baseline noise around center of previous event
        z_enter = adapt_z_enter_from_local(flux_smooth, center_idx, half_window=rolling_w_for_adapt//2, base_z=z_enter_base)

        new_istart, new_iend = find_event_window_from_z(z, z_enter=z_enter, z_exit=z_exit, persist=persist)
        if new_istart is None: # if nothing new, keep previous
            break

        if frame_min is not None and frame_max is not None:
            if not (new_iend < frame_min or new_istart > frame_max): # require overlap with seed window
                prev_istart, prev_iend = new_istart, new_iend
            else:
                prev_istart, prev_iend = frame_min, frame_max # reject: stick with original seed window
                break
        else:
            if (new_istart == prev_istart) and (new_iend == prev_iend):
                prev_istart, prev_iend = new_istart, new_iend
                break
            prev_istart, prev_iend = new_istart, new_iend

    evt_len = max(1, prev_iend - prev_istart + 1) # final baseline mask and error model
    pad = int(max(10, min(500, pad_frac * evt_len)))
    mask_transient = np.zeros(N, bool)
    mask_transient[max(0, prev_istart - pad): min(N, prev_iend + pad + 1)] = True
    final_baseline_mask = (~mask_transient) & finite
    if np.nansum(final_baseline_mask) < want_baseline_pts: # fallback safeguard
        final_baseline_mask = stab & finite

    final_med = float(np.nanmedian(flux_smooth[final_baseline_mask]))
    final_mad = 1.4826 * float(np.nanmedian(np.abs(flux_smooth[final_baseline_mask] - final_med)))
    if not np.isfinite(final_mad) or final_mad == 0:
        final_mad = float(np.nanstd(flux_smooth[final_baseline_mask])) if np.any(np.isfinite(flux_smooth[final_baseline_mask])) else 1.0

    err_model = make_err_model_from_local(raw_flux, flux_smooth, flux_err, final_baseline_mask, w_mad=base_w)

    return int(prev_istart), int(prev_iend), z, final_med, final_mad, final_baseline_mask, err_model

def smoothness_metric(flux, w=21):
    b, a = butter(2, 1.0/w, btype='low')
    low = filtfilt(b, a, flux)
    high = flux - low
    return np.nanvar(high) / np.nanvar(low)

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

def stable_mask(x, w=21, z_tol=1.5):
    """
    Flag "stable" samples where rolling z is small (quasi-flat).
    Used to select baseline regions.
    """
    med, mad = rolling_median_mad(x, w)
    mad = np.where((~np.isfinite(mad)) | (mad == 0), np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826, mad)
    z = (x - med) / mad
    return np.isfinite(z) & (np.abs(z) < z_tol)

def choose_lower_baseline(f, left_mask, right_mask):
    """
    Build candidate baselines on both sides; prefer the side with lower median
    (more conservative against over-estimating signal).
    """
    medL = np.nanmedian(f[left_mask]) if np.any(left_mask) else np.nan
    medR = np.nanmedian(f[right_mask]) if np.any(right_mask) else np.nan

    # pick the lower finite median
    if np.isfinite(medL) and np.isfinite(medR):
        return (left_mask if medL <= medR else right_mask)
    elif np.isfinite(medL):
        return left_mask
    elif np.isfinite(medR):
        return right_mask
    else:
        return np.zeros_like(f, dtype=bool)  # no baseline found

def find_event_window_from_z(z, z_enter=2.5, z_exit=1.0, persist=2, exit_frac=0.5):
    """
    Hysteretic onset/end from Z with persistence.
    - enter when z > z_enter for persist samples
    - exit when at least `exit_frac` of last `persist` samples are < z_exit
    """
    N = z.size
    above = z > z_enter

    # persistence enter
    run = 0
    istart = None
    for i in range(N):
        run = run + 1 if above[i] else 0
        if run >= persist:
            istart = i - persist + 1
            break
    if istart is None:
        return None, None

    # exit condition with fraction-based rule
    iend = N - 1
    for j in range(istart, N):
        window = z[j:j+persist]
        if window.size == persist and np.nanmean(window < z_exit) >= exit_frac:
            iend = j + persist - 1
            break

    return istart, iend

class Implement_reductions:
    def __init__(self, stars, tpf_info, diff, epsf_data, 
                 noise, corrlim=0, difflim=100, 
                 fwhmlim=5, maxlim=10, snrlim=2, roundness=0.8, 
                 poiss_val=1, siglim=1, dist_cut=0.6, ratio_cut = 2,
                 f_dist=50):
        
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
        self.siglim = siglim
        self.ratio_cut = ratio_cut
        
        self.det_min_pts = 3   # Stage-1 (early) clustering len threshold
        self.val_min_pts = 5   # Stage-2 (validation) threshold
        
        self.time = tpf_info.time.value
        self.campaign = tpf_info.campaign
        self.targetid = tpf_info.targetid
        self.cadence = self.time[1] - self.time[0]
        self.noise = noise

        corr = self.filter_detections(self.stars, initial=True).reset_index(drop=True)

        groups = self._grouping(corr, f_dist=f_dist)

        if groups is not None and len(groups) > 0:
            processor = EventProcessor(time_overlap_thresh=0.6)

            filtered_events, surviving_groups = processor.filter_detections_by_clusters(groups)
            if surviving_groups is not None:
                args = self.detected_events(groups, siglim=siglim)
                full_events, filtered_events, lc_mjds_arr, lc_flux_arr, lc_flux_err_arr = args
            
                if full_events is not None:
                    self.filtered_stars = filtered_events
                    self.full_events = full_events
                    self.lc_mjds_arr = lc_mjds_arr
                    self.lc_flux_arr = lc_flux_arr
                    self.lc_flux_err_arr = lc_flux_err_arr
                else:
                    self._build_Nones()
            else:
                self._build_Nones()
        else:
            self._build_Nones()
            
    def _build_Nones(self):
        self.filtered_stars = None
        self.full_events = None
        self.lc_mjds_arr = None
        self.lc_flux_arr = None
        self.lc_flux_err_arr = None
            
    def weighted_std(self, values, weights):
        """
        Weighted standard deviation
        """
        avg = np.sum(values * weights) / np.sum(weights)
        variance = np.sum(weights * (values - avg)**2) / np.sum(weights)
        return np.sqrt(variance)

    def compute_weighted_position_stats(self, corr):
        """
        For each cluster, compute weighted x, y, xstd, ystd
        """
        corr = corr.copy()
        cluster_ids = corr['cluster'].unique()
        corr['x_weighted'] = np.nan
        corr['y_weighted'] = np.nan
        corr['xstd_weighted'] = np.nan
        corr['ystd_weighted'] = np.nan

        for cid in cluster_ids:
            cluster = corr[corr['cluster'] == cid]
            weights = cluster['snr'].values
            xw = np.nansum(cluster['xcentroid'] * weights) / np.nansum(weights)
            yw = np.nansum(cluster['ycentroid'] * weights) / np.nansum(weights)
            xstd_w = self.weighted_std(cluster['xcentroid'].values, weights)
            ystd_w = self.weighted_std(cluster['ycentroid'].values, weights)

            corr.loc[cluster.index, 'x_weighted'] = xw
            corr.loc[cluster.index, 'y_weighted'] = yw
            corr.loc[cluster.index, 'xstd_weighted'] = xstd_w
            corr.loc[cluster.index, 'ystd_weighted'] = ystd_w

        return corr
    
    def filter_by_weighted_sig(self, full_events):
        if full_events is None or len(full_events) == 0:
            return None

        mask_keep = full_events['sig_max_weighted'] > self.siglim # Keep only events above threshold
        filtered = full_events[mask_keep].copy()

        if len(filtered) == 0:
            return None

        # Re-index clusters sequentially
        filtered = filtered.reset_index(drop=True)
        filtered['cluster'] = np.arange(1, len(filtered) + 1)
        
        return filtered
        
    def filter_detections(self, stars, initial=False):
        if initial:
            corr = stars[(stars.correlation >= 0.01) & 
                         (stars.psfdiff <= 2) & 
                         (stars.fwhm <= 8) &  (stars.fwhm >= 0.8) & 
                         (stars.snr >= 2) & (stars.snr < 10000) & 
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
                        roundness, poisson_thresh, xstd, ystd, lc_snr):
        
        return ((correlation >= self.corrlim) & (psfdiff <= self.difflim) &
                (fwhm <= self.fwhmlim) & (fwhm >= 0.8) &
                (abs(snr) >= self.snrlim) & (abs(snr) < 10000) & 
                (abs(roundness) <= self.roundness) &
                (poisson_thresh >= self.poiss_val) &
                (xstd <= self.dist_cut) & (ystd <= self.dist_cut) & 
                (abs(lc_snr) >= self.snrlim))
        
    def _estimate_frame_scale(self, corr, r_xy=0.75, q=0.8, min_pairs=50, default_scale=50):
        """
        Estimate a typical 'association' time scale (in frames) from local (x,y) neighbors.
        Looks at |Δframe| among points within r_xy pixels and returns the q-quantile.
        Falls back to default_scale if not enough pairs.
        """
        if len(corr) < 2:
            return default_scale

        pts = corr[['xcentroid','ycentroid']].to_numpy(np.float32)
        frames = corr['frame'].to_numpy(np.int64)
        tree = cKDTree(pts)
        idx_lists = tree.query_ball_point(pts, r=r_xy)

        dts = []
        for i, neigh in enumerate(idx_lists):
            fi = frames[i]
            for j in neigh:
                if j == i: 
                    continue
                dts.append(abs(fi - frames[j]))

        if len(dts) >= min_pairs:
            return max(5, np.nanpercentile(dts, q*100.0))
        return default_scale
    
    def _grouping(self, corr: pd.DataFrame, f_dist: int = 50, micro_pixel_tol: float = 0.3) -> pd.DataFrame | None:
        """
        Stage A: reproduce your 3D DBSCAN on [x,y,frame] to get 'big' clusters.
        Stage B: inside each big cluster, find micro-splinters (min_samples=1) in (x,y),
                pick the best splinter by weight = sqrt(N)*sqrt(sum(snr)),
                and attach sigma-clipped, SNR-weighted position stats.
        Returns a 'refined' corr containing ONLY the winning splinter rows per big cluster,
        with new per-row columns:
        x_weight, y_weight, x_weight_std, y_weight_std, int_SNR, frame_min, frame_max
        and a re-numbered 'cluster' (1..K) for downstream use.
        """
        if corr.empty:
            return None

        # ---------- Stage A: your spatiotemporal DBSCAN (with adaptive frame scaling) ----------
        f_scale = self._estimate_frame_scale(corr, r_xy=0.75, q=0.8, min_pairs=50, default_scale=f_dist)
        
        f_scale = max(f_scale, f_dist)

        data = corr[['xcentroid', 'ycentroid', 'frame']].to_numpy(np.float32)
        data[:, 2] *= 1.5 / float(f_scale)

        db = DBSCAN(
            eps=1.5,
            min_samples=self.det_min_pts,     # your original seed requirement (e.g., 3)
            metric='euclidean',
            algorithm='auto',
            n_jobs=1
        )
        labels = db.fit_predict(data)
        corrA = corr.assign(cluster_big=labels)
        corrA = corrA[corrA.cluster_big != -1]
        if corrA.empty:
            return None

        # ---------- Stage B: refine each big cluster by spatial micro-splinters ----------
        refined_rows = []
        new_cluster_id = 1

        for big_id, sub in corrA.groupby('cluster_big'):
            xy = sub[['xcentroid', 'ycentroid']].to_numpy(np.float64) # Tight spatial-only DBSCAN inside this big cluster to get splinters
            micro_labels = DBSCAN(
                eps=micro_pixel_tol,
                min_samples=1,                # catch even singletons
                metric='euclidean',
                algorithm='auto',
                n_jobs=1
            ).fit_predict(xy)
            sub = sub.assign(micro=micro_labels)

            best = None # Evaluate each micro-splinter; choose the best by weight = sqrt(N)*sqrt(sum(snr))
            best_payload = None

            for mid, g in sub.groupby('micro'):
                N = len(g)
                snr = g['snr'].to_numpy(float)
                snr_sum = float(np.nansum(snr))
                weight_for_selection = np.sqrt(max(N, 1)) * np.sqrt(max(snr_sum, 1e-12))

                x_raw = g['xcentroid'].to_numpy(float) # Sigma-clip x/y, then compute SNR-weighted mean and std within this micro
                y_raw = g['ycentroid'].to_numpy(float)
                x_cl = sigma_clip(x_raw, sigma=3, maxiters=5)
                y_cl = sigma_clip(y_raw, sigma=3, maxiters=5)
                m = (~x_cl.mask) & (~y_cl.mask)
                if not np.any(m):
                    xs, ys, ws = x_raw, y_raw, np.clip(snr, 1e-6, None)
                else:
                    xs, ys, ws = x_cl.data[m], y_cl.data[m], np.clip(snr[m], 1e-6, None)

                x_w = float(np.average(xs, weights=ws))
                y_w = float(np.average(ys, weights=ws))
                x_var = float(np.average((xs - x_w)**2, weights=ws))
                y_var = float(np.average((ys - y_w)**2, weights=ws))
                x_std = np.sqrt(max(x_var, 0.0))
                y_std = np.sqrt(max(y_var, 0.0))

                payload = {
                    'x_weight': x_w,
                    'y_weight': y_w,
                    'x_weight_std': x_std,
                    'y_weight_std': y_std,
                    'int_SNR': snr_sum,
                    'frame_min': int(g['frame'].min()),
                    'frame_max': int(g['frame'].max()),
                    'rows': g.index.to_numpy()
                }

                if (best is None) or (weight_for_selection > best):
                    best = weight_for_selection
                    best_payload = payload

            if best_payload is None:
                continue

            # Keep only the winning micro rows for this big cluster and attach the stats
            gbest = corr.loc[best_payload['rows']].copy()
            gbest['cluster'] = new_cluster_id  # new compact cluster id expected by your downstream code
            gbest['x_weight'] = best_payload['x_weight']
            gbest['y_weight'] = best_payload['y_weight']
            gbest['x_weight_std'] = best_payload['x_weight_std']
            gbest['y_weight_std'] = best_payload['y_weight_std']
            gbest['int_SNR'] = best_payload['int_SNR']
            gbest['frame_min'] = best_payload['frame_min']
            gbest['frame_max'] = best_payload['frame_max']

            refined_rows.append(gbest)
            new_cluster_id += 1

        if not refined_rows:
            return None
        refined = pd.concat(refined_rows, ignore_index=True)

        # Keep only the columns you need; but having the weights on every row is handy downstream
        return refined

    def detected_events(self, events, siglim=2.0):
        
        mjds_list = []
        lc_fluxes_list = []
        lc_fluxes_err_list = []
        
        cluster_ids = np.unique(events['cluster'])

        full_events = pd.DataFrame(columns=['cluster', 'frame_min', 'frame_max',
                                            'true_start', 'true_end',
                                            'remapped_frame_min', 'remapped_frame_max', 
                                            'x', 'y', 'xstd', 'ystd', 'sig_max', 'sig_95', 
                                            'roundness', 'fwhm', 'snr', 'psfdiff', 
                                            'correlation', 'poisson_thresh', 'smoothness_ratio', 
                                            'period', 'period_confidence', 
                                            'lc_snr_max', 'lc_snr_95',
                                            'e_roundness', 'e_fwhm', 'e_snr', 'e_psfdiff', 
                                            'e_correlation', 'e_poisson_thresh'])

        new_stars = pd.DataFrame(columns=events.columns)
        events = events.copy()
        events['lc_sig'] = -1.0

        for cid in cluster_ids:
            cluster = events[events['cluster'] == cid]
            
            if len(cluster) < self.det_min_pts:
                continue
            
            x = float(cluster['x_weight'].iloc[0])
            y = float(cluster['y_weight'].iloc[0])
            xstd = float(cluster['x_weight_std'].iloc[0])
            ystd = float(cluster['y_weight_std'].iloc[0])
            
            if (xstd >= self.dist_cut) | (ystd >= self.dist_cut):
                continue

            frame_min = int(cluster['frame_min'].iloc[0])
            frame_max = int(cluster['frame_max'].iloc[0])

            # --- build LC at the centroid for this cluster ---
            bjds = deepcopy(self.time)
            lc_raw   = forced_photometry(self.diff,  x, y, self.epsf, bkg = False)
            flux_err = forced_photometry(self.noise, x, y, self.epsf, bkg = False)

            mfin = np.isfinite(lc_raw) & np.isfinite(bjds) & np.isfinite(flux_err)
            
            bjds, lc_raw, flux_err = bjds[mfin], lc_raw[mfin], flux_err[mfin]
            
            og_ind = np.where(mfin)[0]  # original frame indices
            rmap_ind = np.arange(lc_raw.size)
            
            fr_min = rmap_ind[og_ind == frame_min][0] if np.any(og_ind == frame_min) else 0
            fr_max = rmap_ind[og_ind == frame_max][0] if np.any(og_ind == frame_max) else (len(rmap_ind)-1)
            
            _, best_period, confidence = check_periodicity(bjds, flux=lc_raw, flux_err=flux_err, fap_level=0.075)
            
            if confidence < 0.75:
                best_period = 0                

            # lc_sm, lc_sm_err = gauss_smooth(bjds, lc_raw, flux_err) # smooth for significance
            lc_sm = lc_raw.copy()
            lc_sm_err = flux_err.copy()
            
            lc_sm_err = np.maximum(np.sqrt(lc_sm_err**2 + flux_err**2), 1e-6)
            
            lc_snr = lc_sm/lc_sm_err

            # 2) find iterative baseline & z on smoothed LC (fast mode, no GP refit)
            args = iterative_baseline_zscore_fast(time=bjds, flux_smooth=lc_sm, raw_flux=lc_raw, 
                                                  flux_err= lc_sm_err,
                                                  frame_min=fr_min, frame_max=fr_max, 
                                                  base_w=21, z_enter_base=siglim,   # your z_enter default
                                                  z_exit=1.0, persist=2, want_baseline_pts=60, 
                                                  pad_frac=0.5, max_iter=3, rolling_w_for_adapt=200)
            
            istart, iend, lc_sig, med, mad, baseline_mask, err_model = args

            if istart is None:
                continue

            residual = lc_sm - med
            win_lo = max(0, istart)
            win_hi = min(len(lc_sig), iend + 1)
            zevt = np.abs(lc_sig[win_lo:win_hi]) if win_hi > win_lo else np.array([])

            sig_max = float(np.nanmax(zevt)) if zevt.size else -1.0
            sig_95  = float(np.nanpercentile(zevt, 95)) if zevt.size else -1.0

            if istart is not None and iend is not None and (iend - istart) >= 12:
                sm_win = slice(istart + 2, iend + 1)
                ratio = smoothness_metric(lc_sm[sm_win], w=21)
            elif istart is not None and iend is not None and (iend - istart) >= self.val_min_pts:
                ratio = 1
            else:
                ratio = np.nan

            if sig_95 < siglim:
                continue
            
            if ~np.isfinite(ratio):
                continue
            
            if (ratio >= self.ratio_cut):
                continue
            
            filtered_cluster = cluster[cluster['frame'].isin(og_ind)].reset_index(drop=True)
            if len(filtered_cluster) < self.det_min_pts:
                continue
            
            lc_snr_event = lc_snr[win_lo:win_hi]
            lc_snr_max = np.nanmax(lc_snr_event)
            lc_snr_95 = np.nanpercentile(lc_snr_event, 95)

            map_inds = np.array([np.where(og_ind == int(fr))[0][0] for fr in filtered_cluster['frame'].values])
            filtered_cluster = filtered_cluster.assign(lc_sig=lc_sig[map_inds])
            filtered_cluster['cluster'] = len(full_events) + 1
            new_stars = pd.concat([new_stars, filtered_cluster], ignore_index=True)

            roundness, _, e_roundness = sigma_clipped_stats(filtered_cluster['roundness'].values, sigma=3)
            fwhm, _, e_fwhm = sigma_clipped_stats(filtered_cluster['fwhm'].values, sigma=3)
            snr, _, e_snr = sigma_clipped_stats(filtered_cluster['snr'].values, sigma=3)
            psfdiff, _, e_psfdiff = sigma_clipped_stats(filtered_cluster['psfdiff'].values, sigma=3)
            correlation, _, e_correlation = sigma_clipped_stats(filtered_cluster['correlation'].values, sigma=3)
            poisson_thresh, _, e_poisson_thresh = sigma_clipped_stats(filtered_cluster['poisson_thresh'].values, sigma=3)
            
            masking = self.mask_detections(correlation, psfdiff, fwhm, snr, roundness, poisson_thresh, xstd, ystd, lc_snr_95)
            if masking is False:
                continue

            bjds_utc = bjds + 54832.5
            t_bary = Time(bjds_utc, format="mjd", scale="tdb")
            t_mjd_utc = t_bary.utc.mjd
            
            rmap_lo = rmap_ind[og_ind == istart][0] if np.any(og_ind == istart) else 0
            rmap_hi = rmap_ind[og_ind == iend][0] if np.any(og_ind == iend) else (len(rmap_ind)-1)
            
            mjds_list.append([t_mjd_utc])
            lc_fluxes_list.append([lc_sm])
            lc_fluxes_err_list.append([lc_sm_err])
            
            # store event
            full_events.loc[len(full_events)] = [
                len(full_events) + 1, # cluster
                istart, iend, # frame_min, frame_max
                t_mjd_utc[istart], # true_start
                t_mjd_utc[iend], # true_end
                rmap_lo, rmap_hi, # remapped_frame_min, remapped_frame_max
                x, y, xstd, ystd, # x, y, xstd, ystd
                sig_max, sig_95, # sig_max, sig_95
                roundness, fwhm, snr, psfdiff, # roundness, fwhm, snr, psfdiff
                correlation, poisson_thresh, ratio, # correlation, poisson_thresh, smoothness_ratio
                best_period, confidence, # period, period_confidence
                lc_snr_max, lc_snr_95,
                e_roundness, e_fwhm, e_snr, e_psfdiff, # e_roundness, e_fwhm, e_snr, e_psfdiff
                e_correlation, e_poisson_thresh # e_correlation, e_poisson_thresh
            ]

        if len(full_events) == 0:
            return None, None, None, None, None
        else:
            new_stars = new_stars.reset_index(drop=True)
            return full_events, new_stars, np.array(mjds_list), np.array(lc_fluxes_list), np.array(lc_fluxes_err_list)

class EventProcessor:
    def __init__(self, time_overlap_thresh=0.5):
        self.time_overlap_thresh = time_overlap_thresh

    def build_cluster_events(self, groups: pd.DataFrame) -> pd.DataFrame:
        """
        Build per-cluster summary events table.
        """
        events = []
        for cid, dfc in groups.groupby("cluster"):
            frame_min, frame_max = dfc["frame"].min(), dfc["frame"].max()
            int_SNR = dfc["int_SNR"].values[0]  # or weighted sum if you prefer
            events.append({
                "cluster": cid,
                "frame_min": frame_min,
                "frame_max": frame_max,
                "int_SNR": int_SNR,
                "n_points": len(dfc),
            })
        return pd.DataFrame(events)

    def remove_overlapping_clusters(self, events: pd.DataFrame) -> pd.DataFrame | None:
        """
        Removes overlapping cluster events, keeping the stronger one (by int_SNR).
        """
        if events is None or len(events) < 2:
            return events

        starts = events['frame_min'].values
        ends   = events['frame_max'].values
        snrs   = events['int_SNR'].values
        n_events = len(events)

        to_remove = np.zeros(n_events, dtype=bool)

        for i in range(n_events):
            for j in range(i+1, n_events):
                lo = max(starts[i], starts[j])
                hi = min(ends[i], ends[j])
                overlap = max(0, hi - lo + 1)
                if overlap > 0:
                    frac_i = overlap / (ends[i] - starts[i] + 1)
                    frac_j = overlap / (ends[j] - starts[j] + 1)
                    if frac_i >= self.time_overlap_thresh or frac_j >= self.time_overlap_thresh:
                        # Keep the stronger event, drop the weaker one
                        if snrs[i] >= snrs[j]:
                            to_remove[j] = True
                        else:
                            to_remove[i] = True

        filtered = events[~to_remove].reset_index(drop=True)
        return filtered if not filtered.empty else None

    def filter_detections_by_clusters(self, groups: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        High-level: build events -> filter overlaps -> return surviving events and detections.
        """
        events = self.build_cluster_events(groups)
        filtered_events = self.remove_overlapping_clusters(events)

        if filtered_events is not None:
            survivor_clusters = set(filtered_events["cluster"])
            groups = groups[groups["cluster"].isin(survivor_clusters)]
            return filtered_events, groups
        else:
            return None, None
