from Kakapo.photometry import forced_photometry, forced_photometry_psf
# from Kakapo.difference_image im
from Kakapo.cleaning_curve import correction_smoothing_lightcurve, wavelet_denoise, gauss_smooth, binned_averages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.stats import sigma_clipped_stats
from astropy.stats import bayesian_blocks
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
    # fallback
    global_mad = np.nanmedian(np.abs((raw_flux if raw_flux is not None else smooth_flux) - 
                                     np.nanmedian(raw_flux if raw_flux is not None else smooth_flux)))
    mad_local = np.where((~np.isfinite(mad_local)) | (mad_local == 0), global_mad, mad_local)
    err = np.empty_like(smooth_flux, dtype=float)
    # baseline: prefer empirical mad but don't go below measurement error if available
    err[baseline_mask] = np.maximum(mad_local[baseline_mask], flux_err[baseline_mask] if flux_err is not None else mad_local[baseline_mask])
    # transient: keep measured error or local mad
    tmask = ~baseline_mask
    if flux_err is not None:
        err[tmask] = np.maximum(flux_err[tmask], mad_local[tmask])
    else:
        err[tmask] = mad_local[tmask]
    # fix NaNs
    err[~np.isfinite(err)] = float(global_mad if np.isfinite(global_mad) and global_mad > 0 else 1.0)
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

    # initial stable baseline using your stable_mask (or simple rolling mad fallback)
    try:
        stab = stable_mask(flux_smooth, w=base_w, z_tol=1.5)
    except Exception:
        # fallback: any finite points
        stab = finite.copy()

    if np.sum(stab & finite) < want_baseline_pts:
        # fallback to a causal mask or all finite
        if frame_min is not None:
            stab = causal_baseline_mask(flux_smooth, end_idx=max(0, frame_min), min_pts=want_baseline_pts, max_lookback=2000)
        else:
            stab = finite.copy()

    # initial med/mad on smoothed baseline candidate
    med = float(np.nanmedian(flux_smooth[stab & finite]))
    mad = 1.4826 * float(np.nanmedian(np.abs(flux_smooth[stab & finite] - med)))
    if not np.isfinite(mad) or mad == 0:
        mad = float(np.nanstd(flux_smooth[stab & finite])) if np.any(np.isfinite(flux_smooth[stab & finite])) else 1.0

    z = (flux_smooth - med) / mad

    # initial detection window: use frame_min/frame_max as seed if provided
    if frame_min is not None and frame_max is not None:
        # restrict to region around cluster first
        win_lo = max(0, int(frame_min) - 1)
        win_hi = min(N, int(frame_max) + 2)
        # compute local adaptive z_enter using center of cluster
        center_idx = int((frame_min + frame_max) // 2)
        z_enter = adapt_z_enter_from_local(flux_smooth, center_idx, half_window=rolling_w_for_adapt//2, base_z=z_enter_base)
        istart, iend = find_event_window_from_z(z, z_enter=z_enter, z_exit=z_exit, persist=persist)
        # fallback: if no event found from z, use the provided frame_min/frame_max region
        if istart is None:
            istart, iend = max(0, int(frame_min)), min(N-1, int(frame_max))
    else:
        # try global
        center_idx = N//2
        z_enter = adapt_z_enter_from_local(flux_smooth, center_idx, half_window=rolling_w_for_adapt//2, base_z=z_enter_base)
        istart, iend = find_event_window_from_z(z, z_enter=z_enter, z_exit=z_exit, persist=persist)
        if istart is None:
            return None, None, z, med, mad, stab, make_err_model_from_local(raw_flux, flux_smooth, flux_err, stab, w_mad=base_w)

    prev_istart, prev_iend = istart, iend

    for it in range(max_iter):
        evt_len = max(1, prev_iend - prev_istart + 1)
        pad = int(max(10, min(500, pad_frac * evt_len)))
        mask_transient = np.zeros(N, bool)
        mask_transient[max(0, prev_istart - pad): min(N, prev_iend + pad + 1)] = True
        baseline_mask = (~mask_transient) & finite

        if np.sum(baseline_mask) < want_baseline_pts:
            # fall back to stab / causal growth
            baseline_mask = stab & finite
            if np.sum(baseline_mask) < want_baseline_pts and (frame_min is not None):
                baseline_mask = causal_baseline_mask(flux_smooth, end_idx=max(0, frame_min), min_pts=want_baseline_pts, max_lookback=2000)

        # recompute med/mad on baseline (smoothed flux)
        if np.any(baseline_mask):
            baseline_med = float(np.nanmedian(flux_smooth[baseline_mask]))
            baseline_mad = 1.4826 * float(np.nanmedian(np.abs(flux_smooth[baseline_mask] - baseline_med)))
            if not np.isfinite(baseline_mad) or baseline_mad == 0:
                baseline_mad = float(np.nanstd(flux_smooth[baseline_mask])) if np.any(np.isfinite(flux_smooth[baseline_mask])) else mad
        else:
            baseline_med, baseline_mad = med, mad

        z = (flux_smooth - baseline_med) / baseline_mad

        # adapt z_enter from local baseline noise around center of previous event
        center_idx = int(max(0, min(N - 1, (prev_istart + prev_iend) // 2)))
        z_enter = adapt_z_enter_from_local(flux_smooth, center_idx, half_window=rolling_w_for_adapt//2, base_z=z_enter_base)

        new_istart, new_iend = find_event_window_from_z(z, z_enter=z_enter, z_exit=z_exit, persist=persist)
        if new_istart is None:
            # if nothing new, keep previous
            break
        # convergence check (allow tiny shifts)
        if (new_istart == prev_istart) and (new_iend == prev_iend):
            prev_istart, prev_iend = new_istart, new_iend
            break
        prev_istart, prev_iend = new_istart, new_iend

    # final baseline mask and error model
    evt_len = max(1, prev_iend - prev_istart + 1)
    pad = int(max(10, min(500, pad_frac * evt_len)))
    mask_transient = np.zeros(N, bool)
    mask_transient[max(0, prev_istart - pad): min(N, prev_iend + pad + 1)] = True
    final_baseline_mask = (~mask_transient) & finite
    # fallback safeguard
    if np.sum(final_baseline_mask) < want_baseline_pts:
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

def rolling_median_mad(x, w):
    # Utility: rolling median + MAD (replace sigma clipping)
    s = pd.Series(x)
    med = s.rolling(w, min_periods=w//2).median()
    mad = (s - med).abs().rolling(w, min_periods=w//2).median()
    mad = 1.4826 * mad
    return med.to_numpy(), mad.to_numpy()

def stable_mask(x, w=21, z_tol=1.5):
    """
    Flag "stable" samples where rolling z is small (quasi-flat).
    Used to select baseline regions.
    """
    med, mad = rolling_median_mad(x, w)
    mad = np.where((~np.isfinite(mad)) | (mad == 0), np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826, mad)
    z = (x - med) / mad
    return np.isfinite(z) & (np.abs(z) < z_tol)

def grow_causal(mask, end_idx, want_min=60, max_lookback=3000, step=32):
    """
    Grow baseline mask to the left (causal) until it has ≥ want_min finite points
    (or ≥ min_pts=40 if want_min unattainable).
    """
    min_pts = min(40, want_min)  # hard floor 40
    m = np.zeros_like(mask, dtype=bool)
    L = step
    left = max(0, end_idx - L)
    m[left:end_idx] = mask[left:end_idx]
    while (np.sum(m) < want_min) and (left > 0) and ((end_idx - left) < max_lookback):
        add = min(step, left)
        left -= add
        m[left:end_idx] |= mask[left:end_idx]
        if np.sum(m) >= min_pts and (end_idx - left) >= step:
            # allow early stop if at least min_pts and still flat
            pass
    return m

def grow_anti_causal(mask, start_idx, want_min=60, max_lookahead=3000, step=32, N=None):
    """
    Grow baseline mask to the right (anti-causal) until it has ≥ want_min finite points.
    """
    if N is None: 
        N = mask.size
    min_pts = min(40, want_min)
    m = np.zeros_like(mask, dtype=bool)
    R = min(start_idx + step, N)
    m[start_idx:R] = mask[start_idx:R]
    while (np.sum(m) < want_min) and (R < N) and ((R - start_idx) < max_lookahead):
        add = min(step, N - R)
        R += add
        m[start_idx:R] |= mask[start_idx:R]
        if np.sum(m) >= min_pts and (R - start_idx) >= step:
            pass
    return m

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

def find_event_window_from_z(z, z_enter=2.5, z_exit=1.0, persist=2):
    """
    Hysteretic onset/end from Z with persistence.
    - enter when z > z_enter for persist samples
    - exit when z < z_exit after being inside
    Returns (i_start, i_end). If no event, returns (None, None).
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

    # exit hysteresis
    iend = None
    inside = True
    for j in range(istart, N):
        if z[j] < z_exit:
            # require a couple below to avoid chattering
            if j+1 < N and z[j+1] < z_exit:
                iend = j+1
                break
    if iend is None:
        iend = N - 1
    return istart, iend

def lc_significance_two_sided(time, flux, flux_err, frame_min, frame_max,
                              want_baseline_pts=60, side_w=21,
                              z_enter=2.5, z_exit=1.5, persist=2,
                              flux_sign=+1):
    """
    Returns:
      sig_max, sig_84, istart, iend, win_lo, win_hi, z, med, mad, flux
    istart/iend are indices in (time, flux, flux_err) *after* masking (i.e., post-mfin arrays).
    """
    N = flux.size
    stab = stable_mask(flux, w=side_w, z_tol=1.5)

    # --- Adaptive buffer around the cluster envelope ---
    evt_len = max(1, frame_max - frame_min + 1)
    pad = max(10, min(500, int(0.5 * evt_len)))  # 0.5x event length, on [10, 500]

    left_stop   = max(0, frame_min - pad)
    right_start = min(N - 1, frame_max + pad)

    left_mask_raw  = stab.copy();  left_mask_raw[left_stop:]   = False
    right_mask_raw = stab.copy();  right_mask_raw[:right_start] = False

    left_mask  = grow_causal(left_mask_raw,  end_idx=left_stop,   want_min=want_baseline_pts)
    right_mask = grow_anti_causal(right_mask_raw, start_idx=right_start, want_min=want_baseline_pts, N=N)

    # --- Sign-consistent baseline selection ---
    medL = np.nanmedian(flux[left_mask])  if np.any(left_mask)  else np.nan
    medR = np.nanmedian(flux[right_mask]) if np.any(right_mask) else np.nan
    med_evt = np.nanmedian(flux[max(0, frame_min-pad):min(N, frame_max+pad+1)])

    def _fallback():
        return choose_lower_baseline(flux, left_mask, right_mask)

    if flux_sign > 0:
        # want event brighter than baseline
        if np.isfinite(medL) and (med_evt > medL):
            base_mask = left_mask
        elif np.isfinite(medR) and (med_evt > medR):
            base_mask = right_mask
        else:
            base_mask = _fallback()
    else:
        # want event dimmer than baseline
        if np.isfinite(medL) and (med_evt < medL):
            base_mask = left_mask
        elif np.isfinite(medR) and (med_evt < medR):
            base_mask = right_mask
        else:
            base_mask = _fallback()

    if not np.any(base_mask):
        base_mask = causal_baseline_mask(flux, end_idx=max(0, frame_min), min_pts=60, max_lookback=2000)

    med = np.nanmedian(flux[base_mask])
    mad = 1.4826 * np.nanmedian(np.abs(flux[base_mask] - med))
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(flux[base_mask], ddof=1)
    if not np.isfinite(mad) or mad == 0:
        mad = 1.0

    z = (flux_sign * (flux - med)) / mad

    istart, iend = find_event_window_from_z(z, z_enter=z_enter, z_exit=z_exit, persist=persist)
    if istart is None:
        istart, iend = max(0, frame_min), min(N-1, frame_max)

    win_lo = max(0, istart)
    win_hi = min(N, iend + 1)
    zevt = np.abs(z[win_lo:win_hi]) if win_hi > win_lo else np.array([])

    sig_max = float(np.nanmax(zevt)) if zevt.size else -1.0
    sig_84  = float(np.nanpercentile(zevt, 84)) if zevt.size else -1.0

    return sig_max, sig_84, istart, iend, z

class Implement_reductions:
    def __init__(self, stars, tpf_info, diff, epsf_data, 
                 noise, corrlim=0, difflim=100, 
                 fwhmlim=5, maxlim=10, snrlim=1, roundness=0.8, 
                 poiss_val=1, siglim=1, dist_cut=0.2,
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
        
        self.det_min_pts = 3   # Stage-1 (early) clustering len threshold
        self.val_min_pts = 5   # Stage-2 (validation) threshold
        
        self.time = tpf_info.time.value
        self.campaign = tpf_info.campaign
        self.targetid = tpf_info.targetid
        self.cadence = self.time[1] - self.time[0]
        self.noise = noise

        # 1) loose initial cuts
        corr = self.filter_detections(self.stars, initial=True).reset_index(drop=True)

        # 2) loose grouping over *time* (frame) with f_dist ~ 50 frames by default
        groups = self._grouping(corr, f_dist=f_dist)

        if groups is not None and len(groups) > 0:
            full_events, filtered_stars = self.detected_events(groups, siglim=siglim)
            self.filtered_stars = filtered_stars
            self.full_events = full_events
        else:
            self.filtered_stars = None
            self.full_events = None
        
    def filter_detections(self, stars, initial=False):
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
        return ((correlation >= self.corrlim) & (psfdiff <= self.difflim) &
                (fwhm <= self.fwhmlim) & (fwhm >= 0.8) &
                (snr >= self.snrlim) & (snr < 10000) & (abs(roundness) <= self.roundness) &
                (poisson_thresh >= self.poiss_val) &
                (xstd <= self.dist_cut) & (ystd <= self.dist_cut))
        
    def _grouping(self, corr: pd.DataFrame, f_dist: int = 50) -> pd.DataFrame | None:
        if corr.empty:
            return None
        data = corr[['xcentroid', 'ycentroid', 'frame']].values.astype(np.float32)
        data[:, 2] *= 1.5 / f_dist  # makes eps=1.5 span roughly ±f_dist frames

        db = DBSCAN(eps=1.5,
                    min_samples=self.det_min_pts,
                    metric='euclidean',
                    algorithm='auto',
                    n_jobs=1)
        labels = db.fit_predict(data)
        corr = corr.assign(cluster=labels)
        corr = corr[corr.cluster != -1]
        return corr if not corr.empty else None

    def detected_events(self, events, siglim=2.0):
        cluster_ids = np.unique(events['cluster'])

        full_events = pd.DataFrame(columns=['cluster', 'frame_min', 'frame_max',
                                            'true_start', 'true_end',
                                            'remapped_frame_min', 'remapped_frame_max', 'x', 'y', 'xstd', 'ystd', 
                                            'sig_max', 'sig_84', 'mjds', 'flux', 'flux_err',
                                            'roundness', 'fwhm', 'snr', 'psfdiff', 'correlation', 
                                            'poisson_thresh', 'e_roundness', 'e_fwhm', 'e_snr', 'e_psfdiff', 
                                            'e_correlation', 'e_poisson_thresh'])

        new_stars = pd.DataFrame(columns=events.columns)
        events = events.copy()
        events['lc_sig'] = -1.0

        for cid in cluster_ids:
            cluster = events[events['cluster'] == cid]
            if len(cluster) < self.det_min_pts:
                continue

            # centroid compactness (loose, pre-LC)
            x, _, xstd = sigma_clipped_stats(cluster['xcentroid'].values, sigma=3)
            y, _, ystd = sigma_clipped_stats(cluster['ycentroid'].values, sigma=3)
            if (xstd >= self.dist_cut) or (ystd >= self.dist_cut):
                continue

            frame_min = int(cluster['frame'].min())
            frame_max = int(cluster['frame'].max())

            # --- build LC at the centroid for this cluster ---
            bjds = deepcopy(self.time)
            lc_raw   = forced_photometry(self.diff,  x, y, self.epsf)
            flux_err = forced_photometry(self.noise, x, y, self.epsf)
            # lc_raw, flux_err = forced_photometry_psf(self.diff, self.noise, x, y, self.epsf, bkg=True, method='psf')
            
            np.save('lc_flux.npy', lc_raw)
            np.save('lc_time.npy', bjds)
            np.save('lc_flux_err.npy', flux_err)
            
            # og_ind = np.arange(len(lc_raw))

            mfin = np.isfinite(lc_raw) & np.isfinite(bjds) & np.isfinite(flux_err)
            bjds, lc_raw, flux_err = bjds[mfin], lc_raw[mfin], flux_err[mfin]

            # print('G SMOOTH START')
            # lc_sm, lc_sm_err = gauss_smooth(bjds, lc_raw) # smooth for significance
            lc_sm = lc_raw.copy()
            lc_sm_err = flux_err.copy()
            # print('G SMOOTH END')

            # 2) find iterative baseline & z on smoothed LC (fast mode, no GP refit)
            args = iterative_baseline_zscore_fast(time=bjds, flux_smooth=lc_sm, raw_flux=lc_raw, 
                                                  flux_err=lc_sm_err, frame_min=frame_min, 
                                                  frame_max=frame_max, base_w=21, z_enter_base=siglim,   # your z_enter default
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
            sig_84  = float(np.nanpercentile(zevt, 84)) if zevt.size else -1.0

            if istart is not None and iend is not None and (iend - istart) >= 5:
                sm_win = slice(istart + 2, iend + 1)
                ratio = smoothness_metric(lc_sm[sm_win], w=21)
            else:
                ratio = np.nan

            if sig_84 < siglim:
                continue
            
            if ~np.isfinite(ratio):
                continue
            
            if (ratio >= 3):
                continue
            
            og_ind = np.where(mfin)[0]  # original frame indices
            rmap_ind = np.arange(lc_sig.size)

            filtered_cluster = cluster[cluster['frame'].isin(og_ind)].reset_index(drop=True)
            if len(filtered_cluster) < self.det_min_pts:
                continue
            
            np.save('lc_significance.npy', np.column_stack([bjds, lc_sm, lc_sm_err, lc_sig]))
            print('ZZZ \n', 
                  'ISTART & END \n', istart, iend, '\n',
                  'FRAME MIN & MAX \n', 
                  int(filtered_cluster['frame'].min()), int(filtered_cluster['frame'].max()))

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

            bjds_utc = bjds + 54832.5
            t_bary = Time(bjds_utc, format="mjd", scale="tdb")
            t_mjd_utc = t_bary.utc.mjd
            
            rmap_lo = rmap_ind[og_ind == istart][0] if np.any(og_ind == istart) else 0
            rmap_hi = rmap_ind[og_ind == iend][0] if np.any(og_ind == iend) else (len(rmap_ind)-1)
            
            # store event
            full_events.loc[len(full_events)] = [
                len(full_events) + 1,
                istart, iend,
                t_mjd_utc[istart], 
                t_mjd_utc[iend], 
                rmap_lo, rmap_hi,
                x, y, xstd, ystd,
                sig_max, sig_84,
                t_mjd_utc, lc_sm, lc_sm_err,
                roundness, fwhm, snr, psfdiff, correlation, poisson_thresh,
                e_roundness, e_fwhm, e_snr, e_psfdiff, e_correlation, e_poisson_thresh
            ]

        if len(full_events) == 0:
            return None, None
        else:
            new_stars = new_stars.reset_index(drop=True)
            return full_events, new_stars
