from __future__ import division, print_function

import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import median_filter

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')
    
def remove_thermal_rpca_trends(difference, errormap, mask=None, demean=True, lam=None):
    """
    difference: (nt, nx, ny) difference images [e-/s]
    errormap:   (nt, nx, ny) per-pixel errors [e-/s]
    mask:       (nt, nx, ny) boolean, True=ignore pixel (optional)
    demean:     bool, subtract per-frame background median before RPCA
    lam:        RPCA lambda (if None, use default heuristic)
    """
    
    nt, nx, ny = difference.shape
    
    Z_map = difference / np.where(errormap > 0, errormap, np.nanmedian(errormap[errormap > 0]))
    
    if mask is not None:
        Z_map = np.where(~mask, Z_map, 0.0)
    
    if demean:
        for t in range(nt):
            good = np.isfinite(Z_map[t]) if mask is None else (~mask[t]) & np.isfinite(Z_map[t])
            if np.any(good):
                med = np.median(Z_map[t][good])
                Z_map[t] -= med
    
    Z_map_flat = Z_map.reshape(nt, nx*ny)
    
    rpca = R_pca(Z_map_flat, mu=None, lmbda=lam)
    L, S = rpca.fit(max_iter=1000, iter_print=100)
    
    L_cube = L.reshape(nt, nx, ny)
    S_cube = S.reshape(nt, nx, ny)
    
    thermal_hat = L_cube * errormap
    D_clean = difference - thermal_hat
    
    return D_clean, L_cube, S_cube

# ---------- helpers ----------
def ema_highpass_1d(x, alpha=0.02):
    # returns high-pass component: x - EMA(x)
    y = np.zeros_like(x, dtype=float)
    m = 0.0
    for i, xi in enumerate(x):
        m = alpha * xi + (1 - alpha) * m
        y[i] = xi - m
    return y

def fit_remove_sinusoids(y, periods):
    """
    Per-pixel harmonic regression: remove sum_k [a_k cos + b_k sin] at given periods.
    y: shape (T,)
    periods: list of floats (in frames); empty list → no-op
    """
    T = y.shape[0]
    t = np.arange(T, dtype=float)
    cols = []
    for P in periods:
        w = 2*np.pi/P
        cols.append(np.cos(w*t))
        cols.append(np.sin(w*t))
    if not cols:
        return y
    X = np.vstack(cols).T  # (T, 2K)
    coef, _, _, _ = lstsq(X, y, rcond=None)
    y_hat = X @ coef
    return y - y_hat

def project_out_regressors(Z, R):
    """
    Remove effect of external regressors R (T x K) from Z (T x P): Z_res = Z - P_R(Z).
    Works on standardized data; unweighted LS is fine.
    """
    if R is None or R.size == 0:
        return Z
    # QR is stable for large P (features)
    Q, _ = np.linalg.qr(R, mode='reduced')  # T x K, orthonormal columns
    return Z - Q @ (Q.T @ Z)

def prepare_Z(difference, errormap, mask=None, demean_frame=True):
    T, ny, nx = difference.shape
    sigma_safe = np.where(errormap > 0, errormap, np.nanmedian(errormap[errormap > 0]))
    Z = difference / sigma_safe
    if mask is not None:
        Z = np.where(~mask, Z, 0.0)
    if demean_frame:
        for t in range(T):
            good = np.isfinite(Z[t]) if mask is None else (~mask[t]) & np.isfinite(Z[t])
            if np.any(good):
                Z[t] -= np.median(Z[t][good])
    return Z

def rpca_once(M, lam=None, max_iter=500):
    # Try your R_pca; else fallback to truncated SVD
    T, P = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(T, P))
        
        rpca = R_pca(M, mu=None, lmbda=lam)
        L, S = rpca.fit(max_iter=max_iter, iter_print=0)
        return L, S

def sliding_rpca(Z, mask=None, lam=None, W=100, step=50, max_iter=500):
    T, ny, nx = Z.shape
    outL = np.zeros_like(Z)
    wts  = np.zeros(T)
    for t0 in range(0, T, step):
        t1 = min(T, t0+W)
        Zw = Z[t0:t1].reshape(t1 - t0, ny*nx)
        if mask is not None:
            Mw = (~mask[t0:t1]).reshape(t1 - t0, ny*nx).astype(float)
            # fill masked with 0; simple EM: refill with L after each pass
            X = Zw.copy()
            for _ in range(3):
                L, S = rpca_once(X, lam=lam, max_iter=max_iter)
                X = Mw*Zw + (1 - Mw)*L
            L, S = rpca_once(X, lam=lam, max_iter=max_iter)
        else:
            L, S = rpca_once(Zw, lam=lam, max_iter=max_iter)
        outL[t0:t1] += L.reshape(t1 - t0, ny, nx)
        wts[t0:t1] += 1
    outL /= wts[:, None, None]
    return outL

# ---------- main cleaner ----------
def clean_thermal_cadence(
    difference, errormap, *,
    mask=None,
    periods_frames=None,        # e.g. [48] if 48-frame cadence; add harmonics [48, 24, 16] etc.
    ema_alpha=0.02,             # temporal HP strength (0.02 ~ ~50-frame time constant)
    regressors=None,            # dict of {name: (T,) arrays}, e.g. {'bg':B, 'cx':cx, 'cy':cy}
    rpca_lambda_scale=1.5,      # >1.0 → stronger sparsity
    window=120, step=60, max_iter=500,
    demean_frame=True
):
    """
    Returns:
      D_clean: difference minus learned low-rank thermal/cadence (e-/s)
      thermal_hat: low-rank estimate in e-/s
      Lz: low-rank (standardized units)
    """
    T, ny, nx = difference.shape

    # 1) Standardize and (optionally) per-frame de-mean
    Z = prepare_Z(difference, errormap, mask=mask, demean_frame=demean_frame)  # (T, ny, nx)

    # 2) Per-pixel cadence removal + EMA high-pass (on Z)
    if periods_frames or ema_alpha is not None:
        Z2 = Z.copy()
        for y in range(ny):
            for x in range(nx):
                z = Z2[:, y, x]
                if not np.isfinite(z).any():
                    continue
                # (a) notch/harmonics
                if periods_frames:
                    z = fit_remove_sinusoids(z, periods_frames)
                # (b) EMA high-pass
                if ema_alpha is not None and ema_alpha > 0:
                    z = ema_highpass_1d(z, alpha=ema_alpha)
                Z2[:, y, x] = z
        Z = Z2

    # 3) Regress out external drivers in standardized space (pixel-level decorrelation)
    if regressors:
        # Build R (T x K), z-score columns to be safe
        Rcols = []
        for k in sorted(regressors.keys()):
            r = np.asarray(regressors[k], float).reshape(T)
            r = r - np.nanmedian(r)
            s = np.nanstd(r);  s = s if s > 0 else 1.0
            Rcols.append(r / s)
        R = np.vstack(Rcols).T  # T x K

        Z_flat = Z.reshape(T, ny*nx)
        Z_flat = project_out_regressors(Z_flat, R)
        Z = Z_flat.reshape(T, ny, nx)

    # 4) Sliding-window RPCA with masking
    lam = rpca_lambda_scale / np.sqrt(max(T, ny*nx))
    Lz = sliding_rpca(Z, mask=mask, lam=lam, W=window, step=step, max_iter=max_iter)

    # 5) Undo standardization to get thermal/systematics estimate
    thermal_hat = Lz * errormap
    D_clean = difference - thermal_hat
    return D_clean, thermal_hat, Lz


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D.flatten(), ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')