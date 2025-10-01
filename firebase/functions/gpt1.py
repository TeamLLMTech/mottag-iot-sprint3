import math
import numpy as np
from math import isfinite
from scipy.optimize import least_squares

def estimate_position_gpt_1(anchors, rssis, tx_power_dbm=-59.0, path_loss_exponent=2.0,
                      min_anchors=3, max_iter=100, tol=1e-6,
                      robust_loss='huber', robust_loss_scale=1.0,
                      bounds=None, weighting=True, outlier_rejection=True):
    """
    Estimate the 2D position of a device given RSSI measurements from fixed anchors.

    Uses a log-distance path loss model to convert RSSI to distances and performs a robust
    non-linear least squares trilateration to find (x, y) that best fits the distance constraints.
    Robust loss (e.g., Huber) and iterative outlier rejection are applied to handle noise and outliers.

    Parameters
    ----------
    anchors : list of tuple of float
        Coordinates (x, y) of the fixed anchor points. Length N.
    rssis : list of float
        RSSI values (in dBm) measured at each anchor (same order as `anchors`). Length N.
    tx_power_dbm : float, optional
        Reference RSSI (in dBm) at 1 meter distance. Default is -59.0 dBm (typical for BLE).
    path_loss_exponent : float, optional
        Path loss exponent *n* for the log-distance model. Default is 2.0 (free-space; indoor often 2.0-2.4).
        Higher values (>2) indicate stronger attenuation (e.g., walls, obstacles).
    min_anchors : int, optional
        Minimum number of anchors required to attempt a solution. Default is 3 (for 2D).
    max_iter : int, optional
        Maximum number of solver iterations (function evaluations) for the non-linear optimizer. Default 100.
    tol : float, optional
        Tolerance for convergence. When changes in solution or cost fall below this, optimization stops.
        This is used for both position change (xtol) and cost change (ftol).
    robust_loss : str or None, optional
        Robust loss function for least squares. Options: 'huber', 'soft_l1', 'cauchy', 'arctan', 'linear' or None.
        Default is 'huber' for robust estimation. Use 'linear' or None to disable robust loss (standard least squares).
    robust_loss_scale : float, optional
        Scale parameter for the robust loss. For 'huber', this is the threshold (δ) where residuals start to be down-weighted.
        Default is 1.0 (in same units as distance residuals, e.g., meters). Lower values make the loss more aggressively robust.
    bounds : tuple of array_like or None, optional
        Bounds for (x, y) as ((min_x, min_y), (max_x, max_y)). Use None for no bounds (default).
        Each of min_x, min_y, max_x, max_y can be a number or None to indicate no bound on that coordinate.
    weighting : bool, optional
        If True, apply weighting to residuals based on expected RSSI measurement variance. Default True.
        Currently uses weight = 1/(distance^2) for each anchor (assuming distance estimation variance grows with distance).
        This gives closer anchors (stronger signals) more influence. If False, all anchors treated equally.
    outlier_rejection : bool or str, optional
        If True (default), perform an iterative outlier rejection after an initial fit.
        Can also specify method as 'iqr', 'mad', etc. True defaults to 'iqr'.
        Outliers are identified by large residuals and removed, then the position is recomputed.

    Returns
    -------
    result : dict
        Dictionary with the estimated solution and metadata:
        - "x": float, estimated x-coordinate.
        - "y": float, estimated y-coordinate.
        - "cov": 2x2 list of lists, approximate covariance matrix of the estimate (if available).
        - "rmse": float, root-mean-square error of the residuals (in distance units, e.g., meters).
        - "residuals": list of float, residual (predicted_distance - estimated_distance_from_RSSI) for each used anchor.
        - "weights": list of float, weights applied to each used anchor's residual (after robust weighting, normalized to 0-1).
        - "used_anchor_indices": list of int, indices of anchors used in the final solution.
        - "rejected_anchor_indices": list of int, indices of anchors that were identified as outliers or invalid and excluded.
        - "iterations": int, number of iterations (function evaluations) performed in the final optimization.
        - "converged": bool, True if the optimizer converged within tolerance.
        - "model_params": dict, parameters used in the model (tx_power_dbm, n, robust_loss, robust_loss_scale, etc.).
        - "error": str, error message if something went wrong (e.g., insufficient anchors, singular geometry).

    Raises
    ------
    ValueError:
        If input types are incorrect or if SciPy is not available for optimization.
    """
    # Validate inputs
    if anchors is None or rssis is None:
        return {"error": "Inputs 'anchors' and 'rssis' must be provided."}
    if len(anchors) != len(rssis):
        return {"error": f"anchors and rssis must have same length, got {len(anchors)} vs {len(rssis)}."}
    N = len(anchors)
    # Ensure minimum anchors
    if N < min_anchors:
        return {"error": f"Insufficient anchors: need at least {min_anchors}, got {N}."}
    # Convert anchors to numpy array
    try:
        coords = np.array(anchors, dtype=float)
    except Exception as e:
        return {"error": f"Invalid anchor coordinates: {e}"}
    if coords.ndim != 2 or coords.shape[1] != 2:
        return {"error": "anchors must be a sequence of (x, y) pairs."}
    # Validate RSSI values
    try:
        rssi_array = np.array(rssis, dtype=float)
    except Exception as e:
        return {"error": f"Invalid RSSI values: {e}"}
    # Filter out invalid RSSI entries (NaN or inf)
    valid_idx = [i for i in range(N) if isfinite(rssi_array[i])]
    removed = [i for i in range(N) if i not in valid_idx]
    if len(valid_idx) < N:
        coords = coords[valid_idx]
        rssi_array = rssi_array[valid_idx]
        N = len(valid_idx)
    if N < min_anchors:
        return {"error": f"After removing invalid entries, need at least {min_anchors} anchors, got {N}.",
                "rejected_anchor_indices": removed}
    # Check anchor geometry (colinearity)
    if N >= 3:
        centered = coords - np.mean(coords, axis=0)
        cov_anchor = np.dot(centered.T, centered) / (N - 1)
        eigvals = np.linalg.eigvals(cov_anchor)
        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))
        if max_eig > 0 and min_eig / max_eig < 1e-3:
            return {"error": "Anchor geometry is ill-conditioned (anchors nearly colinear)."}
    # Convert RSSI to distance estimates using log-distance model
    tx = float(tx_power_dbm)
    n = float(path_loss_exponent)
    dist_est = 10.0 ** ((tx - rssi_array) / (10.0 * n))
    # Initial position guess (weighted centroid)
    if weighting and N > 0:
        w_init = 1.0 / (dist_est + 1e-6)
    else:
        w_init = np.ones_like(dist_est)
    x0 = float(np.dot(w_init, coords[:, 0]) / np.sum(w_init)) if N > 0 else 0.0
    y0 = float(np.dot(w_init, coords[:, 1]) / np.sum(w_init)) if N > 0 else 0.0
    initial_guess = np.array([x0, y0], dtype=float)
    # Prepare weights for residual function
    w = None
    if weighting:
        w = 1.0 / (dist_est**2 + 1e-9)
    # Residual function for least squares
    def residual_func(params):
        px, py = params
        d_pred = np.sqrt((px - coords[:, 0])**2 + (py - coords[:, 1])**2)
        res = d_pred - dist_est
        if w is not None:
            res = np.sqrt(w) * res
        return res
    if least_squares is None:
        raise ValueError("SciPy is required for non-linear least squares optimization.")
    # Set up bounds
    lsq_bounds = (-np.inf, np.inf)
    if bounds is not None:
        try:
            lb, ub = bounds
            lb = [(-np.inf if b is None else float(b)) for b in lb]
            ub = [(np.inf if b is None else float(b)) for b in ub]
            lsq_bounds = (lb, ub)
        except Exception as e:
            return {"error": f"Invalid bounds format: {e}"}
    # Robust loss setup
    loss = robust_loss if robust_loss is not None else 'linear'
    if loss == 'none':
        loss = 'linear'
    # Initial optimization
    res1 = least_squares(residual_func, initial_guess, loss=loss, f_scale=robust_loss_scale,
                          xtol=tol, ftol=tol, gtol=tol, max_nfev=max_iter, bounds=lsq_bounds)
    if not res1.success:
        return {"error": "Optimization did not converge: " + res1.message, "converged": False}
    # Determine outlier rejection method
    outlier_method = None
    if outlier_rejection:
        if isinstance(outlier_rejection, str):
            method = outlier_rejection.lower()
            if method in ['iqr', 'mad', 'ransac']:
                outlier_method = method
        else:
            outlier_method = 'iqr'
    used_idx = valid_idx.copy()
    rejected_idx = []
    final_res = res1
    final_coords = coords
    final_dist_est = dist_est
    if outlier_method:
        px, py = res1.x
        d_pred_all = np.sqrt((px - coords[:, 0])**2 + (py - coords[:, 1])**2)
        raw_residuals = d_pred_all - dist_est
        abs_res = np.abs(raw_residuals)
        outlier_local = []
        if outlier_method == 'iqr':
            q1 = np.percentile(abs_res, 25)
            q3 = np.percentile(abs_res, 75)
            iqr = q3 - q1
            thresh = q3 + 1.5 * iqr
            outlier_local = [i for i, r in enumerate(abs_res) if r > thresh]
        elif outlier_method == 'mad':
            med = np.median(abs_res)
            mad = np.median(np.abs(abs_res - med))
            if mad > 1e-12:
                thresh = 3.0 * 1.4826 * mad
                outlier_local = [i for i, r in enumerate(abs_res) if np.abs(r - med) > thresh]
        elif outlier_method == 'ransac':
            base_rmse = math.sqrt(np.mean(raw_residuals**2)) if len(raw_residuals) > 0 else 0.0
            best_imp, worst_idx = 0.0, None
            for i in range(len(coords)):
                res_masked = np.delete(raw_residuals, i)
                if res_masked.size == 0:
                    continue
                rmse_masked = math.sqrt(np.mean(res_masked**2))
                imp = base_rmse - rmse_masked
                if imp > best_imp and imp > 1e-6:
                    best_imp = imp
                    worst_idx = i
            if worst_idx is not None:
                outlier_local = [worst_idx]
        outlier_local = sorted(set(outlier_local))
        if outlier_local:
            # Map to original indices
            rejected_idx = [used_idx[i] for i in outlier_local]
            used_idx = [idx for idx in used_idx if idx not in rejected_idx]
            if len(used_idx) < 3:
                return {"error": f"Outlier rejection removed too many anchors; only {len(used_idx)} remain (need >=3).",
                        "used_anchor_indices": used_idx, "rejected_anchor_indices": removed + rejected_idx}
            coords2 = np.delete(coords, outlier_local, axis=0)
            dist_est2 = np.delete(dist_est, outlier_local)
            w2 = None
            if weighting:
                w2 = 1.0 / (dist_est2**2 + 1e-9)
            def residual_func2(params):
                px, py = params
                d_pred = np.sqrt((px - coords2[:, 0])**2 + (py - coords2[:, 1])**2)
                res = d_pred - dist_est2
                if w2 is not None:
                    res = np.sqrt(w2) * res
                return res
            res2 = least_squares(residual_func2, res1.x, loss=loss, f_scale=robust_loss_scale,
                                 xtol=tol, ftol=tol, gtol=tol, max_nfev=max_iter, bounds=lsq_bounds)
            final_res = res2 if res2.success else res1
            final_coords = coords2
            final_dist_est = dist_est2
    # Compute final residuals and metrics
    px, py = final_res.x
    final_pred_d = np.sqrt((px - final_coords[:, 0])**2 + (py - final_coords[:, 1])**2)
    final_raw_res = final_pred_d - final_dist_est
    rmse = math.sqrt(np.mean(final_raw_res**2)) if final_raw_res.size > 0 else 0.0
    # Covariance matrix approximation
    cov_matrix = None
    if final_res.jac is not None and final_res.jac.size > 0:
        J = final_res.jac
        try:
            JTJ = np.dot(J.T, J)
            inv_JTJ = np.linalg.inv(JTJ)
            dof = max(1, len(final_raw_res) - 2)
            sigma2 = np.sum(final_raw_res**2) / dof
            cov_est = inv_JTJ * sigma2
            cov_matrix = [[float(cov_est[0, 0]), float(cov_est[0, 1])],
                          [float(cov_est[1, 0]), float(cov_est[1, 1])]]
        except Exception:
            cov_matrix = None
    # Compute robust weights for final residuals (0-1 scale)
    weights_out = []
    if loss != 'linear':
        for i, r in enumerate(final_raw_res):
            # Determine weighted residual (if initial weighting used) for robust weight calc
            weighted_r = r
            if weighting:
                # Find corresponding original index of this residual to get weight
                # If outliers were removed, final_coords corresponds to some subset of original
                # We can approximate by using w from original valid indices if index mapping not trivial.
                if i < len(final_coords):
                    # Approximating index mapping by position after removal:
                    weighted_r = math.sqrt(w[ i if len(final_coords) == len(w) else used_idx[i] - (sum(j < used_idx[i] for j in removed) + sum(j < used_idx[i] for j in rejected_idx)) ]) * r
            # Robust weight depending on loss
            if loss == 'huber':
                if abs(weighted_r) <= robust_loss_scale:
                    rw = 1.0
                else:
                    rw = robust_loss_scale / abs(weighted_r)
            elif loss == 'soft_l1':
                rw = 1.0 / math.sqrt(1.0 + (weighted_r/robust_loss_scale)**2)
            elif loss == 'cauchy':
                rw = 1.0 / (1.0 + (weighted_r/robust_loss_scale)**2)
            elif loss == 'arctan':
                rw = 1.0 / (1.0 + (weighted_r/robust_loss_scale)**4)
            else:
                rw = 1.0
            rw = max(0.0, min(1.0, rw))
            weights_out.append(round(rw, 3))
    else:
        weights_out = [1.0] * len(final_raw_res)
    # Model parameters used
    model_params = {
        "tx_power_dbm": tx_power_dbm,
        "n": path_loss_exponent,
        "robust_loss": (robust_loss if robust_loss is not None else "none")
    }
    if robust_loss is not None and loss != 'linear':
        model_params["robust_loss_scale"] = robust_loss_scale
    model_params["weighting"] = bool(weighting)
    model_params["outlier_rejection"] = (outlier_method or True) if outlier_rejection else False
    return {
        "x": float(px),
        "y": float(py),
        "cov": cov_matrix,
        "rmse": float(rmse),
        "residuals": [float(r) for r in final_raw_res],
        "weights": weights_out,
        "used_anchor_indices": used_idx,
        "rejected_anchor_indices": removed + rejected_idx if removed or rejected_idx else [],
        "iterations": getattr(final_res, 'nfev', None),
        "converged": bool(final_res.success),
        "model_params": model_params
    }