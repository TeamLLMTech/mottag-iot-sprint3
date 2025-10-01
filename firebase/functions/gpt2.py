import math
import numpy as np

def estimate_position_gpt_2(anchors, rssis, measured_power=-59.0, path_loss_exponent=2.0,
                      method='auto', robust_loss='soft_l1', robust_loss_scale=1.0,
                      ransac_threshold=1.0, ransac_trials=100):
    """
    Estimate the 2D position of a device using trilateration from RSSI measurements.

    This function computes the (x, y) position of a device given a set of fixed anchor positions
    and corresponding RSSI (Received Signal Strength Indicator) measurements from the device at those anchors.
    It converts RSSI values to distance estimates using a log-distance path loss model and then performs
    a trilateration algorithm to estimate the most likely position of the device.
    The algorithm is designed to be robust to noise, measurement inaccuracies, outliers, and inconsistent readings.

    **Parameters:**
    - **anchors** (`list` of `tuple(float, float)`): Coordinates of fixed anchor points. For example: `[(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]`.
      There must be at least 3 anchors for 2D trilateration. More anchors can improve accuracy.
    - **rssis** (`list` of `float`): RSSI values (in dBm) measured at the corresponding anchors (same order as `anchors`).
      Each RSSI value should ideally be a negative number (as RSSI in dBm is typically negative). The list length must match `anchors`.
    - **measured_power** (`float`, optional): The expected RSSI at 1 meter distance (also known as TxPower or 1m RSSI calibration value).
      This is used in the path loss model to convert RSSI to distance. Defaults to -59.0 dBm, a typical value for BLE beacons.
      If known, you should set this to the calibrated RSSI at 1m for your device and environment for best results.
    - **path_loss_exponent** (`float`, optional): The path loss exponent (environmental factor). Typically 2.0 for free space,
      and 2-4 for indoor environments. Defaults to 2.0.
    - **method** (`str`, optional): Algorithm to use for trilateration. Choices are:
        - `'auto'` (default): Uses a robust method by default (non-linear least squares with robust loss).
        - `'linear'`: Linear least squares solution by linearizing the equations. Fast but less accurate if there's noise.
        - `'nonlinear'`: Non-linear least squares solution (minimizes distance error) for higher accuracy.
        - `'robust'`: Non-linear least squares with a robust loss function to reduce outlier impact.
        - `'ransac'`: RANSAC approach to ignore outliers by trying different anchor subsets.
    - **robust_loss** (`str`, optional): Loss function for robust non-linear method. Applicable if `method` is `'robust'` or `'auto'`.
      Options as defined in `scipy.optimize.least_squares`: `'linear'` (no robustness), `'soft_l1'`, `'huber'`, `'cauchy'`, etc.
      Default is `'soft_l1'`, which is a smooth approximation of L1 loss, reducing influence of outliers.
    - **robust_loss_scale** (`float`, optional): Scaling parameter for the robust loss. Default is 1.0. Higher values make the loss function closer to linear (less robust).
      Lower values give more robustness by down-weighting outliers more strongly.
    - **ransac_threshold** (`float`, optional): Distance error threshold (in meters) for inliers in RANSAC. Default is 1.0 meter.
      Anchors with distance residuals below this threshold are considered inliers for a candidate solution.
    - **ransac_trials** (`int`, optional): Number of iterations (random subsets to try) in RANSAC. If the number of anchors is small, it may try all combinations of 3 anchors.
      Default is 100 trials (for larger numbers of anchors, a random subset of combinations will be tried).

    **Returns:**
    - A dictionary with the following keys on successful localization:
        - `"position"`: tuple `(x, y)` of the estimated device coordinates.
        - `"uncertainty_radius"`: A float indicating an approximate uncertainty radius (in meters) for the estimated position.
          This can be interpreted as a rough radius around the estimated position where the device likely lies.
        - `"uncertainty_bbox"`: A tuple `(xmin, ymin, xmax, ymax)` giving a simplistic bounding box for the estimate (usually just `position ± uncertainty_radius` in each direction).
        - `"anchor_count"`: Total number of anchors provided.
        - `"anchors_used"`: Number of anchors effectively used in the final estimation (after any outlier removal).
        - `"residuals"`: A list of distance residuals for each anchor used in the final solution (difference between estimated distance to anchor and distance inferred from that anchor's RSSI).
        - `"rmse"`: The root-mean-square error of the distance residuals (in meters) for the final solution, as a measure of fit quality (lower is better).
        - `"method_used"`: The method/algorithm that was actually used for the computation (could be adjusted from the `method` parameter in some cases).
        - Other metadata or diagnostic info may be added as needed.
    - If the position cannot be estimated due to insufficient or inconsistent data, returns a dictionary with an `"error"` key describing the issue. For example:
        - `"error": "Not enough anchors (requires >= 3)."`
        - `"error": "Colinear anchor configuration, solution not unique."`
        - `"error": "RSSI values are all very weak or invalid."`
        - etc.

    **Algorithmic details:**
    1. **RSSI to Distance Conversion:** Each RSSI value is converted to an estimated distance using the log-distance path loss model:
          `distance = 10 ** ((measured_power - rssi) / (10 * path_loss_exponent))`.
       If any distances come out as NaN or infinite (due to invalid inputs), an error is returned.
       If some RSSI values are extremely out-of-range, those anchors may be considered unreliable.
    2. **Initial Outlier Check:** If more than the minimum required anchors are provided, the algorithm can attempt to detect and handle outliers:
       - If an anchor's RSSI suggests a distance that is impossible or far inconsistent with others, it may be ignored or down-weighted.
    3. **Trilateration Methods:**
       - *Linear method:* Forms a system of linear equations by subtracting distance equations relative to a reference anchor. Solves using least squares (normal equations).
       - *Non-linear least squares:* Defines a cost function based on the difference between the distance to the estimated position and the measured distance for each anchor, and minimizes the sum of squared errors (this is done via iterative optimization).
       - *Robust least squares:* Same as non-linear but using a robust loss (like Huber or soft L1) to reduce influence of anchors with large errors.
       - *RANSAC:* Randomly selects minimal sets of anchors (3 anchors) to compute candidate positions (via linear trilateration), then selects the solution with the most inliers (anchors within a certain error threshold). The final solution can be refined using a non-linear fit on all inliers.
    4. **Uncertainty Estimation:** The uncertainty radius is approximated from the residual errors (e.g., roughly the RMSE or max error). A bounding box is also provided as a simple representation of position uncertainty.
    5. The function is stateless; each call is independent and relies only on its inputs. No internal state is preserved between calls.

    **Examples:**
    ```
    anchors = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)]
    rssis = [-50.0, -60.0, -55.0, -65.0]
    result = estimate_position(anchors, rssis)
    if "error" in result:
        print("Estimation failed:", result["error"])
    else:
        print("Estimated position:", result["position"])
        print("Uncertainty (radius):", result["uncertainty_radius"])
        print("RMSE of fit:", result["rmse"])
    ```
    """
    # Convert inputs to numpy arrays for convenience
    try:
        anchor_array = np.array(anchors, dtype=float)
    except Exception as e:
        return {"error": f"Invalid anchors input: {e}"}
    anchor_count = anchor_array.shape[0]
    if anchor_count < 3:
        return {"error": f"Not enough anchors (requires >= 3, got {anchor_count})."}
    if anchor_array.ndim != 2 or anchor_array.shape[1] != 2:
        return {"error": "Anchors must be a list of (x, y) coordinates."}
    try:
        rssi_array = np.array(rssis, dtype=float)
    except Exception as e:
        return {"error": f"Invalid RSSI input: {e}"}
    if rssi_array.shape[0] != anchor_count:
        return {"error": f"anchors and rssis must have same length (got {anchor_count} and {rssi_array.shape[0]})."}
    # Remove any anchors with invalid RSSI (NaN or Inf)
    valid_mask = np.isfinite(rssi_array)
    if not np.all(valid_mask):
        anchor_array = anchor_array[valid_mask]
        rssi_array = rssi_array[valid_mask]
        removed = anchor_count - anchor_array.shape[0]
        anchor_count = anchor_array.shape[0]
        if anchor_count < 3:
            return {"error": f"Insufficient valid RSSI values after removing {removed} invalid entries."}
    # Convert RSSI to distances using log-distance path loss model
    distances = np.power(10.0, (measured_power - rssi_array) / (10.0 * path_loss_exponent))
    if not np.all(np.isfinite(distances)):
        return {"error": "Non-finite distance calculated from RSSI (check measured_power and path_loss_exponent)."}
    if np.all(distances > 1e6):
        return {"error": "RSSI values indicate extremely large distances (device likely out of range)."}
    # Identify obvious outliers based on distances
    outlier_mask = np.zeros(anchor_count, dtype=bool)
    if anchor_count > 3:
        med_dist = float(np.median(distances))
        mad_dist = float(np.median(np.abs(distances - med_dist)))
        if mad_dist == 0:
            mad_dist = med_dist * 0.1
        high_thr = med_dist + 3 * mad_dist
        low_thr = max(0.0, med_dist - 3 * mad_dist)
        for i, d in enumerate(distances):
            if d > high_thr or d < low_thr:
                outlier_mask[i] = True
        if np.count_nonzero(outlier_mask) > anchor_count / 2:
            outlier_mask[:] = False
    chosen_method = str(method).lower()
    if chosen_method == 'auto':
        chosen_method = 'robust'  # default to robust approach
    if chosen_method == 'linear':
        # Solve linearized equations
        ref_idx = 0
        x1, y1 = anchor_array[ref_idx]
        d1 = distances[ref_idx]
        A = []
        b = []
        for j in range(anchor_count):
            if j == ref_idx: continue
            xj, yj = anchor_array[j]
            dj = distances[j]
            A.append([2*(xj - x1), 2*(yj - y1)])
            b.append(d1**2 - dj**2 + (xj**2 - x1**2) + (yj**2 - y1**2))
        A = np.array(A)
        b = np.array(b)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception as e:
            return {"error": f"Linear solve failed: {e}"}
        px, py = float(sol[0]), float(sol[1])
        est_dists = np.sqrt(np.sum((anchor_array - np.array([px, py]))**2, axis=1))
        resids = est_dists - distances
        method_used = 'linear'
        anchors_used_count = anchor_count
    elif chosen_method == 'ransac':
        # RANSAC: try random triples to find best inliers
        best_inliers = -1
        best_xy = None
        best_inlier_mask = None
        from itertools import combinations
        combos = list(combinations(range(anchor_count), 3))
        if len(combos) > ransac_trials:
            np.random.shuffle(combos)
            combos = combos[:ransac_trials]
        for combo in combos:
            idx0, idx1, idx2 = combo
            # Solve linear for this triple
            x1, y1 = anchor_array[idx0]; d1 = distances[idx0]
            A = []
            b = []
            for j in [idx1, idx2]:
                xj, yj = anchor_array[j]; dj = distances[j]
                A.append([2*(xj - x1), 2*(yj - y1)])
                b.append(d1**2 - dj**2 + (xj**2 - x1**2) + (yj**2 - y1**2))
            A = np.array(A)
            b = np.array(b)
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            cand_x, cand_y = float(sol[0]), float(sol[1])
            # Count inliers
            dists_cand = np.sqrt(np.sum((anchor_array - np.array([cand_x, cand_y]))**2, axis=1))
            residuals_cand = np.abs(dists_cand - distances)
            inlier_mask = residuals_cand <= ransac_threshold
            inliers = int(np.count_nonzero(inlier_mask))
            if inliers > best_inliers or (inliers == best_inliers and best_inliers != -1):
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_xy = (cand_x, cand_y)
                    best_inlier_mask = inlier_mask
                else:
                    # tie-break by total inlier error
                    current_error = float(np.sum(residuals_cand[inlier_mask]))
                    best_error = float(np.sum((np.abs(np.sqrt(np.sum((anchor_array - np.array(best_xy))**2, axis=1)) - distances))[best_inlier_mask]))
                    if current_error < best_error:
                        best_xy = (cand_x, cand_y)
                        best_inlier_mask = inlier_mask
        if best_xy is None:
            return {"error": "RANSAC failed to find a solution."}
        # refine using inliers via non-linear least squares (if any outliers exist)
        if best_inliers < anchor_count:
            in_idx = np.where(best_inlier_mask)[0]
            anchor_in = anchor_array[best_inlier_mask]
            dist_in = distances[best_inlier_mask]
            # refine with non-linear least squares on inliers
            def resid_fn(X):
                return np.sqrt(np.sum((anchor_in - X)**2, axis=1)) - dist_in
            try:
                bounds = (np.min(anchor_in, axis=0) - np.max(dist_in), np.max(anchor_in, axis=0) + np.max(dist_in))
                res_opt = __import__('scipy').optimize.least_squares(resid_fn, x0=np.array(best_xy), bounds=bounds, loss='linear')
                px, py = float(res_opt.x[0]), float(res_opt.x[1])
            except Exception:
                px, py = best_xy
            used_mask = best_inlier_mask
        else:
            px, py = best_xy
            used_mask = np.ones(anchor_count, dtype=bool)
        est_dists = np.sqrt(np.sum((anchor_array[used_mask] - np.array([px, py]))**2, axis=1))
        resids = est_dists - distances[used_mask]
        method_used = 'ransac'
        anchors_used_count = int(np.count_nonzero(used_mask))
        excluded_idx = [int(i) for i in range(anchor_count) if not used_mask[i]]
    else:
        # Non-linear (with optional robust loss)
        # initial guess: use linear estimate on non-outlier anchors
        init_guess = None
        if anchor_count >= 3:
            anchor_sub = anchor_array[~outlier_mask] if np.any(outlier_mask) else anchor_array
            dist_sub = distances[~outlier_mask] if np.any(outlier_mask) else distances
            if anchor_sub.shape[0] >= 3:
                x1, y1 = anchor_sub[0]; d1 = dist_sub[0]
                A = []
                b = []
                for j in range(anchor_sub.shape[0]):
                    if j == 0: continue
                    xj, yj = anchor_sub[j]; dj = dist_sub[j]
                    A.append([2*(xj - x1), 2*(yj - y1)])
                    b.append(d1**2 - dj**2 + (xj**2 - x1**2) + (yj**2 - y1**2))
                A = np.array(A)
                b = np.array(b)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    init_guess = np.array([sol[0], sol[1]], dtype=float)
                except Exception:
                    init_guess = None
        if init_guess is None:
            # fallback to weighted average by inverse distance
            w = 1.0 / (distances + 1e-6)
            init_guess = np.array([np.sum(anchor_array[:,0]*w)/np.sum(w), np.sum(anchor_array[:,1]*w)/np.sum(w)], dtype=float)
        # Define residual function for all anchors
        def resid_all(X):
            return np.sqrt(np.sum((anchor_array - X)**2, axis=1)) - distances
        try:
            ls_res = __import__('scipy').optimize.least_squares(resid_all, x0=init_guess, loss=(robust_loss if chosen_method=='robust' else 'linear'), f_scale=robust_loss_scale)
            px, py = float(ls_res.x[0]), float(ls_res.x[1])
        except Exception as e:
            return {"error": f"Nonlinear optimization failed: {e}"}
        est_dists = np.sqrt(np.sum((anchor_array - np.array([px, py]))**2, axis=1))
        resids = est_dists - distances
        method_used = 'robust' if chosen_method == 'robust' else 'nonlinear'
        anchors_used_count = anchor_count
    # Compute uncertainty and prepare output
    residuals_list = [float(r) for r in resids]
    rmse = math.sqrt(np.mean(np.array(resids)**2)) if len(residuals_list) > 0 else 0.0
    max_error = float(np.max(np.abs(resids))) if len(residuals_list) > 0 else 0.0
    uncertainty_radius = max_error if max_error > 2*rmse else 2*rmse
    xmin = float((px - uncertainty_radius))
    ymin = float((py - uncertainty_radius))
    xmax = float((px + uncertainty_radius))
    ymax = float((py + uncertainty_radius))
    result = {
        "position": (float(px), float(py)),
        "uncertainty_radius": float(uncertainty_radius),
        "uncertainty_bbox": (xmin, ymin, xmax, ymax),
        "anchor_count": int(anchor_count),
        "anchors_used": int(anchors_used_count),
        "residuals": residuals_list,
        "rmse": float(rmse),
        "method_used": method_used
    }
    if method_used == 'ransac' and 'excluded_idx' in locals() and len(excluded_idx) > 0:
        result["excluded_anchors"] = excluded_idx
    return result