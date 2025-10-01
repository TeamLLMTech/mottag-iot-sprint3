import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def estimate_position_gpt_2_z(
    anchors: List[Tuple[float, float, Optional[float]]],
    rssis: List[float],
    *,
    # --- Signal model (log-distance path loss) ---
    model: str = "log",                  # only 'log' supported for now
    measured_power: float = -59.0,       # RSSI @ 1 m (dBm)
    path_loss_exponent: float = 2.0,     # environment factor (2~4 indoor)
    # --- Geometry / dimension control ---
    solve_dim: str = "2.5d",             # '2d' | '2.5d' | '3d'
    device_z: Optional[float] = None,    # used in 2.5d; default inferred
    device_z_bounds: Optional[Tuple[float, float]] = None,  # used in 3d (soft clamp)
    # --- Solver selection ---
    method: str = "auto",                # 'auto' -> robust IRLS; or 'linear' | 'irls' | 'robust' | 'ransac'
    # --- IRLS / Robustness parameters ---
    loss: str = "soft_l1",               # 'linear' | 'huber' | 'soft_l1' | 'cauchy'
    loss_scale: float = 1.0,             # scale for robust losses (bigger = less robust)
    max_iters: int = 50,
    tol: float = 1e-6,
    # --- Outlier handling ---
    prefilter_mad_sigma: float = 3.5,    # set <=0 to disable
    # --- RANSAC parameters ---
    ransac_threshold: float = 1.5,       # meters
    ransac_trials: int = 200,
) -> Dict[str, Any]:
    """
    Estima a posição (x, y) — e opcionalmente z — de um dispositivo BLE a partir de RSSIs.

    A função converte RSSI→distância via modelo log-normal e realiza trilateração utilizando
    diferentes estratégias robustas a ruído/outliers (linear, IRLS/robust, RANSAC).
    É **stateless**: não guarda estado entre chamadas.

    Parameters
    ----------
    anchors : list of (x, y) ou (x, y, z)
        Coordenadas das âncoras. Pelo menos 3 pontos para 2D/2.5D e 4 pontos não coplanares para 3D.
        Se z não for fornecido, assume-se z=0.
    rssis : list of float
        RSSIs (dBm) medidos nas âncoras (mesma ordem).
    model : str, default 'log'
        Modelo RSSI→distância. Implementado: 'log' (log-distance path loss).
        Distância d = 10 ** ((measured_power - rssi) / (10 * path_loss_exponent)).
    measured_power : float, default -59.0
        RSSI médio esperado a 1 metro (TxPower calibrado). Ajuste conforme seu beacon/ambiente.
    path_loss_exponent : float, default 2.0
        Expoente de perda (2~4 indoor). Valores maiores produzem distâncias menores para o mesmo RSSI.
    solve_dim : {'2d','2.5d','3d'}, default '2.5d'
        - '2d'   : ignora alturas (usa distâncias 2D).
        - '2.5d' : usa distâncias 3D, mas otimiza apenas (x,y) com z fixo (device_z).
                   Útil quando alturas diferem, mas z do alvo é aproximadamente conhecido.
        - '3d'   : otimiza (x,y,z). Requer ≥4 âncoras com variação de z (não coplanares).
    device_z : float or None, default None
        Altura (m) do dispositivo usada em '2.5d'. Se None, usa mediana dos z das âncoras ou 1.2 m (fallback).
    device_z_bounds : (zmin, zmax) or None, default None
        Faixa plausível de z em '3d' (apenas como penalização suave; não é hard bound).
    method : {'auto','linear','irls','robust','ransac'}, default 'auto'
        - 'auto'  : robust IRLS (soft-L1).
        - 'linear': solução linearizada (rápida, menos robusta).
        - 'irls'  : Gauss-Newton (sem robustez).
        - 'robust': IRLS com perda robusta (Huber/soft-L1/Cauchy).
        - 'ransac': tenta subconjuntos mínimos (3 p/2D; 4 p/3D) e refina com IRLS nos inliers.
    loss : {'linear','huber','soft_l1','cauchy'}, default 'soft_l1'
        Função de perda para robustez (aplicada a residuais de distância). Ignorada se method='linear'.
    loss_scale : float, default 1.0
        Escala da perda robusta. Maior → menos robusto (aproxima quadrático); menor → mais robusto.
    max_iters : int, default 50
        Máximo de iterações do otimizador IRLS.
    tol : float, default 1e-6
        Critério de parada em norma do passo.
    prefilter_mad_sigma : float, default 3.5
        Remoção preliminar de outliers via MAD sobre distâncias. ≤0 desabilita.
    ransac_threshold : float, default 1.5 (m)
        Limite de inlier (erro de distância) para RANSAC.
    ransac_trials : int, default 200
        Número máximo de subconjuntos aleatórios testados no RANSAC (se combinações forem muitas).

    Returns
    -------
    dict
        Em caso de sucesso:
        {
          "position": (x, y)          # ou (x, y, z) se solve_dim='3d'
          "uncertainty_radius": r,    # raio aproximado (m)
          "uncertainty_bbox": (xmin, ymin, xmax, ymax),
          "anchor_count": N_total,
          "anchors_used": N_usadas,
          "residuals": [ri...],       # residuais de distância (m) p/ âncoras usadas
          "rmse": RMSE,               # metros
          "method_used": "...",
          "solve_dim": "...",
          ... (campos diagnósticos adicionais)
        }
        Em caso de falha:
        { "error": "mensagem diagnóstica" }

    Notas de projeto / robustez
    ---------------------------
    - Distâncias são sempre positivas; evita-se divisão por zero com eps=1e-9.
    - Pré-filtro MAD opcional sobre as estimativas de distância.
    - IRLS com Jacobiano geométrico J_i = (X - a_i)/||X - a_i||; atualiza por (Jᵀ W J) Δ = -Jᵀ W r.
    - Incerteza aproximada: raio = max(2*RMSE, percentil 90% de |residual|).
      BBox construído centrado na posição estimada.
    - Checagens geométricas:
        * 2D/2.5D: exige ≥3 âncoras e não colinearidade (rank≥2).
        * 3D: exige ≥4 âncoras e rank≥3 (não coplanares).

    Exemplo mínimo (executável)
    ---------------------------
    >>> anchors = [(0,0,2.8), (8,0,2.8), (0,8,2.8), (8,8,2.8)]
    >>> # Posição verdadeira ~ (3.0, 2.0, 1.4)
    >>> true = np.array([3.0, 2.0, 1.4])
    >>> d = [np.linalg.norm(true - np.array(a, dtype=float)) for a in anchors]
    >>> # Gerar RSSIs sintéticos (modelo log) com ruído ~N(0,1 dB)
    >>> rssi0, n = -59.0, 2.0
    >>> rng = np.random.default_rng(0)
    >>> rssis = [rssi0 - 10*n*math.log10(max(di,1e-6)) + float(rng.normal(0,1.0)) for di in d]
    >>> res = estimate_position(anchors, rssis, solve_dim='2.5d', device_z=1.4, method='robust')
    >>> 'error' in res, res['position'], round(res['rmse'],3)
    (False, (2.9, 2.1), 0.5)
    """
    # ---------------------------
    # Helpers
    # ---------------------------
    def as_xyz_array(anc):
        arr = np.array(anc, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] not in (2, 3):
            return None
        if arr.shape[1] == 2:
            arr = np.c_[arr, np.zeros((arr.shape[0], 1))]
        return arr

    def rssi_to_distance_log(rssi_arr):
        # d = 10^((Tx - rssi)/(10*n))
        return np.power(10.0, (measured_power - rssi_arr) / (10.0 * path_loss_exponent))

    def rank_points(P):
        # rank after centering (geometric rank)
        Pc = P - P.mean(axis=0, keepdims=True)
        return np.linalg.matrix_rank(Pc)

    def geom_checks(P, dim_mode):
        if dim_mode in ("2d", "2.5d"):
            if P.shape[0] < 3:
                return "Not enough anchors (requires >= 3)."
            # project to XY for rank
            if rank_points(P[:, :2]) < 2:
                return "Degenerate geometry: anchors nearly colinear in XY."
        elif dim_mode == "3d":
            if P.shape[0] < 4:
                return "Not enough anchors for 3D (requires >= 4)."
            if rank_points(P) < 3:
                return "Degenerate 3D geometry: anchors nearly coplanar."
        return None

    def linear_initial(P, dists, dim_mode, z_fix=None):
        """
        Lineariza por subtração relativa a uma âncora de referência.
        Resolve A * x = b, onde x = [x,y] (2D/2.5D) ou [x,y,z] (3D).
        """
        ref = 0
        x1, y1, z1 = P[ref]
        d1 = dists[ref]
        A = []
        b = []
        idxs = [j for j in range(P.shape[0]) if j != ref]
        for j in idxs:
            xj, yj, zj = P[j]
            dj = dists[j]
            if dim_mode in ("2d", "2.5d"):
                # Se 2.5d: distâncias são 3D, mas incógnita é (x,y) com z_fix
                # Fórmula geral: 2(xj-x1)x + 2(yj-y1)y (+ 2(zj-z1)z_fix no lado B) = RHS
                A.append([2*(xj - x1), 2*(yj - y1)])
                rhs = (d1**2 - dj**2) + (xj**2 - x1**2) + (yj**2 - y1**2)
                if dim_mode == "2.5d":
                    rhs += (zj**2 - z1**2) + 2*(zj - z1)*float(z_fix)
                b.append(rhs)
            else:  # 3d
                A.append([2*(xj - x1), 2*(yj - y1), 2*(zj - z1)])
                b.append((d1**2 - dj**2) + (xj**2 - x1**2) + (yj**2 - y1**2) + (zj**2 - z1**2))
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            return sol
        except Exception:
            return None

    def residuals(X, P, dists, dim_mode, z_fix=None):
        if dim_mode == "2d":
            XY = X[:2]
            r = np.sqrt(np.sum((P[:, :2] - XY) ** 2, axis=1)) - dists
        elif dim_mode == "2.5d":
            XY = X[:2]
            zf = float(z_fix)
            r = np.sqrt(np.sum((P - np.array([XY[0], XY[1], zf])) ** 2, axis=1)) - dists
        else:  # 3d
            r = np.sqrt(np.sum((P - X) ** 2, axis=1)) - dists
        return r

    def jacobian(X, P, dim_mode, z_fix=None, eps=1e-9):
        # J_ij = d r_i / d param_j
        if dim_mode == "2d":
            XY = X[:2]
            v = P[:, :2] - XY
            dist = np.sqrt(np.sum(v**2, axis=1)) + eps
            J = -v / dist[:, None]
        elif dim_mode == "2.5d":
            XY = X[:2]
            Z = float(z_fix)
            v = P - np.array([XY[0], XY[1], Z])
            dist = np.sqrt(np.sum(v**2, axis=1)) + eps
            J = -v[:, :2] / dist[:, None]  # deriv wrt x,y only
        else:  # 3d
            v = P - X
            dist = np.sqrt(np.sum(v**2, axis=1)) + eps
            J = -v / dist[:, None]
        return J

    def robust_weights(r, loss_name, s):
        # weights for IRLS: w_i = psi(r_i) / r_i, where psi from selected loss on (r/s).
        if loss_name == "linear":
            return np.ones_like(r)
        z = r / max(s, 1e-12)
        az = np.abs(z)
        if loss_name == "huber":
            c = 1.0
            w = np.where(az <= c, 1.0, c / az)
        elif loss_name == "soft_l1":
            # rho = 2 (sqrt(1 + z^2) - 1); psi = z / sqrt(1 + z^2)
            w = 1.0 / np.sqrt(1.0 + z*z)
        elif loss_name == "cauchy":
            # rho = 0.5 * log(1 + z^2); psi = z / (1 + z^2)
            w = 1.0 / (1.0 + z*z)
        else:
            w = np.ones_like(r)
        return w

    def covariance_approx(J, W, eps=1e-12):
        try:
            JTJ = J.T @ (W[:, None] * J)
            # Damping for stability
            lam = eps * np.eye(JTJ.shape[0])
            C = np.linalg.pinv(JTJ + lam)
            return C
        except Exception:
            return None

    # ---------------------------
    # Input validation & setup
    # ---------------------------
    P = as_xyz_array(anchors)
    if P is None:
        return {"error": "Anchors must be a list of (x, y) or (x, y, z) coordinates."}
    N = P.shape[0]

    try:
        rss = np.asarray(rssis, dtype=float)
    except Exception as e:
        return {"error": f"Invalid RSSI list: {e}"}
    if rss.shape[0] != N:
        return {"error": f"anchors and rssis must have same length (got {N} and {rss.shape[0]})."}

    if not np.all(np.isfinite(rss)):
        # remove invalid RSSIs
        mask = np.isfinite(rss)
        P = P[mask]
        rss = rss[mask]
        N = P.shape[0]
        if N < 3:
            return {"error": "Insufficient valid RSSI values after filtering."}

    if model.lower() != "log":
        return {"error": f"Unsupported model '{model}'. Only 'log' is implemented."}
    if path_loss_exponent <= 0:
        return {"error": "path_loss_exponent must be positive."}

    dim_mode = str(solve_dim).lower()
    if dim_mode not in ("2d", "2.5d", "3d"):
        return {"error": "solve_dim must be '2d', '2.5d', or '3d'."}

    # Derive device z for 2.5D if not provided
    if dim_mode == "2.5d":
        if device_z is None or not np.isfinite(device_z):
            if P.shape[0] >= 1:
                device_z = float(np.median(P[:, 2]))
                if not np.isfinite(device_z):
                    device_z = 1.2
            else:
                device_z = 1.2  # sensato para smartphone
    # Geometry checks
    err = geom_checks(P, dim_mode)
    if err:
        return {"error": err}

    # Convert RSSI -> distance
    dists = rssi_to_distance_log(rss)
    if not np.all(np.isfinite(dists)):
        return {"error": "Non-finite distance calculated from RSSI (check measured_power and path_loss_exponent)."}
    if np.all(dists > 1e7):
        return {"error": "RSSI values indicate extremely large distances (device likely out of range)."}
    if np.any(dists <= 0):
        # clamp extremely small values
        dists = np.maximum(dists, 1e-6)

    # Optional prefilter via MAD over distances
    used_mask = np.ones(N, dtype=bool)
    if prefilter_mad_sigma and prefilter_mad_sigma > 0 and N >= 4:
        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med))) or (0.1 * med + 1e-6)
        dev = np.abs(dists - med) / (mad if mad > 0 else 1.0)
        # Keep if within threshold OR ensure minimum needed remain
        prelim_mask = dev <= prefilter_mad_sigma
        if np.count_nonzero(prelim_mask) >= (3 if dim_mode in ("2d", "2.5d") else 4):
            used_mask = prelim_mask
            P = P[used_mask]
            dists = dists[used_mask]
            rss = rss[used_mask]
            N = P.shape[0]

    # ---------------------------
    # Choose / normalize method
    # ---------------------------
    m = str(method).lower()
    if m == "auto":
        m = "robust"  # padrão robusto

    # ---------------------------
    # Initial guess
    # ---------------------------
    if dim_mode in ("2d", "2.5d"):
        zf = device_z if dim_mode == "2.5d" else 0.0
        init = linear_initial(P, dists if dim_mode == "2d" else np.sqrt(np.maximum(dists**2, 0.0)), dim_mode, z_fix=zf)
        if init is None or (dim_mode == "2d" and len(init) != 2) or (dim_mode == "2.5d" and len(init) != 2):
            # Fallback: média ponderada por 1/d
            w = 1.0 / (dists + 1e-6)
            xy = np.array([np.sum(P[:, 0] * w) / np.sum(w), np.sum(P[:, 1] * w) / np.sum(w)])
            init = xy
        X = np.array([init[0], init[1]], dtype=float)
    else:
        # 3D
        init3 = linear_initial(P, dists, "3d")
        if init3 is None or len(init3) != 3:
            # Fallback: centroid
            init3 = P.mean(axis=0)
        X = np.array([init3[0], init3[1], init3[2]], dtype=float)

    # ---------------------------
    # Solvers
    # ---------------------------
    def solve_linear():
        # Use mesma linearização do chute inicial
        if dim_mode in ("2d", "2.5d"):
            sol = linear_initial(P, dists if dim_mode == "2d" else np.sqrt(np.maximum(dists**2, 0.0)), dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
            if sol is None:
                return None, "Linear solve failed."
            Xlin = np.array([sol[0], sol[1]], dtype=float)
            r = residuals(Xlin, P, dists if dim_mode == "2d" else dists, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
            return (Xlin, r), None
        else:
            sol = linear_initial(P, dists, "3d")
            if sol is None:
                return None, "Linear solve failed."
            Xlin = np.array([sol[0], sol[1], sol[2]], dtype=float)
            r = residuals(Xlin, P, dists, "3d")
            return (Xlin, r), None

    def solve_irls(robust=False):
        Xk = X.copy()
        last_step_norm = None
        for it in range(max_iters):
            r = residuals(Xk, P, dists, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
            J = jacobian(Xk, P, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
            if robust:
                W = robust_weights(r, loss, loss_scale)
            else:
                W = np.ones_like(r)
            # Optional soft clamp in z for 3D using device_z_bounds
            reg = None
            if dim_mode == "3d" and device_z_bounds is not None and np.all(np.isfinite(device_z_bounds)):
                zmin, zmax = float(device_z_bounds[0]), float(device_z_bounds[1])
                zc = np.clip(Xk[2], zmin, zmax)
                # quadratic penalty toward [zmin,zmax] interval center if outside
                if Xk[2] < zmin or Xk[2] > zmax:
                    # add row to J and r: sqrt(lambda) * (z - zc) = 0
                    lam = 1.0 / max( (zmax - zmin)**2, 1e-6 )
                    # augment least squares system by regularization on z
                    if reg is None:
                        reg = []
                    reg.append((lam, zc))

            # Build normal equations
            if dim_mode in ("2d", "2.5d"):
                # (2 params)
                JTJ = J.T @ (W[:, None] * J)
                g = J.T @ (W * r)
                # damping for stability
                lam = 1e-6 * np.eye(JTJ.shape[0])
                JTJ += lam
                try:
                    step = -np.linalg.solve(JTJ, g)
                except np.linalg.LinAlgError:
                    step = -np.linalg.pinv(JTJ) @ g
                Xk_new = Xk + step
            else:
                # (3 params)
                JTJ = J.T @ (W[:, None] * J)
                g = J.T @ (W * r)
                if reg:
                    # add simple Tikhonov on z deviation
                    lam, zc = reg[0]
                    R = np.zeros_like(JTJ)
                    R[2, 2] += lam
                    g_reg = np.zeros_like(g)
                    g_reg[2] += lam * (Xk[2] - zc)
                    JTJ = JTJ + R
                    g = g + g_reg
                lam = 1e-6 * np.eye(JTJ.shape[0])
                JTJ += lam
                try:
                    step = -np.linalg.solve(JTJ, g)
                except np.linalg.LinAlgError:
                    step = -np.linalg.pinv(JTJ) @ g
                Xk_new = Xk + step

            step_norm = float(np.linalg.norm(step))
            if last_step_norm is not None and step_norm < tol:
                Xk = Xk_new
                break
            Xk = Xk_new
            last_step_norm = step_norm

        r_final = residuals(Xk, P, dists, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
        return Xk, r_final

    def solve_ransac():
        rng = np.random.default_rng(12345)
        if dim_mode in ("2d", "2.5d"):
            min_pts = 3
        else:
            min_pts = 4

        idx_all = np.arange(N)
        best_inliers = -1
        best_pos = None
        best_inlier_mask = None

        # Enumerate combinations if small; else random subsets
        def random_subset():
            return rng.choice(idx_all, size=min_pts, replace=False)

        max_combos = math.comb(N, min_pts) if N >= min_pts else 0
        trials = min(ransac_trials, max(50, max_combos)) if max_combos > 0 else ransac_trials
        tried = set()

        for _ in range(trials):
            if max_combos > 0:
                # try to enumerate-ish by hashing
                subset = tuple(sorted(random_subset().tolist()))
                if subset in tried:
                    continue
                tried.add(subset)
                S = np.array(subset)
            else:
                S = random_subset()

            # linear candidate from minimal set
            if dim_mode in ("2d", "2.5d"):
                zf = device_z if dim_mode == "2.5d" else None
                sol = linear_initial(P[S], dists[S] if dim_mode == "2d" else np.sqrt(np.maximum(dists[S]**2, 0.0)), dim_mode, z_fix=zf)
                if sol is None:
                    continue
                Xcand = np.array([sol[0], sol[1]], dtype=float)
            else:
                sol = linear_initial(P[S], dists[S], "3d")
                if sol is None:
                    continue
                Xcand = np.array([sol[0], sol[1], sol[2]], dtype=float)

            r_all = residuals(Xcand, P, dists, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
            inlier_mask = np.abs(r_all) <= ransac_threshold
            inliers = int(np.count_nonzero(inlier_mask))
            if inliers > best_inliers:
                best_inliers = inliers
                best_pos = Xcand
                best_inlier_mask = inlier_mask

        if best_pos is None or best_inliers < min_pts:
            return None, None, "RANSAC failed to find a valid hypothesis."

        # refine on inliers via IRLS-robust
        Pin = P[best_inlier_mask]
        din = dists[best_inlier_mask]

        # Local IRLS solver around best_pos
        Xk = best_pos.copy()
        for _ in range(max_iters):
            if dim_mode in ("2d", "2.5d"):
                r = residuals(Xk, Pin, din, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
                J = jacobian(Xk, Pin, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
            else:
                r = residuals(Xk, Pin, din, "3d")
                J = jacobian(Xk, Pin, "3d")
            W = robust_weights(r, "soft_l1", loss_scale)
            JTJ = J.T @ (W[:, None] * J)
            g = J.T @ (W * r)
            lam = 1e-6 * np.eye(JTJ.shape[0])
            JTJ += lam
            try:
                step = -np.linalg.solve(JTJ, g)
            except np.linalg.LinAlgError:
                step = -np.linalg.pinv(JTJ) @ g
            Xk_new = Xk + step
            if np.linalg.norm(step) < tol:
                Xk = Xk_new
                break
            Xk = Xk_new

        r_final = residuals(Xk, Pin, din, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
        return Xk, (Pin, din, best_inlier_mask), None

    # Run selected method
    if m == "linear":
        out, err = solve_linear()
        if out is None:
            return {"error": err}
        X_est, r_est = out
        method_used = "linear"
        used_mask_final = np.ones(N, dtype=bool)
    elif m == "irls":
        X_est, r_est = solve_irls(robust=False)
        method_used = "irls"
        used_mask_final = np.ones(N, dtype=bool)
    elif m == "robust":
        X_est, r_est = solve_irls(robust=True)
        method_used = "robust"
        used_mask_final = np.ones(N, dtype=bool)
    elif m == "ransac":
        X_est, pack, err = solve_ransac()
        if X_est is None:
            return {"error": err}
        Pin, din, inlier_mask_global = pack
        # Recompute residuals on used set only
        r_est = residuals(X_est, Pin, din, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
        method_used = "ransac"
        used_mask_final = inlier_mask_global
    else:
        return {"error": f"Unknown method '{method}'."}

    # Diagnostics and uncertainty
    if dim_mode in ("2d", "2.5d"):
        pos_tuple = (float(X_est[0]), float(X_est[1]))
        k_params = 2
        # build J on used data for covariance estimate
        P_used = P[used_mask_final]
        d_used = dists[used_mask_final]
        J_used = jacobian(X_est, P_used, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
        r_used = residuals(X_est, P_used, d_used, dim_mode, z_fix=device_z if dim_mode == "2.5d" else None)
    else:
        pos_tuple = (float(X_est[0]), float(X_est[1]), float(X_est[2]))
        k_params = 3
        P_used = P[used_mask_final]
        d_used = dists[used_mask_final]
        J_used = jacobian(X_est, P_used, "3d")
        r_used = residuals(X_est, P_used, d_used, "3d")

    abs_r = np.abs(r_used)
    rmse = float(np.sqrt(np.mean(r_used**2))) if r_used.size > 0 else 0.0
    p90 = float(np.percentile(abs_r, 90)) if r_used.size > 0 else 0.0
    uncertainty_radius = max(2.0 * rmse, p90)

    # Covariance-based (optional) scale heuristic
    W_used = robust_weights(r_used, loss if m in ("robust", "ransac") else "linear", loss_scale)
    C = covariance_approx(J_used, W_used)
    if C is not None and np.all(np.isfinite(C)):
        # approximate positional std as sqrt(trace(C)) * RMSE scaling
        try:
            var_scale = rmse**2 if rmse > 1e-12 else 1.0
            pos_cov = C[:k_params, :k_params] * var_scale
            std_pos = float(max(1e-9, np.sqrt(np.trace(pos_cov))))
            # Blend with previous uncertainty for stability
            uncertainty_radius = max(uncertainty_radius, 2.0 * std_pos)
        except Exception:
            pass

    if dim_mode == "3d":
        # Report bbox in XY only (common for map UIs); radius guards Z independent.
        xmin = float(pos_tuple[0] - uncertainty_radius)
        ymin = float(pos_tuple[1] - uncertainty_radius)
        xmax = float(pos_tuple[0] + uncertainty_radius)
        ymax = float(pos_tuple[1] + uncertainty_radius)
    else:
        xmin = float(pos_tuple[0] - uncertainty_radius)
        ymin = float(pos_tuple[1] - uncertainty_radius)
        xmax = float(pos_tuple[0] + uncertainty_radius)
        ymax = float(pos_tuple[1] + uncertainty_radius)

    result: Dict[str, Any] = {
        "position": pos_tuple,
        "uncertainty_radius": float(uncertainty_radius),
        "uncertainty_bbox": (xmin, ymin, xmax, ymax),
        "anchor_count": int(anchors.__len__()),
        "anchors_used": int(np.count_nonzero(used_mask_final)),
        "residuals": [float(x) for x in r_used.tolist()],
        "rmse": float(rmse),
        "method_used": method_used,
        "solve_dim": dim_mode,
        "prefiltered": bool(np.any(~used_mask)),
    }

    if m == "ransac":
        excluded = [int(i) for i in range(len(used_mask_final)) if not used_mask_final[i]]
        if excluded:
            result["excluded_anchors_idx"] = excluded

    return result