"""
Trilateration robusta no espaço de RSSI (MLE), adequada a execução stateless
(p.ex. funções serverless). Inclui:

- Ajuste direto no modelo log-distância em dBm (evita viés de RSSI->dist)
- Multi-start (vários chutes) e perda robusta (Cauchy/soft_l1)
- Estimação opcional de A (measured_power @1m) e n (expoente de perda)
- Filtragem de beacons fracos e pesos consistentes com variância
- Clamps numéricos (RSSI floor, distância mínima/máxima)
- Diagnóstico de incerteza (covariância, DRMS, CEP50, R95) por chamada
- RANSAC leve opcional para lidar com outliers severos/NLOS

Dependências: numpy, scipy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, List
import numpy as np
from scipy.optimize import least_squares


@dataclass
class EstimateResult:
    x: float
    y: float
    success: bool
    n_iter: int
    cost: float  # 0.5 * ||resid||^2 (pós-peso)
    A_est: Optional[float]
    n_est: Optional[float]
    # incertezas (metros)
    drms_m: Optional[float]
    cep50_m: Optional[float]
    r95_m: Optional[float]
    # qualidade
    n_used_beacons: int
    n_inliers: int
    median_abs_resid_dbm: float
    max_abs_resid_dbm: float
    # depuração
    message: str


class TrilaterationRSSIMLE:
    def __init__(
        self,
        beacons: Iterable[Tuple[float, float]],
        measured_power: float = -69.0,  # A @1m (dBm)
        path_loss_exponent: float = 1.8,  # n
        bounds_xy: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> None:
        self.beacons = np.asarray(beacons, dtype=float)
        if self.beacons.ndim != 2 or self.beacons.shape[1] != 2:
            raise ValueError("beacons deve ser lista/array de (x,y)")
        self.A = float(measured_power)
        self.n = float(path_loss_exponent)
        self.bounds_xy = bounds_xy

    # ---------------------------- utilidades ---------------------------- #
    def _normalize(self):
        c = self.beacons.mean(axis=0)
        B = self.beacons - c
        scale = max(np.linalg.norm(B, axis=1).mean(), 1e-6)
        return c, scale, B / scale

    @staticmethod
    def _weights_from_quality_or_rssi(rssis: np.ndarray, quality: Optional[np.ndarray]) -> np.ndarray:
        if quality is not None:
            w = np.asarray(quality, dtype=float)
            w = np.maximum(w, 1e-9)
            return w / w.max()
        # Heurística: RSSI mais alto = menor variância ⇒ mais peso.
        # Normaliza para [0.1, 1.0]
        r_min, r_max = float(np.min(rssis)), float(np.max(rssis))
        rng = max(r_max - r_min, 1e-3)
        w = (rssis - r_min) / rng
        return 0.1 + 0.9 * np.clip(w, 0.0, 1.0)

    @staticmethod
    def _make_multistart_grid(num: int) -> np.ndarray:
        # pontos em coords normalizadas (~unidade). Inclui (0,0) e cantos
        base = np.array([
            [0.0, 0.0],
            [ 0.5,  0.5], [ 0.5, -0.5], [-0.5,  0.5], [-0.5, -0.5],
            [ 0.75, 0.0], [-0.75, 0.0], [0.0, 0.75], [0.0, -0.75],
            [ 0.25, 0.0], [0.0, 0.25], [-0.25, 0.0], [0.0, -0.25],
        ])
        if num <= len(base):
            return base[:num]
        # amostra radial extra
        k = num - len(base)
        angles = np.linspace(0, 2*np.pi, k, endpoint=False)
        extra = np.c_[0.6*np.cos(angles), 0.6*np.sin(angles)]
        return np.vstack([base, extra])

    # --------------------------- estimativa ---------------------------- #
    def estimate(
        self,
        rssis: Iterable[float],
        quality: Optional[Iterable[float]] = None,
        *,
        estimate_A: bool = True,
        estimate_n: bool = False,
        loss: str = "cauchy",  # "soft_l1" também é bom
        f_scale: float = 2.0,
        max_nfev: int = 400,
        min_good_rssi_dbm: float = -90.0,
        min_beacons_after_filter: int = 4,
        rssi_floor_dbm: float = -95.0,
        d_min_max_m: Tuple[float, float] = (0.5, 50.0),
        multistart: int = 8,
        ransac_max_trials: int = 0,
        ransac_subset_size: Optional[int] = None,
        rssi_inlier_threshold_dbm: float = 6.0,
        random_seed: Optional[int] = None,
    ) -> EstimateResult:
        """
        Calcula posição por MLE no espaço de RSSI (dBm).

        Parâmetros chave:
          - estimate_A/estimate_n: estimar A (@1m) e/ou n junto com posição
          - multistart: número de chutes de inicialização
          - ransac_max_trials: >0 ativa RANSAC leve (iterações)
          - rssi_floor_dbm: piso para RSSI antes do ajuste
          - d_min_max_m: clamp de distância [d_min, d_max] nas iterações
        """
        rng = np.random.default_rng(random_seed)
        rssis = np.asarray(list(rssis), dtype=float)
        if rssis.shape[0] != self.beacons.shape[0]:
            raise ValueError("Número de RSSIs difere do número de beacons")
        # aplica piso para reduzir explosões numéricas
        rssis = np.maximum(rssis, rssi_floor_dbm)
        quality_arr = None if quality is None else np.asarray(list(quality), dtype=float)

        # 1) filtro de beacons muito fracos, se possível
        mask = np.ones_like(rssis, dtype=bool)
        if rssis.size >= min_beacons_after_filter:
            strong = rssis > min_good_rssi_dbm
            if strong.sum() >= min_beacons_after_filter:
                mask = strong
        B_all = self.beacons
        B = B_all[mask]
        r_used = rssis[mask]
        q_used = None if quality_arr is None else quality_arr[mask]
        n_used = B.shape[0]
        if n_used < 3:
            raise ValueError("Beacons insuficientes após filtro (mín. 3)")

        # pesos iniciais por RSSI/quality
        w = self._weights_from_quality_or_rssi(r_used, q_used)
        wsqrt = np.sqrt(w)

        # normalização de geometria
        c, scale, Bn = self._norm_from_custom_beacons(B)

        # bounds dos parâmetros
        lb_xy = np.array([-np.inf, -np.inf])
        ub_xy = np.array([ np.inf,  np.inf])
        if self.bounds_xy is not None:
            (xmin, ymin), (xmax, ymax) = self.bounds_xy
            bmin = (np.array([xmin, ymin]) - c) / scale
            bmax = (np.array([xmax, ymax]) - c) / scale
            lb_xy = np.maximum(lb_xy, bmin)
            ub_xy = np.minimum(ub_xy, bmax)

        theta0_base: List[float] = []
        theta_lb: List[float] = []
        theta_ub: List[float] = []

        # chute inicial: centróide ponderado em coords normalizadas
        w_pos = w / w.sum()
        p0n = (Bn * w_pos[:, None]).sum(axis=0)

        def make_theta0(p_guess_n: np.ndarray) -> np.ndarray:
            t = [p_guess_n[0], p_guess_n[1]]
            if estimate_A:
                t.append(self.A)
            if estimate_n:
                t.append(self.n)
            return np.array(t, dtype=float)

        theta_lb = [lb_xy[0], lb_xy[1]]
        theta_ub = [ub_xy[0], ub_xy[1]]
        if estimate_A:
            theta_lb.append(self.A - 10.0)
            theta_ub.append(self.A + 10.0)
        if estimate_n:
            theta_lb.append(1.2)
            theta_ub.append(3.5)
        theta_lb = np.asarray(theta_lb, float)
        theta_ub = np.asarray(theta_ub, float)

        d_min, d_max = float(d_min_max_m[0]), float(d_min_max_m[1])

        def residuals(theta: np.ndarray) -> np.ndarray:
            x_n, y_n = theta[0], theta[1]
            A_hat = theta[2] if estimate_A else self.A
            if estimate_A and estimate_n:
                n_hat = theta[3]
            elif (not estimate_A) and estimate_n:
                n_hat = theta[2]
            else:
                n_hat = self.n

            p = np.array([x_n, y_n]) * scale + c  # volta para escala original
            d = np.linalg.norm(p - B, axis=1)
            d = np.clip(d, d_min, d_max)
            rssi_pred = A_hat - 10.0 * n_hat * np.log10(d)
            return wsqrt * (rssi_pred - r_used)

        # -------------------- Multi-start (e opcional RANSAC) -------------------- #
        starts = self._make_multistart_grid(multistart)
        starts[0] = p0n  # garante que inclui o chute do centróide

        def solve_from(pn: np.ndarray) -> Dict[str, Any]:
            theta0 = make_theta0(pn)
            res = least_squares(
                residuals,
                x0=theta0,
                bounds=(theta_lb, theta_ub),
                loss=loss,
                f_scale=f_scale,
                max_nfev=max_nfev,
            )
            # métricas básicas
            # res.cost = 0.5 * sum(resid^2)
            abs_res = np.abs(res.fun / wsqrt)  # tira o peso para métrica em dBm
            med_abs = float(np.median(abs_res))
            max_abs = float(np.max(abs_res))

            # inliers por threshold em dBm (não ponderado)
            inliers = int(np.sum(abs_res <= rssi_inlier_threshold_dbm))

            # incerteza somente para (x,y)
            drms = cep50 = r95 = None
            A_out = None
            n_out = None
            try:
                J = res.jac  # shape: N x P
                N = len(r_used)
                P = len(res.x)
                sigma2 = 2.0 * res.cost / max(N - P, 1)
                JTJ = J.T @ J
                cov_full = np.linalg.pinv(JTJ) * sigma2
                cov_xy = cov_full[:2, :2]
                varx, vary = float(cov_xy[0, 0]), float(cov_xy[1, 1])
                # voltar variâncias para escala original (coords estavam normalizadas)
                # Var(x_orig) = (scale^2) * Var(x_n)
                varx *= (scale ** 2)
                vary *= (scale ** 2)
                # assumindo correlação moderada, usa sigma_eff pela média
                sigma_eff = np.sqrt(0.5 * (varx + vary))
                drms = float(np.sqrt(varx + vary))
                cep50 = float(1.17741 * sigma_eff)  # mediana do Rayleigh
                r95 = float(2.44775 * sigma_eff)    # ~chi2 95% 2D
            except Exception:
                pass

            x_n, y_n = res.x[0], res.x[1]
            p_est = np.array([x_n, y_n]) * scale + c

            if estimate_A:
                A_out = float(res.x[2])
            if estimate_n and estimate_A:
                n_out = float(res.x[3])
            elif estimate_n and (not estimate_A):
                n_out = float(res.x[2])

            return dict(
                x=float(p_est[0]),
                y=float(p_est[1]),
                cost=float(res.cost),
                success=bool(res.success),
                n_iter=int(res.nfev),
                med_abs_dbm=med_abs,
                max_abs_dbm=max_abs,
                inliers=inliers,
                drms=drms,
                cep50=cep50,
                r95=r95,
                A_est=A_out,
                n_est=n_out,
                message=str(res.message),
            )

        def run_multistart() -> Dict[str, Any]:
            best = None
            for pn in starts:
                sol = solve_from(pn)
                if (best is None) or (sol["cost"] < best["cost"]):
                    best = sol
            return best

        def run_ransac() -> Dict[str, Any]:
            if ransac_max_trials <= 0 or n_used < 4:
                return run_multistart()
            best = None
            k = ransac_subset_size or max(4, min(5, n_used))
            idx_all = np.arange(n_used)
            trials = int(ransac_max_trials)
            for _ in range(trials):
                idx = rng.choice(idx_all, size=k, replace=False)
                # salva contexto e reduz arrays
                B_sub = B.copy()
                r_sub = r_used.copy()
                w_sub = w.copy()
                B_loc = B_sub[idx]
                r_loc = r_sub[idx]
                w_loc = w_sub[idx]

                # funções locais temporárias
                c_loc, scale_loc, _ = self._norm_from_custom_beacons(B_loc)

                def residuals_loc(theta: np.ndarray) -> np.ndarray:
                    x_n, y_n = theta[0], theta[1]
                    A_hat = theta[2] if estimate_A else self.A
                    if estimate_A and estimate_n:
                        n_hat = theta[3]
                    elif (not estimate_A) and estimate_n:
                        n_hat = theta[2]
                    else:
                        n_hat = self.n
                    p = np.array([x_n, y_n]) * scale_loc + c_loc
                    d = np.linalg.norm(p - B_loc, axis=1)
                    d = np.clip(d, d_min, d_max)
                    rssi_pred = A_hat - 10.0 * n_hat * np.log10(d)
                    return np.sqrt(w_loc) * (rssi_pred - r_loc)

                # chutes locais
                c_loc2, scale_loc2, Bn_loc = self._norm_from_custom_beacons(B_loc)
                w_pos_loc = w_loc / w_loc.sum()
                p0n_loc = (Bn_loc * w_pos_loc[:, None]).sum(axis=0)
                starts_loc = self._make_multistart_grid(max(4, multistart // 2))
                starts_loc[0] = p0n_loc
                theta_lb_loc = [ -np.inf, -np.inf ]
                theta_ub_loc = [  np.inf,  np.inf ]
                if estimate_A:
                    theta_lb_loc.append(self.A - 10.0)
                    theta_ub_loc.append(self.A + 10.0)
                if estimate_n:
                    theta_lb_loc.append(1.2)
                    theta_ub_loc.append(3.5)
                theta_lb_loc = np.asarray(theta_lb_loc, float)
                theta_ub_loc = np.asarray(theta_ub_loc, float)

                best_loc = None
                for pn in starts_loc:
                    theta0 = make_theta0(pn)
                    res = least_squares(
                        residuals_loc, x0=theta0, bounds=(theta_lb_loc, theta_ub_loc),
                        loss=loss, f_scale=f_scale, max_nfev=max_nfev//2
                    )
                    # avalia em TODOS os pontos originais
                    cand = solve_from(pn)  # reusa solve_from com todos os beacons
                    if (best_loc is None) or (cand["cost"] < best_loc["cost"]):
                        best_loc = cand
                if (best is None) or (best_loc["cost"] < best["cost"]):
                    best = best_loc
            return best if best is not None else run_multistart()

        best = run_ransac() if ransac_max_trials > 0 else run_multistart()

        return EstimateResult(
            x=best["x"],
            y=best["y"],
            success=True,
            n_iter=best["n_iter"],
            cost=best["cost"],
            A_est=best["A_est"],
            n_est=best["n_est"],
            drms_m=best["drms"],
            cep50_m=best["cep50"],
            r95_m=best["r95"],
            n_used_beacons=n_used,
            n_inliers=best["inliers"],
            median_abs_resid_dbm=best["med_abs_dbm"],
            max_abs_resid_dbm=best["max_abs_dbm"],
            message=best["message"],
        )

    # versão de _normalize que aceita beacons arbitrários
    @staticmethod
    def _norm_from_custom_beacons(B: np.ndarray):
        c = B.mean(axis=0)
        BC = B - c
        scale = max(np.linalg.norm(BC, axis=1).mean(), 1e-6)
        return c, scale, BC / scale