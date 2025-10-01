import numpy as np
from scipy.optimize import least_squares

class Trilateration3:
    def __init__(self, beacons, measured_power=-69.0, path_loss_exponent=1.8,
                 bounds_xy=None):
        """
        beacons: lista de tuplas (x, y)
        measured_power: RSSI @ 1m (dBm) calibrado para seu TX/RX
        path_loss_exponent: expoente de perda calibrado por ambiente
        bounds_xy: ((xmin, ymin), (xmax, ymax)) ou None
        """
        self.beacons = np.asarray(beacons, dtype=float)
        self.measured_power = float(measured_power)
        self.n = float(path_loss_exponent)
        self.bounds_xy = bounds_xy

    def rssi_to_distance(self, rssi):
        # Modelo log-distância: d = 10^((A - RSSI)/(10 * n)), A = measured_power @1m
        return 10 ** ((self.measured_power - rssi) / (10.0 * self.n))

    def _normalize(self):
        # translada p/ centróide e escala p/ tamanho típico
        c = self.beacons.mean(axis=0)
        B = self.beacons - c
        scale = max(np.linalg.norm(B, axis=1).mean(), 1e-6)
        return c, scale, B / scale

    def estimate(self, rssis, quality=None):
        """
        rssis: lista de RSSI (dBm) na mesma ordem dos beacons
        quality: opcional, pesos de qualidade por leitura (maior=melhor).
                 Se None, usa heurística baseada no próprio RSSI.
        Retorna (x, y)
        """
        rssis = np.asarray(rssis, dtype=float)
        if rssis.shape[0] != self.beacons.shape[0]:
            raise ValueError("Número de RSSIs difere do número de beacons")

        d_hat = np.array([self.rssi_to_distance(r) for r in rssis])

        # pesos (maior peso para sinais mais fortes ou qualidade informada)
        if quality is None:
            # heurística: w = 1 / (k + d_hat) — menos peso para distâncias maiores
            w = 1.0 / (1.0 + d_hat)
        else:
            w = np.asarray(quality, dtype=float)
            w = np.maximum(w, 1e-6)
            w = w / w.max()

        # normalização geométrica
        c, scale, Bn = self._normalize()

        # chute inicial: centróide ponderado
        w_pos = w / w.sum()
        p0 = (Bn * w_pos[:, None]).sum(axis=0)
        s0 = 1.0  # fator de escala inicial

        def residuals(theta):
            x, y, s = theta
            p = np.array([x, y])
            # resíduos em metros normalizados: ||p - bi|| - s * (d_i / scale)
            ri = np.linalg.norm(p - Bn, axis=1) - s * (d_hat / scale)
            return np.sqrt(w) * ri  # WLS

        # bounds (opcionais) em xy se fornecido; s>=0
        if self.bounds_xy is not None:
            (xmin, ymin), (xmax, ymax) = self.bounds_xy
            # normaliza bounds
            bmin = (np.array([xmin, ymin]) - c) / scale
            bmax = (np.array([xmax, ymax]) - c) / scale
            lb = np.array([bmin[0], bmin[1], 0.0])
            ub = np.array([bmax[0], bmax[1], 10.0])
        else:
            lb = np.array([-np.inf, -np.inf, 0.0])
            ub = np.array([ np.inf,  np.inf, 10.0])

        res = least_squares(
            residuals,
            x0=np.array([p0[0], p0[1], s0]),
            bounds=(lb, ub),
            loss="soft_l1",   # robusto a outliers
            f_scale=1.0,      # sensibilidade da perda robusta
            max_nfev=200
        )

        x_n, y_n, s = res.x
        # desscale p/ coordenadas originais
        p_est = np.array([x_n, y_n]) * scale + c
        return float(p_est[0]), float(p_est[1])
