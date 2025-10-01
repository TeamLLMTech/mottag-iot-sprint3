from typing import List, Sequence, Dict, Any, Optional, Union
import numpy as np
import math

def rssi_filter_ukf_1(rssis: Sequence[Union[float, Sequence[Optional[float]]]], **kwargs) -> Dict[str, Any]:
    """
    Filtra uma série temporal de medidas de RSSI (em dBm) usando um Filtro de Kalman Não-Linear do tipo Unscented (UKF),
    com mecanismos robustos a ruído, outliers e inconsistências entre antenas (bias por antena).
    
    A função é **stateless**: não mantém estado entre execuções. Cada chamada é independente.
    
    Parâmetros
    ----------
    rssis : Sequence[float] | Sequence[Sequence[float | None]]
        Série temporal de RSSI em dBm. Pode ser:
        - 1D (ex.: [-70, -72, None, -68, ...]) para uma única antena; ou
        - 2D, onde cada item no tempo é um vetor de medidas de múltiplas antenas
          (ex.: [[-70, -72], [-69, None], [-71, -73], ...]). Valores `None`/NaN são tratados como ausentes.
          A ordem é por timestamp (primeiro índice é o tempo).
    
    Parâmetros opcionais (kwargs)
    -----------------------------
    dt : float, default=1.0
        Passo temporal (constante) entre amostras.
    alpha : float, default=0.5
        Parâmetro do UKF para espalhamento dos sigma points (típico 1e-3 a 1). Valores maiores aumentam a dispersão.
    beta : float, default=2.0
        Parâmetro do UKF. Para distribuições quase-Gaussianas recomenda-se beta=2.
    kappa : float, default=0.0
        Parâmetro do UKF (escala dos sigma points). Geralmente 0 para muitos casos.
    q_level : float, default=1.0
        Variância do ruído de processo do nível (dBm^2). Controla quão rápido o nível pode variar.
    q_drift : float, default=0.01
        Variância do ruído de processo do drift (dBm^2 por passo). Controla variação do drift.
    q_bias : float, default=0.1
        Variância do ruído de processo dos vieses por antena (dBm^2). Maior => vieses mais "móveis".
    r_meas : float | Sequence[float], default=4.0
        Variância base do ruído de medida (dBm^2). Pode ser escalar (aplicado a todas as antenas) ou
        um vetor por antena.
    drift_damping : float, default=0.05
        Fator de amortecimento do drift por passo (0..1). 0=sem amortecimento; 0.05 reduz lentamente o drift.
    bias_forget : float, default=0.01
        Fator de esquecimento do viés por antena (0..1). Ajuda a acomodar inconsistências entre antenas ao longo do tempo.
    sat_min : float, default=-120.0
        Limite inferior físico/plausível do RSSI.
    sat_max : float, default=-20.0
        Limite superior físico/plausível do RSSI.
    huber_delta : float, default=1.5
        Parâmetro do estimador Huber para reduzir a influência de outliers no passo de medição.
    gating_threshold : float, default=3.5
        Limiar (em desvios-padrão) para gate/remoção de outliers severos na inovação.
    init_level_var : float, default=25.0
        Variância inicial para o nível.
    init_drift_var : float, default=1.0
        Variância inicial para o drift.
    init_bias_var : float, default=9.0
        Variância inicial para cada viés de antena.
    init_level : float | None, default=None
        Valor inicial do nível. Se None, usa a mediana das primeiras observações válidas.
    init_drift : float, default=0.0
        Valor inicial do drift.
    init_bias : float | Sequence[float] | None, default=None
        Viés inicial por antena (0 se None).
    
    Retorno
    -------
    dict com as chaves:
        - "rssi_filtered": lista de floats com o RSSI filtrado por timestamp (nível estimado, já limitado a [sat_min, sat_max]).
        - "rssi_variance": lista de floats com a variância a posteriori do nível em cada passo.
        - "meta": dicionário com metadados úteis (parâmetros usados, taxas de outlier/faltantes, etc.).
        - "state_history": lista de dicts com amostras do estado a cada passo: {"level", "drift", "biases", "used_antennas", "outliers_step"}.
    
    Modelo de Estado
    ----------------
    x = [level, drift, bias_0, bias_1, ..., bias_{K-1}]
    Evolução: 
        level_k+1 = level_k + dt * drift_k + w_level
        drift_k+1 = (1 - drift_damping) * drift_k + w_drift
        bias_i,k+1 = (1 - bias_forget) * bias_i,k + w_bias_i
    Medição (para antena i observada em k):
        z_i,k = clip(level_k + bias_i,k, [sat_min, sat_max]) + v_i
    
    Robustez
    --------
    - Gate por inovação: componentes com |inov| > gating_threshold * sigma são descartados no passo k.
    - Peso Huber por componente restante, ajustando a matriz R efetiva (R_eff = R_base / w^2), reduzindo a influência de outliers.
    - Tratamento de faltantes (None/NaN) por componente e por passo.
    
    Observações
    -----------
    - Os defaults são conservadores e seguros; ajuste q_*, r_meas, e thresholds conforme seu ambiente.
    - A função não realiza I/O de arquivos ou rede e é independente de execuções anteriores.
    
    Exemplo mínimo
    --------------
    >>> # Exemplo 2 antenas com vieses distintos e outliers
    >>> rssis = [
    ...   [-70.0, -72.5],
    ...   [-69.0, -71.8],
    ...   [None,  -73.0],
    ...   [-68.2, -69.9],
    ...   [-120.0, -10.0],  # outliers
    ...   [-67.8, -70.2],
    ... ]
    >>> result = filter(rssis)
    >>> print(result["rssi_filtered"][:3])  # primeiros valores filtrados
    >>> print(result["meta"]["outlier_rate"])
    """
    # ----------------------
    # Helpers
    # ----------------------
    def _as_time_by_ant_matrix(data):
        # Detecta se é 1D ou 2D. Retorna (T, K) np.ndarray com NaNs nos ausentes.
        if len(data) == 0:
            return np.zeros((0, 1), dtype=float)
        first = data[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            # garantir todos no mesmo comprimento K (completando com NaN se necessário)
            K = max(len(row) if isinstance(row, (list, tuple, np.ndarray)) else 1 for row in data)
            T = len(data)
            M = np.full((T, K), np.nan, dtype=float)
            for t, row in enumerate(data):
                if isinstance(row, (list, tuple, np.ndarray)):
                    for i, v in enumerate(row):
                        M[t, i] = np.nan if v is None else float(v)
                else:
                    M[t, 0] = np.nan if row is None else float(row)
            return M
        else:
            # 1D
            M = np.array([np.nan if v is None else float(v) for v in data], dtype=float).reshape(-1, 1)
            return M

    def _soft_clip(x, lo, hi):
        # Hard clip é suficiente; manter como função separada caso queira "soft" futuramente.
        return np.clip(x, lo, hi)

    def _sigma_points(mean, cov, alpha, kappa):
        n = mean.shape[0]
        lam = alpha**2 * (n + kappa) - n
        S = None
        try:
            S = np.linalg.cholesky((n + lam) * cov)
        except np.linalg.LinAlgError:
            # fallback numérico: usar decomposição por autovalores (mais estável, mais caro)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-9)
            S = eigvecs @ np.diag(np.sqrt((n + lam) * eigvals)) @ eigvecs.T
        X = np.zeros((n, 2 * n + 1))
        X[:, 0] = mean
        for i in range(n):
            X[:, i + 1] = mean + S[:, i]
            X[:, i + 1 + n] = mean - S[:, i]
        return X, lam

    def _ukf_weights(n, lam, alpha, beta):
        Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wc = np.copy(Wm)
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
        return Wm, Wc

    def _process_fn(Xsig, dt, drift_damping, bias_forget):
        # Xsig: (n, 2n+1)
        n, cols = Xsig.shape
        # Derivações: [level, drift, biases...]
        Xp = np.copy(Xsig)
        level = Xsig[0, :]
        drift = Xsig[1, :]
        Xp[0, :] = level + dt * drift
        Xp[1, :] = (1.0 - drift_damping) * drift
        if n > 2:
            biases = Xsig[2:, :]
            Xp[2:, :] = (1.0 - bias_forget) * biases
        return Xp

    def _measurement_fn(Xsig_pred, ant_indices, sat_min, sat_max):
        # Retorna Zsig de dim (m, 2n+1), com m = len(ant_indices)
        level = Xsig_pred[0, :]
        if len(ant_indices) == 0:
            return np.zeros((0, Xsig_pred.shape[1]))
        if Xsig_pred.shape[0] == 2:
            # Sem biases no estado (K=1 tratado sem viés)
            z = _soft_clip(level, sat_min, sat_max)
            return z.reshape(1, -1).repeat(len(ant_indices), axis=0)
        else:
            # Com biases por antena
            biases = Xsig_pred[2:, :]  # (K, 2n+1)
            Z = np.zeros((len(ant_indices), Xsig_pred.shape[1]))
            for idx_pos, ant_idx in enumerate(ant_indices):
                b = biases[ant_idx, :]
                Z[idx_pos, :] = _soft_clip(level + b, sat_min, sat_max)
            return Z

    def _reconstruct_R(base_r, indices):
        if np.isscalar(base_r):
            return np.full(len(indices), float(base_r), dtype=float)
        else:
            base_r = np.asarray(base_r, dtype=float).ravel()
            if len(base_r) < (max(indices)+1 if indices else 0):
                # completar com último valor caso r_meas tenha sido parcial
                last = base_r[-1] if base_r.size > 0 else 4.0
                ext = np.full(max(indices)+1, last, dtype=float)
                ext[:base_r.size] = base_r
                base_r = ext
            return base_r[indices]

    # ----------------------
    # Parse inputs
    # ----------------------
    M = _as_time_by_ant_matrix(rssis)  # (T, K)
    T, K = M.shape

    # Defaults
    dt = float(kwargs.get("dt", 1.0))
    alpha = float(kwargs.get("alpha", 0.5))
    beta = float(kwargs.get("beta", 2.0))
    kappa = float(kwargs.get("kappa", 0.0))

    q_level = float(kwargs.get("q_level", 1.0))
    q_drift = float(kwargs.get("q_drift", 0.01))
    q_bias = float(kwargs.get("q_bias", 0.1))

    r_meas = kwargs.get("r_meas", 4.0)  # pode ser escalar ou vetor

    drift_damping = float(kwargs.get("drift_damping", 0.05))
    bias_forget = float(kwargs.get("bias_forget", 0.01))

    sat_min = float(kwargs.get("sat_min", -120.0))
    sat_max = float(kwargs.get("sat_max", -20.0))

    huber_delta = float(kwargs.get("huber_delta", 1.5))
    gating_threshold = float(kwargs.get("gating_threshold", 3.5))

    init_level_var = float(kwargs.get("init_level_var", 25.0))
    init_drift_var = float(kwargs.get("init_drift_var", 1.0))
    init_bias_var = float(kwargs.get("init_bias_var", 9.0))

    init_level = kwargs.get("init_level", None)
    init_drift = float(kwargs.get("init_drift", 0.0))
    init_bias = kwargs.get("init_bias", None)

    # Estado: [level, drift, biases (0..K-1)]
    n = 2 + (K if K > 1 else 0)  # para K=1 podemos omitir bias para reduzir dimensão
    use_biases = K > 1

    # Inicialização do estado
    # Estimativa robusta do nível inicial: mediana das primeiras N válidas
    def _nanmedian(a):
        try:
            return float(np.nanmedian(a))
        except:
            vals = [x for x in a if x==x]  # not NaN
            return float(vals[len(vals)//2]) if vals else -80.0

    if init_level is None:
        # usa as primeiras min(10, T) * K observações válidas
        first_slice = M[:min(10, T), :]
        init_level = _nanmedian(first_slice[np.isfinite(first_slice)] if np.isfinite(first_slice).any() else np.array([-80.0]))
    x = np.zeros(n, dtype=float)
    x[0] = float(init_level)
    x[1] = float(init_drift)
    if use_biases:
        if init_bias is None:
            x[2:] = 0.0
        else:
            b = np.asarray(init_bias, dtype=float).ravel()
            if b.size < K:
                b = np.pad(b, (0, K - b.size), mode='edge') if b.size>0 else np.zeros(K, dtype=float)
            x[2:] = b[:K]

    # Covariância inicial
    P = np.zeros((n, n), dtype=float)
    P[0, 0] = init_level_var
    P[1, 1] = init_drift_var
    if use_biases:
        for i in range(K):
            P[2 + i, 2 + i] = init_bias_var

    # Ruído de processo
    Q = np.zeros((n, n), dtype=float)
    Q[0, 0] = q_level
    Q[1, 1] = q_drift
    if use_biases:
        for i in range(K):
            Q[2 + i, 2 + i] = q_bias

    # Saídas
    rssi_filtered = []
    rssi_variance = []
    state_history = []
    outlier_total = 0
    missing_total = 0
    used_meas_total = 0

    eps = 1e-9

    # ----------------------
    # Loop temporal
    # ----------------------
    for t in range(T):
        out_this = 0
        used_antennas = 0
        z_t_full = M[t, :]  # (K,)
        valid_mask = np.isfinite(z_t_full)
        missing_total += int(K - np.count_nonzero(valid_mask))

        # Predição
        Xsig, lam = _sigma_points(x, P, alpha, kappa)
        Wm, Wc = _ukf_weights(n, lam, alpha, beta)
        Xsig_pred = _process_fn(Xsig, dt, drift_damping, bias_forget)
        x_pred = Xsig_pred @ Wm
        # Cov predito
        Xm = Xsig_pred - x_pred.reshape(-1, 1)
        P_pred = Q + (Xm * Wc) @ Xm.T

        # Atualização (se houver medições válidas)
        if np.any(valid_mask):
            ant_indices_all = np.where(valid_mask)[0].tolist()
            z_obs = z_t_full[valid_mask]
            # Medição predita com todos válidos
            Zsig = _measurement_fn(Xsig_pred, ant_indices_all, sat_min, sat_max)  # (m, 2n+1)
            z_pred = Zsig @ Wm
            Zm = Zsig - z_pred.reshape(-1, 1)
            # R base (para as antenas válidas)
            R_base_diag = _reconstruct_R(r_meas, ant_indices_all)
            S = np.diag(R_base_diag) + (Zm * Wc) @ Zm.T
            # Inovação e gating
            S_diag = np.maximum(np.diag(S), eps)
            innov = z_obs - z_pred
            std = np.sqrt(S_diag)
            keep = np.abs(innov) <= gating_threshold * std
            out_this = int(np.size(keep) - np.count_nonzero(keep))
            outlier_total += out_this

            if np.any(keep):
                ant_indices = [ant_indices_all[i] for i, kpt in enumerate(keep) if kpt]
                z_obs_k = z_obs[keep]
                # Recalcula Zsig/z_pred/S apenas com antenas mantidas
                Zsig_k = _measurement_fn(Xsig_pred, ant_indices, sat_min, sat_max)
                z_pred_k = Zsig_k @ Wm
                Zm_k = Zsig_k - z_pred_k.reshape(-1, 1)
                R_base_k = _reconstruct_R(r_meas, ant_indices)
                S_k = np.diag(R_base_k) + (Zm_k * Wc) @ Zm_k.T
                S_k_diag = np.maximum(np.diag(S_k), eps)
                innov_k = z_obs_k - z_pred_k
                std_k = np.sqrt(S_k_diag)

                # Pesos Huber (0<w<=1). Se |r| <= delta*std => w=1; senão w=(delta*std)/|r|
                w = np.ones_like(innov_k)
                abs_r = np.abs(innov_k)
                thresh = huber_delta * std_k
                mask_out = abs_r > thresh
                w[mask_out] = (thresh[mask_out] / (abs_r[mask_out] + eps))
                # Ajuste de R efetivo
                R_eff = R_base_k / (w**2 + eps)
                S_k = np.diag(R_eff) + (Zm_k * Wc) @ Zm_k.T

                # Cross-cov
                C = (Xm * Wc) @ Zm_k.T
                # Ganho de Kalman
                try:
                    S_inv = np.linalg.inv(S_k)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S_k)
                K_gain = C @ S_inv
                # Atualiza
                x = x_pred + K_gain @ (z_obs_k - z_pred_k)
                P = P_pred - K_gain @ S_k @ K_gain.T
                used_meas_total += len(z_obs_k)
                used_antennas = len(z_obs_k)
            else:
                # Sem medições confiáveis -> usa apenas predição
                x = x_pred
                P = P_pred
                used_antennas = 0
        else:
            # Sem medições -> usa apenas predição
            x = x_pred
            P = P_pred
            used_antennas = 0

        # Garantir simetria numérica e PSD
        P = 0.5 * (P + P.T)
        # Corrigir pequenos negativos na diagonal por erros numéricos
        d = np.diag(P).copy()
        d[d < eps] = eps
        np.fill_diagonal(P, d)

        # Saída por passo
        level_est = float(_soft_clip(x[0], sat_min, sat_max))
        rssi_filtered.append(level_est)
        rssi_variance.append(float(P[0, 0]))
        state_history.append({
            "level": level_est,
            "drift": float(x[1]),
            "biases": ([] if not use_biases else [float(b) for b in x[2:]]),
            "used_antennas": int(used_antennas),
            "outliers_step": int(out_this),
        })

    # Metadados
    total_meas = int(np.isfinite(M).sum())
    meta = {
        "params": {
            "dt": dt, "alpha": alpha, "beta": beta, "kappa": kappa,
            "q_level": q_level, "q_drift": q_drift, "q_bias": q_bias,
            "r_meas": r_meas,
            "drift_damping": drift_damping, "bias_forget": bias_forget,
            "sat_min": sat_min, "sat_max": sat_max,
            "huber_delta": huber_delta, "gating_threshold": gating_threshold,
            "init_level_var": init_level_var, "init_drift_var": init_drift_var, "init_bias_var": init_bias_var,
        },
        "samples": T,
        "antennas": K,
        "missing_count": int(missing_total),
        "missing_rate": float(missing_total / (T * K) if T*K>0 else 0.0),
        "outlier_count": int(outlier_total),
        "outlier_rate": float(outlier_total / total_meas) if total_meas>0 else 0.0,
        "used_measurements": int(used_meas_total),
    }

    return {
        "rssi_filtered": rssi_filtered,
        "rssi_variance": rssi_variance,
        "meta": meta,
        "state_history": state_history,
    }