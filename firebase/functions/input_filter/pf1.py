from typing import List, Union, Optional, Dict, Any
import numpy as np
import math

def rssi_filter_pf_1(
    rssis: Union[List[float], List[List[Optional[float]]], np.ndarray],
    *,
    num_particles: int = 1024,
    process_noise_std: float = 1.5,
    meas_noise_std: Optional[float] = None,
    outlier_prob: float = 0.05,
    outlier_scale: float = 8.0,
    resample_threshold: float = 0.5,
    estimate_bias: bool = True,
    bias_clamp_db: float = 12.0,
    antenna_weights: Optional[Union[List[float], np.ndarray, str]] = "auto",
    init_mean: Optional[float] = None,
    init_std: float = 8.0,
    jitter_after_resample_std: float = 0.2,
    return_particles: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Filtra uma série temporal de RSSI (dBm) medida por uma ou mais antenas fixas,
    usando um Filtro de Partículas (PF) robusto a ruído, outliers e inconsistências
    entre antenas. A implementação é *stateless*: cada chamada é independente.

    Parâmetros
    ----------
    rssis : list | np.ndarray
        - 1D: [T] → série de uma antena ao longo do tempo (dBm).
        - 2D: [T, K] → série temporal com K antenas por timestamp (dBm).
          Use `None`/`np.nan` para medições ausentes.
    num_particles : int, padrão 1024
        Número de partículas do PF. Valores maiores tendem a aumentar a robustez e o custo computacional.
    process_noise_std : float, padrão 1.5
        Desvio-padrão do ruído de processo (modelo dinâmico x_t = x_{t-1} + w_t), em dB.
    meas_noise_std : float | None
        Desvio-padrão base do ruído de medição por antena, em dB. Se None, é estimado de forma robusta
        a partir dos dados (MAD) e limitado a um intervalo seguro (2–6 dB).
    outlier_prob : float, padrão 0.05
        Probabilidade de outlier na medição. O modelo de observação é uma mistura:
        (1 - p) N(0, σ^2) + p N(0, (ασ)^2).
    outlier_scale : float, padrão 8.0
        Fator α que infla a variância do componente de outlier (ασ).
    resample_threshold : float, padrão 0.5
        Gatilho de reamostragem por ESS: reamostrar quando ESS < threshold * N.
    estimate_bias : bool, padrão True
        Se True, estima e remove um viés constante por antena (offset) de forma robusta.
    bias_clamp_db : float, padrão 12.0
        Limite absoluto de correção de viés por antena (±bias_clamp_db).
    antenna_weights : list | np.ndarray | "auto" | None
        Pesos de confiabilidade por antena (K valores). Se "auto" (padrão), calcula pesos robustos
        via inverso da variância robusta (MAD^2). Se lista/array, usa como pesos base.
        Caso None, usa pesos iguais.
    init_mean : float | None
        Média inicial do estado latente (dBm). Se None, usa a mediana robusta do 1º timestamp observado.
    init_std : float, padrão 8.0
        Desvio-padrão inicial das partículas (dB).
    jitter_after_resample_std : float, padrão 0.2
        Jitter gaussiano pequeno (dB) aplicado às partículas após reamostragem (regularização).
    return_particles : bool, padrão False
        Se True, retorna também as partículas do último passo e seus pesos.
    seed : int | None
        Semente do gerador aleatório para reprodutibilidade.

    Retorno
    -------
    dict com chaves:
        - 'filtered' : List[float]
            Estimativa filtrada por timestamp (média ponderada das partículas), em dBm.
        - 'variance' : List[float]
            Variância das partículas por timestamp (dB^2).
        - 'ess' : List[float]
            Tamanho efetivo da amostra por timestamp.
        - 'used_antennas' : List[int]
            Número de antenas efetivamente usadas (com observação válida) em cada timestamp.
        - 'n_resamples' : int
            Número de reamostragens realizadas.
        - 'antenna_bias' : List[float]
            Viés estimado (dBm) por antena removido das medições (0.0 se K=1 ou estimate_bias=False).
        - 'antenna_weights' : List[float]
            Pesos finais de confiabilidade por antena (normalizados para média 1.0 entre antenas observadas).
        - 'params' : dict
            Parâmetros efetivos (incluindo σ de medição final utilizado).
        - 'particles' : np.ndarray (opcional)
            Partículas do último timestamp [N], se return_particles=True.
        - 'weights' : np.ndarray (opcional)
            Pesos normalizados do último timestamp [N], se return_particles=True.

    Modelagem e Robustez
    --------------------
    - Estado latente x_t: "RSSI verdadeiro" (1D), evolui como passeio aleatório (random walk).
    - Observação y_{t,j} por antena j:
          p(y_{t,j} | x_t) = (1-p) N(x_t, σ_j^2) + p N(x_t, (α σ_j)^2)
      com σ_j ajustado pela confiabilidade da antena.
    - Viés por antena (opcional): estimado via mediana dos desvios em relação à mediana entre antenas,
      com *clamp* para evitar correções absurdas.
    - Pesos de antena automáticos: w_j ∝ 1 / (MAD_j^2 + ε), normalizados para média 1.0.
      A variância efetiva por antena é σ_j^2 / max(w_j, ε), reduzindo a influência de antenas ruidosas.
    - Ausências (NaN/None): são ignoradas no passo de atualização. Se todas ausentes em t,
      só há predição (sem atualização).

    Defaults Seguros
    ----------------
    - Se `meas_noise_std` não for informado, é estimado de forma robusta (MAD) e
      restrito ao intervalo [2.0, 6.0] dB.
    - Inicialização: se `init_mean` for None, usa mediana do 1º timestamp com dados; se não houver nenhum
      dado em toda a série, assume -65 dBm por convenção de ambientes indoor.

    Exemplo mínimo
    --------------
    >>> # Exemplo com 2 antenas, viés e outliers
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = 60
    >>> true = -60 + np.cumsum(rng.normal(0, 0.3, size=T))
    >>> # Antena A: viés +2 dB ; Antena B: viés -3 dB
    >>> yA = true + 2.0 + rng.normal(0, 2.5, size=T)
    >>> yB = true - 3.0 + rng.normal(0, 3.5, size=T)
    >>> # 5 outliers grandes
    >>> idx = rng.choice(T, 5, replace=False)
    >>> yA[idx] += rng.normal(0, 15.0, size=5)
    >>> yB[idx] += rng.normal(0, 12.0, size=5)
    >>> Y = np.stack([yA, yB], axis=1)  # [T, K=2]
    >>> out = filter(Y, seed=123)
    >>> print(round(out['filtered'][-1], 2), round(np.var(out['filtered'][-10:]), 2))
    """
    # ---------- utilitários internos ----------
    def _to_2d(arr) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        elif a.ndim != 2:
            raise ValueError("rssis deve ser 1D [T] ou 2D [T, K].")
        return a

    def _nan_median(x, axis=None):
        return np.nanmedian(x, axis=axis)

    def _mad(x, axis=None):
        """MAD escalado (≈ desvio-padrão robusto)."""
        med = _nan_median(x, axis=axis)
        dev = np.abs(x - med)
        mad = _nan_median(dev, axis=axis)
        return 1.4826 * mad  # escala para ~sigma

    def _systematic_resample(rng, particles, weights):
        N = len(particles)
        positions = (rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(weights)
        idx = np.searchsorted(cumsum, positions)
        return particles[idx]

    def _gauss_pdf_log(err2_over_var):
        # log N(0, var) sem a constante 0.5*log(2π*var):
        # aqui usamos apenas a parte dependente de err/var; a constante cancela na normalização.
        return -0.5 * err2_over_var

    def _log_mix_gauss(e2, var, p_out, scale_out):
        # log[(1-p)N(0,var) + p N(0, var_out)] (sem constantes), numericamente estável
        var_out = var * (scale_out ** 2)
        a = _gauss_pdf_log(e2 / var) - 0.5 * np.log(var)
        b = _gauss_pdf_log(e2 / var_out) - 0.5 * np.log(var_out)
        # log( (1-p) e^a + p e^b )
        # usa logaddexp: log( e^(log(1-p)+a) + e^(log p + b) )
        return np.logaddexp(np.log1p(-p_out) + a, np.log(p_out) + b)

    # ---------- preparo dos dados ----------
    Y = _to_2d(rssis)  # [T, K]
    T, K = Y.shape
    rng = np.random.default_rng(seed)

    # Estimar/Aplicar viés por antena (constante) de forma robusta
    if estimate_bias and K > 1:
        med_across_ant = _nan_median(Y, axis=1)[:, None]  # [T,1]
        residuals = Y - med_across_ant  # desvios por antena vs mediana entre antenas
        bias = _nan_median(residuals, axis=0)
        # clamp
        bias = np.clip(np.where(np.isnan(bias), 0.0, bias), -bias_clamp_db, bias_clamp_db)
    else:
        bias = np.zeros(K, dtype=float)

    Y_adj = Y - bias  # remove offset estimado

    # Pesos de antena
    if isinstance(antenna_weights, (list, np.ndarray)):
        w_ant = np.asarray(antenna_weights, dtype=float)
        if w_ant.shape != (K,):
            raise ValueError("antenna_weights deve ter K elementos.")
        w_ant = np.where(np.isnan(w_ant) | (w_ant <= 0), 1.0, w_ant)
    elif antenna_weights == "auto" and K >= 1:
        # peso ∝ 1 / (MAD_j^2 + eps), normalizado para média 1
        mad_j = _mad(Y_adj, axis=0)
        eps = 1e-6
        invvar = 1.0 / (mad_j ** 2 + eps)
        w_ant = invvar
        # substitui NaN/inf por 1.0
        w_ant = np.where(~np.isfinite(w_ant), 1.0, w_ant)
    else:
        w_ant = np.ones(K, dtype=float)

    # Normaliza pesos para média 1.0
    if np.all(w_ant <= 0):
        w_ant = np.ones(K)
    w_ant = w_ant * (K / np.sum(w_ant))  # média = 1

    # Estimar σ de medição base, se necessário
    if meas_noise_std is None:
        # estimativa robusta por timestamp contra mediana entre antenas
        resid = Y_adj - _nan_median(Y_adj, axis=1)[:, None]
        sigma_est = float(np.nanmedian(_mad(resid, axis=0)))
        if not np.isfinite(sigma_est) or sigma_est <= 0:
            sigma_est = 3.0
        meas_noise_std_eff = float(np.clip(sigma_est, 2.0, 6.0))
    else:
        meas_noise_std_eff = float(meas_noise_std)

    # σ efetivo por antena será ajustado pelos pesos: var_j = (σ^2) / max(w_j, eps)
    w_eps = 1e-6
    var_base = meas_noise_std_eff ** 2
    var_per_ant = var_base / np.maximum(w_ant, w_eps)  # [K]

    # Inicialização do estado
    # init_mean: mediana do primeiro timestamp observável, ou -65 se nada disponível
    if init_mean is None:
        first_obs_idx = None
        for t in range(T):
            if np.any(np.isfinite(Y_adj[t])):
                first_obs_idx = t
                break
        if first_obs_idx is not None:
            init_mean_eff = float(_nan_median(Y_adj[first_obs_idx]))
            if not np.isfinite(init_mean_eff):
                init_mean_eff = -65.0
        else:
            init_mean_eff = -65.0
    else:
        init_mean_eff = float(init_mean)

    # Partículas e pesos
    particles = rng.normal(loc=init_mean_eff, scale=init_std, size=num_particles)
    weights = np.ones(num_particles, dtype=float) / num_particles

    filtered = []
    variance = []
    ess_list = []
    used_antennas = []
    n_resamples = 0

    # ---------- loop temporal ----------
    for t in range(T):
        # Predição
        particles = particles + rng.normal(0.0, process_noise_std, size=num_particles)

        # Observações disponíveis em t
        y_t = Y_adj[t]  # [K]
        valid = np.isfinite(y_t)
        k_used = int(np.sum(valid))
        used_antennas.append(k_used)

        if k_used > 0:
            yv = y_t[valid]                # [k_used]
            var_v = var_per_ant[valid]     # [k_used]

            # Likelihood robusta (mistura Gaussiana) por antena e partícula
            # Calcula log-verossimilhança somando sobre antenas
            # Para eficiência, expandimos partículas para [N,1]
            x = particles[:, None]  # [N,1]
            # erro por antena para cada partícula: (yv - x)
            err = yv[None, :] - x   # [N, k_used]
            e2 = err ** 2           # [N, k_used]
            # log-likelihood por antena: [N, k_used]
            ll_j = _log_mix_gauss(e2, var_v[None, :], outlier_prob, outlier_scale)
            # soma sobre antenas (ignoramos constantes comuns)
            ll = np.sum(ll_j, axis=1)  # [N]

            # atualização de pesos (no espaço log para estabilidade)
            # weights_new ∝ weights * exp(ll)
            # Para estabilizar, subtraímos max(ll)
            m = np.max(ll)
            w_unnorm = weights * np.exp(ll - m)
            sumw = np.sum(w_unnorm)
            if sumw <= 0 or not np.isfinite(sumw):
                # fallback numérico: reset pesos uniformes
                weights = np.ones(num_particles) / num_particles
            else:
                weights = w_unnorm / sumw
        else:
            # sem observações: apenas predição, pesos inalterados
            pass

        # ESS e reamostragem
        ess = 1.0 / np.sum(weights ** 2)
        ess_list.append(float(ess))

        if ess < resample_threshold * num_particles:
            particles = _systematic_resample(rng, particles, weights)
            weights.fill(1.0 / num_particles)
            if jitter_after_resample_std > 0:
                particles = particles + rng.normal(0.0, jitter_after_resample_std, size=num_particles)
            n_resamples += 1

        # estimativas
        mean_t = float(np.sum(particles * weights))
        var_t = float(np.sum(((particles - mean_t) ** 2) * weights))
        filtered.append(mean_t)
        variance.append(var_t)

    result: Dict[str, Any] = {
        "filtered": filtered,
        "variance": variance,
        "ess": ess_list,
        "used_antennas": used_antennas,
        "n_resamples": int(n_resamples),
        "antenna_bias": [float(b) for b in bias],
        "antenna_weights": [float(w) for w in w_ant],
        "params": {
            "num_particles": int(num_particles),
            "process_noise_std": float(process_noise_std),
            "meas_noise_std": float(meas_noise_std_eff),
            "outlier_prob": float(outlier_prob),
            "outlier_scale": float(outlier_scale),
            "resample_threshold": float(resample_threshold),
            "estimate_bias": bool(estimate_bias),
            "bias_clamp_db": float(bias_clamp_db),
            "init_mean": float(init_mean_eff),
            "init_std": float(init_std),
            "jitter_after_resample_std": float(jitter_after_resample_std),
            "seed": seed if seed is not None else None,
        },
    }

    if return_particles:
        result["particles"] = particles.copy()
        result["weights"] = weights.copy()

    return result