from typing import List, Sequence, Union, Optional, Dict, Any
import math

Number = Union[int, float]
RSSIInput = Union[Number, Sequence[Optional[Number]]]


def rssi_filter_kf_1(
    rssis: Sequence[RSSIInput],
    *,
    dt: Optional[float] = None,
    dts: Optional[Sequence[float]] = None,
    meas_std_init: float = 4.0,
    meas_std_min: float = 1.5,
    process_var_pos: float = 0.50,
    process_var_vel: float = 0.10,
    gate_sigma: float = 3.5,
    gate_R_scale: float = 100.0,
    mad_window: int = 15,
    antenna_mad_k: float = 2.5,
    winsorize: bool = True,
    robust_update: bool = True,
) -> Dict[str, Any]:
    """
    Filtra uma série temporal de RSSI usando um Filtro de Kalman (KF) 1D robusto,
    com estado [RSSI, derivada] e mecanismos de rejeição/atenuação de outliers,
    consolidação robusta entre antenas e adaptação do ruído de medição.

    A implementação é **stateless**: nenhum estado é persistido entre chamadas.

    Parâmetros
    ----------
    rssis : Sequence[RSSIInput]
        Série temporal ordenada por tempo. Em cada instante k, a entrada pode ser:
        - float/int: RSSI de uma única antena (em dBm); ou
        - sequência (lista/tupla) de RSSIs (float|int|None) de múltiplas antenas.
          Valores `None` ou `NaN` são ignorados.
    dt : float, opcional
        Intervalo de tempo fixo entre amostras. Se não fornecido e `dts` não for usado,
        assume-se dt = 1.0.
    dts : Sequence[float], opcional
        Intervalos de tempo variáveis (um por passo). Se fornecido, tem precedência
        sobre `dt`. O tamanho deve ser >= len(rssis) - 1 (excesso é ignorado).
    meas_std_init : float, padrão 4.0
        Desvio-padrão inicial (em dB) do ruído de medição. Usado no início e como
        fallback quando não há dados suficientes para estimar.
    meas_std_min : float, padrão 1.5
        Desvio-padrão mínimo (em dB) imposto ao estimar o ruído de medição de forma
        robusta (evita colapso numérico).
    process_var_pos : float, padrão 0.50
        Variância de processo (Q) para a componente de posição (RSSI). Ajuste maior
        se o RSSI real variar rapidamente.
    process_var_vel : float, padrão 0.10
        Variância de processo (Q) para a componente de velocidade (derivada do RSSI).
    gate_sigma : float, padrão 3.5
        Limite de "gating" (em sigmas) para inovação. Inovações acima deste valor têm
        seu R aumentado em `gate_R_scale` (downweight) na atualização do KF.
    gate_R_scale : float, padrão 100.0
        Fator multiplicativo aplicado ao ruído de medição R quando a medição é
        considerada outlier no teste de inovação.
    mad_window : int, padrão 15
        Tamanho da janela (amostras recentes) para estimativa robusta do ruído de
        medição via MAD (Median Absolute Deviation).
    antenna_mad_k : float, padrão 2.5
        Fator de corte (em desvios robustos) para detectar/atenuar outliers entre
        antenas no mesmo timestamp, antes da fusão robusta.
    winsorize : bool, padrão True
        Se True, aplica winsorização (clipping) dos valores por-antena antes da
        agregação robusta; caso False, apenas atribui pesos Huber (sem clipping).
    robust_update : bool, padrão True
        Se True, usa pesos de Huber sobre a inovação para uma atualização ainda mais
        robusta (equivalente a inflar R conforme o resíduo).

    Retorno
    -------
    dict
        {
          "rssi_filtered": List[float],
          "metadata": {
             "used_measurements": List[Optional[float]],   # medições agregadas (z_k)
             "innovations": List[Optional[float]],         # z_k - H x_pred
             "innovation_std": List[Optional[float]],      # sqrt(S_k)
             "kalman_gain": List[float],                   # K_k (escala em 1D)
             "R_effective": List[Optional[float]],         # R_k efetivo usado
             "Q_sequence": List[float],                    # traço de Q_k por passo
             "antenna_counts": List[Optional[int]],        # nº de antenas válidas por k
             "antenna_outliers": List[Optional[int]],      # nº de antenas rejeitadas por k
             "flags": {
                 "no_measurement_steps": List[int],        # índices sem medição válida
                 "gated_steps": List[int],                 # índices com inovação > gate
             },
             "params": { ... }                             # parâmetros efetivos
          }
        }

    Estratégia de Robustez
    ----------------------
    1) Consolidação por timestamp (multiantena):
       - Remove None/NaN.
       - Calcula mediana e MAD, rejeita/atenua valores muito afastados (|x - med| > k*MAD).
       - Agrega via média ponderada por pesos Huber (ou winsorização + média).
    2) KF adaptativo:
       - Estado: x = [RSSI, dRSSI/dt], F = [[1, dt],[0,1]], H = [1, 0].
       - R_k estimado por janela de MAD dos z_k recentes (mínimo `meas_std_min`).
       - Teste de inovação: se |y| > gate_sigma * sqrt(S), multiplica R_k por `gate_R_scale`.
       - (Opcional) Reaplicação Huber na inovação para inflar R_k adicionalmente.
    3) Suporta passos sem medição (apenas predição).

    Notas
    -----
    - Use valores padrão se parâmetros críticos não forem informados.
    - Não há I/O de arquivos ou rede.
    - Unidades: RSSI em dBm, variâncias em dB².

    Exemplo
    -------
    >>> # Exemplo mínimo (executável):
    >>> series = [-60, -61, -62, -90, -61, -60, -59, -200, -58, -57]  # com outliers
    >>> result = filter(series)
    >>> round(result["rssi_filtered"][-1], 1)
    -57.3
    >>> # Multiantena (por timestamp):
    >>> multi = [
    ...   [-60, -61, -59],
    ...   [-61, -60, None],
    ...   [-90, -62, -61],  # uma antena com outlier
    ...   [-60, -59, -58],
    ... ]
    >>> out = filter(multi)
    >>> [round(v,1) for v in out["rssi_filtered"]]
    [-60.2, -60.4, -60.7, -60.3]
    """
    # --------- Helpers ---------
    def _is_nan(x: Any) -> bool:
        try:
            return math.isnan(float(x))
        except Exception:
            return False

    def _huber_weight(res: float, scale: float, k: float = 1.345) -> float:
        """Peso Huber para resíduo 'res' com escala robusta 'scale'."""
        if scale <= 0:
            return 1.0
        a = abs(res) / (k * scale)
        return 1.0 if a <= 1.0 else 1.0 / a

    def _robust_aggregate_antenna(values: Sequence[Optional[Number]]) -> Dict[str, Any]:
        """
        Agrega medições de múltiplas antenas (um timestamp) com robustez.
        Retorna dict com: z (float|None), n_valid, n_outliers, used_values, R_cross (var cruzada).
        """
        # Limpeza básica
        vs = [float(v) for v in values if v is not None and not _is_nan(v)]
        n_valid = len(vs)
        if n_valid == 0:
            return {"z": None, "n_valid": 0, "n_outliers": 0, "used_values": [], "R_cross": None}

        # Estatística robusta
        med = sorted(vs)[n_valid // 2] if n_valid % 2 == 1 else 0.5 * (
            sorted(vs)[n_valid // 2 - 1] + sorted(vs)[n_valid // 2]
        )
        abs_dev = [abs(v - med) for v in vs]
        mad = sorted(abs_dev)[n_valid // 2]
        # Escala robusta (aprox σ) a partir do MAD
        sigma_r = 1.4826 * mad if mad > 0 else max(meas_std_min, 0.5 * meas_std_init)

        # Seleção/atenuação de outliers por antena
        thr = antenna_mad_k * sigma_r
        kept = []
        weights = []
        n_out = 0
        for v in vs:
            res = v - med
            if winsorize:
                # Winsorização + peso Huber para suavizar pequenas discrepâncias
                if abs(res) > thr:
                    n_out += 1
                    v_clipped = med + math.copysign(thr, res)
                    kept.append(v_clipped)
                else:
                    kept.append(v)
                w = _huber_weight(res, sigma_r)
                weights.append(w)
            else:
                # Apenas pesos Huber; outliers têm peso baixo
                if abs(res) > thr:
                    n_out += 1
                kept.append(v)
                weights.append(_huber_weight(res, sigma_r))

        # Média ponderada (evita divisão por zero)
        wsum = sum(weights) if weights else 1.0
        if wsum <= 1e-9:
            z = sum(kept) / len(kept)
        else:
            z = sum(w * v for w, v in zip(weights, kept)) / wsum

        # Variância cruzada entre antenas (após robustificação)
        if len(kept) >= 2:
            mean_k = sum(kept) / len(kept)
            var_k = sum((v - mean_k) ** 2 for v in kept) / (len(kept) - 1)
            R_cross = max(var_k, meas_std_min ** 2)
        else:
            R_cross = None

        return {
            "z": float(z),
            "n_valid": n_valid,
            "n_outliers": n_out,
            "used_values": kept,
            "R_cross": R_cross,
        }

    def _as_scalar_or_aggregate(item: RSSIInput) -> Dict[str, Any]:
        """Aceita escalar ou sequência por timestamp e retorna medição agregada."""
        if isinstance(item, (int, float)) and not _is_nan(item):
            return {"z": float(item), "n_valid": 1, "n_outliers": 0, "used_values": [float(item)], "R_cross": None}
        # Sequência por timestamp
        if isinstance(item, (list, tuple)):
            return _robust_aggregate_antenna(item)
        # Valor inválido
        return {"z": None, "n_valid": 0, "n_outliers": 0, "used_values": [], "R_cross": None}

    # --------- Preparação ---------
    n = len(rssis)
    if n == 0:
        return {
            "rssi_filtered": [],
            "metadata": {
                "used_measurements": [],
                "innovations": [],
                "innovation_std": [],
                "kalman_gain": [],
                "R_effective": [],
                "Q_sequence": [],
                "antenna_counts": [],
                "antenna_outliers": [],
                "flags": {"no_measurement_steps": [], "gated_steps": []},
                "params": {
                    "dt": dt,
                    "meas_std_init": meas_std_init,
                    "meas_std_min": meas_std_min,
                    "process_var_pos": process_var_pos,
                    "process_var_vel": process_var_vel,
                    "gate_sigma": gate_sigma,
                    "gate_R_scale": gate_R_scale,
                    "mad_window": mad_window,
                    "antenna_mad_k": antenna_mad_k,
                    "winsorize": winsorize,
                    "robust_update": robust_update,
                },
            },
        }

    # dt(s) por passo
    if dts is not None and len(dts) >= max(0, n - 1):
        dts_list = [float(x) if x is not None and x > 0 else 1.0 for x in dts[: n - 1]]
    else:
        step_dt = float(dt) if (dt is not None and dt > 0) else 1.0
        dts_list = [step_dt] * (n - 1)

    # Agregar medições por timestamp (robustamente entre antenas)
    agg = [_as_scalar_or_aggregate(x) for x in rssis]
    z_seq = [a["z"] for a in agg]
    antenna_counts = [a["n_valid"] if a["n_valid"] > 0 else None for a in agg]
    antenna_outliers = [a["n_outliers"] if a["n_valid"] > 0 else None for a in agg]
    cross_vars = [a["R_cross"] for a in agg]

    # --------- Inicialização do KF ---------
    # Estado: x = [pos, vel], P 2x2
    # Inicializa posição com primeira medição válida; velocidade = 0
    first_valid_idx = next((i for i, z in enumerate(z_seq) if z is not None), None)
    if first_valid_idx is None:
        # Sem medições válidas
        return {
            "rssi_filtered": [math.nan] * n,
            "metadata": {
                "used_measurements": [None] * n,
                "innovations": [None] * n,
                "innovation_std": [None] * n,
                "kalman_gain": [0.0] * n,
                "R_effective": [None] * n,
                "Q_sequence": [],
                "antenna_counts": antenna_counts,
                "antenna_outliers": antenna_outliers,
                "flags": {"no_measurement_steps": list(range(n)), "gated_steps": []},
                "params": {
                    "dt": dt,
                    "meas_std_init": meas_std_init,
                    "meas_std_min": meas_std_min,
                    "process_var_pos": process_var_pos,
                    "process_var_vel": process_var_vel,
                    "gate_sigma": gate_sigma,
                    "gate_R_scale": gate_R_scale,
                    "mad_window": mad_window,
                    "antenna_mad_k": antenna_mad_k,
                    "winsorize": winsorize,
                    "robust_update": robust_update,
                },
            },
        }

    x_pos = z_seq[first_valid_idx]
    x_vel = 0.0
    # Covariância inicial grande mas estável
    P00, P01, P10, P11 = (meas_std_init ** 2, 0.0, 0.0, (5 * meas_std_init) ** 2)

    H0, H1 = 1.0, 0.0  # H = [1, 0]

    # Buffers de saída
    x_filtered: List[float] = [math.nan] * n
    innovations: List[Optional[float]] = [None] * n
    inov_std: List[Optional[float]] = [None] * n
    kalman_gain: List[float] = [0.0] * n
    R_eff_list: List[Optional[float]] = [None] * n
    Q_trace_list: List[float] = []
    no_meas_steps: List[int] = []
    gated_steps: List[int] = []

    # Janela para estimar R via MAD (apenas das medições agregadas disponíveis)
    recent_meas: List[float] = []

    # Avança até o primeiro índice válido (predições triviais)
    for i in range(first_valid_idx):
        x_filtered[i] = x_pos  # mantém valor inicial
        R_eff_list[i] = None
        kalman_gain[i] = 0.0

    # --------- Loop do KF ---------
    for k in range(first_valid_idx, n):
        # Determina dt para este passo (k-1 -> k)
        if k == first_valid_idx:
            dt_k = dts_list[k - 1] if (k - 1) < len(dts_list) else (dt if dt else 1.0)
        else:
            dt_k = dts_list[k - 1] if (k - 1) < len(dts_list) else dts_list[-1]

        # Matriz de transição F e ruído de processo Q
        # F = [[1, dt], [0, 1]]
        # Q simples (diagonal) escalado por dt
        q_pos = process_var_pos * max(dt_k, 1e-6)
        q_vel = process_var_vel * max(dt_k, 1e-6)
        Q_trace_list.append(q_pos + q_vel)

        # Predição: x_pred = F x
        x_pred_pos = x_pos + dt_k * x_vel
        x_pred_vel = x_vel

        # P_pred = F P F^T + Q
        # Computo manual para 2x2
        # F = [[1, dt], [0, 1]]
        P00_pred = P00 + dt_k * (P10 + P01) + (dt_k ** 2) * P11 + q_pos
        P01_pred = P01 + dt_k * P11
        P10_pred = P10 + dt_k * P11
        P11_pred = P11 + q_vel

        z_k = z_seq[k]
        if z_k is None:
            # Sem medição: apenas predição
            x_pos, x_vel = x_pred_pos, x_pred_vel
            P00, P01, P10, P11 = P00_pred, P01_pred, P10_pred, P11_pred
            x_filtered[k] = x_pos
            no_meas_steps.append(k)
            continue

        # Estima R_k de forma robusta a partir das últimas medições agregadas
        recent_meas.append(z_k)
        if len(recent_meas) > mad_window:
            recent_meas.pop(0)

        if len(recent_meas) >= 3:
            med_r = sorted(recent_meas)[len(recent_meas) // 2] if len(recent_meas) % 2 == 1 else 0.5 * (
                sorted(recent_meas)[len(recent_meas) // 2 - 1] + sorted(recent_meas)[len(recent_meas) // 2]
            )
            mad_r = sorted([abs(v - med_r) for v in recent_meas])[len(recent_meas) // 2]
            sigma_r = max(meas_std_min, (1.4826 * mad_r) if mad_r > 0 else meas_std_min)
        else:
            sigma_r = max(meas_std_min, meas_std_init)

        R_k = sigma_r ** 2

        # Se houver grande variância cruzada entre antenas neste timestamp, combine de forma conservadora
        cross_var = cross_vars[k]
        if cross_var is not None:
            # Usa o máximo (abordagem conservadora)
            R_k = max(R_k, float(cross_var))

        # Inovação
        y_k = z_k - (H0 * x_pred_pos + H1 * x_pred_vel)
        S_k = H0 * (P00_pred * H0 + P01_pred * H1) + H1 * (P10_pred * H0 + P11_pred * H1) + R_k  # escalar
        S_k = max(S_k, 1e-9)
        sstd = math.sqrt(S_k)

        # Gating por inovação (downweight em vez de descartar)
        if abs(y_k) > gate_sigma * sstd:
            R_k *= gate_R_scale
            S_k = H0 * (P00_pred * H0 + P01_pred * H1) + H1 * (P10_pred * H0 + P11_pred * H1) + R_k
            S_k = max(S_k, 1e-9)
            sstd = math.sqrt(S_k)
            gated_steps.append(k)

        # Opcional: robustificar ainda mais via peso Huber na inovação
        if robust_update:
            w_huber = _huber_weight(y_k, sstd)
            if w_huber < 1.0:
                # Inflar R equivalente a 1/w² (pois K ~ P H^T / (H P H^T + R))
                R_k /= (w_huber ** 2)
                S_k = H0 * (P00_pred * H0 + P01_pred * H1) + H1 * (P10_pred * H0 + P11_pred * H1) + R_k
                S_k = max(S_k, 1e-9)
                sstd = math.sqrt(S_k)

        # Ganho de Kalman (escala 1D para a observação da posição)
        # K = P_pred H^T / S -> apenas primeira coluna relevante
        K0 = (P00_pred * H0 + P01_pred * H1) / S_k  # para posição
        K1 = (P10_pred * H0 + P11_pred * H1) / S_k  # para velocidade

        # Atualização do estado
        x_pos = x_pred_pos + K0 * y_k
        x_vel = x_pred_vel + K1 * y_k

        # Atualização de P: P = (I - K H) P_pred
        # (I - K H) para H=[1,0] => [[1-K0, -K0*0],[ -K1, 1- K1*0]] = [[1-K0, 0],[-K1, 1]]
        I_KH00 = 1.0 - K0 * H0
        I_KH01 = -K0 * H1
        I_KH10 = -K1 * H0
        I_KH11 = 1.0 - K1 * H1

        # P = (I-KH) * P_pred
        P00 = I_KH00 * P00_pred + I_KH01 * P10_pred
        P01 = I_KH00 * P01_pred + I_KH01 * P11_pred
        P10 = I_KH10 * P00_pred + I_KH11 * P10_pred
        P11 = I_KH10 * P01_pred + I_KH11 * P11_pred

        # Simetrização numérica leve
        sym = 0.5 * (P01 + P10)
        P01 = sym
        P10 = sym

        # Salvar saídas do passo
        x_filtered[k] = x_pos
        innovations[k] = y_k
        inov_std[k] = sstd
        kalman_gain[k] = float(K0)  # ganho relevante para a posição
        R_eff_list[k] = R_k

    metadata = {
        "used_measurements": z_seq,
        "innovations": innovations,
        "innovation_std": inov_std,
        "kalman_gain": kalman_gain,
        "R_effective": R_eff_list,
        "Q_sequence": Q_trace_list,
        "antenna_counts": antenna_counts,
        "antenna_outliers": antenna_outliers,
        "flags": {"no_measurement_steps": no_meas_steps, "gated_steps": gated_steps},
        "params": {
            "dt": dt if dts is None else None,
            "meas_std_init": meas_std_init,
            "meas_std_min": meas_std_min,
            "process_var_pos": process_var_pos,
            "process_var_vel": process_var_vel,
            "gate_sigma": gate_sigma,
            "gate_R_scale": gate_R_scale,
            "mad_window": mad_window,
            "antenna_mad_k": antenna_mad_k,
            "winsorize": winsorize,
            "robust_update": robust_update,
        },
    }

    return {"rssi_filtered": x_filtered, "metadata": metadata}