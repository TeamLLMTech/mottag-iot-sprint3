def estimate_position_gpt_3_z(anchors, rssis, **kwargs):
    """
    Estima a posição 2D de um dispositivo baseado em medições de RSSI de múltiplas âncoras (antenas fixas).
    
    Usa trilateração em 2D (x, y) incorporando a altura (z) das âncoras no cálculo de distância (3D) para cada âncora.
    O algoritmo converte RSSI em distâncias aproximadas usando um modelo de perda de percurso (path loss) e então 
    resolve a posição via mínimos quadrados, com mecanismos para lidar com ruído e outliers nas medições.
    A implementação é *stateless*: cada chamada calcula a posição de forma independente.
    
    Parâmetros:
    - anchors: lista de tuplas (x, y, z) com coordenadas conhecidas das âncoras fixas. Deve conter pelo menos 3 âncoras 
      para determinar uma posição 2D unicamente. Exemplo: [(0.0, 0.0, 3.0), (5.0, 0.0, 3.0), (0.0, 5.0, 3.0), ...]
    - rssis: lista de valores RSSI (em dBm) medidos pelas respectivas âncoras, na mesma ordem da lista de anchors. 
      Cada valor corresponde ao RSSI recebido do dispositivo pela âncora naquela posição.
    - **kwargs: parâmetros opcionais para ajuste do algoritmo. Principais opções:
        - path_loss_exponent (float): Expoente de perda de percurso para conversão RSSI->distância. 
          Valor típico em espaço livre ~2.0, em ambiente indoor 2.0-3.0. [Padrão: 2.0]
        - reference_distance (float): Distância de referência em metros para o valor calibrado de RSSI. Geralmente 1.0 m. [Padrão: 1.0]
        - reference_rssi (float): RSSI esperado (dBm) na distância de referência. [Padrão: -59.0 dBm a 1 metro, valor comum para BLE calibrado]
        - device_height (float): Altura z assumida para o dispositivo (em metros). Isso influencia o cálculo de distância real até cada âncora.
          Por padrão 0.0 (ex: dispositivo no piso). Se o dispositivo estiver a alguma altura conhecida, pode ser fornecida.
        - robust (bool): Se True, ativa remoção de outliers de medidas de âncoras inconsistentes. [Padrão: True]
        - outlier_threshold (float): Limite de erro (em metros) para considerar uma âncora como outlier. Se None, o algoritmo decide baseado 
          na mediana dos erros residuais. [Padrão: None]
        - max_iterations (int): Máximo de iterações de refinamento/remoção de outliers. [Padrão: 10]
        - initial_position (tuple): Posição inicial de palpite (x, y) para o dispositivo. Se None, o algoritmo calcula um palpite inicial.
          [Padrão: None]
    
    Retorno:
    - dict com resultado da estimativa, contendo:
        - "position": tupla (x, y) estimada.
        - "uncertainty": medida de incerteza da estimativa. Pode ser um dicionário com "radius" indicando um raio (em metros) de incerteza 
          aproximado, ou None se não for aplicável.
        - "used_anchors": número de âncoras efetivamente usadas no cálculo final (desconsiderando outliers descartados).
        - "total_anchors": número total de âncoras recebidas na entrada.
        - "outliers": lista de índices das âncoras consideradas outliers e excluídas (lista vazia se nenhuma foi excluída).
        - Possíveis metadados adicionais, por exemplo "error_estimate" com o erro médio quadrático residual (RMSE) das distâncias.
        - Em caso de falha ou dados insuficientes, retorna {"error": "descrição do problema"}.
    
    Exemplo mínimo de uso:
    ```python
    anchors = [(0.0, 0.0, 3.0), (5.0, 0.0, 3.0), (0.0, 5.0, 3.0)]
    rssis = [-50.0, -60.0, -55.0]
    result = estimate_position(anchors, rssis)
    if "error" in result:
        print("Erro na estimativa:", result["error"])
    else:
        print("Posição estimada:", result["position"])
        print("Incerteza (raio aprox.):", result["uncertainty"]["radius"])
        print("Âncoras usadas:", result["used_anchors"], "de", result["total_anchors"])
    ```
    """
    import math
    
    # Validação básica das entradas:
    if anchors is None or rssis is None:
        return {"error": "Entradas anchors ou rssis não fornecidas."}
    try:
        anchors_list = list(anchors)
        rssis_list = list(rssis)
    except Exception as e:
        return {"error": f"Falha ao ler listas de anchors e RSSIs: {e}"}
    if len(anchors_list) != len(rssis_list):
        return {"error": "Tamanhos de listas anchors e rssis diferem."}
    N = len(anchors_list)
    if N < 3:
        return {"error": "Número insuficiente de âncoras. São necessárias pelo menos 3 âncoras para posição 2D."}
    
    # Checar se coordenadas das âncoras são válidas (tupla de 3 elementos numéricos)
    for idx, a in enumerate(anchors_list):
        if a is None or len(a) != 3:
            return {"error": f"Coordenadas da âncora {idx} inválidas (esperado tupla de 3 valores)."}
        # Check numérico:
        x_i, y_i, z_i = a
        try:
            float(x_i); float(y_i); float(z_i)
        except:
            return {"error": f"Coordenadas da âncora {idx} não numéricas."}
    # Checar RSSIs são numéricos:
    for idx, r in enumerate(rssis_list):
        try:
            float(r)
        except:
            return {"error": f"Valor RSSI {idx} ('{r}') não é numérico."}
    
    # Parâmetros do modelo de propagação e opções:
    n = kwargs.get('path_loss_exponent', 2.0)
    ref_dist = kwargs.get('reference_distance', 1.0)
    ref_rssi = kwargs.get('reference_rssi', -59.0)
    device_z = kwargs.get('device_height', 0.0)
    robust = kwargs.get('robust', True)
    outlier_threshold = kwargs.get('outlier_threshold', None)
    max_iter = kwargs.get('max_iterations', 10)
    initial_position = kwargs.get('initial_position', None)
    
    # Converter RSSI para distâncias estimadas usando modelo de perda de percurso logarítmico:
    distances = []
    for rssi in rssis_list:
        # Fórmula: RSSI = ref_rssi - 10 * n * log10(d / ref_dist)
        # => d = ref_dist * 10^((ref_rssi - RSSI) / (10 * n))
        d = ref_dist * (10 ** ((ref_rssi - rssi) / (10.0 * n)))
        if d == float('inf') or d != d:  # verifica inf ou NaN
            return {"error": "Valor de distância inválido calculado a partir de RSSI. Verifique parâmetros do modelo."}
        distances.append(d)
    
    # Função auxiliar para solver linear com lista de âncoras ativas (usadas)
    def solve_linear(indices):
        # Usa método dos mínimos quadrados linearizado (diferença de equações) para obter (x, y).
        if len(indices) < 3:
            raise ValueError("Menos de 3 âncoras para solver linear.")
        # Escolhe uma âncora de referência (última do índice fornecido)
        ref_idx = indices[-1]
        x_ref, y_ref, z_ref = anchors_list[ref_idx]
        d_ref = distances[ref_idx]
        # Monta sistema linear A*pos = B
        A = []
        B = []
        for i in indices:
            if i == ref_idx:
                continue
            xi, yi, zi = anchors_list[i]
            di = distances[i]
            # Coeficientes lineares:
            A_x = xi - x_ref
            A_y = yi - y_ref
            # Termo constante:
            B_val = (xi**2 + yi**2 + zi**2 - di**2) - (x_ref**2 + y_ref**2 + z_ref**2 - d_ref**2)
            B_val *= 0.5
            A.append([A_x, A_y])
            B.append(B_val)
        try:
            import numpy as np
            A = np.array(A, dtype=float)
            B = np.array(B, dtype=float)
            sol, _, rank, _ = np.linalg.lstsq(A, B, rcond=None)
            if rank < 2:
                # Sistema degenerado (âncoras colineares em 2D)
                raise ValueError("Âncoras colineares ou sistema singular.")
            x_sol, y_sol = float(sol[0]), float(sol[1])
            return x_sol, y_sol
        except ImportError:
            # Fallback se numpy não disponível (solução analítica para 3 âncoras)
            if len(indices) != 3:
                raise ValueError("Necessário numpy para resolver caso com múltiplas âncoras (>3).")
            # Resolver sistema 2x2 manualmente
            A11, A12 = A[0]
            A21, A22 = A[1]
            B1 = B[0]
            B2 = B[1]
            det = A11 * A22 - A12 * A21
            if abs(det) < 1e-9:
                raise ValueError("Âncoras colineares ou sistema singular (determinante ~0).")
            x_sol = (B1 * A22 - B2 * A12) / det
            y_sol = (A11 * B2 - A21 * B1) / det
            return x_sol, y_sol
    
    # Posição inicial:
    if initial_position is None:
        try:
            x_est, y_est = solve_linear(list(range(N)))
        except Exception as e:
            return {"error": f"Falha no cálculo inicial de posição: {e}"}
    else:
        try:
            x_est = float(initial_position[0])
            y_est = float(initial_position[1])
        except Exception as e:
            return {"error": f"Parâmetro initial_position inválido: {e}"}
    
    # Iterativamente refinar e remover outliers, se ativado:
    used_indices = list(range(N))
    outliers_removed = []
    iterations = 0
    while robust and iterations < max_iter:
        # Calcular resíduos das distâncias para âncoras atualmente usadas:
        residuals = []
        for i in used_indices:
            xi, yi, zi = anchors_list[i]
            di = distances[i]
            dist_pred = math.sqrt((xi - x_est)**2 + (yi - y_est)**2 + ((device_z - zi) ** 2))
            residuals.append(dist_pred - di)
        abs_res = [abs(r) for r in residuals]
        if not abs_res:
            break
        max_err = max(abs_res)
        max_idx_local = abs_res.index(max_err)
        max_idx = used_indices[max_idx_local]
        # Critério de outlier:
        if outlier_threshold is not None:
            threshold_val = outlier_threshold
        else:
            med_err = sorted(abs_res)[len(abs_res)//2]  # mediana dos erros
            threshold_val = 1e-3 if med_err < 1e-9 else 3 * med_err
        # Remove âncora com maior erro se exceder limite e ainda houver âncoras suficientes
        if max_err > threshold_val and len(used_indices) > 3:
            outliers_removed.append(max_idx)
            used_indices.remove(max_idx)
            try:
                x_est, y_est = solve_linear(used_indices)
            except Exception as e:
                return {"error": f"Falha após remover outlier (âncora {max_idx}): {e}"}
        else:
            break
        iterations += 1
    
    # Calcular incerteza básica (raio) e qualidade do ajuste:
    final_residuals = []
    for i in used_indices:
        xi, yi, zi = anchors_list[i]
        di = distances[i]
        dist_pred = math.sqrt((xi - x_est)**2 + (yi - y_est)**2 + ((device_z - zi) ** 2))
        final_residuals.append(dist_pred - di)
    rmse = None
    if final_residuals:
        rmse = math.sqrt(sum((r**2) for r in final_residuals) / len(final_residuals))
    
    # Montar resultado
    result = {
        "position": (x_est, y_est),
        "uncertainty": None,
        "used_anchors": len(used_indices),
        "total_anchors": N,
        "outliers": outliers_removed
    }
    if rmse is not None:
        result["uncertainty"] = {"radius": 2 * rmse}
        result["error_estimate"] = rmse
    return result
