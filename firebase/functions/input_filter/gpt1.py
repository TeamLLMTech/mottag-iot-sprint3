import numpy as np

def rssi_filter_gpt_1(rssi_array, win=5, ema_alpha=0.3, hampel_k=3, initial_var=25.0):
    """
    Stateless RSSI filter. Processes the entire array and returns filtered RSSI and variance.
    """
    buf = rssi_array[-win:] if len(rssi_array) >= win else rssi_array
    x = np.array(buf)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) + 1e-6
    # Hampel: clip outliers
    x_clipped = np.clip(x, med - hampel_k * mad, med + hampel_k * mad)
    med2 = np.median(x_clipped)
    # EMA + variance (stateless: use med2 as initial EMA)
    ema = med2
    err = med2 - ema
    var = ema_alpha * (err * err) + (1 - ema_alpha) * initial_var
    return ema, max(var, 1.0)  # filtered RSSI, variance