
from firebase_functions import db_fn
from firebase_admin import initialize_app, db
from tl1 import TrilaterationController
from tl2 import TrilaterationController2
from tl3 import Trilateration3
from pf import RSSI_Localizer
from mle1 import TrilaterationRSSIMLE
from gpt1 import estimate_position_gpt_1
from gpt2 import estimate_position_gpt_2
from gpt2z import estimate_position_gpt_2_z
from gpt3z import estimate_position_gpt_3_z

from input_filter.gpt1 import rssi_filter_gpt_1
from input_filter.pf1 import rssi_filter_pf_1
from input_filter.kf1 import rssi_filter_kf_1
from input_filter.ukf1 import rssi_filter_ukf_1
from typing import Any, Dict, Tuple, List

# Constants
OLD_THRESHOLD_MS = 30000
DEFAULT_CALC_STRATEGY = "tl1"
DEFAULT_INPUT_FILTER_STRATEGY = "none"
DEFAULT_INPUT_FILTER_WINDOW = 5

app = initialize_app()

def get_recent_scans(scans: Dict[str, Any], now_timestamp: float, threshold: int = OLD_THRESHOLD_MS) -> List[Dict[str, Any]]:
    """
    Filter scans to only those with a server_timestamp within the threshold.
    """
    return [
        {**scan, "scan_id": scan_id}
        for scan_id, scan in scans.items()
        if "server_timestamp" in scan and now_timestamp - scan["server_timestamp"] < threshold
    ]

def get_antenna_positions(recent_scans: List[Dict[str, Any]], antenas_data: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Get the positions of the three antennas from the recent scans and antenna data.
    """
    return [
        (antenas_data[scan["scan_id"]]["x"], antenas_data[scan["scan_id"]]["y"])
        for scan in recent_scans
    ]

def build_pf_antenna_config(positions: List[Tuple[float, float]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the configuration list for the PF strategy antennas.
    """
    return [
        {
            "signalAttenuation": config.get("signal_attenuation", 3),
            "location": {"x": pos[0], "y": pos[1]},
            "reference": {
                "distance": config.get("reference_distance", 1),
                "signal": config.get("reference_signal", -70)
            }
        }
        for pos in positions
    ]

@db_fn.on_value_written(reference=r"/feed/{hardware_id}")
def on_db_write(event: db_fn.Event[dict]) -> None:
    """
    Triggered when a value is written to the database feed. Processes scan data and updates tag position.
    """
    print(event)
    hardware_id = event.params["hardware_id"]

    config_ref = db.reference(f"/config")
    config_data = config_ref.get() or {}

    input_filter_strategy = config_data.get("input_filter_strategy", DEFAULT_INPUT_FILTER_STRATEGY)
    if input_filter_strategy is None or input_filter_strategy == "none":
        print("No filter")
        scans = event.data.after or {}
    else:
        scans = filter_rssi_values(
            input_filter_strategy,
            hardware_id, 
            config_data["input_filter"].get(input_filter_strategy, {})
        )

    now_timestamp = event.time.timestamp() * 1000
    recent_scans = get_recent_scans(scans, now_timestamp)

    if len(recent_scans) < 3:
        print("There are not enough recent scans.")
        return

    # Order recent scans by RSSI (descending)
    recent_scans.sort(key=lambda x: x["rssi"], reverse=True)
    print(recent_scans)

    antenas_ref = db.reference("/antenas")
    antenas_data = antenas_ref.get() or {}
    antenna_positions = get_antenna_positions(recent_scans, antenas_data)

    calc_strategy = config_data.get("calc_strategy", DEFAULT_CALC_STRATEGY)

    estimated_position = calculate_position(
        strategy=calc_strategy,
        antenna_positions=antenna_positions,
        antenna_rssi=[scan["rssi"] for scan in recent_scans],
        config=config_data["calc"].get(calc_strategy, {})
    )

    print(estimated_position)

    tags_ref = db.reference("/tags")
    tags = tags_ref.get() or {}
    tag = next((tag for tag in tags.values() if tag.get("hardware_id") == hardware_id), None)

    if not tag:
        print("Tag not found.")
        return

    posicao_ref = db.reference(f"/posicoes/{tag['id']}")
    data = {
        "tag_id": tag["id"],
        "x": estimated_position[0],
        "y": estimated_position[1],
        "timestamp": now_timestamp
    }
    posicao_ref.set(data)

def calculate_position(
    strategy: str,
    antenna_positions: List[Tuple[float, float]],
    antenna_rssi: List[float],
    config: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Calculate the estimated position using the selected strategy.
    """
    if strategy == "tl1":
        print("Using trilateration strategy 1 with config:")
        print(config)
        trilateration = TrilaterationController(
            bp_1=antenna_positions[0],
            bp_2=antenna_positions[1],
            bp_3=antenna_positions[2],
            scale=config.get("scale", 64),
            measured_power=config.get("measured_power", -69),
            path_loss_exponent=config.get("path_loss_exponent", 1.8),
        )
        return trilateration.get_position(
            rssi_1=antenna_rssi[0],
            rssi_2=antenna_rssi[1],
            rssi_3=antenna_rssi[2],
        )
    elif strategy == "tl2":
        print("Using trilateration strategy 2 with config:")
        print(config)
        trilateration = TrilaterationController2(
            bp_1=antenna_positions[0],
            bp_2=antenna_positions[1],
            bp_3=antenna_positions[2],
            measured_power=config.get("measured_power", -69),
            path_loss_exponent=config.get("path_loss_exponent", 1.8),
        )
        return trilateration.get_position(
            rssi_1=antenna_rssi[0],
            rssi_2=antenna_rssi[1],
            rssi_3=antenna_rssi[2],
        )
    elif strategy == "tl3":
        print("Using trilateration strategy 3 with config:")
        print(config)
        if config.get("bounds_xy"):
            bounds_xy = (config["bounds_xy"]["xmin"], config["bounds_xy"]["ymin"]), (config["bounds_xy"]["xmax"], config["bounds_xy"]["ymax"])
        else:
            bounds_xy = None
        trilateration = Trilateration3(
            beacons=antenna_positions,
            measured_power=config.get("measured_power", -69),
            path_loss_exponent=config.get("path_loss_exponent", 1.8),
            bounds_xy=bounds_xy
        )
        return trilateration.estimate(
            rssis=antenna_rssi
        )
    elif strategy == "pf":
        print("Using pf strategy with config:")
        print(config)
        antenna_config = build_pf_antenna_config(antenna_positions, config)
        localizer = RSSI_Localizer(antenna_config)
        estimated_position = localizer.getNodePosition(antenna_rssi)
        return (estimated_position[0][0], estimated_position[1][0])
    
    elif strategy == "mle1":
        print("Using mle1 strategy with config:")
        print(config)
        if config.get("bounds_xy"):
            bounds_xy = (config["bounds_xy"]["xmin"], config["bounds_xy"]["ymin"]), (config["bounds_xy"]["xmax"], config["bounds_xy"]["ymax"])
        else:
            bounds_xy = None
        trilateration = TrilaterationRSSIMLE(
            beacons=antenna_positions,
            measured_power=config.get("measured_power", -69),
            path_loss_exponent=config.get("path_loss_exponent", 1.8),
            bounds_xy=bounds_xy
        )
        res = trilateration.estimate(
            rssis=antenna_rssi,
            estimate_A=config.get("estimate_A", True),      # estima A (@1m)
            estimate_n=config.get("estimate_n", False),     # mantenha n fixo (ative somente com ≥4-5 beacons bons)
            multistart=config.get("multistart", 8),
            ransac_max_trials=config.get("ransac_max_trials", 0), # 0 para desativar
            loss=config.get("loss", "cauchy"),
        )
        print(res)
        return res.x, res.y
    elif strategy == "gpt1":
        print("Using gpt1 strategy with config:")
        print(config)
        result = estimate_position_gpt_1(
            anchors=antenna_positions,
            rssis=antenna_rssi,
            **config
        )
        print(result)
        return result["x"], result["y"]
    elif strategy == "gpt2":
        print("Using gpt2 strategy with config:")
        print(config)
        result = estimate_position_gpt_2(
            anchors=antenna_positions,
            rssis=antenna_rssi,
            **config
        )
        print(result)
        return result["position"]
    elif strategy == "gpt2z":
        print("Using gpt2z strategy with config:")
        print(config)
        result = estimate_position_gpt_2_z(
            anchors=[(a[0], a[1], 0) for a in antenna_positions],
            rssis=antenna_rssi,
            **config
        )
        print(result)
        return result["position"]
    elif strategy == "gpt3z":
        print("Using gpt3z strategy with config:")
        print(config)
        result = estimate_position_gpt_3_z(
            anchors=[(a[0], a[1], 0) for a in antenna_positions],
            rssis=antenna_rssi,
            **config
        )
        print(result)
        return result["position"]
    elif strategy == "center4":
        print("Using center4 strategy with config:")
        print(config)
        # Find the 4 antennas with greatest RSSI
        if len(antenna_positions) < 4:
            raise ValueError("center4 strategy requires at least 4 antennas")
        # Get indices of top 4 RSSI values
        top_indices = sorted(range(len(antenna_rssi)), key=lambda i: antenna_rssi[i], reverse=True)[:4]
        top_positions = [antenna_positions[i] for i in top_indices]
        top_rssi = [antenna_rssi[i] for i in top_indices]
        # Calculate center (average) of the 4 positions
        center_x = sum(pos[0] for pos in top_positions) / 4
        center_y = sum(pos[1] for pos in top_positions) / 4
        # Check if RSSI values are close (e.g., max-min < threshold)
        rssi_range = max(top_rssi) - min(top_rssi)
        threshold = config.get("center4_rssi_threshold", 2.0)  # dBm, can be tuned
        if rssi_range <= threshold:
            # All RSSI are close, return center
            return (center_x, center_y)
        else:
            # Move proportionally towards the strongest RSSI
            # Weighted average by normalized RSSI (shift to positive)
            min_rssi = min(top_rssi)
            norm_rssi = [r - min_rssi + 1e-6 for r in top_rssi]  # avoid zero
            total = sum(norm_rssi)
            weighted_x = sum(pos[0] * w for pos, w in zip(top_positions, norm_rssi)) / total
            weighted_y = sum(pos[1] * w for pos, w in zip(top_positions, norm_rssi)) / total
            # Interpolate between center and weighted position
            alpha = min(rssi_range / (threshold * 2), 1.0)  # 0=center, 1=weighted
            x = center_x * (1 - alpha) + weighted_x * alpha
            y = center_y * (1 - alpha) + weighted_y * alpha
            return (x, y)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def filter_rssi_values(
    strategy: str,
    hardware_id: str,
    config: Dict[str, Any]
):
    history_ref = db.reference(f"/rssi_history/{hardware_id}")
    history = history_ref.get()

    if history is None:
        raise ValueError(f"No RSSI history found: {hardware_id}")
    
    if strategy == "avg1":
        print("Using avg1 filter with config:")
        print(config)
        filtered_history = {}
        for scan_id, entries in history.items():
            if len(entries) == 0:
                continue
            # Take the last N entries
            avg_rssi = sum(entry["rssi"] for entry in entries) / len(entries)
            filtered_history[scan_id] = {
                "rssi": avg_rssi,
                "server_timestamp": entries[-1]["timestamp"]
            }
        return filtered_history
    elif strategy == "gpt1":
        print("Using gpt1 filter with config:")
        print(config)
        filtered_history = {}
        for scan_id, entries in history.items():
            if len(entries) == 0:
                continue
            # Apply the RSSI filter
            filter_result, _ = rssi_filter_gpt_1(
                rssi_array=[entry["rssi"] for entry in entries],
                **config
            )
            filtered_history[scan_id] = {
                "rssi": filter_result,
                "server_timestamp": entries[-1]["timestamp"]
            }
        return filtered_history
    elif strategy == "pf1":
        print("Using pf1 filter with config:")
        print(config)
        filtered_history = {}
        for scan_id, entries in history.items():
            if len(entries) == 0:
                continue
            # Apply the RSSI filter
            filter_result = rssi_filter_pf_1(
                [entry["rssi"] for entry in entries],
                **config
            )
            filtered_history[scan_id] = {
                "rssi": round(filter_result["filtered"][-1], 2),
                "server_timestamp": entries[-1]["timestamp"]
            }
        return filtered_history
    elif strategy == "kf1":
        print("Using kf1 filter with config:")
        print(config)
        filtered_history = {}
        for scan_id, entries in history.items():
            if len(entries) == 0:
                continue
            # Apply the RSSI filter
            filter_result = rssi_filter_kf_1(
                [entry["rssi"] for entry in entries],
                **config
            )
            filtered_history[scan_id] = {
                "rssi": round(filter_result["rssi_filtered"][-1], 2),
                "server_timestamp": entries[-1]["timestamp"]
            }
        return filtered_history
    elif strategy == "ukf1":
        print("Using ukf1 filter with config:")
        print(config)
        filtered_history = {}
        for scan_id, entries in history.items():
            if len(entries) == 0:
                continue
            # Apply the RSSI filter
            filter_result = rssi_filter_ukf_1(
                [entry["rssi"] for entry in entries],
                **config
            )
            filtered_history[scan_id] = {
                "rssi": round(filter_result["rssi_filtered"][-1], 2),
                "server_timestamp": entries[-1]["timestamp"]
            }
        return filtered_history
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ---------------------------------------------------------------------

@db_fn.on_value_written(reference=r"/feed/{hardware_id}/{scan_id}")
def save_rssi_history(event: db_fn.Event[dict]) -> None:

    print(event)

    data = event.data

    if not data.after:
        # If the data was deleted, do nothing
        return
    
    data = event.data.after

    if not data.get("rssi") or not data.get("server_timestamp"):
        # If rssi or server_timestamp is missing, do nothing
        print("Invalid data")
        return
    
    hardware_id = event.params["hardware_id"]
    scan_id = event.params["scan_id"]
    
    history_ref = db.reference(f"/rssi_history/{hardware_id}/{scan_id}")
    history = history_ref.get() or []

    # if timestamp is older than the last entry, do nothing
    if len(history) > 0 and data["server_timestamp"] <= history[-1]["timestamp"]:
        print("Old timestamp, ignoring...")
        return
    
    config_ref = db.reference(f"/config/input_filter_window")
    input_filter_window = config_ref.get() or DEFAULT_INPUT_FILTER_WINDOW

    history.append({"rssi": data["rssi"], "timestamp": data["server_timestamp"]})
    history = history[-input_filter_window:]
    history_ref.set(history)