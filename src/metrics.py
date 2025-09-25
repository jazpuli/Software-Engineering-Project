import time

def compute_metrics_for_model(url: str) -> dict:
    model_name = url.split("/")[-1]
    start = time.time()

    # Placeholder metric values (replace with real computations later)
    ramp_up_time = 0.5
    bus_factor = 0.5
    performance_claims = 0.5
    license_score = 0.5
    dataset_and_code_score = 0.5
    dataset_quality = 0.5
    code_quality = 0.5
    size_score = {
        "raspberry_pi": 0.5,
        "jetson_nano": 0.5,
        "desktop_pc": 0.5,
        "aws_server": 0.5,
    }

    # --- NetScore calculation with weights ---
    net_score = (
        0.3 * ramp_up_time +
        0.2 * bus_factor +
        0.2 * license_score +
        0.2 * performance_claims +
        0.1 * size_score["desktop_pc"]
    )

    result = {
        "name": model_name,
        "category": "MODEL",
        "net_score": net_score,
        "ramp_up_time": ramp_up_time,
        "bus_factor": bus_factor,
        "performance_claims": performance_claims,
        "license": license_score,
        "dataset_and_code_score": dataset_and_code_score,
        "dataset_quality": dataset_quality,
        "code_quality": code_quality,
        "size_score": size_score,
    }

    # Add latency values (same dummy latency for now)
    elapsed = int((time.time() - start) * 1000)
    for key in list(result.keys()):
        if key not in ("name", "category", "size_score"):
            result[f"{key}_latency"] = elapsed

    return result
