import requests
import time

HF_API = "https://huggingface.co/api/models/"

def compute_metrics_for_model(url: str) -> dict:
    """
    Hugging Face API, compute metrics
    """
    model_name = url.split("/")[-1]
    start = time.time()

    try:
        response = requests.get(f"{HF_API}{model_name}", timeout=10)
        data = response.json()
    except Exception:
        data = {}

    # Example metric: ramp_up_time = proxy for documentation length
    ramp_up_time = len(data.get("cardData", {}).get("long_description", "")) / 1000
    ramp_up_time = min(ramp_up_time, 1.0)

    # Example metric: bus_factor = proxy for number of files in repo
    bus_factor = len(data.get("siblings", [])) / 10
    bus_factor = min(bus_factor, 1.0)

    # License check: 1 if exists, 0 otherwise
    license = 1.0 if data.get("license") else 0.0

    # Placeholders for now (to be expanded later)
    performance_claims = 0.5
    dataset_and_code_score = 0.5
    dataset_quality = 0.5
    code_quality = 0.5

    # Size score placeholder
    size_score = {
        "raspberry_pi": 0.5,
        "jetson_nano": 0.5,
        "desktop_pc": 0.5,
        "aws_server": 0.5,
    }

    # Simple NetScore: average of all metrics
    net_score = (
        ramp_up_time + bus_factor + license + performance_claims +
        dataset_and_code_score + dataset_quality + code_quality +
        sum(size_score.values())/4
    ) / 8

    elapsed = int((time.time() - start) * 1000)

    result = {
        "name": model_name,
        "category": "MODEL",
        "net_score": net_score,
        "ramp_up_time": ramp_up_time,
        "bus_factor": bus_factor,
        "performance_claims": performance_claims,
        "license": license,
        "dataset_and_code_score": dataset_and_code_score,
        "dataset_quality": dataset_quality,
        "code_quality": code_quality,
        "size_score": size_score,
    }

    # Add latency metrics
    for key in list(result.keys()):
        if key not in ("name", "category", "size_score"):
            result[f"{key}_latency"] = elapsed

    return result
