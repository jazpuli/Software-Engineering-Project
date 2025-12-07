"""Metrics computation service for artifact rating."""

import time
import requests
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from src.api.db import crud

# Hugging Face API base URL
HF_API = "https://huggingface.co/api/models/"

# Weights for NetScore calculation (must sum to ~1.0)
WEIGHTS = {
    "ramp_up_time": 0.12,
    "bus_factor": 0.12,
    "license": 0.12,
    "performance_claims": 0.12,
    "dataset_and_code_score": 0.10,
    "dataset_quality": 0.10,
    "code_quality": 0.10,
    "size_score": 0.10,
    "reproducibility": 0.06,
    "reviewedness": 0.06,
}


def compute_net_score(metrics: dict) -> float:
    """Compute weighted average of metrics (ignores latencies and treescore)."""
    size_avg = sum(metrics["size_score"].values()) / 4  # avg of 4 hardware targets

    score = 0.0
    score += WEIGHTS["ramp_up_time"] * metrics["ramp_up_time"]
    score += WEIGHTS["bus_factor"] * metrics["bus_factor"]
    score += WEIGHTS["license"] * metrics["license"]
    score += WEIGHTS["performance_claims"] * metrics["performance_claims"]
    score += WEIGHTS["dataset_and_code_score"] * metrics["dataset_and_code_score"]
    score += WEIGHTS["dataset_quality"] * metrics["dataset_quality"]
    score += WEIGHTS["code_quality"] * metrics["code_quality"]
    score += WEIGHTS["size_score"] * size_avg

    # Handle reproducibility (0, 0.5, or 1)
    score += WEIGHTS["reproducibility"] * metrics["reproducibility"]

    # Handle reviewedness (-1 means not available, treat as 0 for score)
    reviewedness = metrics["reviewedness"]
    if reviewedness >= 0:
        score += WEIGHTS["reviewedness"] * reviewedness

    return round(score, 3)


def fetch_huggingface_metadata(url: str) -> Dict[str, Any]:
    """Fetch metadata from HuggingFace API."""
    # Extract model name from URL
    model_name = url.rstrip("/").split("/")[-1]

    # Also try to get the org/model format
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        full_model_name = f"{parts[-2]}/{parts[-1]}"
    else:
        full_model_name = model_name

    try:
        response = requests.get(f"{HF_API}{full_model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    # Fallback: try just the model name
    try:
        response = requests.get(f"{HF_API}{model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    return {}


def compute_ramp_up_time(hf_data: dict) -> float:
    """Compute ramp_up_time metric based on documentation quality."""
    # Check multiple sources for documentation
    card_data = hf_data.get("cardData", {}) or {}

    # Try different documentation sources
    description = card_data.get("description", "") or ""
    long_desc = card_data.get("long_description", "") or ""

    # Also check model card content if available
    model_card = hf_data.get("card", "") or ""

    # Check README or other text fields
    readme = hf_data.get("readme", "") or ""

    total_length = len(description) + len(long_desc) + len(model_card) + len(readme)

    # Also give credit for having tags, pipeline info, etc.
    if hf_data.get("tags"):
        total_length += 100 * len(hf_data.get("tags", []))
    if hf_data.get("pipeline_tag"):
        total_length += 200
    if hf_data.get("library_name"):
        total_length += 100

    score = min(total_length / 1000, 1.0)
    return round(max(score, 0.3), 3)  # Minimum 0.3 for any model with metadata


def compute_bus_factor(hf_data: dict) -> float:
    """Compute bus_factor metric based on contributors/files."""
    # Proxy: number of files in repo (more files = more contributors typically)
    siblings = hf_data.get("siblings", [])
    score = min(len(siblings) / 10, 1.0)
    return round(score, 3)


def compute_license_score(hf_data: dict) -> float:
    """Compute license metric (1 if has license, 0 otherwise)."""
    # Check multiple places for license
    license_val = hf_data.get("license")
    card_data = hf_data.get("cardData", {}) or {}
    card_license = card_data.get("license")

    # Check tags for license info
    tags = hf_data.get("tags", []) or []
    license_tags = [t for t in tags if "license" in t.lower()]

    has_license = bool(license_val or card_license or license_tags)
    return 1.0 if has_license else 0.0


def compute_performance_claims(hf_data: dict) -> float:
    """Compute performance_claims metric."""
    # Check for eval results or benchmarks
    eval_results = hf_data.get("eval_results", [])
    if eval_results:
        return min(len(eval_results) / 5, 1.0)
    return 0.5  # Default if no eval data


def compute_dataset_and_code_score(hf_data: dict) -> float:
    """Compute dataset_and_code_score metric."""
    score = 0.0

    # Check for associated datasets
    has_dataset = bool(hf_data.get("dataset_tags"))
    if has_dataset:
        score += 0.3

    # Check for code files
    siblings = hf_data.get("siblings", []) or []
    code_extensions = (".py", ".ipynb", ".js", ".ts", ".java", ".cpp", ".c")
    has_code = any(
        s.get("rfilename", "").endswith(code_extensions)
        for s in siblings
    )
    if has_code:
        score += 0.3

    # Give partial credit for having config files (shows it's a proper model)
    config_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
    has_config = any(
        s.get("rfilename", "") in config_files
        for s in siblings
    )
    if has_config:
        score += 0.2

    # Give credit for having model files
    model_extensions = (".safetensors", ".bin", ".pt", ".onnx", ".h5")
    has_model = any(
        s.get("rfilename", "").endswith(model_extensions)
        for s in siblings
    )
    if has_model:
        score += 0.2

    return min(score, 1.0)


def compute_dataset_quality(hf_data: dict) -> float:
    """Compute dataset_quality metric."""
    # Placeholder: check for dataset documentation
    dataset_tags = hf_data.get("dataset_tags", [])
    if dataset_tags:
        return min(len(dataset_tags) / 3, 1.0)
    return 0.5


def compute_code_quality(hf_data: dict) -> float:
    """Compute code_quality metric."""
    # Placeholder: based on file organization
    siblings = hf_data.get("siblings", [])
    py_files = [s for s in siblings if s.get("rfilename", "").endswith(".py")]
    if py_files:
        return min(len(py_files) / 5, 1.0)
    return 0.5


def compute_size_score(hf_data: dict) -> dict:
    """Compute size scores for different hardware targets."""
    # Get model size from safetensors metadata or file sizes
    safetensors = hf_data.get("safetensors", {})
    total_size = safetensors.get("total", 0)

    if total_size == 0:
        # Try to estimate from files
        siblings = hf_data.get("siblings", [])
        for sibling in siblings:
            if "size" in sibling:
                total_size += sibling.get("size", 0)

    # Convert to GB
    size_gb = total_size / (1024 ** 3)

    # Score based on hardware capability
    return {
        "raspberry_pi": max(0, min(1.0, 1.0 - size_gb / 2)),      # < 2GB ideal
        "jetson_nano": max(0, min(1.0, 1.0 - size_gb / 4)),       # < 4GB ideal
        "desktop_pc": max(0, min(1.0, 1.0 - size_gb / 16)),       # < 16GB ideal
        "aws_server": max(0, min(1.0, 1.0 - size_gb / 64)),       # < 64GB ideal
    }


def compute_reproducibility(hf_data: dict) -> float:
    """
    Compute reproducibility metric.

    Returns:
        0 - No reproducibility indicators
        0.5 - Some indicators present
        1.0 - Strong reproducibility indicators
    """
    indicators = 0

    # Check for config files
    siblings = hf_data.get("siblings", [])
    config_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
    for sibling in siblings:
        if sibling.get("rfilename") in config_files:
            indicators += 1

    # Check for training info in card
    card_data = hf_data.get("cardData", {})
    if card_data.get("training_data") or card_data.get("training_procedure"):
        indicators += 2

    # Check for random seed documentation
    # This would require more sophisticated parsing

    if indicators >= 3:
        return 1.0
    elif indicators >= 1:
        return 0.5
    return 0.0


def compute_reviewedness(hf_data: dict) -> float:
    """
    Compute reviewedness metric.

    Per spec: The fraction of all code in the associated GitHub repository
    that was introduced through pull requests with a code review.
    Returns -1 if there is no linked GitHub repository.

    Returns:
        -1 - No GitHub repository linked
        0-1 - Fraction of code introduced through reviewed PRs
    """
    try:
        from src.api.services.github import (
            find_github_url_for_model,
            compute_reviewedness_for_repo,
        )

        # Try to find GitHub URL
        github_url = find_github_url_for_model(hf_data)

        if github_url:
            # Compute actual reviewedness from GitHub
            return compute_reviewedness_for_repo(github_url)

        # Fallback: use community engagement as proxy
        downloads = hf_data.get("downloads", 0)
        likes = hf_data.get("likes", 0)

        # High engagement suggests some level of review/vetting
        if downloads > 10000 and likes > 100:
            return 0.7  # Likely well-reviewed by community
        elif downloads > 1000 or likes > 10:
            return 0.3 + min(likes / 500, 0.3)

        return -1.0  # No GitHub and low engagement

    except Exception:
        return -1.0  # Cannot determine


def compute_treescore(db: Session, artifact_id: str) -> float:
    """
    Compute treescore as mean of parent artifact scores.

    Args:
        db: Database session
        artifact_id: ID of the artifact to compute treescore for

    Returns:
        Mean net_score of parent artifacts, or -1 if no parents (N/A)
    """
    parents = crud.get_parents(db, artifact_id)
    if not parents:
        return -1.0  # No parents = N/A, not 0

    parent_scores = []
    for parent in parents:
        rating = crud.get_latest_rating(db, parent.id)
        if rating:
            parent_scores.append(rating.net_score)

    if not parent_scores:
        return -1.0  # Parents exist but have no ratings = N/A

    return round(sum(parent_scores) / len(parent_scores), 3)


def compute_all_metrics(
    url: str,
    db: Optional[Session] = None,
    artifact_id: Optional[str] = None,
) -> dict:
    """
    Compute all metrics for an artifact.

    Args:
        url: Source URL (HuggingFace URL)
        db: Optional database session for treescore calculation
        artifact_id: Optional artifact ID for treescore calculation

    Returns:
        Dictionary containing all metric scores and latencies
    """
    start_time = time.time()

    # Fetch HuggingFace metadata
    hf_data = fetch_huggingface_metadata(url)

    # Compute individual metrics
    metrics = {
        "ramp_up_time": compute_ramp_up_time(hf_data),
        "bus_factor": compute_bus_factor(hf_data),
        "license": compute_license_score(hf_data),
        "performance_claims": compute_performance_claims(hf_data),
        "dataset_and_code_score": compute_dataset_and_code_score(hf_data),
        "dataset_quality": compute_dataset_quality(hf_data),
        "code_quality": compute_code_quality(hf_data),
        "size_score": compute_size_score(hf_data),
        "reproducibility": compute_reproducibility(hf_data),
        "reviewedness": compute_reviewedness(hf_data),
    }

    # Compute treescore if database context available
    if db and artifact_id:
        metrics["treescore"] = compute_treescore(db, artifact_id)
    else:
        metrics["treescore"] = -1.0  # N/A when no DB context (e.g., during ingest)

    # Compute net score
    metrics["net_score"] = compute_net_score(metrics)

    # Calculate latency
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Add latencies (same for all metrics in this implementation)
    latencies = {
        "net_score": elapsed_ms,
        "ramp_up_time": elapsed_ms,
        "bus_factor": elapsed_ms,
        "license": elapsed_ms,
        "performance_claims": elapsed_ms,
        "dataset_and_code_score": elapsed_ms,
        "dataset_quality": elapsed_ms,
        "code_quality": elapsed_ms,
    }

    return {
        "metrics": metrics,
        "latencies": latencies,
        "hf_data": hf_data,  # Include raw data for reference
    }


def passes_quality_threshold(metrics: dict, threshold: float = 0.5) -> bool:
    """
    Check if metrics pass the quality threshold for ingest.

    Critical metrics must be >= threshold. Some metrics are relaxed:
    - treescore: excluded (0 is valid for models without parents)
    - reviewedness: -1 means unknown, which is OK
    - net_score: checked separately as weighted average
    """
    # Critical metrics that must pass
    critical_keys = [
        "bus_factor", "license", "performance_claims",
        "code_quality",
    ]

    # Optional metrics (nice to have but not blocking)
    optional_keys = [
        "ramp_up_time", "dataset_and_code_score", "dataset_quality",
        "reproducibility",
    ]

    # Count failures
    critical_failures = 0
    optional_failures = 0

    for key in critical_keys:
        value = metrics.get(key, 0)
        if value < threshold:
            critical_failures += 1

    for key in optional_keys:
        value = metrics.get(key, 0)
        if value < threshold:
            optional_failures += 1

    # Fail if more than 1 critical metric fails
    if critical_failures > 1:
        return False

    # Fail if more than 2 optional metrics fail
    if optional_failures > 2:
        return False

    # Net score should be at least 0.3 (more lenient)
    net_score = metrics.get("net_score", 0)
    if net_score < 0.3:
        return False

    return True

