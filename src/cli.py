import sys
import json
from pathlib import Path
from typing import List
from src.metrics import compute_metrics_for_model

def validate_file(url_file: str) -> Path:
    """Validate and return the file path, exiting if invalid."""
    path = Path(url_file)
    if not path.exists():
        print(f"Error: File not found: {url_file}", file=sys.stderr)
        sys.exit(1)
    return path

def read_urls_from_file(path: Path) -> List[str]:
    """Read and return list of URLs from file."""
    return path.read_text().splitlines()

def is_valid_model_url(url: str) -> bool:
    """Check if URL is a valid Hugging Face model URL (not dataset)."""
    return "huggingface.co/" in url and "/datasets/" not in url

def process_urls(urls: List[str]) -> None:
    """Process each URL and print metrics for valid model URLs."""
    for url in urls:
        if is_valid_model_url(url):
            try:
                result = compute_metrics_for_model(url)
                print(json.dumps(result))
            except Exception as e:
                print(f"Error processing {url}: {e}", file=sys.stderr)

def main(url_file: str) -> None:
    path = validate_file(url_file)
    urls = read_urls_from_file(path)
    process_urls(urls)
