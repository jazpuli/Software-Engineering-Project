# Software Engineering Project – CLI for Model Reuse

This project is a command-line tool that evaluates machine learning models hosted on Hugging Face.
It computes a set of metrics (ramp-up time, bus factor, license, etc.) and outputs results in JSON format.

## Team Members
- D'laney Lopez
- Jazmin Pulido

## Requirements
- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
```bash
# Install dependencies
./run install

# Run tests
./run test

# Process URLs from a file
./run urls.txt
```

## Project Structure
- `src/cli.py`: Main CLI entry point
- `src/metrics.py`: Metrics computation logic
- `tests/`: Test files
- `run`: Executable script for common operations
