"""Project tasks. `uv run build.py gate` runs the checks every commit must pass."""

import argparse
import subprocess
import sys

# Each check runs as `python -m <check>` in this interpreter's environment.
GATE: list[list[str]] = [
    ["ruff", "check", "tunery", "tests", "build.py"],
    ["pytest", "-q"],
]


def gate(checks: list[list[str]] = GATE) -> int:
    """Run the checks in order; return the first failing exit code, else 0."""
    for check in checks:
        print(f"$ {' '.join(check)}", flush=True)
        returncode = subprocess.run([sys.executable, "-m", *check], check=False).returncode
        if returncode != 0:
            print(f"gate failed: {' '.join(check)} exited with {returncode}")
            return returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["gate"], help="gate: lint and test")
    parser.parse_args()
    return gate()


if __name__ == "__main__":
    sys.exit(main())
