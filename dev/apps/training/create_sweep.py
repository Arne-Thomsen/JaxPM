#!/usr/bin/env python
"""Helper script to create a wandb sweep and return only the sweep ID.

This script suppresses all wandb output and prints only the sweep ID,
making it safe to use in bash scripts.
"""

import sys
import os
import yaml

# Ensure we don't import from the local wandb/ directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

# Suppress wandb output
os.environ["WANDB_SILENT"] = "true"


def create_sweep(config_path: str, project: str) -> str:
    """Create a sweep and return only the sweep ID.

    The sweep name should be defined in the config file itself under the 'name' key.
    """
    import wandb

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Redirect outputs to suppress wandb's print statements
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect to devnull
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull

        sweep_id = wandb.sweep(config, project=project)

    finally:
        # Restore outputs
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    return sweep_id


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: create_sweep.py <config_path> <project>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    project = sys.argv[2]

    try:
        sweep_id = create_sweep(config_path, project)
        # Print ONLY the sweep ID to stdout
        print(sweep_id)
    except Exception as e:
        print(f"Error creating sweep: {e}", file=sys.stderr)
        sys.exit(1)
