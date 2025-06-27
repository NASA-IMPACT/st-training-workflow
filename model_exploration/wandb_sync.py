import os
import subprocess
import sys

# --- Configuration ---
wandb_entity = "sajil"
# Make sure this project name is EXACTLY correct!
wandb_project = "nasa_st_traning"
# Provide the full, correct path to the offline run directory
offline_run_dir = "wandb/run-20250504_175149-gni4inyd"

# --- Authentication Check (Optional but Recommended) ---
# Ensure the WANDB_API_KEY environment variable is set if not logged in via CLI
api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    print("Warning: WANDB_API_KEY environment variable not found.")
    # Add logic here if needed - e.g., try to read from a file, or exit
    # For now, we'll let `wandb sync` try to find credentials itself

# --- Construct the command ---
# Use sys.executable to ensure we use the python env's wandb installation
# Using "-m wandb" is often more robust than calling "wandb" directly
command = [
    sys.executable,  # Path to the current Python interpreter
    "-m",
    "wandb",  # Execute the wandb module
    "sync",
    "--project",
    wandb_project,
    "--entity",
    wandb_entity,
    offline_run_dir,
]

print(f"Running command: {' '.join(command)}")

# --- Execute the command ---
try:
    # Use check=True to raise an error if the command fails
    # capture_output=True captures stdout/stderr (optional)
    # text=True decodes stdout/stderr as text (requires Python 3.7+)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Sync command completed successfully.")
    print("stdout:\n", result.stdout)
    if result.stderr:
        print("stderr:\n", result.stderr)  # wandb often prints status to stderr

except subprocess.CalledProcessError as e:
    print(f"Error running wandb sync: {e}", file=sys.stderr)
    print(f"Return code: {e.returncode}", file=sys.stderr)
    print(f"stdout: {e.stdout}", file=sys.stderr)
    print(f"stderr: {e.stderr}", file=sys.stderr)  # Error details are often here
    # Handle the error appropriately

except FileNotFoundError:
    print(
        f"Error: Could not find the Python executable '{sys.executable}' or the 'wandb' module.",
        file=sys.stderr,
    )
    print(
        "Ensure wandb is installed in the Python environment ('pip install wandb').",
        file=sys.stderr,
    )
