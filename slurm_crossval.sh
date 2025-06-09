#!/bin/bash
#SBATCH --job-name=challenge2
#SBATCH --output=slurm_output_%j.log
#SBATCH --error=slurm_error_%j.log
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

echo "Starting train_crossval.py"
python train_crossval.py

RESULT_DIR=$(ls -td results/* | head -n 1)

if [ -d "$RESULT_DIR" ]; then
    echo "Found results directory: $RESULT_DIR"
    echo "Starting test_crossval.py"
    python test_crossval.py "$RESULT_DIR"
else
    echo "No results directory found. Skipping test phase."
    exit 1
fi
