#!/bin/bash
#SBATCH --job-name=python_test_job          # Job name
#SBATCH --output=python_output_%j.txt       # Output file (%j expands to jobID)
#SBATCH --error=python_error_%j.txt         # Error file (%j expands to jobID)
#SBATCH --time=00:05:00                     # Wall time (hh:mm:ss)
#SBATCH --partition=gpu                     # Partition name
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=1G                            # Memory per node

# Load the necessary module (if required)
module load python/3.8  # Adjust the module and version as needed

# Run the Python script
python simple_script.py
