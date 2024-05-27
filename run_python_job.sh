#!/bin/bash
#SBATCH --job-name=python_test
#SBATCH --output=python_output_%j.txt
#SBATCH --error=python_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --mem=2G
#SBATCH --gres=gpu:1

# Source the environment setup script (modify the path if necessary)
source /usr/share/modules/init/bash

# Load any required modules
module load python/3.8.5

# Execute the Python script
python3 simple_script.py
