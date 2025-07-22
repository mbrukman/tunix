#!/bin/bash

# This script sets up a virtual environment with tunix and r2egym installed from local repositories.
# It requires three arguments: the path for the new venv, the path to the local tunix repo,
# and the path to the local R2E-Gym repo.

# Example Usage:
# bash tunix/experimental/deep_swe/scripts/setup_env.sh r2egym_tunix_venv . ../R2E-Gym

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_create_venv> <path_to_tunix_repo> <path_to_r2e_gym_repo>"
    exit 1
fi

VENV_PATH=$1
TUNIX_PATH=$2
R2E_GYM_PATH=$3

echo "Creating virtual environment at: ${VENV_PATH}"
# Create the virtual environment
python3 -m venv "${VENV_PATH}"

# Activate the virtual environment and install packages
source "${VENV_PATH}/bin/activate"

echo "Installing tunix from: ${TUNIX_PATH}"
pip install -e "${TUNIX_PATH}"

echo "Installing R2E-Gym from: ${R2E_GYM_PATH}"
pip install -e "${R2E_GYM_PATH}"

pip install pathwaysutils
pip install jax[tpu]

echo "Installing Jupyter"
pip install jupyter

echo "Virtual environment '${VENV_PATH}' is ready."
echo "To activate, run: source ${VENV_PATH}/bin/activate"
