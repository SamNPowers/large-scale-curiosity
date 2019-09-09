#!/bin/sh

CUDA_VISIBLE="$1"
CONFIG_NAME="$2"

#CONDA_ENV="venv_curiosity"
CONDA_ENV="venv_curiosity_tf1_12"
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

#module load cuda-10.0
#module load cudnn-10.0-7.3
module load cuda-9.0
module load cudnn-9.0-7.0.5

export PYTHONPATH="$CURRENT_DIR/../..":"$CURRENT_DIR/../../baselines"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE

echo "Starting experiment with gpu $CUDA_VISIBLE_DEVICES and config $CONFIG_NAME"

cd "$CURRENT_DIR/.." || exit
nice -19 python3 curiosity_runner.py --experiment-config "experiment_configs/$CONFIG_NAME" --output-dir "experiment_configs/tmp"

sleep infinity