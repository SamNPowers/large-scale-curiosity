#!/bin/bash

TMUX_NEW_NAME="$1"
CUDA_VISIBLE="$2"
CONFIG_NAME="$3"

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

tmux new -s "$TMUX_NEW_NAME" "$CURRENT_DIR/internal_yoda_runner.sh $CUDA_VISIBLE $CONFIG_NAME"
