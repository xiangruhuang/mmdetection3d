#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29816}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --resume-from work_dirs/3dssd_kitti-3d-3class/epoch_6.pth
