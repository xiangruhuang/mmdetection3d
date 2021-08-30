#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29816}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=3 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
	--launcher pytorch ${@:3}

#--resume-from work_dirs/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus/epoch_9.pth \
