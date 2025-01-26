#! /bin/bash
export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=${2:-10065}
export NODE_RANK=${3:-0}
export OMP_NUM_THREADS=6

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$NODE_RANK

echo $MASTER_ADDR
echo $MASTER_PORT
echo $NODE_RANK

NODE_RANK=$NODE_RANK CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python main.py fit -c configs/mini-vqgan-1.yaml