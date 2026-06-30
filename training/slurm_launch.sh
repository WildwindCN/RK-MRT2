#!/bin/bash
#SBATCH --job-name=spectrostream-codec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --output=logs/codec_%j.out
#SBATCH --error=logs/codec_%j.err

# SpectroStream Codec 训练 — Slurm 启动脚本
# 8×A800, 200 epochs, ~50-70h

set -e

# 环境配置
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# 数据路径 (修改为实际路径)
DATA_DIR="${DATA_DIR:-/data/ace_studio_48khz}"
CHECKPOINT_DIR="./checkpoints"
mkdir -p "$CHECKPOINT_DIR" logs

echo "============================================"
echo "SpectroStream Codec Training"
echo "  Nodes: $SLURM_NNODES"
echo "  GPUs:  $SLURM_GPUS_PER_NODE"
echo "  Data:  $DATA_DIR"
echo "  Output: $CHECKPOINT_DIR"
echo "============================================"

# 单节点 DDP 启动
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    training/train_codec.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_DIR" \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --precision bf16 \
    --num_workers 4 \
    --segment_seconds 10.0

echo "Training complete."
