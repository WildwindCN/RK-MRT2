#!/bin/bash
#SBATCH --job-name=spectrostream-codec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=640G
#SBATCH --time=48:00:00
#SBATCH --output=logs/codec_%j.out
#SBATCH --error=logs/codec_%j.err

# SpectroStream Codec 训练 — 8×A100 80GB, 120s segments
# 1000h ≈ 30,000 segments, batch=96 → 312 steps/epoch
# 预计: Phase1 ~21min/epoch, Phase2 ~36min/epoch
# 48h → ~100 epochs (Phase1: 50, Phase2: ~50)

set -e

# --- NCCL 性能调优 (8×A100 NVLink) ---
export NCCL_DEBUG=WARN
export NCCL_ALGO=Ring
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_BUFFSIZE=2097152
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- CPU/IO 优化 ---
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data/synthetic-data/audio/ace-step-xl-turbo-factory/wav}"
CHECKPOINT_DIR="./checkpoints"
mkdir -p "$CHECKPOINT_DIR" logs

echo "============================================"
echo "SpectroStream Codec Training (8×A100 80GB, 120s segments)"
echo "  Model: 46M trainable, DDP bf16"
echo "  GPUs:  $SLURM_GPUS_PER_NODE × 80GB"
echo "  Batch: 12/GPU × 8 = 96 effective"
echo "  Segment: 120s (5.76M samples)"
echo "  Data:  $DATA_DIR"
echo "============================================"

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    training/train_codec.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_DIR" \
    --epochs 200 \
    --batch_size 12 \
    --lr 3e-4 \
    --precision bf16 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --segment_seconds 120.0

echo "Training complete."
