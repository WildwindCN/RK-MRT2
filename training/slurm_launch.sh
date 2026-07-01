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

# SpectroStream Codec — 8×80GB, max GPU utilization
# ~2GB/sample 激活 (120s bf16), batch=24 → ~48GB/GPU (60% safe)
# 1000h ≈ 30,000 segments, batch=192 → 156 steps/epoch
# Phase1 ~12min/epoch, Phase2 ~20min/epoch, 48h → ~150 epochs

set -e

# GPU perf lock (A800 300W)
nvidia-smi -pm 1 2>/dev/null || true
nvidia-smi -ac 1215,1410 2>/dev/null || true

# NCCL (NVSwitch)
export NCCL_DEBUG=WARN
export NCCL_ALGO=Tree
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data/synthetic-data/audio/ace-step-xl-turbo-factory/wav}"
CHECKPOINT_DIR="./checkpoints"
mkdir -p "$CHECKPOINT_DIR" logs

echo "=== SpectroStream Codec (8×80GB, batch=192) ==="
echo "Data: $DATA_DIR"

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    training/train_codec.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_DIR" \
    --epochs 200 \
    --batch_size 24 \
    --lr 3e-4 \
    --precision bf16 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --segment_seconds 120.0

echo "Done."
