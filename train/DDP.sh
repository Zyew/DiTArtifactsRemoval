#!/bin/bash -l

#SBATCH --chdir=/home/hpc/iwi5/iwi5220h/DiTArtifactsRemoval

#SBATCH --partition=rtx3080         # 请求 RTX 3080 GPU 分区
#SBATCH --job-name=ddpm  # 作业名称
#SBATCH --time=24:00:00             # 设定最大运行时间为 24 小时
#SBATCH --gres=gpu:4                # 请求 4 个 RTX 3080 GPU
#SBATCH --cpus-per-task=12          # 每个任务使用 12 个 CPU 核心
#SBATCH --output=/home/hpc/iwi5/iwi5220h/DiTArtifactsRemoval/slurm/%j_%x_%Y%m%d.out   # 输出文件
#SBATCH --error=/home/hpc/iwi5/iwi5220h/DiTArtifactsRemoval/slurm/%j_%x_%Y%m%d_error.out  # 错误输出文件
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ziye.wang@fau.de


unset SLURM_EXPORT_ENV

source /home/hpc/iwi5/iwi5220h/miniconda3/bin/activate DiT
#conda activate DiT

echo "Job started on $(hostname) at $(date)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29800
export WORLD_SIZE=4
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=12
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    /home/hpc/iwi5/iwi5220h/DiTArtifactsRemoval/ddpm_ddp.py
    

echo "Job completed at $(date)"