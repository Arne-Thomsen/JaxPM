#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=single
#SBATCH --output="./logs/single_%j.log"

srun --cpu-bind=threads --gpu-bind=none \
    python run_training.py \
        --hparams_config configs/hparams_cnn_64.yaml \
        --sim_config configs/sim_64.yaml \
        --loss_config configs/particle_loss_last.yaml \
        --use_wandb