#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4          
#SBATCH --ntasks-per-node=4        
#SBATCH --gpus-per-task=1          
#SBATCH --cpus-per-task=32         
#SBATCH --job-name=parallel_sweep
#SBATCH --output="./logs/parallel_sweep_%j.log"

# Create the sweep once and get its ID
echo "Creating wandb sweep..."
SWEEP_ID=$(python create_sweep.py configs/hparams_sweep_64.yaml JaxHPM)

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep"
    exit 1
fi

echo "Created sweep with ID: $SWEEP_ID"
echo "View sweep at: https://wandb.ai/eth-cosmo/JaxHPM/sweeps/$SWEEP_ID"

srun --cpu-bind=threads --gpu-bind=single:1 \
    python run_training.py \
        --sweep_name "64_v1" \
        --sweep_id $SWEEP_ID \
        --sim_config configs/sim_64.yaml \
        --loss_config configs/particle_loss.yaml \
        --use_wandb