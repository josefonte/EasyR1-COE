#!/bin/bash

#SBATCH --job-name=easyr1
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=400GB
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "🚀 Starting EasyR1 training job..."
echo "📅 Job started at: $(date)"

# Container and image names
CONTAINER_NAME="easyr1_docker_efficient"

# Start the container and run the training script
# Mount both workspace and storage directories
enroot start --root --rw \
    -m /home/hk-project-p0022560/lmu_eob1101/EfficientR1V:/workspace \
    "$CONTAINER_NAME" bash -c "
    bash examples/mllm_scripts/run_qwen_mllm_geoqa.sh
"