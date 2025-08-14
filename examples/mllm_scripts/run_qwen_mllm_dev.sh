#!/bin/bash

# stop and clear all ray jobs
ray stop --force
# ensure no stale ray processes
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true

# prefer fresh local ray unless user explicitly provides an address
unset RAY_ADDRESS

set -x

# Set memory threshold for Ray to prevent OOM errors
export RAY_memory_usage_threshold=0.95

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME="Qwen2.5-VL-7B-GRPO-HOREKA-${TIMESTAMP}"

export PYTHONUNBUFFERED=1
export WANDB_ENTITY="Compression_MLLM"  # Team name
export WANDB_PROJECT="EasyR1-MLLM-COE"  # Project name
export WANDB_RUN_NAME="$RUN_NAME"
export WANDB_LOG_MODEL="end"
export WANDB_BASE_URL="https://api.wandb.ai/"
export WANDB_API_KEY="4707f68896fcc0f6d74763ecf5747773294205b8"

# Additional stability environment variables
export RAY_DEDUP_LOGS=0
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_LOGGING_LEVEL=WARNING


# Try to login to wandb with error handling
echo "🔑 Attempting to login to Weights & Biases..."
if wandb login "$WANDB_API_KEY" 2>/dev/null; then
    echo "✅ Wandb login successful (minimal disk usage mode)"
    echo "📊 Logging to: ${WANDB_ENTITY}/${WANDB_PROJECT}"
else
    echo "⚠️  Warning: Wandb login failed, proceeding without wandb tracking"
    echo "💡 Check your API key and internet connection"
    REPORT_TO="none"
fi


echo "🚀 Starting training..."
echo "📊 Model: ${MODEL_PATH}"
echo "🏷️  Run name: ${RUN_NAME}"

# Function to cleanup Ray processes
cleanup_ray() {
    echo "🧹 Cleaning up Ray processes..."
    ray stop --force || true
    sleep 2
    # Kill any remaining Ray processes
    pkill -f "ray::" || true
    pkill -f "raylet" || true
    sleep 1
}

# Set trap to cleanup on exit
trap cleanup_ray EXIT

# Run training with proper error handling
if python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.rollout_batch_size=64 \
    data.val_batch_size=64 \
    worker.actor.global_batch_size=16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.reward.reward_function=examples/reward_function/training_coe.py:compute_score \
    worker.reward.reward_type=coe_batch \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name="${WANDB_RUN_NAME}" \
    trainer.save_metadata=true; then
    echo "✅ Training completed successfully!"
    TRAINING_SUCCESS=true
else
    echo "❌ Training failed with exit code $?"
    TRAINING_SUCCESS=false
fi

# Manual cleanup before exit
echo "🧹 Performing manual cleanup..."
cleanup_ray

# Exit with appropriate code
if [ "$TRAINING_SUCCESS" = true ]; then
    echo "🎉 Job completed successfully!"
    exit 0
else
    echo "💥 Job failed during training"
    exit 1
fi
