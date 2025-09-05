#!/bin/bash

# Development version with smaller batches for hidden states extraction testing

# Comprehensive Ray cleanup
echo "🧹 Stopping all Ray processes..."
ray stop --force || true
sleep 10
echo "✅ Comprehensive Ray cleanup completed"

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME="Qwen2.5-VL-7B-GRPO-DEV-${TIMESTAMP}"

export PYTHONUNBUFFERED=1
export WANDB_ENTITY="Compression_MLLM"  # Team name
export WANDB_PROJECT="EasyR1-MLLM-COE-DEV"  # Development project
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

echo "🚀 Starting DEVELOPMENT training with smaller batches..."
echo "📊 Model: ${MODEL_PATH}"
echo "🏷️  Run name: ${RUN_NAME}"
echo "🔬 Hidden states extraction enabled!"

# Function to cleanup Ray processes
cleanup_ray() {
    echo "🧹 Cleaning up Ray processes..."
    ray stop --force || true
    sleep 10
    echo "✅ Comprehensive Ray cleanup completed"
}

# Set trap to cleanup on exit
trap cleanup_ray EXIT

# Run training with SMALLER BATCH SIZES for faster testing
if python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.rollout_batch_size=64 \
    data.mini_rollout_batch_size=32 \
    data.val_batch_size=32 \
    worker.actor.global_batch_size=32 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.n=3 \
    trainer.total_epochs=2 \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name="${WANDB_RUN_NAME}"; then
    echo "✅ Development training completed successfully!"
    TRAINING_SUCCESS=true
else
    echo "❌ Development training failed with exit code $?"
    TRAINING_SUCCESS=false
fi

# Manual cleanup before exit
echo "🧹 Performing manual cleanup..."
cleanup_ray

# Show hidden states output if available
if [ -d "./hidden_states_output" ]; then
    echo "📁 Hidden states output directory contents:"
    ls -la ./hidden_states_output/
    echo "📊 Total files: $(ls -1 ./hidden_states_output/ | wc -l)"
else
    echo "⚠️  No hidden states output directory found"
fi

# Exit with appropriate code
if [ "$TRAINING_SUCCESS" = true ]; then
    echo "🎉 Development job completed successfully!"
    echo "🔬 Check ./hidden_states_output/ for extracted tensors"
    exit 0
else
    echo "💥 Development job failed during training"
    exit 1
fi
