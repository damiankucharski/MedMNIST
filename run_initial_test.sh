#!/bin/bash

# Create output directory for results
mkdir -p ./output

# Use just one dataset and model for initial testing
dataset="pathmnist"
model="resnet18"
epochs=3

echo "Running benchmark for $dataset with $model..."
uv run python original_benchmarks/train_and_eval_pytorch.py \
    --data_flag $dataset \
    --output_root ./output \
    --num_epochs $epochs \
    --model_flag $model \
    --download \
    --resize \
    --run "initial_test_224"

echo "Initial test completed!"