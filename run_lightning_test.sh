#!/bin/bash

# Create output directory for results
mkdir -p ./lightning_output

# Use just one dataset and model for initial testing
dataset="pathmnist"
model="resnet18"
epochs=3

echo "Running Lightning implementation for $dataset with $model..."
uv run python lightning_implementation.py \
    --data_flag $dataset \
    --output_root ./lightning_output \
    --num_epochs $epochs \
    --model_name $model \
    --load_size 28 \
    --resize_to 224 \
    --download \
    --gpu

echo "Lightning test completed!"