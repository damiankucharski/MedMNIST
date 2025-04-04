#!/bin/bash

# Create output directory for results
mkdir -p ./output

# Array of 2D datasets to test
datasets=("dermamnist")
# Array of models to test - Only ResNet18 for faster run
models=("resnet18")
# Number of epochs - paper used 100 epochs
epochs=100

# Run benchmark for each dataset and model
for data in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running benchmark for $data with $model..."
        uv run python original_benchmarks/train_and_eval_pytorch.py \
            --data_flag $data \
            --output_root ./output \
            --num_epochs $epochs \
            --model_flag $model \
            --download \
            --resize \
            --as_rgb \
            --run "benchmark_${model}_224"
        echo "Finished benchmark for $data with $model"
    done
done

echo "All benchmarks completed!"