#!/bin/bash

# Create output directory for results
mkdir -p ./lightning_output

# Array of 2D datasets to test - same as original for comparison
# Array of models to test - Only ResNet18 for faster run
models=("resnet18")
# Number of epochs - paper used 100 epochs
# datasets=(pathmnist chestmnist dermamnist octmnist pneumoniamnist retinamnist breastmnist bloodmnist tissuemnist organamnist organcmnist organsmnist)
datasets=(pathmnist)

# Run benchmark for each dataset and model
for data in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running Lightning implementation for $data with $model..."
        uv run python lightning_implementation.py \
            --data_flag $data \
            --output_root ./lightning_output \
            --model_name $model \
            --load_size 224 \
            --as_rgb \
            --gpu \
            --num_epochs 100 \
            # --weigh_loss \
            # --k_folds 5 \
            # --val_fold 0 
            # --resize_to 224 \
            # --download \

            # Compute and log TorchMetrics metrics
        echo "Finished Lightning implementation for $data with $model"
    done
done

echo "All Lightning benchmarks completed!"