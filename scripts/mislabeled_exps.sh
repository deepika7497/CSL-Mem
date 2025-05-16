#!/bin/bash

# Load the data directory path from config.json using Python
ROOT_PATH=$(python -c "import json; print(json.load(open('config.json'))['data_dir'])")

# Define common parameters
CONTAINER_NAME="learning-dynamics-models"
DATASETS=("cifar10" "cifar100")
NOISE_LEVELS=(0.01 0.02 0.05 0.1)
SEEDS=(1 2 3)
PARTS=(0 1)
EPOCHS=$(seq 0 199)

# Training Jobs

# Loop for training mislabelled models
echo "Starting training for mislabelled models..."
for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "Training mislabelled model: Dataset=${DATASET}, Seed=${SEED}, Noise=${NOISE_LEVEL}"
            python ./mislabelled_exps/train_mislabelled.py --dataset ${DATASET} --random_seed ${SEED} --label_noise ${NOISE_LEVEL}
        done
    done
done

# Loop for training k-fold confidence learning models
echo "Starting training for k-fold confidence learning models..."
for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "Training k-fold confidence learning model: Dataset=${DATASET}, Seed=${SEED}, Noise=${NOISE_LEVEL}"
            python ./mislabelled_exps/train_k_fold_conf_learning.py --dataset ${DATASET} --random_seed ${SEED} --label_noise ${NOISE_LEVEL}
        done
    done
done

# Loop for training SSFT models
echo "Starting training for SSFT models..."
for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for PART in "${PARTS[@]}"; do
                echo "Training SSFT model: Dataset=${DATASET}, Seed=${SEED}, Noise=${NOISE_LEVEL}, Part=${PART}"
                python ./mislabelled_exps/train_ssft.py --label_noise ${NOISE_LEVEL} --dataset ${DATASET} --random_seed ${SEED} --part ${PART}
            done
        done
    done
done

# Scoring Jobs

# Method to score the baseline metrics and our models
echo "Starting scoring for baseline metrics and our models..."
for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    for EPOCH in ${EPOCHS}; do
        for SEED in "${SEEDS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                MODEL_NAME="${DATASET}/${DATASET}_resnet18_noisy_idx_${SEED}_epoch_${EPOCH}_noise_${NOISE_LEVEL}.ckpt"
                echo "Scoring: Dataset=${DATASET}, Seed=${SEED}, Epoch=${EPOCH}, Noise=${NOISE_LEVEL}"
                python ./mislabelled_exps/score_mislabelled.py \
                    --dataset ${DATASET} \
                    --container_name ${CONTAINER_NAME} \
                    --model_name ${MODEL_NAME} \
                    --root_path ${ROOT_PATH} \
                    --load_from_azure_blob
            done
        done
    done
done

# Scorer to get learning time
echo "Starting learning time scoring..."
for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    for EPOCH in ${EPOCHS}; do
        for SEED in "${SEEDS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                MODEL_NAME="${DATASET}/${DATASET}_resnet18_noisy_idx_${SEED}_epoch_${EPOCH}_noise_${NOISE_LEVEL}.ckpt"
                echo "Scoring learning time: Dataset=${DATASET}, Seed=${SEED}, Epoch=${EPOCH}, Noise=${NOISE_LEVEL}"
                python ./mislabelled_exps/learning_time_scorer.py \
                    --dataset ${DATASET} \
                    --container_name ${CONTAINER_NAME} \
                    --model_name ${MODEL_NAME} \
                    --root_path ${ROOT_PATH}
            done
        done
    done
done

# Scoring confident learning
echo "Starting scoring for confident learning models..."
for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do

            echo "Scoring confident learning: Dataset=${DATASET}, Seed=${SEED}, Noise=${NOISE_LEVEL}"
            python ./mislabelled_exps/score_conf_learning.py \
                --dataset ${DATASET} \
                --random_seed ${SEED} \
                --label_noise ${NOISE_LEVEL}

        done
    done
done

echo "All training and scoring jobs complete."
