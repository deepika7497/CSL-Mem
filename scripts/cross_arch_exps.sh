#!/bin/bash

# Load the data directory path from config.json using Python
ROOT_PATH=$(python -c "import json; print(json.load(open('config.json'))['data_dir'])")

# Define common parameters
DATASET="cifar100"
SEED=1
ARCHS=("vgg16" "fz_inception" "mobilenetv2")
EPOCHS=$(seq 0 199)
CONTAINER_NAME="learning-dynamics-models"

# Train the models for each architecture
echo "Starting training for different architectures..."
for ARCH in "${ARCHS[@]}"; do
    echo "Training: Architecture=${ARCH}, Dataset=${DATASET}, Seed=${SEED}"
    python train.py --dataset ${DATASET} --arch ${ARCH} --random_seed ${SEED}
done

# Score the models for each architecture across all epochs
echo "Starting scoring for different architectures..."
for EPOCH in ${EPOCHS}; do
    for ARCH in "${ARCHS[@]}"; do
        MODEL_NAME="${DATASET}/${DATASET}_${ARCH}_seed_${SEED}_epoch_${EPOCH}.ckpt"
        echo "Scoring: Architecture=${ARCH}, Epoch=${EPOCH}"
        python ./mislabelled_exps/score_mislabelled.py \
            --container_name ${CONTAINER_NAME} \
            --model_name ${MODEL_NAME} \
            --dataset ${DATASET} \
            --root_path ${ROOT_PATH} \
            --load_from_azure_blob \
            --arch ${ARCH}
    done
done

echo "Training and scoring complete."
