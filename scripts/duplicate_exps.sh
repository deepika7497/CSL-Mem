#!/bin/bash

# SSFT Train Jobs
echo "Starting SSFT training jobs..."
for DATASET in 'cifar10_duplicate' 'cifar100_duplicate'; do
    for LABEL_NOISE in 0.0; do
        for SEED in {1..3}; do
            for PART in 0 1; do
                echo "Training SSFT: Dataset=${DATASET}, Seed=${SEED}, Part=${PART}, Label Noise=${LABEL_NOISE}"
                python ./mislabelled_exps/train_ssft.py \
                    --label_noise ${LABEL_NOISE} \
                    --dataset ${DATASET} \
                    --random_seed ${SEED} \
                    --part ${PART}
            done
        done
    done
done

# Standard Training Models for CIFAR-10 and CIFAR-100 Duplicate Datasets
echo "Starting standard training for CIFAR-10 and CIFAR-100 duplicate datasets..."
for SEED in {1..3}; do
    echo "Training standard model: Dataset=cifar100_duplicate, Seed=${SEED}"
    python ./duplicate_exps/train_duplicates.py \
        --random_seed ${SEED} \
        --dataset cifar100_duplicate

    echo "Training standard model: Dataset=cifar10_duplicate, Seed=${SEED}"
    python ./duplicate_exps/train_duplicates.py \
        --random_seed ${SEED} \
        --dataset cifar10_duplicate
done

# Train Confident Learning Models
echo "Starting confident learning training jobs..."
for DATASET in "cifar100" "cifar10"; do
    for SEED in {1..3}; do
        echo "Training confident learning model: Dataset=${DATASET}_duplicate_noisy, Seed=${SEED}"
        python ./duplicate_exps/train_cl_k_fold_models.py \
            --dataset ${DATASET}_duplicate_noisy \
            --random_seed ${SEED}
    done
done

# Scoring Jobs

# Confident Learning Score Jobs
echo "Starting confident learning scoring jobs..."
for DATASET in 'cifar100' 'cifar10'; do
    for SEED in {1..3}; do
        echo "Scoring confident learning model: Dataset=${DATASET}_duplicate_noisy, Seed=${SEED}"
        python ./duplicate_exps/score_conf_learning.py \
            --dataset ${DATASET}_duplicate_noisy \
            --random_seed ${SEED}
    done
done

# Score Metrics (Loss Curvature, Gradient, Learning Time)
echo "Starting score metrics for CIFAR-100 duplicate dataset..."
for SEED in {1..3}; do
    for EPOCH in {0..199}; do
        echo "Scoring metrics: Dataset=cifar100_duplicate, Seed=${SEED}, Epoch=${EPOCH}"
        python ./duplicate_exps/score_loss_curv_lt.py \
            --epoch ${EPOCH} \
            --random_seed ${SEED} \
            --dataset cifar100_duplicate
    done
done

echo "Starting score metrics for CIFAR-10 duplicate dataset..."
for SEED in {1..3}; do
    for EPOCH in {0..199}; do
        echo "Scoring metrics: Dataset=cifar10_duplicate, Seed=${SEED}, Epoch=${EPOCH}"
        python ./duplicate_exps/score_loss_curv_lt.py \
            --epoch ${EPOCH} \
            --random_seed ${SEED} \
            --dataset cifar10_duplicate
    done
done

echo "All training and scoring jobs complete."
