# ICML 2025: Towards Memorization Estimation: Fast, Formal and Free

This repository contains the code and scripts to reproduce the experiments described in the ICML '25 paper _"Towards Memorization Estimation: Fast, Formal and Free"_. Follow the instructions below to set up and run the various experiments. If you found our work useful please considering citing our work.

```bibtex
@inproceedings{
    ravikumar2025towards,
    title={Towards Memorization Estimation: Fast, Formal and Free},
    author={Deepak Ravikumar and Efstathia Soufleri and Abolfazl Hashemi and Kaushik Roy},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=KZlQEoEtiu}
}
```

## Setup

To properly set up your environment, you will need to:
1. Install necessary dependencies.
2. Deploy a MinIO Docker container for storage.
3. Configure your environment with the correct `config.json` and credentials files.
4. Download and place the **FZ Influence Matrix checkpoints**.

### Step 1: Install Dependencies

Ensure that your Python environment includes the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Step 2: Deploy MinIO Docker Container

MinIO will act as your object storage service. To deploy a MinIO Docker container locally or on a remote server, follow these steps:

1. **Pull the MinIO Docker image**:
    ```bash
    docker pull minio/minio
    ```

2. **Run the MinIO container**:
    ```bash
    docker run -p 9000:9000 -p 9001:9001 --name minio \
        -e "MINIO_ROOT_USER=<your-access-key>" \
        -e "MINIO_ROOT_PASSWORD=<your-secret-key>" \
        minio/minio server /data --console-address ":9001"
    ```

    Replace `<your-access-key>` and `<your-secret-key>` with your desired credentials. This command runs MinIO on ports `9000` (for S3 access) and `9001` (for the MinIO console).

3. **Access MinIO Console**: Once the container is running, you can access the MinIO web console by navigating to:
    ```
    http://<your-server-ip>:9001
    ```

    Login using the access and secret keys you provided when running the container.

### Step 3: Create Configuration Files

You need to create two configuration files: `config.json` and `credentials.json`. 

#### **config.json**

Create a `config.json` file in the root directory of the project. This file should look like the following:

```json
{
    "log_dir": "./logs",
    "seeds_dir": "./seeds",
    "data_dir": "/local/path/to/your/data/directory/",
    "model_save_dir": "./pretrained/"
}
```

- Update the `"data_dir"` field to point to your local data directory where the datasets are stored.

#### **credentials.json**

Create a `credentials.json` file to store your MinIO credentials. This file should look like the following:

```json
{
    "url": "http://<minio-server-ip>:9001/api/v1/service-account-credentials",
    "endpoint": "<minio-server-ip>:9000",
    "accessKey": "<your-access-key>",
    "secretKey": "<your-secret-key>",
    "api": "s3v4",
    "path": "auto"
}
```

- Replace `<minio-server-ip>` with the IP address or domain where your MinIO service is hosted.
- Replace `<your-access-key>` and `<your-secret-key>` with the same values you used when setting up the MinIO Docker container.

### Step 4: Download FZ Influence Matrix Checkpoints

You will need to download the **FZ Influence Matrix checkpoints** and place them in the following directory:

```bash
analysis_checkpoints/dataset
```

Make sure the checkpoints are correctly placed in this directory for the analysis to work properly.

### Step 5: Verify the Setup

Ensure that your environment is configured correctly and that you have access to MinIO. You can use the MinIO console to verify that you can upload and access files from your data directory.

## Experiments

This section outlines the steps to reproduce the experiments from the paper. 

### 1. Mislabelled Experiment

To reproduce the mislabelled experiment results, follow these steps:

#### Step 1: Run the Mislabelled Experiment Script

The mislabelled experiment involves training multiple models (mislabelled, k-fold confidence learning, and SSFT models) and then scoring them. To run all the training and scoring jobs for the mislabelled experiments, execute the following script:

```bash
sh ./scripts/mislabeled_exps.sh
```

This script will:
- Train the mislabelled models, k-fold confidence learning models, and SSFT models for both CIFAR-10 and CIFAR-100 datasets, across different noise levels and seeds.
- Score the models for each epoch to compute baseline metrics and learning time.

#### Step 2: Analyze the Mislabelled Results

After running the mislabelled experiments, analyze the results using the `analyze_mislabeled.ipynb` notebook. This notebook will compute relevant metrics and generate visualizations as described in the paper.

To launch the notebook, run:

```bash
jupyter notebook analyze_mislabeled.ipynb
```

---

### 2. Duplicate Detection Experiment

To reproduce the duplicate detection experiment results, follow these steps:

#### Step 1: Run the Duplicate Detection Experiment Script

The duplicate detection experiment involves training SSFT models, confident learning models, and standard models for the CIFAR-10 and CIFAR-100 duplicate datasets, followed by scoring them using various metrics.

To run all the training and scoring jobs for the duplicate detection experiments, execute the following script:

```bash
sh ./scripts/duplicate_exps.sh
```

This script will:
- Train SSFT models with different seeds and parts for both CIFAR-10 and CIFAR-100 duplicate datasets.
- Train standard models for the duplicate datasets.
- Train confident learning models for both CIFAR-10 and CIFAR-100 duplicate datasets.
- Score the models and compute metrics such as loss curvature, gradient, and learning time for both datasets.

#### Step 2: Analyze the Duplicate Detection Results

Once the duplicate detection experiment has completed, analyze the results using the `analyze_duplicates.ipynb` notebook. This notebook will compute and visualize the relevant metrics as described in the paper.

To analyze the results, run the following command to launch the notebook:

```bash
jupyter notebook analyze_duplicates.ipynb
```

---


### 3. Cross-Architecture Memory Score Similarity Experiment

To reproduce the cross-architecture memory score similarity experiment results, follow these steps:

#### Step 1: Run the Cross-Architecture Experiment Script

The cross-architecture memory score similarity experiment involves training models using three architectures (`vgg16`, `fz_inception`, `mobilenetv2`) on the CIFAR-100 dataset and then scoring them across all epochs (0-199).

To run the cross-architecture experiment, execute the following script:

```bash
sh ./scripts/cross_architecture_exps.sh
```

This script will:
- Train the models using the three architectures.
- Score the models for each epoch to compute the cross-architecture memory score similarity.

#### Step 2: Analyze Cross-Architecture Results

Once the experiment is complete, you can analyze the results using the `mem_vs_arch.ipynb` notebook. This notebook will compute and visualize relevant metrics across the different architectures as described in the paper.

To launch the notebook, run:

```bash
jupyter notebook mem_vs_arch.ipynb
```

### Notes:
- Ensure that your `config.json` is properly configured with the correct `data_dir` path.
