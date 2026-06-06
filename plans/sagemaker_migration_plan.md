# SageMaker Migration Plan: NYC Collision Severity Prediction

## Phase 1: AWS Environment Setup
1.  **S3 Bucket:** Create an S3 bucket (e.g., `sagemaker-aai540-group6-collisions`) to store raw data, processed features, and model artifacts.
2.  **IAM Role:** Ensure your SageMaker Execution Role has `AmazonS3FullAccess` and `AmazonSageMakerFullAccess`.

## Phase 2: Code Migration
1.  **Upload Scripts:** Upload the `src/` directory to your SageMaker Studio environment or clone the GitHub repository.
    *   `data_loader.py`: For fetching data into S3.
    *   `preprocessing.py` & `features.py`: For the SageMaker Processing Job.
    *   `sagemaker_train.py`: The entry point for the SageMaker Training Job.
2.  **Requirements:** Use the `requirements.txt` to ensure the SageMaker environment has `imbalanced-learn`.

## Phase 3: Orchestrating the Pipeline (The CI/CD DAG)
Use a SageMaker Studio Notebook to run the following steps:

### Step 1: Data Ingestion (To S3)
*   Run `data_loader.py` within a notebook cell or a small processing job to fetch a large sample (e.g., 500k+ rows) and save it directly to `s3://{bucket}/data/raw/`.

### Step 2: SageMaker Processing Job
*   Use the `SKLearnProcessor` from the SageMaker SDK.
*   Run a job that uses `preprocessing.py` and `features.py` to clean the data and engineer features.
*   The job will split the data into `train.csv` and `test.csv` and save them back to S3.

### Step 3: SageMaker Training Job
*   Use the `SKLearn` Estimator with `sagemaker_train.py` as the entry point.
*   Scale up: Choose a powerful instance (e.g., `ml.m5.2xlarge`) to handle the full dataset and SMOTE memory requirements.
*   Pass hyperparameters like `n_estimators=200` and `max_depth=15`.

### Step 4: Model Registration
*   Use the `model.register()` method to add the successful model to the **SageMaker Model Registry**.
*   This satisfies the "Model Registry" requirement for your deliverable.

## Phase 4: Validation & Deliverable
1.  Verify the model appears in the SageMaker console under **Models** -> **Model Registry**.
2.  Export the final Notebook as a PDF/HTML to include in your project submission as proof of SageMaker execution.
