# Introduction

This project focuses on building a machine learning model for binary classification in credit scoring. Utilizing a dataset containing various financial and demographic attributes of loan applicants, the goal is to predict whether an applicant will default on a loan. The project demonstrates the end-to-end workflow, including local deployment using TensorFlow Serving and cloud deployment with Vertex AI on Google Cloud Platform. This model aims to assist financial institutions in making informed lending decisions.

The source of the dataset is Kaggle and the link is: <https://www.kaggle.com/datasets/nikhil1e9/loan-default>

# Technologies Used

1. Tensorflow
2. Tensorflow Extended (TFX)
3. Keras
4. Keras Tuner
5. Tensorflow Metadata
6. Tensorflow Datasets
7. Docker
8. Tensorflow serving
9. Python
10. Numpy
11. Pandas
12. Scikit Learn
13. GCP
14. Vertex AI
15. Google Cloud Container Registry
16. Google Cloud Buckets
17. Google Cloud Build
18. Github
19. Github/GCP integration (for container building)
20. Tensor Board
21. GCP Vertex AI Logging
22. GCP Vertex AI Monitoring
23. GCP Vertex AI Online Model Deployment
24. GCP Vertex AI Online Model Endpoint

# Description of the Dataset

This dataset represents information about individual loan applications, containing various features related to the applicant's personal, financial, and loan-specific details. Here's a detailed explanation of each column:

1. **LoanID:** A unique identifier for each loan application. Example: I38PQUQS96.
2. **Age:** The age of the applicant. Example: 56 years old.
3. **Income:** The annual income of the applicant in dollars. Example: $85,994.
4. **LoanAmount:** The amount of money requested or approved for the loan in dollars. Example: $50,587.
5. **CreditScore:** The credit score of the applicant, a numerical expression based on the applicant's credit files, to represent the creditworthiness of the applicant. Example: 520.
6. **MonthsEmployed:** The number of months the applicant has been employed in their current job. Example: 80 months (or approximately 6 years and 8 months).
7. **NumCreditLines:** The number of credit lines (credit cards, loans, etc.) that the applicant has. Example: 4 credit lines.
8. **InterestRate:** The interest rate on the loan as a percentage. Example: 15.23%.
9. **LoanTerm:** The term of the loan in months. Example: 36 months (or 3 years).
10. **DTIRatio (Debt-to-Income Ratio):** The ratio of the applicant's monthly debt payments to their monthly income. Example: 0.44 (44%).
11. **Education:** The highest level of education attained by the applicant. Example: Bachelor's degree.
12. **EmploymentType:** The type of employment of the applicant. Example: Full-time.
13. **MaritalStatus:** The marital status of the applicant. Example: Divorced.
14. **HasMortgage:** A binary indicator (Yes/No) showing whether the applicant has a mortgage. Example: Yes.
15. **HasDependents:** A binary indicator (Yes/No) showing whether the applicant has dependents. Example: Yes.
16. **LoanPurpose:** The purpose for which the loan is being taken. Example: "Other."
17. **HasCoSigner:** A binary indicator (Yes/No) showing whether the loan has a co-signer. Example: Yes.
18. **Default:** The target (LABEL) variable indicating whether the loan has defaulted (1) or not (0). Example: 0.


# The Goal of This Project

The goal of this machine learning project is to develop a predictive model for loan risk assessment, utilizing a dataset of individual loan applications. The model aims to predict the likelihood of loan default based on various applicant features, such as age, income, loan amount, credit score, employment details, and more. By accurately predicting defaults, financial institutions can better manage risk, make informed lending decisions, and reduce potential losses. The insights gained from this model can also help tailor loan products to different customer profiles, enhancing overall financial stability and customer satisfaction.

## Steps

The following sections outline the steps in this process.

###

### Initial Data Introspection

To gain a comprehensive understanding of our dataset and ensure we are well-prepared for building the pipeline, we start with a simple Jupyter notebook. Note however that we do not perform any transformation processes in this notebook as they will be handled in the pipeline. We use the notebook to determine the state of the data in order to plan these transformation and intemediate processing in the pipeline. This step is crucial for several reasons:

1. **Content Exploration:** We first explore the contents of the dataset to get a sense of the data types, sample values, and overall structure. This helps us identify any immediate issues or areas that need attention.
2. **Schema Analysis:** We define and analyze the schema of the data. This step ensures that we have a clear understanding of the expected data types, ranges, and constraints for each feature, which is critical for subsequent processing and validation.
3. **Statistical Overview:** By generating descriptive statistics, we can quickly spot central tendencies, dispersion, and the presence of outliers. This statistical overview is essential for informing our preprocessing and feature engineering steps.
4. **Anomaly Detection**: We use the notebook to identify any anomalies in the data. Detecting anomalies early helps prevent unexpected issues during model training and ensures the quality and reliability of our data. In our case there were no anomalies.
5. **Data Skew and Imbalance**: Analyzing data skew and imbalance is vital for developing robust models. Understanding these aspects allows us to implement appropriate techniques, such as resampling or reweighting, to mitigate their impact on model performance. There was significant data imbalance in the Label column with a ratio of 88:12 for Default:Non-default clients.

By performing these preliminary analyses, we lay the groundwork for a smooth and efficient pipeline development process, ensuring that our data is well-understood and prepared for subsequent machine learning tasks.

# Building the Local Pipeline using TFX

Building a local machine learning pipeline using TensorFlow Extended (TFX) involves several well-defined steps. Each step is crucial for transforming raw data into a deployable machine learning model. Here’s a detailed outline of the steps we used:

## 1\. Data Ingestion

**Component:** ExampleGen

**Module file:** N/A

**Description**: The ExampleGen component ingests the raw data into the TFX pipeline. It splits the data into training and evaluation sets. This step ensures the data is in a consistent format and ready for further processing. Our split was 70:30. Where 70 was used for training and 30 used for evaluation.

## 2\. Data Validation

**Component**: StatisticsGen, SchemaGen, ExampleValidator

**Module file:** N/A

**Description**:

- StatisticsGen: Generates statistics for the dataset to understand the data distribution.
- SchemaGen: Automatically generates a schema for the data, defining the expected data types and ranges.
- ExampleValidator: Uses the schema and statistics to detect anomalies and missing values in the data.

Given we had already reviewed the data using our Jupyter notebook file, we did not have to make any transformations to correct for anomalies or schema errors. The data was anomaly and schema free.

## 3\. Data Transformation

**Component**: Transform

**Module file:** transform.py

**Description:** The Transform component preprocesses the data. This includes scaling numerical features and categorizing categorical features. The transformation logic is defined in a preprocessing function within the transformer.py file, ensuring that the same transformations are applied during both training and serving.

To prepare the data for training we had to perform scaling and categorization. We split the data columns into Numerical and Categorical features before we began.

### **Scaling:**

Normalizes numerical features to have a mean of zero and a standard deviation of one.

### Categorization

1. **Compute Vocabulary:** For each categorical feature, a vocabulary is computed to list all possible values.
2. **Apply Vocabulary:** The computed vocabulary is applied to transform the categorical features into integer indices.
3. **OOV Buckets:** Out-of-vocabulary (OOV) buckets are added to handle any categories in the test or future data that were not seen during training.
4. **One-Hot Encoding:** Converts the integer indices of categorical features into a binary matrix representation.
5. **Transformed Data Naming:** The transformed features are given new names to avoid overwriting the original data using the transformed_name function.

All these are handled in the transformer.py module.

## 4\. Tuning the Machine Learning Model with Bayesian Optimization

In our machine learning pipeline, hyperparameter tuning is a critical step to optimize the model's performance. This process involves searching for the best combination of hyperparameters that yield the highest model accuracy or other relevant metrics. Given the constraints of our machine resources, we employed the Bayesian Optimization approach for hyperparameter tuning using the BayesianOptimization tuner from Keras Tuner and a custom model builder function. The Hyperband tuner showed better promise at lower tuning configurations and would be considered the best tuner of choice for this process but it requires more powerful and expensive cpu/gpu configurations to execute at production levels.

### Steps in the Hyperparameter Tuning Process

#### Model Builder Function

**Description**: The model builder function defines the structure of the model and includes hyperparameters that will be tuned. This function is essential for the tuning process as it specifies which hyperparameters will be searched and their possible ranges.

Components:

1. **Input Layers:** Defines the input shape of the data.
2. **Hidden Layers:** Specifies the number of layers, units in each layer, and activation functions.
3. **Output Layer:** Defines the output layer with the appropriate activation function for the task (e.g., softmax for classification).
4. **Hyperparameters:** Includes learning rate, number of units in each hidden layer, dropout rates, and other relevant parameters.

#### Bayesian Optimization Tuner

**Description**: The Bayesian optimization tuner is used to search for the best hyperparameters efficiently. Bayesian Optimization builds a probabilistic model of the objective function and uses it to select the most promising hyperparameters to evaluate.

Advantages: This method is more sample-efficient compared to random search and grid search, making it suitable for scenarios with limited computational resources.

#### Integration with TFX Pipeline

**Description**: The hyperparameter tuning step is integrated into the TFX pipeline to ensure that the tuned model is used for further training and evaluation.

**Components:**

Tuner Component: A custom TFX component that encapsulates the hyperparameter tuning logic.

## 5\. Training the Machine Learning Model

After determining the best hyperparameters through tuning, we proceed to train the model using these optimized settings. In our case, we settled for a feed-forward deep neural network (DNN) as it showed more promise compared to a convolutional neural network (CNN) for our structured tabular data.

### Overview of the Training Process

#### Model Selection

**Feed-Forward Deep Neural Network**: Based on our data's nature and preliminary experiments, we found that a feed-forward DNN outperformed other architectures, including CNNs, in terms of predictive accuracy and computational efficiency. The DNN's architecture includes multiple dense layers, each followed by activation functions and dropout layers to prevent overfitting.

#### Hyperparameter Tuning Results

**Best Hyperparameters:** Through the Bayesian Optimization tuner, we identified the optimal hyperparameters for our model. This was used for the training data. An example of this information from one of our training sessions is shown below:

```json
{
  "space": [
    {
      "class_name": "Int",
      "config": {
        "name": "hidden_layers",
        "default": 1,
        "conditions": [],
        "min_value": 1,
        "max_value": 5,
        "step": 1,
        "sampling": "linear"
      }
    },
    {
      "class_name": "Choice",
      "config": {
        "name": "learning_rate",
        "default": 0.001,
        "conditions": [],
        "values": [0.01, 0.001, 0.0001],
        "ordered": true
      }
    },
    {
      "class_name": "Int",
      "config": {
        "name": "unit0",
        "default": 64,
        "conditions": [],
        "min_value": 8,
        "max_value": 256,
        "step": 64,
        "sampling": "linear"
      }
    },
    {
      "class_name": "Int",
      "config": {
        "name": "unit1",
        "default": 64,
        "conditions": [],
        "min_value": 8,
        "max_value": 256,
        "step": 64,
        "sampling": "linear"
      }
    },
    {
      "class_name": "Int",
      "config": {
        "name": "unit2",
        "default": 64,
        "conditions": [],
        "min_value": 8,
        "max_value": 256,
        "step": 64,
        "sampling": "linear"
      }
    },
    {
      "class_name": "Int",
      "config": {
        "name": "unit3",
        "default": 64,
        "conditions": [],
        "min_value": 8,
        "max_value": 256,
        "step": 64,
        "sampling": "linear"
      }
    },
    {
      "class_name": "Int",
      "config": {
        "name": "unit4",
        "default": 64,
        "conditions": [],
        "min_value": 8,
        "max_value": 256,
        "step": 64,
        "sampling": "linear"
      }
    }
  ],
  "values": {
    "hidden_layers": 2,
    "learning_rate": 0.001,
    "unit0": 72,
    "unit1": 72,
    "unit2": 8,
    "unit3": 136,
    "unit4": 72
  }
}
```

**Module**: trainer_local.py (which is specifically configured for our local training environment)

**Components**:

Data Preparation: Load and preprocess the training and validation datasets from the Tfx pipeline.

**Model Definition**: Use the best hyperparameters to build the model architecture.

Compilation: Compile the model with the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

**Training**: Train the model on the training data and validate it on the validation data. You can view the tensorboard logs by running:
```tensorboard --logdir=gs://vertex-ai-train-393/tb```

**Evaluation**: Assess the model's performance using metrics such as accuracy, precision, recall, and AUC.

## 6\. Model Resolver

**Components**: Resolver

**Module**: N/A

**Description**: The Model Resolver component is responsible for retrieving the latest blessed model. A blessed model is one that has passed all validation checks and is deemed suitable for deployment. By ensuring that only the latest and most effective model is used for further validation and comparison, the Model Resolver helps maintain a high standard for the models deployed in production.

## 7\. Model Evaluator

**Components**: Tensorflow Model Analysis

**Module**: N/A

**Description**: The Evaluator component is essential for assessing the performance of the newly trained model. It uses TensorFlow Model Analysis (TFMA) to compute various metrics and determine if the new model meets the required performance thresholds compared to the baseline model.

## Pusher Component

**Module**: N/A

**Components**: Pusher

**Input Dependencies**: The Pusher component typically depends on the outputs of both the Trainer and Evaluator components.

It requires the trained model artifact (model) from the Trainer component.

It also needs the model blessing (model_blessing) from the Evaluator component to ensure that only validated models are pushed to production.

**Functionality**:

1. **Validation**: The Pusher verifies that the model has been blessed (validated) before proceeding with deployment.
2. **Versioning**: It manages versioning of models to ensure that each deployment is uniquely identified and can be tracked.
3. **Deployment**: The Pusher orchestrates the deployment of the validated model to a specified deployment target (e.g., TensorFlow Serving, Kubernetes, etc.).

# File and Folders Structure

This is the file and folder structure for the project

- dataset/
  - Loan_default.csv
- extras/
- local_build/
- models/
- modules/
  - trainer_local.py
  - trainer.py
  - tuner.py
  - transformer.py
- pipelines/
  - pipeline_gcp.py
  - pipeline_local.py
- tb/
- .env
- cloudbuild.yaml
- Dockerfile
- generate_json_tf_request.py
- kubeflow_dag_runner.py
- local_dag_runner.py
- read_env.py
- requirements.txt
- serve_tf_serving.bash
- serve_with_tf_serving.py


# How to train, serve and test locally

1. CD to the root folder of the project
2. Run the follwoing code: ```python local_dag_runner.py```
3. This will begin the training process and once it is complete you should see a model file created in the models/local/ folder
4. Next, run the following in your terminal:
   ```python serve_with_tf_serving.py --port 8501 --model_name 1720308768 --model_path ./models/local/```

where --model_name is the name of the model you want to serve: --model_path is the path of the model which is ./models/local/ in our case

This command brings up a dockerized tensorflow serving server and after a few moments it is ready to accept requests.

1. You can test the model by running the following curl request in terminal:
```bash
curl -d '{
  "instances": [
    {
      "examples": {
        "b64": "Cp8DChgKBkxvYW5JRBIOCgwKCkkzOFBRVVFTOTYKDAoDQWdlEgUaAwoBOAoRCgZJbmNvbWUSBxoFCgPqnwUKFQoKTG9hbkFtb3VudBIHGgUKA5uLAwoVCgtDcmVkaXRTY29yZRIGGgQKAogEChcKDk1vbnRoc0VtcGxveWVkEgUaAwoBUAoXCg5OdW1DcmVkaXRMaW5lcxIFGgMKAQQKGAoMSW50ZXJlc3RSYXRlEggSBgoEFK5zQQoRCghMb2FuVGVybRIFGgMKASQKFAoIRFRJUmF0aW8SCBIGCgSuR+E+ChsKCUVkdWNhdGlvbhIOCgwKCkJhY2hlbG9yJ3MKHwoORW1wbG95bWVudFR5cGUSDQoLCglGdWxsLXRpbWUKHQoNTWFyaXRhbFN0YXR1cxIMCgoKCERpdm9yY2VkChYKC0hhc01vcnRnYWdlEgcKBQoDWWVzChgKDUhhc0RlcGVuZGVudHMSBwoFCgNZZXMKGAoLTG9hblB1cnBvc2USCQoHCgVPdGhlcgoWCgtIYXNDb1NpZ25lchIHCgUKA1llcw=="
      }
    }
  ]
}' -X POST http://localhost:8501/v1/models/1720308768:predict
```
# Training, Serving and Testing on Vertex AI

Integrate GitHub with Google Container Registry: This step enables building a custom image essential for the training process on Vertex AI. The image includes all dependencies specified in requirements.txt, which are copied into the Docker container during the build phase.

## Service Account Setup

Configure a service account with the necessary permissions to interact with Google Cloud resources. Generate a JSON-formatted service account key for secure access to your Google Cloud account.

## Google Cloud Storage Folder Structure

1\. Configure your Google Cloud buckets as Vertex AI uses them to store all generated artifacts during model building and training, including the final model itself.

- Data Folder: Store your dataset (data/).
- Modules Folder: Place module files (trainer.py, tuner.py, transformer.py, etc.) in the modules directory (modules/).
- Models Folder: Dedicated for storing final models (models/).
- TensorBoard Logs Folder: Store TensorBoard logs (tb/).
- Base Data: Contains artifacts and metadata crucial for the build process (base/).

**The folder structure is shown below:**

- vertex-ai-train-393/
  - base/
  - data/
    - Loan_Default.csv
  - models/
  - modules/
    - trainer.py
    - tuner.py
    - transformer.py
  - tb/

2\. kubeflow_dag_runner.py serves as the pipeline orchestrator for our Vertex AI deployment. It generates a pipeline instructions file using Kubeflow configuration and deploys it to the Vertex AI pipeline on GCP.

3\. The Vertex AI pipeline mirrors the local training steps to manage the training process. After training completes, you should be able to view the model in the models folder

### Training

On your local machine, run the following in terminal:

python kubeflow_dag_runner.py

This triggers the build process and pushes the pipeline code to Vertex AI for processing.

After processing is complete, you should see the trained model in the models folder

**Deploying And Serving the Model**

Deploying the Model to Vertex AI via UI

a. Navigate to Vertex AI

1. Go to Google Cloud Console: Navigate to the Google Cloud Console.
2. Open Vertex AI: Select the "Vertex AI" from the left-hand navigation menu.

b. Create a New Model

1. Navigate to Models: Click on "Models" in the Vertex AI section.
2. Create a New Model: Click on the "Create Model" button.
3. Fill in Details: Enter a name for your model and optionally, a description.

c. Upload Model Artifacts

1. Choose Model Type: Select the model type based on your artifact format (e.g., TensorFlow, custom prediction).
2. Upload Model: Click on "Upload Model" and select "Choose files from Cloud Storage."
3. Select Model Artifacts: Browse your Google Cloud Storage bucket and select the directory containing your model artifacts.
4. Configure Serving Container: Vertex AI provides default serving containers for common frameworks like TensorFlow. You can select the appropriate version.

d. Deploy Model

1. Deploy Model: Once the model artifacts are uploaded and configured, click "Deploy" to deploy your model.
2. Configure Deployment: Choose a deployment name, select the machine type (CPU/GPU), and set the number of replicas for scalability.
3. Advanced Settings: Optionally, configure advanced settings such as traffic splitting for A/B testing or canary deployments.
4. Confirm Deployment: Review your deployment settings and click "Deploy" to initiate the deployment process.

### Verify Deployment

Once deployed, Vertex AI will provision resources to serve your model. You can verify the deployment status and health through the Vertex AI UI.

### Monitor and Manage

**Monitoring:** Use Vertex AI’s monitoring tools to track model performance metrics, including latency, errors, and traffic.

**Scaling:** Adjust the number of replicas based on traffic demands to ensure optimal performance.