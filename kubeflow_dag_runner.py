import os
import time
from absl import logging
from typing import NamedTuple

from google.cloud import aiplatform
from google.oauth2 import service_account
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from pipeline import create_pipeline

from kfp.dsl import pipeline, component, OutputPath, InputPath, Output, Metrics
from kfp import compiler

from google_cloud_pipeline_components.v1.model import ModelUploadOp

# Define constants
SERVICE_ACCOUNT_FILE = 'gcp_key.json'
PROJECT_ID = 'carbide-theme-428210-v5'
PIPELINE_ROOT = 'gs://vertex-ai-train-393/base/'
PIPELINE_METADATA = 'gs://vertex-ai-train-393/metadata/'
DATA_PATH = 'gs://vertex-ai-train-393/data/'
SERVING_DIR = 'gs://vertex-ai-train-393/models/'
MODULES_DIR = 'gs://vertex-ai-train-393/modules/'
REGION = 'us-central1'

# Generate unique pipeline name using current time
suffix = int(time.time())
PIPELINE_NAME = f'loan-default-prediction-{suffix}'
# Load credentials
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
# Initialize the Vertex AI client with the loaded credentials
print('credentials successfully retrieved')
aiplatform.init(credentials=credentials, project=PROJECT_ID, location=REGION)


def run():
    # Configure metadata
    # metadata_config = kubeflow_v2_dag_runner.get_default_kubeflow_metadata_config()

    # Define TFX image (optional)
    tfx_image = 'gcr.io/carbide-theme-428210-v5/ml_classification'
    runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig()

    # Run Kubeflow DagRunner with the created pipeline
    kubeflow_v2_dag_runner.KubeflowV2DagRunner(config=runner_config, output_filename=PIPELINE_NAME+".json").run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            serving_dir=SERVING_DIR,
            data_path=DATA_PATH,
            module_path=MODULES_DIR
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()

    # Create and run a pipeline job in Vertex AI
    job = aiplatform.PipelineJob(
        template_path=PIPELINE_NAME + ".json",
        display_name=PIPELINE_NAME
    )
    job.run(sync=False)

# tfx pipeline create --pipeline-path=kubeflow_dag_runner.py --endpoint=https://6c04783b0063a2f7-dot-us-central1.pipelines.googleusercontent.com/
