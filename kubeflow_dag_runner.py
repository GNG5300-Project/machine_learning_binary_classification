import time
from absl import logging
from google.cloud import aiplatform
from google.oauth2 import service_account
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from pipelines.pipeline_gcp import create_pipeline
import read_env

# Generate unique pipeline name using current time
suffix = int(time.time())
PIPELINE_NAME = read_env.PIPELINE_NAME_PREFIX + str(suffix)  # Full pipeline name with suffix
SERVICE_ACCOUNT_FILE = read_env.SERVICE_ACCOUNT_FILE  # Path to service account file
PROJECT_ID = read_env.PROJECT_ID  # GCP project ID
PIPELINE_ROOT = read_env.PIPELINE_ROOT  # Root directory for pipeline
PIPELINE_METADATA = read_env.PIPELINE_METADATA  # Metadata directory for pipeline
DATA_PATH = read_env.DATA_PATH  # Path to data
SERVING_DIR = read_env.SERVING_DIR  # Directory for serving the model
MODULES_DIR = read_env.MODULES_DIR  # Directory for modules
REGION = read_env.REGION  # GCP region

# Load credentials from service account file
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
# Initialize the Vertex AI client with the loaded credentials
print('credentials successfully retrieved')
aiplatform.init(credentials=credentials, project=PROJECT_ID, location=REGION)


def run():
    # Define TFX image (optional)
    tfx_image = 'gcr.io/carbide-theme-428210-v5/ml_classification:latest'  # TFX image for running the pipeline
    runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
        default_image=tfx_image)  # Configuration for Kubeflow DagRunner

    # Run Kubeflow DagRunner with the created pipeline
    kubeflow_v2_dag_runner.KubeflowV2DagRunner(config=runner_config, output_filename=PIPELINE_NAME+".json").run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,  # Name of the pipeline
            pipeline_root=PIPELINE_ROOT,  # Root directory for the pipeline
            serving_dir=SERVING_DIR,  # Directory for serving the model
            data_path=DATA_PATH,  # Path to the data
            module_path=MODULES_DIR  # Directory for modules
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)  # Set logging verbosity to INFO
    run()  # Run the pipeline

    # Create and run a pipeline job in GCP Vertex AI
    job = aiplatform.PipelineJob(
        template_path=PIPELINE_NAME + ".json",  # Path to the pipeline JSON file
        display_name=PIPELINE_NAME  # Display name for the pipeline job
    )
    job.run(sync=False)  # Run the job asynchronously
