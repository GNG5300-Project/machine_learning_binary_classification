import time
from absl import logging
from google.cloud import aiplatform
from google.oauth2 import service_account
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from pipelines.pipeline_gcp import create_pipeline
import read_env

# Generate unique pipeline name using current time
suffix = int(time.time())
PIPELINE_NAME = read_env.PIPELINE_NAME_PREFIX + str(suffix)
SERVICE_ACCOUNT_FILE = read_env.SERVICE_ACCOUNT_FILE
PROJECT_ID = read_env.PROJECT_ID
PIPELINE_ROOT = read_env.PIPELINE_ROOT
PIPELINE_METADATA = read_env.PIPELINE_METADATA
DATA_PATH = read_env.DATA_PATH
SERVING_DIR = read_env.SERVING_DIR
MODULES_DIR = read_env.MODULES_DIR
REGION = read_env.REGION

# Load credentials
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE)
# Initialize the Vertex AI client with the loaded credentials
print('credentials successfully retrieved')
aiplatform.init(credentials=credentials, project=PROJECT_ID, location=REGION)


def run():
    # Define TFX image (optional)
    tfx_image = 'gcr.io/carbide-theme-428210-v5/ml_classification:latest'
    runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
        default_image=tfx_image)

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

    # Create and run a pipeline job in GCP Vertex AI
    job = aiplatform.PipelineJob(
        template_path=PIPELINE_NAME + ".json",
        display_name=PIPELINE_NAME
    )
    job.run(sync=False)
