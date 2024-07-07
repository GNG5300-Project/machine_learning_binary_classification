import os
from dotenv import load_dotenv

path= os.path.join('.', '.env')
load_dotenv(dotenv_path=path)

# local build constants
PIPELINE_ROOT_LOCAL = os.getenv('PIPELINE_ROOT_LOCAL')
DATA_PATH_LOCAL = os.getenv('DATA_PATH_LOCAL')
SERVING_DIR_LOCAL = os.getenv('SERVING_DIR_LOCAL')
MODULES_DIR_LOCAL = os.getenv('MODULES_DIR_LOCAL')
PIPELINE_METADATA_LOCAL = os.getenv('PIPELINE_METADATA_LOCAL')
DB_LOCAL = os.getenv('DB_LOCAL')
PIPELINE_NAME_PREFIX_LOCAL = os.getenv('PIPELINE_NAME_PREFIX_LOCAL')

# GCP (Google cloud platforms) Vertex AI constants

# Define constants
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
PROJECT_ID = os.getenv('PROJECT_ID')
PIPELINE_ROOT = os.getenv('PIPELINE_ROOT')
PIPELINE_METADATA = os.getenv('PIPELINE_METADATA')
DATA_PATH = os.getenv('DATA_PATH')
SERVING_DIR = os.getenv('SERVING_DIR')
MODULES_DIR = os.getenv('MODULES_DIR')
REGION = os.getenv('REGION')
PIPELINE_NAME_PREFIX = 'loan-default-prediction-'
