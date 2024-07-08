import os
from dotenv import load_dotenv

# Load environment variables from .env file
path= os.path.join('.', '.env')
load_dotenv(dotenv_path=path)

# Local build constants
PIPELINE_ROOT_LOCAL = os.getenv('PIPELINE_ROOT_LOCAL')  # Root directory for local pipeline
DATA_PATH_LOCAL = os.getenv('DATA_PATH_LOCAL')  # Path to local data
SERVING_DIR_LOCAL = os.getenv('SERVING_DIR_LOCAL')  # Directory for serving local model
MODULES_DIR_LOCAL = os.getenv('MODULES_DIR_LOCAL')  # Directory for local modules
PIPELINE_METADATA_LOCAL = os.getenv('PIPELINE_METADATA_LOCAL')  # Metadata directory for local pipeline
DB_LOCAL = os.getenv('DB_LOCAL')  # Local database connection string
PIPELINE_NAME_PREFIX_LOCAL = os.getenv('PIPELINE_NAME_PREFIX_LOCAL')  # Prefix for local pipeline names

# GCP (Google Cloud Platform) Vertex AI constants

# Define constants for GCP Vertex AI
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')  # Path to service account file
PROJECT_ID = os.getenv('PROJECT_ID')  # GCP project ID
PIPELINE_ROOT = os.getenv('PIPELINE_ROOT')  # Root directory for GCP pipeline
PIPELINE_METADATA = os.getenv('PIPELINE_METADATA')  # Metadata directory for GCP pipeline
DATA_PATH = os.getenv('DATA_PATH')  # Path to data in GCP
SERVING_DIR = os.getenv('SERVING_DIR')  # Directory for serving model in GCP
MODULES_DIR = os.getenv('MODULES_DIR')  # Directory for GCP modules
REGION = os.getenv('REGION')  # GCP region
PIPELINE_NAME_PREFIX = 'loan-default-prediction-'  # Prefix for GCP pipeline names
