import os
import time
from absl import logging
import tfx.v1 as tfx
from pipelines.pipeline_local import create_pipeline
import read_env

# Define constants
suffix = int(time.time())  # Suffix for pipeline name based on current time
PIPELINE_NAME = read_env.PIPELINE_NAME_PREFIX_LOCAL + \
    str(suffix)  # Full pipeline name with suffix
PIPELINE_ROOT = read_env.PIPELINE_ROOT_LOCAL  # Root directory for pipeline
# Metadata directory for pipeline
PIPELINE_METADATA = read_env.PIPELINE_METADATA_LOCAL
DATA_PATH = read_env.DATA_PATH_LOCAL  # Path to data
SERVING_DIR = read_env.SERVING_DIR_LOCAL  # Directory for serving the model
MODULES_DIR = read_env.MODULES_DIR_LOCAL  # Directory for modules
DB = read_env.DB_LOCAL  # Local database connection string


def run():
    # Configure metadata
    # Run Local DagRunner with the created pipeline
    tfx.orchestration.LocalDagRunner().run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,  # Name of the pipeline
            # Root directory for the pipeline
            pipeline_root=os.path.join('.', PIPELINE_ROOT,),
            enable_cache=False,  # Disable caching
            # Directory for serving the model
            serving_dir=os.path.join('.', SERVING_DIR,),
            data_path=os.path.join('.', DATA_PATH,),  # Path to the data
            # Directory for modules
            module_dir=os.path.join('.', MODULES_DIR,),
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
                # Metadata connection configuration
                os.path.join(".", PIPELINE_METADATA, DB)
            )
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)  # Set logging verbosity to INFO
    run()  # Run the pipeline
