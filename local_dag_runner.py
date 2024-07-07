import os
import time
from absl import logging
import tfx.v1 as tfx
from pipelines.pipeline_local import create_pipeline
import read_env
# Define constants
suffix = int(time.time())
PIPELINE_NAME = read_env.PIPELINE_NAME_PREFIX_LOCAL + str(suffix)
PIPELINE_ROOT = read_env.PIPELINE_ROOT_LOCAL
PIPELINE_METADATA = read_env.PIPELINE_METADATA_LOCAL
DATA_PATH = read_env.DATA_PATH_LOCAL
SERVING_DIR = read_env.SERVING_DIR_LOCAL
MODULES_DIR = read_env.MODULES_DIR_LOCAL
DB = read_env.DB_LOCAL

print(SERVING_DIR, os.path.join('.', SERVING_DIR,))
def run():
    # Configure metadata
    # Run Local DagRunner with the created pipeline
    tfx.orchestration.LocalDagRunner().run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=os.path.join('.', PIPELINE_ROOT,),
            enable_cache=False,
            serving_dir=os.path.join('.', SERVING_DIR,),
            data_path=os.path.join('.', DATA_PATH,),
            module_dir=os.path.join('.', MODULES_DIR,),
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
                os.path.join(".", PIPELINE_METADATA, DB))
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()