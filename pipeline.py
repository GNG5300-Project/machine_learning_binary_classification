from cmath import pi
from posixpath import split
from tfx.proto import example_gen_pb2
from tfx.orchestration import pipeline 
import os  
from tfx.components import CsvExampleGen 
from tfx.components import StatisticsGen
from tfx.components import SchemaGen 
from tfx.components import ExampleValidator
from tfx.components import Transform 
from tfx.components import Tuner 
from tfx.proto import trainer_pb2

transformation = 'classification_transformation.py'

def create_pipeline(
    pipeline_name,
    pipeline_root,
    data_path,
    serving_dir,
    beam_pipeline_args = None,
    metadata_connection_config=None
):
  components =[]

  output = proto.Output(
      split_config=example_gen_pb2.SplitConfig(splits=[
          proto.SplitConfig.Split(name='train', hash_buckets=7),
          proto.SplitConfig.Split(name='eval', hash_buckets=3)
      ]))
  example_gen = tfx.components.CsvExampleGen(
      input_base=data_path, output_config=output)