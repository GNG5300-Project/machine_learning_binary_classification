from cmath import pi
from posixpath import split
import tfx
from tfx.proto import example_gen_pb2
from tfx.orchestration import pipeline
import os
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Tuner
from tfx.components import Trainer
from tfx.proto import trainer_pb2
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
import tensorflow_model_analysis as tfma
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.proto import pusher_pb2

transformation_module = 'classification_transformation.py'
tuner_module = 'classificationtuner.py'
trainer_module = 'classificationtrainer.py'

PROJECT_ID = 'carbide-theme-428210-v5'


def create_pipeline(
    pipeline_name,
    pipeline_root,
    data_path,
    serving_dir,
    module_path,
    beam_pipeline_args=None,
    metadata_connection_config=None
):

    components = []

    # example gen starts
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=7),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=3)
        ]))
    example_gen = CsvExampleGen(input_base=data_path, output_config=output)
    components.append(example_gen)
    # example gen end

    # statistics gen
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    components.append(statistics_gen)
    # statistics gen end

    # schema gen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    components.append(schema_gen)
    # schema gen end

    # validator
    validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )
    components.append(validator)
    # validator ends

    # transform
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_path + transformation_module
    )
    components.append(transform)
    # transform ends

    # tuner
    tuner = Tuner(
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=1),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=1),
        module_file=module_path + tuner_module
    )
    components.append(tuner)
    # tuner ends

    # trainer starts
    TRAINING_STEPS = 10
    EVALUATION_STEPS = 10

    trainer = Trainer(
        module_file=trainer_module,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAINING_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVALUATION_STEPS))

    components.append(trainer)
    # trainer ends

    # model resolver begins
    # Get the latest blessed model for model validation.
    model_resolver = Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')
    components.append(model_resolver)
# model resolver ends

# Evaluator starts
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Default_xf',
                                    preprocessing_function_names=['transform_features'])],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.1}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.000001})))
            ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )
    components.append(evaluator)
# evaluator ends

# pusher
    pusher = Pusher(
        model=trainer.outputs['model'],
        # disabled due to reduced steps for demonstration purposes
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_dir))
    )
    components.append(pusher)
# pusher ends

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config
    )
