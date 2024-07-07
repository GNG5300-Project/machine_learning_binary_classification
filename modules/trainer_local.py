# Define imports
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
from tfx_bsl.public import tfxio
from typing import List
import logging
import keras_tuner
from absl import logging
import numpy as np
import os

_BATCH_SIZE = 128
_TENSOR_BOARD_LOG_DIR = os.path.join('.', '../tb')
CATEGORICAL_COLUMNS = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'LoanPurpose', 'HasMortgage', 'HasDependents', 'HasCoSigner'
]
CATEGORICAL_COLUMNS_DICT = {
    'Education': 4, 'EmploymentType': 4, 'MaritalStatus': 3,
    'LoanPurpose': 5, 'HasMortgage': 2, 'HasDependents': 2, 'HasCoSigner': 2
}
NUMERICAL_COLUMNS = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]
LABEL_KEY = "Default"
NUM_OOV_BUCKETS = 2


def transformed_name(key):
    return key + '_xf'


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
    processed = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=transformed_name(LABEL_KEY)),
        schema).repeat()
    return processed


def _get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = keras_tuner.HyperParameters()
    # Define search space.
    hp.Choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2], default=1e-2)
    hp.Int('hidden_layers', min_value=1, max_value=4, default=1)
    return hp


def _build_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
    input_numeric = [
        tf.keras.layers.Input(name=transformed_name(colname), shape=(1,), dtype=tf.float32) for colname in NUMERICAL_COLUMNS
    ]

    input_categorical = [
        tf.keras.layers.Input(name=transformed_name(colname), shape=(vocab_size + NUM_OOV_BUCKETS,), dtype=tf.float32) for colname, vocab_size in CATEGORICAL_COLUMNS_DICT.items()
    ]

    input_layers = input_numeric + input_categorical

    input_numeric_concat = tf.keras.layers.concatenate(input_numeric)
    input_categorical_concat = tf.keras.layers.concatenate(input_categorical)

    deep = tf.keras.layers.concatenate(
        [input_numeric_concat, input_categorical_concat])

    num_hidden_layers = hparams.get('hidden_layers')
    hp_learning_rate = hparams.get('learning_rate')

    for i in range(num_hidden_layers):
        num_nodes = hparams.Int('unit'+str(i), min_value=8,
                                max_value=256, step=64, default=64)
        deep = tf.keras.layers.Dense(num_nodes, activation='relu')(deep)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(deep)

    model = keras.Model(inputs=input_layers, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[
            'binary_accuracy',
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ]
    )

    model.summary(print_fn=logging.info)
    return model


def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tensorflow.Example`."""
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(LABEL_KEY)
        raw_features = tf.io.parse_example(
            serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(
            serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def dataset_to_numpy(dataset):
    features, labels = [], []

    for feature, label in dataset.as_numpy_iterator():
        features.append(feature)
        labels.append(label)

    return np.array(features), np.array(labels)


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    schema = tf_transform_output.transformed_metadata.schema

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=_BATCH_SIZE)

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=_BATCH_SIZE)

    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(
            fn_args.hyperparameters)
    else:
        hparams = _get_hyperparameters()
    logging.info('HyperParameters for training: %s', hparams.get_config())

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(hparams)

    tb_log_dir = _TENSOR_BOARD_LOG_DIR
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_dir, update_freq='epoch')

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        class_weight={0: 0.5681818181818182, 1: 4.166666666666667}
    )

    signatures = {
        'serving_default':
            _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features':
            _get_transform_features_signature(model, tf_transform_output)
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
