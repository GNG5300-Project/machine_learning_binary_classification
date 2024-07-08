# Import necessary libraries and modules
from keras_tuner.engine import base_tuner
import kerastuner as kt
from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow as tf
import tensorflow_transform as tft
import numpy as np

# Define constants for categorical and numerical columns
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
NUM_OOV_BUCKETS = 2
LABEL_KEY = "Default"

# Utility function to get transformed feature names


def transformed_name(key):
    return key + '_xf'


# Define the result type for the tuner function
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

# Early stopping callback to avoid overfitting
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Function to read gzipped TFRecord files


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Function to create an input dataset


def _input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=128) -> tf.data.Dataset:
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    # Create batches of features and labels
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

# Function to build the model


def model_builder(hp):
    num_hidden_layers = hp.Int(
        'hidden_layers', min_value=1, max_value=5, default=1)
    hp_learning_rate = hp.Choice('learning_rate', values=[
                                 1e-2, 1e-3, 1e-4], default=1e-3)

    # Define input layers for numerical features
    input_numeric = [
        tf.keras.layers.Input(name=transformed_name(colname), shape=(1,), dtype=tf.float32) for colname in NUMERICAL_COLUMNS
    ]

    # Define input layers for categorical features
    input_categorical = [
        tf.keras.layers.Input(name=transformed_name(colname), shape=(vocab_size + NUM_OOV_BUCKETS,), dtype=tf.float32) for colname, vocab_size in CATEGORICAL_COLUMNS_DICT.items()
    ]

    input_layers = input_numeric + input_categorical

    # Concatenate numerical and categorical inputs
    input_numeric = tf.keras.layers.concatenate(input_numeric)
    input_categorical = tf.keras.layers.concatenate(input_categorical)
    deep = tf.keras.layers.concatenate([input_numeric, input_categorical])

    # Add hidden layers
    for i in range(num_hidden_layers):
        num_nodes = hp.Int('unit'+str(i), min_value=8,
                           max_value=256, step=64, default=64)
        deep = tf.keras.layers.Dense(num_nodes, activation='relu')(deep)

    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(deep)

    model = tf.keras.Model(input_layers, output)

    # Compile the model
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

    model.summary()

    return model

# Function to flatten the dataset


def flatten_dataset(dataset):
    flat_dataset = dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x))
    return flat_dataset

# Tuner function


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    # Initialize the tuner
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=5,  # Specify the number of trials
        executions_per_trial=5,
        directory=fn_args.working_dir,
        project_name='kt_random_search'
    )

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create training and validation datasets
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 1)
    val_data_set = _input_fn(fn_args.eval_files, tf_transform_output, 1)

    # Return the tuner and fit arguments
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            "x": train_dataset,
            "validation_data": val_data_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "class_weight": {0: 0.5681818181818182, 1: 4.166666666666667}
        }
    )
