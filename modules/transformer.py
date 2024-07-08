import tensorflow as tf
import tensorflow_transform as tft

# Define the categorical and numerical columns along with their properties
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
NUM_OOV_BUCKETS = 2  # Number of out-of-vocabulary buckets for categorical features
LABEL_KEY = "Default"  # Label column

# Utility function to get transformed feature names


def transformed_name(key):
    return key + "_xf"

# Preprocessing function for the TFX pipeline


def preprocessing_fn(inputs):
    outputs = {}

    # Process numerical columns
    for key in NUMERICAL_COLUMNS:
        # Scale numerical features to [0, 1]
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[transformed_name(key)] = tf.reshape(
            scaled, [-1])  # Reshape to ensure correct dimensions

    # Process categorical columns
    for key, vocab_size in CATEGORICAL_COLUMNS_DICT.items():
        indices = tft.compute_and_apply_vocabulary(
            inputs[key], num_oov_buckets=NUM_OOV_BUCKETS)  # Compute and apply vocabulary
        # One-hot encode the indices
        one_hot = tf.one_hot(indices, vocab_size + NUM_OOV_BUCKETS)
        outputs[transformed_name(key)] = tf.reshape(
            one_hot, [-1, vocab_size + NUM_OOV_BUCKETS])  # Reshape to ensure correct dimensions

    # Process label column
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        inputs[LABEL_KEY], tf.float32)  # Cast label to float32

    return outputs
