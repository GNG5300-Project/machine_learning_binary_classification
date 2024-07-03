import tensorflow as tf
import tensorflow_transform as tft

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


def tranformed_name(key):
    return key + "_xf"

def preprocessing_fn(inputs):
  outputs = {}
  for key in NUMERICAL_COLUMNS:
    scaled = tft.scale_to_0_1(inputs[key])
    outputs[tranformed_name(key)] = tf.reshape(scaled, [-1])
    
  for key, vocab_size in CATEGORICAL_COLUMNS_DICT.items():
    indices = tft.compute_and_apply_vocabulary(inputs[key], num_oov_buckets=NUM_OOV_BUCKETS)
    one_hot = tf.one_hot(indices,vocab_size + NUM_OOV_BUCKETS)
    outputs[tranformed_name(key)] = tf.reshape(one_hot,[ -1, vocab_size + NUM_OOV_BUCKETS])
    
  outputs[tranformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)
  return outputs
