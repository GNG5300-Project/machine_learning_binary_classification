import base64
import json
import tensorflow as tf

# Original features dictionary
features = {
    'LoanID': 'I38PQUQS96',
    'Age': 56,
    'Income': 85994,
    'LoanAmount': 50587,
    'CreditScore': 520,
    'MonthsEmployed': 80,
    'NumCreditLines': 4,
    'InterestRate': 15.23,
    'LoanTerm': 36,
    'DTIRatio': 0.44,
    'Education': "Bachelor's",
    'EmploymentType': 'Full-time',
    'MaritalStatus': 'Divorced',
    'HasMortgage': 'Yes',
    'HasDependents': 'Yes',
    'LoanPurpose': 'Other',
    'HasCoSigner': 'Yes'
}

features1 = {
    "LoanID": "C1OZ6DPJ8Y",
    "Age": 46,
    "Income": 84208,
    "LoanAmount": 129188,
    "CreditScore": 451,
    "MonthsEmployed": 26,
    "NumCreditLines": 3,
    "InterestRate": 21.17,
    "LoanTerm": 24,
    "DTIRatio": 0.31,
    "Education": "Master's",
    "EmploymentType": "Unemployed",
    "MaritalStatus": "Divorced",
    "HasMortgage": "Yes",
    "HasDependents": "Yes",
    "LoanPurpose": "Auto",
    "HasCoSigner": "No"
}


def create_tf_feature(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    elif isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
    else:
        raise ValueError(f"Unsupported type for value: {type(value)}")


# Create tf.train.Feature for each item in features
feature_spec = {key: create_tf_feature(value)
                for key, value in features.items()}

# Create tf.train.Example from feature_spec
example = tf.train.Example(features=tf.train.Features(feature=feature_spec))

# Serialize tf.train.Example to string
serialized_example = example.SerializeToString()
print(serialized_example)

# Encode serialized Example using Base64
b64_example = base64.b64encode(serialized_example).decode()

# Create a JSON object with Base64 encoded Example
json_payload = {
    "instances": [
        {
            'examples': {
                'b64': b64_example
            }
        }
    ]
}

# Print or use the JSON payload
print(json.dumps(json_payload, indent=2))
