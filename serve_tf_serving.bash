#!/bin/bash

MODEL_NAME=""
PORT=8501
MODEL_PATH="/models/half_plus_two"

# Parse arguments
while getopts 'n:p:m:' flag; do
  case "${flag}" in
    n) MODEL_NAME="${OPTARG}" ;;
    p) PORT="${OPTARG}" ;;
    m) MODEL_PATH="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [[ -z "${MODEL_NAME}" ]]; then
  echo "Error: MODEL_NAME is required."
  exit 1
fi

echo "Starting TensorFlow Serving for model ${MODEL_NAME} on port ${PORT} with model path ${MODEL_PATH} ..."

docker run -t --rm -p ${PORT}:${PORT} \
    -v "${MODEL_PATH}:/models/${MODEL_NAME}" \
    -e MODEL_NAME=${MODEL_NAME} \
    tensorflow/serving

