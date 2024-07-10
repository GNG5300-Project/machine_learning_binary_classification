
# create the container
docker build -t train_with_docker_image:latest -f ./TrainDockerfile .

## build the model
models_path="models/local/"
absolute_file_path=$(readlink -f "$models_path")
docker run -it --rm -v $absolute_file_path:/app/models/local/ train_with_docker_image