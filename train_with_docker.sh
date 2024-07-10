
# create the container
docker build -t train_with_docker_image:latest -f ./TrainDockerfile .

## build the model
models_path="models/local/"
absolute_file_path=$(readlink -f "$models_path")
docker run -it --rm -v $absolute_file_path:/app/models/local/ train_with_docker_image

This guide will walk you through the process of running the machine learning training code in a Docker container using a Dockerfile (TrainDockerfile). We  have already written all the steps involved in a bash script called 'train_with_docker.sh'. The script will build the docker image locally and use it train the model and finally write it into the mapped volume - models/local/

To use the script:
1. cd into the root directory of the project
2. Add execution rights to the script by running the following in the terminal  ```chmod +x ./train_with_docker.sh```
3. Now run the shell script with ./train_with_dokcer.sh 

The process will print all the steps from building to model training to the terminal and exit once complete.