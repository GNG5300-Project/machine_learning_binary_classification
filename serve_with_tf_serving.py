import subprocess
import argparse
import os

def serve_model(port, model_name, model_path):
    # Resolve model_path to an absolute path
    model_path = os.path.abspath(model_path)
    
    cmd = [
        "docker", "run", "-t", "--rm",
           "-d", 
        "-p", f"{port}:8501",
        "-v", f"{model_path}:/models/{model_name}",
        "-e", f"MODEL_NAME={model_name}",
        "tensorflow/serving"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a TensorFlow model using TensorFlow Serving in a Docker container.")
    parser.add_argument("--port", type=int, default=8501, help="Port number to expose TensorFlow Serving")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to serve")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")

    args, _ = parser.parse_known_args()  # Use _ to capture unrecognized arguments

    serve_model(args.port, args.model_name, args.model_path)
