"""Script to generate protobuf files from worker.proto."""

import os
import subprocess
from pathlib import Path

def generate_proto():
    """Generate protobuf files from worker.proto."""
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    
    # Path to the proto file
    proto_file = current_dir / "worker.proto"
    
    # Paths to generated files
    py_out = current_dir / "worker_pb2.py"
    grpc_py_out = current_dir / "worker_pb2_grpc.py"
    
    # Remove old generated files if they exist
    for f in [py_out, grpc_py_out]:
        if f.exists():
            f.unlink()
    
    # Generate Python files
    subprocess.run([
        "python3", "-m", "grpc_tools.protoc",
        f"--proto_path={current_dir}",
        f"--python_out={current_dir}",
        f"--grpc_python_out={current_dir}",
        str(proto_file)
    ], check=True)
    
    print(f"Protobuf files generated and old files overwritten in current dir: {current_dir}.")

if __name__ == "__main__":
    generate_proto() 