#!/usr/bin/env python3
"""Script to download the testcasque/ppe-detection-qlq3d model weights from Roboflow

Requirements:
    pip install roboflow
Usage:
    python download_model.py --api_key <YOUR_ROBOFLOW_API_KEY>
"""

import os
import sys
import argparse
import shutil
import zipfile
import yaml
import subprocess

def download_model(api_key: str, version: int = 1):
    print("Installing roboflow SDK if not present...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
    except Exception as e:
        print(f"Warning: Failed to install roboflow: {e}")

    from roboflow import Roboflow

    # 1. Initialize Roboflow
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    # 2. Get the workspace and project
    try:
        project = rf.workspace("testcasque").project("ppe-detection-qlq3d")
        version_obj = project.version(version)
    except Exception as e:
        print(f"Error accessing project: {e}")
        print("Please check that your ROBOFLOW_API_KEY is valid and has access to this public workspace.")
        return False

    # 3. Create models directory
    os.makedirs("models", exist_ok=True)
    temp_dir = "models/temp_roboflow"
    os.makedirs(temp_dir, exist_ok=True)

    # 4. Download model weights
    print(f"Downloading model version {version} weights...")
    try:
        # Download in PyTorch format
        version_obj.model.download(format="pt", location=temp_dir)
    except Exception as e:
        print(f"Error downloading weights: {e}")
        # Cleanup
        shutil.rmtree(temp_dir)
        return False

    # 5. Extract and rename the weights file
    # Roboflow model download saves weights in a subdirectory, typically 'weights/best.pt'
    # inside the target directory. Let's find it.
    weights_found = False
    for root, dirs, files in os.walk(temp_dir):
        if "best.pt" in files:
            source_path = os.path.join(root, "best.pt")
            target_path = "models/ppe-detection-qlq3d.pt"
            shutil.copy(source_path, target_path)
            print(f"Successfully extracted weights to {target_path}!")
            weights_found = True
            break

    # Cleanup temp directory
    shutil.rmtree(temp_dir)

    if not weights_found:
        print("Error: Could not locate 'best.pt' in the downloaded files.")
        return False

    # 6. Update config.yaml to point to this model
    config_path = "config.yaml"
    if os.path.exists(config_path):
        print(f"Updating {config_path} with new model path...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if "models" in config and "ppe" in config["models"]:
            config["models"]["ppe"]["path"] = "models/ppe-detection-qlq3d.pt"
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print("Successfully updated config.yaml!")
        else:
            print("Warning: Could not find 'models.ppe' structure in config.yaml.")
    else:
        print("Warning: config.yaml not found, skip updating config path.")

    print("\nImplementation complete! Run 'python main.py' to run with the new PPE model.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Roboflow PPE model weights.")
    parser.add_argument("--api_key", required=True, help="Your Roboflow API Key.")
    parser.add_argument("--version", type=int, default=1, help="Model version to download.")
    args = parser.parse_args()
    
    download_model(args.api_key, args.version)
