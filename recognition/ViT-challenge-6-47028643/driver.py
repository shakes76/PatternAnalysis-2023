"""
This file is used to run the created model/algorithm
either via first running train.py or predict.py, depending on whether the model exists.

Author: Felix Hall 
Student number: 47028643
"""
import os
import subprocess

# Define the paths to your model and Python files
model_path = "saved_models/best_model_45.pth"
predict_script = "predict.py"
train_script = "train.py"

# Check if the model file exists
if os.path.exists(model_path):
    # If the model exists, call predict.py
    try:
        subprocess.run(["python", predict_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {predict_script}: {e}")
else:
    # If the model doesn't exist, run train.py
    try:
        subprocess.run(["python", train_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {train_script}: {e}")