import os
import shutil
from sklearn.model_selection import train_test_split


print("hello")


# Define paths to your data directories
data_dir = 'AD_NC'  # Update with the actual path to your data directory

# Create directories for train and test sets
train_dir = os.path.join(data_dir, 'train/AD')
test_dir = os.path.join(data_dir, 'test/AD')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the proportion of data to allocate to the training set
train_split = 0.7  # 70% for training, adjust as needed

# Iterate through the "AD" and "NC" subdirectories and split the data
for class_name in ["AD", "NC"]:
    class_dir = os.path.join(data_dir, 'adnc-train', class_name)
    all_files = os.listdir(class_dir)

    # Split the data into training and testing sets for each class
    train_files, test_files = train_test_split(all_files, test_size=1 - train_split, random_state=42)

    # Move the files to their respective directories
    for file in train_files:
        source_path = os.path.join(class_dir, file)
        dest_path = os.path.join(train_dir, class_name, file)
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        shutil.move(source_path, dest_path)

    for file in test_files:
        source_path = os.path.join(class_dir, file)
        dest_path = os.path.join(test_dir, class_name, file)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        shutil.move(source_path, dest_path)
