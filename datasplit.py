import os
import shutil
from sklearn.model_selection import train_test_split

print("Starting data split process...")

# Define paths to your data directories
data_dir = 'AD_NC'  # Update with the actual path to your data directory

# Define the proportion of data to allocate to the training set
train_split = 0.7  # 70% for training, adjust as needed
print(f"Allocating {train_split * 100}% of data for training...")

# Iterate through the "train" and "test" subdirectories and split the data
for data_split in ["train", "test"]:
    for class_name in ["AD", "NC"]:
        class_dir = os.path.join(data_dir, data_split, class_name)
        all_files = os.listdir(class_dir)
        total_files = len(all_files)
        print(f"Total {class_name} files in {data_split} set: {total_files}")

        if total_files > 0:
            # Split the data into training and testing sets for each class
            train_files, test_files = train_test_split(all_files, test_size=1 - train_split, random_state=42)
            print(f"Allocating {len(train_files)} files for training and {len(test_files)} files for testing for class {class_name}...")

            # Move the files to their respective directories
            for file in train_files:
                source_path = os.path.join(class_dir, file)
                dest_path = os.path.join(data_dir, 'train' if data_split == 'train' else 'test', class_name, file)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if not os.path.isdir(source_path) and source_path != dest_path:
                    shutil.copy(source_path, dest_path)

            for file in test_files:
                source_path = os.path.join(class_dir, file)
                dest_path = os.path.join(data_dir, 'train' if data_split == 'train' else 'test', class_name, file)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if not os.path.isdir(source_path) and source_path != dest_path:
                    shutil.copy(source_path, dest_path)
        else:
            print(f"No files available for class {class_name} in {data_split} set.")

print("Data split completed.")
