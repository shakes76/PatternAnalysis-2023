import os
from PIL import Image
import os
import shutil
from tqdm import tqdm

def downsample_images(source_folder, target_folder, n):
    # Ensure the target directory exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Get all files from the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and f.endswith('.jpg')]
    
    for file in tqdm(files, desc="Processing images"):
        file_path = os.path.join(source_folder, file)
        
        # Try to open the file with PIL
        try:
            with Image.open(file_path) as img:
                # Get new dimensions
                width, height = img.size
                new_dimensions = (width//n, height//n)
                
                # Resize and save
                resized_img = img.resize(new_dimensions)
                resized_img.save(os.path.join(target_folder, file))
                
        except Exception as e:
            print(f"Failed to process {file}. Error: {e}")




def copy_txt_files(source_folder, target_folder):
    """
    Copy all .txt files from source_folder to target_folder.
    
    Parameters:
    - source_folder: Path to the source directory
    - target_folder: Path to the target directory
    """
    
    # Ensure the target directory exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Iterate through all files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.txt'):
            source_file_path = os.path.join(source_folder, file_name)
            target_file_path = os.path.join(target_folder, file_name)
            
            shutil.copy2(source_file_path, target_file_path)
            #print(f"Copied {file_name} to {target_folder}")



if __name__ == "__main__":
    source = "/Users/larsmoan/Documents/UQ/COMP3710/PatternAnalysis-2023/lesion_detection_larsmoan/data/ISIC_2017/val"
    target = "/Users/larsmoan/Documents/UQ/COMP3710/PatternAnalysis-2023/lesion_detection_larsmoan/data/ISIC_2017_0.5/val"
    n = 2  # Downsample by factor of 2
    downsample_images(source, target, n)
    copy_txt_files(source_folder=source, target_folder=target)
