from dataset import load_test_data, load_train_data

#path = ...  # Laptop Path
path = r"C:\Users\deepp\Documents\Offline Projects\ML Datasets\ADNI" # PC path

if __name__ == "__main__":
    
    transforms = None
    train_dataset = load_train_data(path, 64, transforms = transforms)
