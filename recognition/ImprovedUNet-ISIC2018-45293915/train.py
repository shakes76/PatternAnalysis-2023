from dataset import get_dataloaders
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)

try:
    train_dataloader, test_dataloader, valid_dataloader = get_dataloaders()

    print("aaaaa", train_dataloader)
    
    if not train_dataloader:
        print("Error: train_dataloader is empty")

    # 

except Exception as e:
    print("An error occurred: ", str(e))
