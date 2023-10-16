from dataset import *
from modules import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision

if __name__ == "__main__":
    dataset_path = r"C:\Users\raulm\Desktop\Uni\Sem2.2023\Patterns\ISIC-2017_Training_Data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    batch_size = 4
    lr = 1e-4

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)
    
    train_steps = get_steps(train_x, batch_size)
    valid_steps = get_steps(valid_x, batch_size)
    
    print(train_steps,  valid_steps)

    model = Unet((H, W, 3))
    metrics = [Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    model.summary()