from dataset import *
from modules import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision


if __name__ == "__main__":
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not found. Using CPU")

    dataset_path = r"C:\Users\raulm\Desktop\Uni\Sem2.2023\Patterns\ISIC-2017_Training_Data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    batch_size = 4
    lr = 1e-4
    num_epoch = 2
    
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)
    
    train_steps = get_steps(train_x, batch_size)
    valid_steps = get_steps(valid_x, batch_size)

    model = Unet((H, W, 3))
    model = model.to(device)
    metrics = [Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    model.summary()
    
    # Start training the model
    model.fit(
        train_dataset,
        epochs=num_epoch,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps
    )