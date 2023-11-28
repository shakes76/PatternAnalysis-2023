from dataset import *
from modules import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from utils import dice_coef, DiceThresholdStop, parse_args

# Initalise variables needed
model_path = "files/model.h5"
csv_path = "files/data.csv"
batch_size = 4
lr = 1e-4
num_epoch = 25
split = 0.2


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.path

    # list of all systems gpus
    physical_devices = tf.config.list_physical_devices('GPU') 
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU")
    else:
        print("Warning: No GPU found, using CPU")
    
    # Load the data to a tf dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path, split)
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)
    
    # Get training & valid steps per epoch
    train_steps = get_steps(train_x, batch_size)
    valid_steps = get_steps(valid_x, batch_size)
    
    # Inisalise model
    model = Unet((H, W, 3))
    metrics = [dice_coef, Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    
    # Add callbacks for model
    callbacks = [
        # Save model at best val_dice_ceof
        ModelCheckpoint(model_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        # Stop model if val_dice_ceof > 0.8 end of epoch
        DiceThresholdStop(threshold=0.8)
    ]
    
    # Start training the model
    model.fit(
        train_dataset,
        epochs=num_epoch,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )