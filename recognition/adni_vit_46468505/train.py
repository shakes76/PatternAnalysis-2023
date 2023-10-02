import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from dataset import parse_data, tf_dataset
from modules import create_vit_classifier

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 25
image_size = 72  # We'll resize input images to this size
patch_size = image_size // 8  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
num_classes = 2
input_size = 256
#input_shape = (input_size,input_size,1)

train_dir = '/kaggle/input/adni-preprocessed/AD_NC/train/'
test_dir = '/kaggle/input/adni-preprocessed/AD_NC/test/'
(x_train,y_train) = parse_data(train_dir)
train_set = tf_dataset(x_train,y_train,batch_size=batch_size)
(x_test,y_test) = parse_data(test_dir)
test_set = tf_dataset(x_test,y_test,batch_size =len(y_test))

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            #keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/kaggle/working/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_set,
        batch_size=batch_size,
        epochs=num_epochs,
        #validation_split=0.1,
        callbacks=[checkpoint_callback],
    )
    model.load_weights(checkpoint_filepath)

    return history

vit_classifier = create_vit_classifier()
#vit_classifier.load_weights('/kaggle/working/model_72bit_acc0.6740.h5')
history = run_experiment(vit_classifier)

loss,acc = vit_classifier.evaluate(test_set)

fl = f"/kaggle/working/model_{image_size}bit_acc{acc:.4f}.h5"
vit_classifier.save_weights(fl)