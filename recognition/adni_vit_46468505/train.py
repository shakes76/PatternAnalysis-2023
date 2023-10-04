import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa;

from dataset import parse_data, tf_dataset
from modules import create_vit_classifier, model_compile
from dataset import batch_size, image_size


num_epochs = 25
input_size = 256

#input_shape = (input_size,input_size,1)

train_dir = './recognition/adni_vit_46468505/test/'
#test_dir = '/kaggle/input/adni-preprocessed/AD_NC/test/'
(x_train,y_train) = parse_data(train_dir)
train_set = tf_dataset(x_train,y_train,batch_size=batch_size)
#(x_test,y_test) = parse_data(test_dir)
#test_set = tf_dataset(x_test,y_test,batch_size =len(y_test))

def run_experiment(model):
    #compile
    model = model_compile(model)
    #save checkpoints
    checkpoint_filepath = "./recognition/adni_vit_46468505/model_checkpoints/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    #fit!
    history = model.fit(
        train_set,
        batch_size=batch_size,
        epochs=num_epochs,
        #validation_split=0.1,
        callbacks=[checkpoint_callback],
    )
    #upload weights
    model.load_weights(checkpoint_filepath)

    return history

vit_classifier = create_vit_classifier()
#vit_classifier.load_weights('/kaggle/working/model_72bit_acc0.6740.h5')
history = run_experiment(vit_classifier)

#loss,acc = vit_classifier.evaluate(test_set)
loss,acc = vit_classifier.evaluate(train_set)

fl = f"./recognition/adni_vit_46468505/model_checkpoints/model_{image_size}bit_acc{acc:.4f}.h5"
vit_classifier.save_weights(fl)