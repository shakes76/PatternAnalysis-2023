import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
#import tensorflow_addons as tfa;

from dataset import parse_data, tf_dataset
from modules import create_vit_classifier, model_compile
from dataset import batch_size, image_size

from sklearn.model_selection import train_test_split

num_epochs = 1
input_size = 256

train_dir = '/Users/georgiapower/Documents/UNI/engineering4:2/COMP3710/PatternAnalysis-2023/recognition/adni_vit_46468505/AD_NC/train/'
test_dir = '/Users/georgiapower/Documents/UNI/engineering4:2/COMP3710/PatternAnalysis-2023/recognition/adni_vit_46468505/AD_NC/train/'
(x_train,y_train) = parse_data(train_dir)
(x_test,y_test) = parse_data(test_dir)

(x_train,x_valid,y_train,y_valid) = train_test_split(x_train,y_train, test_size = 0.2)

train_set = tf_dataset(x_train,y_train,batch_size=batch_size)
valid_set = tf_dataset(x_valid,y_valid,batch_size=batch_size)
test_set = tf_dataset(x_test,y_test,batch_size=batch_size)

def run_experiment(model):
    #compile
    model = model_compile(model)

    #fit!
    history = model.fit(
        train_set,
        validation_data = valid_set,
        shuffle=True,
        batch_size=batch_size,
        epochs=num_epochs
    )

    return history

vit_classifier = create_vit_classifier()
vit_classifier.load_weights('/Users/georgiapower/Documents/UNI/engineering4:2/COMP3710/PatternAnalysis-2023/recognition/adni_vit_46468505/model_checkpoints//model_256bit_acc0.7271.h5')
history = run_experiment(vit_classifier)

loss,acc = vit_classifier.evaluate(test_set)

fl = f"/Users/georgiapower/Documents/UNI/engineering4:2/COMP3710/PatternAnalysis-2023/recognition/adni_vit_46468505/model_checkpoints/model_{image_size}bit_acc{acc:.4f}.h5"
try:
    vit_classifier.save_weights(fl)
except:
    print("uhoh saving fail! ")

# Plot the training loss and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the training accuracy and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
