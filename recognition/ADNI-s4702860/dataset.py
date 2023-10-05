import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

'''
Function used to load ADNI datasets. 
Assumes image size is 256x240

Inputs:
    base_path - the base path (test, train) used for classifying either the train or test data

Outputs:
    data - the image converted to a tensorflow tensor
    labels - the classification label, 1 for having alzeimers, else 0
'''
def load_dataset(base_path, image_size=(256, 240)):
    data = []
    labels = []

    # Iterate through the AD and NC classes from the path
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        class_label = 1 if class_name == "AD" else 0  # 1 for 'AD', 0 for 'NC'
        
        # Iterate through each file and append the image data and label
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            image = load_img(file_path, target_size=image_size)
            image = img_to_array(image)
            image = image / 255.0  # Normalize the image

            data.append(image)
            labels.append(class_label)


    data = tf.convert_to_tensor(data, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    return data, labels

def main():
    train_path = "AD_NC/train"
    test_path = "AD_NC/test"

    print("Begin training")
    train_data, train_labels = load_dataset(train_path)
    print("Finished training")

    test_data, test_labels = load_dataset(test_path)
    print("Finished testing")


    print("Shape of train_data:", train_data.shape)
    print("Shape of train_labels:", train_labels.shape)

    print("Shape of test_data:", test_data.shape)
    print("Shape of test_labels:", test_labels.shape)



if __name__=="__main__":
    main()