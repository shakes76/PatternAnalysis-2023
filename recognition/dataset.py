import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf

# preset img size
H = 256
W = 256


def load_data(dataset_path, split=0.2):
    # grab only imgs with jpg
    images = sorted(glob(os.path.join(dataset_path, "ISIC-2017_Training_Data", "*.jpg")))
    # grab all masks
    masks = sorted(glob(os.path.join(dataset_path, "ISIC-2017_Training_Part1_GroundTruth", "*.png")))

    test_size = int(len(images) * split)

    # Split into 60/20/20
    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    # (H, W, 3) as RBG 3 chanels
    x = cv2.imread(path, cv2.IMREAD_COLOR)  
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    # (256, 256, 3)
    return x                     

def read_mask(path):
    path = path.decode()
     ## (H, W) no channels present
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    x = cv2.resize(x, (W, H))
    x = x/255.0
    ## (256, 256)
    x = x.astype(np.float32) 
    # Add chanel for gs (256, 256, 1)            
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    # wrap function to tf 
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    # Procces batches on CPU while GPU in use (consumer/prod overlap)
    dataset = dataset.prefetch(10)
    return dataset
