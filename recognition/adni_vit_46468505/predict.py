import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from dataset import parse_data, tf_dataset
from modules import create_vit_classifier,model_compile


input_size = 256

test_dir = './recognition/adni_vit_46468505/test/'
(x_test,y_test) = parse_data(test_dir)
test_set = tf_dataset(x_test,y_test,batch_size =len(y_test))

vit_classifier = create_vit_classifier()
vit_classifier.load_weights('./recognition/adni_vit_46468505/model_256bit_acc0.7271.h5')
vit_classifier=model_compile(vit_classifier)

loss,acc = vit_classifier.evaluate(test_set,verbose=1)
