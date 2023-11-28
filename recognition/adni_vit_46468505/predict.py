import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from dataset import read_test_path,tf_dataset
from modules import create_vit_classifier,model_compile


class_decoder = {0: "Normal",1:"Alzheimers"}

input_size = 256

#test_dir = './recognition/adni_vit_46468505/test/'
#(x_test,y_test) = parse_data(test_dir)
#test_set = tf_dataset(x_test,y_test,batch_size =len(y_test))

image_path = str(input("Path of Image to Analyse: "))
point = tf_dataset([image_path],[0])
image = read_test_path(image_path)

plt.imshow(image)
plt.title("Test Image")
plt.show()

vit_classifier = create_vit_classifier()
vit_classifier.load_weights('/Users/georgiapower/Documents/UNI/engineering4:2/COMP3710/PatternAnalysis-2023/recognition/adni_vit_46468505/model_checkpoints/model_256bit_acc0.7179.h5')
vit_classifier=model_compile(vit_classifier)

probs = vit_classifier.predict(point)
predicted_class = np.argmax(probs)
print(f"Test Image is Class {class_decoder[predicted_class]} with Confidence {probs[0][predicted_class]:.4f}")