import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from dataset import read_test_path,tf_dataset,parse_data
from modules import create_vit_classifier,model_compile

class_decoder = {0: "NC",1:"AD"}

max_samples = 5

test_dir = '/Users/georgiapower/Documents/UNI/engineering4:2/COMP3710/PatternAnalysis-2023/recognition/adni_vit_46468505/AD_NC/train/'
(x_test,y_test) = parse_data(test_dir)
x_test = x_test[0:max_samples]
y_test = y_test[0:max_samples]

test_set = tf_dataset(x_test,y_test,batch_size=1)

vit_classifier = create_vit_classifier()
vit_classifier.load_weights('/Users/georgiapower/Documents/UNI/engineering4:2/COMP3710/PatternAnalysis-2023/recognition/adni_vit_46468505/model_checkpoints/model_256bit_acc0.7271.h5')
vit_classifier=model_compile(vit_classifier)

probs = vit_classifier.predict(test_set)


imags=[]
labels=[]
for imag,label in test_set.take(max_samples):
    imag = imag.numpy().reshape((256,256,1))
    imags.append(imag)

    labels.append(label.numpy()[0])
print(labels)
print(imags[0].shape)

fig, axs = plt.subplots(1,max_samples)

for p,ax,imag,lab in zip(probs,axs,imags,labels):
    predicted_class = np.argmax(p)
    clasname = class_decoder[predicted_class]
    predconf = p[predicted_class]
    ax.imshow(imag)
    ax.set_title(f"Pred: {clasname}, Truth: {class_decoder[lab]}")
plt.show()