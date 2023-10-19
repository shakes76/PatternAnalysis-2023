# %%

import tensorflow as tf
from dataset import image_list
from dataset import load_images
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


#define varibles
height = 128
width = 128
dimension = 1
batch_size = 32
# %%

#load the trained weights of the neural network
siamese_model = tf.keras.saving.load_model("C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/PatternAnalysis-2023/recognition/DanielPfisterSiameseNetwork/SiameseNeuralNetworkFinal.h5")

# %%
#define path test images
path_test_images_AD = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/AD/"
path_test_images_NC = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/NC/"

#create test image list and count number of test images
list_test_AD, number_test_AD = image_list(path_test_images_AD)
list_test_NC, number_test_NC = image_list(path_test_images_NC)

#create a copie of the list and shuffle only the AD2 and NC2 list
list_test_AD2 = list_test_AD[1:number_test_AD]
list_test_NC2 = list_test_NC[1:number_test_NC]
temp_AD2 = list_test_AD[0]
temp_NC2 = list_test_NC[0]
print(list_test_AD)
print(list_test_NC)
print(list_test_AD2)
print(list_test_NC2)
random.Random(42).shuffle(list_test_AD2)
random.Random(42).shuffle(list_test_NC2)
list_test_AD2.append(temp_AD2)
list_test_NC2.append(temp_NC2)

print(list_test_AD)
print(list_test_NC)
print(list_test_AD2)
print(list_test_NC2)

#counter variable
number_of_correct = 0
number_of_incorrect = 0

#test loop for accuracy
for a in range(0, number_test_AD//batch_size*batch_size, batch_size):
  print(a)
  #1 load 32 images
  test_image_chunk_AD1 = list_test_AD[a:a+batch_size]
  test_image_chunk_NC1 = list_test_NC[a:a+batch_size]
  test_image_chunk_AD2 = list_test_AD2[a:a+batch_size]
  test_image_chunk_NC2 = list_test_NC2[a:a+batch_size]
  print(test_image_chunk_AD1)
  print(test_image_chunk_NC1)
  print(test_image_chunk_AD2)
  print(test_image_chunk_NC2)
  test_image_AD1 = load_images(path_test_images_AD, test_image_chunk_AD1, height, width)
  test_image_NC1 = load_images(path_test_images_NC, test_image_chunk_NC1, height, width)
  test_image_AD2 = load_images(path_test_images_AD, test_image_chunk_AD2, height, width)
  test_image_NC2 = load_images(path_test_images_NC, test_image_chunk_NC2, height, width)

  testimage1_list = []
  testimage2_list = []
  label_test_list = []
  #2 create image pairs
  for b in range(batch_size):
    testimage1_list.append(test_image_AD1[b,:,:,:])
    testimage2_list.append(test_image_AD2[b,:,:,:])
    label_test_list.append(1)
    testimage1_list.append(test_image_AD1[b,:,:,:])
    testimage2_list.append(test_image_NC1[b,:,:,:])
    label_test_list.append(0)
    testimage1_list.append(test_image_NC1[b,:,:,:])
    testimage2_list.append(test_image_NC2[b,:,:,:])
    label_test_list.append(1)
    testimage1_list.append(test_image_NC2[b,:,:,:])
    testimage2_list.append(test_image_AD2[b,:,:,:])
    label_test_list.append(0)

  testimage1_list_array = np.array(testimage1_list)
  testimage2_list_array = np.array(testimage2_list)
  label_array = np.array(label_test_list)
  #3 make prediction
  test_pair = [testimage1_list_array, testimage2_list_array]
  float_formatter = "{:.2f}".format
  prediction = siamese_model.predict(test_pair, steps = 1)
  print(np.round(prediction[:,0],3),(label_array))
  #4 defines the classification of the image pair depending on the score
  prediction_label = []
  for c in prediction[:,0]:
    if c >= 0.50: 
        prediction_label.append(1)
    else:
        prediction_label.append(0)

  prediction_label = np.array(prediction_label)
  print(prediction_label)
  text_prediction = []
  #5 compare prediction label with correct label
  for d in range(0,len(prediction_label)):
      if prediction_label[d] == label_array[d]:
        number_of_correct = number_of_correct + 1
        text_prediction.append("correct")
      else:
        number_of_incorrect = number_of_incorrect + 1
        text_prediction.append("incorrect")
  print(number_of_correct)
  print(number_of_incorrect)
  print(text_prediction)
# %%
#calculate the accuray test result 
accuray_test = number_of_correct/(len(list_test_AD)*4)
print("The overall accuracy of the saimese network is %.2f."% (accuray_test))
# %%
#plot the result of the first ten results
for i in range(0,10):
  titel = "The prediciton of the saimese network is %0.f and the actual label is %.0f. \nAs a result the prediciton is %s."% (prediction_label[i],label_array[i],text_prediction[i])
  fig = plt.figure()
  plt.suptitle(titel)
  ax1 = fig.add_subplot(2,2,1)
  ax1.imshow(testimage1_list_array[i,:,:,:])
  ax2 = fig.add_subplot(2,2,2)
  ax2.imshow(testimage2_list_array[i,:,:,:])
# %%
