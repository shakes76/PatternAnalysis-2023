#%%
import tensorflow as tf
from tensorflow.keras import layers

def siamese_network(height,width,dimension):

    #deifne input of siamese network
    input_shape = (height, width, dimension)
    left_input = layers.Input(input_shape)
    right_input = layers.Input(input_shape)
    #define standard model of the left and right siamese network which is a VGG16 without all the dense layers
    vgg16 = tf.keras.Sequential([ layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu'),
                                layers.MaxPool2D(pool_size =2, strides =2, padding ='same'),

                                layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu'),
                                layers.MaxPool2D(pool_size =2, strides =2, padding ='same'),

                                layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu'),
                                layers.MaxPool2D(pool_size =2, strides =2, padding ='same'),

                                layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
                                layers.MaxPool2D(pool_size =2, strides =2, padding ='same'),

                                layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
                                layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
                                layers.MaxPool2D(pool_size =2, strides =2, padding ='same'),

                                layers.Flatten(),
                                layers.Dense(256, activation='sigmoid'), #60 precent 1024 vgg16 128 128
                                ])

    #save the features from the left and right network in two variables
    feature_vector_left_output = vgg16(left_input)
    feature_vector_right_output = vgg16(right_input)

    #merge layer which claulates the distance between both networks
    merge_layer = layers.Lambda(lambda features: tf.abs(features[0] - features[1]))([feature_vector_left_output, feature_vector_right_output])


    #fully connected layers
    output = layers.Dense(1, activation='sigmoid')(merge_layer)
    #create whole neural network model
    model = tf.keras.Model(inputs=[left_input, right_input], outputs=output)

    model.summary()
    
    #Configurates the loss funciton, optimizer type and metrics of the model for training.
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy'])
    
    return model

# %%
