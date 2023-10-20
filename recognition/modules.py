from keras.layers import *
from keras.models import Sequential, Model
from keras.regularizers import l2

input_A = Input((75, 80, 1), name='input_A')
input_B = Input((75, 80, 1), name='input_B')

def get_block(depth):
    return Sequential([Conv2D(depth, 3, 1, kernel_regularizer=l2(0.01)), # 3x3 padding with a stride of 1 with L2 Regularisation
                       BatchNormalization(),
                       LeakyReLU()])

DEPTH = 128

cnn = Sequential([Reshape((75, 80, 1)),
                  get_block(DEPTH),
                  Dropout(0.5),  # Add dropout with a 50% dropout rate
                  get_block(DEPTH * 2),
                  get_block(DEPTH * 4),
                  Dropout(0.5),  # Add dropout with a 50% dropout rate
                  get_block(DEPTH * 8),
                  GlobalAveragePooling2D(),
                  Dense(64, activation='sigmoid')])

# As we are using multiple inputs, we concatenate the inputs' feature vectors
feature_vector_A = cnn(input_A)
feature_vector_B = cnn(input_B)

# After Concatenation, add another Dropout
feature_vectors = Concatenate()([feature_vector_A, feature_vector_B])
feature_vectors = Dropout(0.5)(feature_vectors)  # Add dropout with a 50% dropout rate

# Add a Dense layer for non-linearity
dense = Dense(64, activation='sigmoid')(feature_vectors)

# Choose Sigmoid to ensure it's a probability
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[input_A, input_B], outputs = output)

def get_model():
    return model