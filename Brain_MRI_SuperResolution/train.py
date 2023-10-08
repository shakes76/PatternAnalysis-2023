from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from models.sub_pixel_cnn import sub_pixel_cnn

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32

# Create an image data generator
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory('data/ad_nc/train', target_size=(128, 128), batch_size=BATCH_SIZE, class_mode='input')
test_gen = datagen.flow_from_directory('data/ad_nc/test', target_size=(128, 128), batch_size=BATCH_SIZE, class_mode='input')

# Model
model = sub_pixel_cnn()
model.compile(optimizer=Adam(), loss=mean_squared_error)

# Train
model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS)

# Save
model.save("saved_models/sub_pixel_cnn.h5")
