#--------------
# Train config
LEARNING_RATE = 2e-5
NUM_EPOCHS = 8
BATCH_SIZE = 16
WEIGHT_DECAY = 0
NUM_WORKERS = 2
PIN_MEMORY = True
SAVE_MODEL_FILE = "model.pytorch"

#--------------
# Predict config
LOAD_MODEL_FILE = "model.pytorch"

#--------------
# Data information
classify_file = './recognition/YOLOv1 s4698053/.tmp/ISIC-2017_Training_Part3_GroundTruth.csv'
mask_dir = './recognition/YOLOv1 s4698053/.tmp/ISIC-2017_Training_Part1_GroundTruth/'
photo_dir = './recognition/YOLOv1 s4698053/.tmp/ISIC-2017_Training_Data/'
train_size = 0.8
test_size = 0.2
image_size = (448, 448)
torch_seed = 56