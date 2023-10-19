#--------------
# Train config
num_epochs=1

#--------------
# Data information
classify_file = './recognition/Mask RCNN s4698053/.tmp/ISIC-2017_Training_Part3_GroundTruth.csv'
mask_dir = './recognition/Mask RCNN s4698053/.tmp/ISIC-2017_Training_Part1_GroundTruth/'
photo_dir = './recognition/Mask RCNN s4698053/.tmp/ISIC-2017_Training_Data/'
train_size = 0.8
test_size = 0.2
image_size = (1000, 700)