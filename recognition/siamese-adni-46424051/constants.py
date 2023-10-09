##################################   constants.py   ##################################
import os

cuda = False
batch_size = 8
test_size = 1000
learning_rate = 0.0001
epochs = 1
train_path = os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/train")
test_path = os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/test")
# train_path = os.path.expanduser("PatternAnalysis-2023/recognition/siamese-adni-46424051/images/train")
# test_path = os.path.expanduser("PatternAnalysis-2023/recognition/siamese-adni-46424051/images/test")