##################################   constants.py   ##################################
import os

cuda = False
batch_size = 1
test_size = 1000
learning_rate = 0.005
epochs = 1
train_path = os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/train")
test_path = os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/test")
# train_path = os.path.expanduser("images/train")
# test_path = os.path.expanduser("images/test")