##################################   constants.py   ##################################
import os

cuda = True
batch_size = 128
test_size = 1000
train_iters = 10000
learning_rate = 0.0001
train_path = os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/train")
test_path = os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/test")