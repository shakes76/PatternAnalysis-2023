##################################   predict.py   ##################################

from train import Train

train = Train()
model = Train.train(train)
Train.test(train)