# Source code of the component for the model
import dataset
import train
import dgl

graph, train_mask, test_mask = dataset.load_data()

train.train_model(graph, 128, 22470, train_mask, test_mask)

#print("Training info: ", train_set)
#print("Validation info: ", val_set)
#print("Testing info: ", test_set)
