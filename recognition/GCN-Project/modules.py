# Source code of the component for the model
# Thomas Bennion s4696627
import dataset
import train
import dgl

#Load and preprocess the data
graph, train_mask, test_mask, num_features = dataset.load_data()

# Initialize the GCN model
model = train.GCN(input_feats=128, hidden_size=64, num_classes=num_features, num_layers=2)

#Train the model
train.train_model(graph, train_mask, test_mask, model, 100, 0.01)