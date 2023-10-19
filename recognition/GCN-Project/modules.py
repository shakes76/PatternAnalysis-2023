# Source code of the component for the model
import dataset

x, y, z = dataset.load_data()

print("Edges info: ", x.shape)
print("Features info: ", y.shape)
print("Target info: ", z.shape)
