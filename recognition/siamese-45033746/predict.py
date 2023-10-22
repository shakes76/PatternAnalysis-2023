from modules import BinaryClassifier, SiameseNetwork
from torch.utils.data import DataLoader
def predict(model: BinaryClassifier, net: SiameseNetwork, testDataLoader: DataLoader):
    model.eval()
    net.eval()
