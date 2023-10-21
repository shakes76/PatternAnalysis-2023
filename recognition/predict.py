from modules import TripletSiameseNetwork
from dataset import gen_loaders
import torch
import pickle

def predict(input=gen_loaders()['test'], triplet_path = 'trip_model.pth', classifier_path='classifier.pickle'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trip_model = TripletSiameseNetwork()
    trip_model = torch.load(triplet_path)
    with open(classifier_path, 'rb') as f:
        clf = pickle.load(f)
    outputs = None

    for i, (img, _, _, _) in enumerate(input):
        img = img.to(device)
        with torch.no_grad():
            features = trip_model.forward_once(img)

        if outputs is None:
            outputs = features.cpu()
        else:
            outputs = torch.cat((outputs, features.cpu()), dim=0)
    
    pred = clf.predict(outputs)
    return pred

