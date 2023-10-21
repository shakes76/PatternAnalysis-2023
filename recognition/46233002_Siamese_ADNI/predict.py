import torch
from torch.utils.data import DataLoader
from dataset import adniPredictionDataset
from modules import Siamese, Classifier
from utils import predict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)

# Path
path = '/home/groups/comp3710'
model_path = '/home/Student/s4623300/PatternAnalysis-2023/recognition/46233002_Siamese_ADNI/easysemihard'

# Create train/test/validation dataset objects 
predset = adniPredictionDataset(path + "/ADNI/AD_NC/test/AD")
pred_loader = DataLoader(predset, shuffle=False, batch_size=32)

# Construct encoder 
siamese = Siamese().to(device)
classifier = Classifier().to(device)
siamese.load_state_dict(torch.load(model_path + '/siamese.pt'))
classifier.load_state_dict(torch.load(model_path + '/classifier.pt'))

# Generate predictions
predict(siamese, classifier, pred_loader, device)
