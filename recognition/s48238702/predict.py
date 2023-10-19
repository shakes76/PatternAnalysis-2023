import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from modules import SiameseNetwork 
from dataset import load_classify_data

# Load pretrained Siamese network
model = SiameseNetwork()
model.load_state_dict(torch.load('SNN.pth', map_location=torch.device('cpu')))
model.eval()

# Load test data
test_loader = load_classify_data(testing=True, batch_size=32) 

threshold = 0.5  

def predict(model, test_loader, threshold):

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            
            output1, output2 = model(img1, img2)  
            
            # Calculate cosine similarity 
            similarity = F.cosine_similarity(output1, output2)
            
            # Predict label based on similarity
            predictions = (similarity > threshold).int()
            
            correct += (predictions == labels).sum()
            total += len(labels)
            
    accuracy = correct/total
    print(f'Accuracy: {accuracy:.2%}')
        
predict(model, test_loader, threshold)