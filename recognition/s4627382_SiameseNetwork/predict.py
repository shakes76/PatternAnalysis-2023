# showing example usage of your trained model. 
# Print out any results and / or provide visualisations where applicable

import modules
import torch
import pickle
import torchvision.transforms as transforms
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda')

def predict_image(image_path):
    # load the image
    image = Image.open(image_path)
    
    # trainsform the data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # move to gpu
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # define and load model
    embeddingNet = modules.Embedding()
    model = modules.SiameseNet(embeddingNet)
    model = model.to(device)
    model.load_state_dict(torch.load("D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/SiameseNet.pth"))
    with open("D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/knn.pkl", "rb") as f:
        knn = pickle.load(f)

    # extract embedding
    embedding = model.get_embedding(image_tensor)
    embedding_numpy = embedding.detach().cpu().numpy()

    # predict using KNN
    prediction = knn.predict(embedding_numpy)
    
    # convert prediction to 'ad' or 'nc'
    label_map = {0: 'ad', 1: 'nc'}
    predicted_label = label_map[prediction[0]]
    
    return predicted_label


# display given image
def display_image(image_path):
    # load image from given path
    img = mpimg.imread(image_path)

    # plot img
    plt.imshow(img)
    plt.axis("off")
    plt.show

# Example usage:
image_path = "D:/Study/MLDataSet/AD_NC/test/AD/388206_78.jpeg"
predicted_label = predict_image(image_path)
display_image(image_path)
print(f"Predicted label: {predicted_label}, Ture label: AD")
