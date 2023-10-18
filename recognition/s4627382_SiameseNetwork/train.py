import dataset, modules
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda')

# data path
train_folder_path = "D:/Study/MLDataSet/AD_NC/train"
train_ad_path = "D:/Study/MLDataSet/AD_NC/train/AD"
train_nc_path = "D:/Study/MLDataSet/AD_NC/train/NC"
test_ad_path = "D:/Study/MLDataSet/AD_NC/test/AD"
test_nc_path = "D:/Study/MLDataSet/AD_NC/test/NC"

margin = 1
epoches = 10

# create data loader
train_loader, validation_loader, test_loader = dataset.load_data(
    train_folder_path, train_ad_path, train_nc_path, test_ad_path, test_nc_path, batch_size=32)

# define models
embbeding = modules.Embedding()
model = modules.SiameseNet(embbeding)
model.to(device)

# define loss function
criterion = modules.ContrastiveLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(train_loader, epoches):
    for epoch in range(epoches):
        # set model to train mode
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        for img1, img2, labels in train_loader:
            # move data to gpu
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # front propagation
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            
            # calculate accuracy
            batch_accuracy = calculate_accuracy(emb1, emb2, labels)
            total_loss += loss.item() * img1.size(0)
            total_accuracy += batch_accuracy * img1.size(0)
            total_samples += img1.size(0)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate accuracy and loss
        avg_accuracy = total_accuracy / total_samples
        avg_loss = total_loss / total_samples

        validate_loss, validate_accuracy = validate(validation_loader)

        print(f"Epoch [{epoch+1}/{epoches}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, validate loss: {validate_loss:.4f}, validate accuracy: {validate_accuracy:.4f}")

        if epoch % 2 == 0:
            visualize_embeddings(train_loader, model)

        # save the model
        torch.save(model.state_dict(),
                    "D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/SiameseNet.pth")
        print("Model saved")

def validate(validation_loader):
    # set model to evaluation mode
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for img1, img2, labels in validation_loader:
            # move data to gpu
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # front propagation
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)

            # get loss and accuracy
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(emb1, emb2, labels)
    
    # calculate average loss and average accuracy
    validate_loss = total_loss/len(validation_loader)
    validate_accuracy = total_accuracy/len(validation_loader)

    return validate_loss, validate_accuracy


# calculate accuracy, 
# if  distance < threshold, these two samples will be considered same
def calculate_accuracy(img1, img2, labels, threshold=0.5):
    
    # calculate the distance between two samples
    distance = (img1 - img2).pow(2).sum(1).sqrt()

    # Predict similarity: 0 for same (distance < threshold), 1 for diff
    predicts = (distance >= threshold).float()

    # Calculate accuracy by comparing predictions to labels
    correct = (predicts == labels).float()
    accuracy = correct.sum().item() / len(labels)
    
    return accuracy

def visualize_embeddings(loader, model, num_samples=300):
    model.eval()
    embeddings = []
    labels_list = []

    # Define label to color mapping
    label_to_color = {0: 'red', 1: 'blue'}

    with torch.no_grad():
        for i, (img1, img2, labels) in enumerate(loader):
            if i * loader.batch_size > num_samples:
                break
            
            img1, img2 = img1.to(device), img2.to(device)
            
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            
            embeddings.append(emb1.cpu())
            embeddings.append(emb2.cpu())
            
            labels_list.extend(labels.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Convert labels to colors
    color_labels = [label_to_color[label] for label in labels_list]

    embeddings = torch.cat(embeddings, dim=0)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=color_labels, s=50, alpha=0.6)
    plt.title("Embedding visualization")
    plt.show()


def main():
    mode = 0
    if mode == 0:
        print("Training")
        train(train_loader, epoches)
        modules.knn(train_loader, validation_loader, model, n_neighbors=5)

    if mode == 1:
        print("Train classifier")
        model.load_state_dict(torch.load("D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/SiameseNet.pth"))
        modules.knn(train_loader, validation_loader, model, n_neighbors=5)

    elif mode == 1:
        print("Testing")


if __name__ == "__main__":
    main()