from torchvision import transforms

from dataset import load_test_data, load_train_data
from modules import *

#path = r"C:\Users\dcp\Documents\OFFLINE-Projects\DATASETS\ADNI"  # Laptop Path
path = r"C:\Users\deepp\Documents\Offline Projects\ML Datasets\ADNI" # PC path

image_shape = (240, 240)
d_latent = 256
embed_dim = 32
transformer_depth = 1
num_heads = 1
n_perceiver_blocks = 1
num_epochs = 1
batch_size = 1
n_classes = 2
lr = 0.005

if __name__ == "__main__":
    
    transforms = transforms.Compose([
        transforms.CenterCrop(240), 
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Lambda(lambda x: torch.flatten(x, start_dim = 1))
    ])

    #  TESTING
    train_dataset = load_train_data(path, batch_size, transforms = transforms)

    model = Perceiver(
        d_latent,
        embed_dim,
        num_heads,
        transformer_depth,
        n_perceiver_blocks,
        n_classes,
        batch_size
    )

    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for _ in range(num_epochs):
        for i, (samples, labels) in enumerate(train_dataset):
            
            optim.zero_grad()
            outputs = model(samples)

            c_loss = loss(outputs, labels)
            c_loss.backward()
            optim.step()


            print(f"Sample batch no. {i}: {labels}")