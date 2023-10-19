from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import BCELoss, CrossEntropyLoss
from tqdm import tqdm, trange
import numpy as np
from PIL import Image
import cv2


root = '/home/groups/comp3710/ADNI/AD_NC/'
# root = '/home/callum/AD_NC/'
image_size = 256
batch_size = 64
image_crop = 210

# class CustomImageFolder(Dataset):
#     def __init__(self, root, transform=None):
#         self.data = datasets.ImageFolder(root, transform=transform)
    
#     def __getitem__(self, index):
#         image, class_index = self.data[index]
#         # Convert the class index to binary label (1 for AD, 0 for NC)
#         binary_label = 1 if class_index == 0 else 0
#         return image, binary_label

#     def __len__(self):
#         return len(self.data)

class CropBrainScan:
    def __call__(self, image):
        # Convert the image to a NumPy array if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Ensure the image is in the CV_8UC1 format
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to separate the brain scan from the background
        _, thresholded = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (the brain scan region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the coordinates of the bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to keep only the brain scan region
        cropped_image = image[y:y + h, x:x + w]

        # Convert the NumPy array back to a PIL image
        cropped_image = Image.fromarray(cropped_image)
        
        return cropped_image

train_transform = transforms.Compose([
    # transforms.CenterCrop((image_crop, image_crop)),
    CropBrainScan(),
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1155], std=[0.2224]) # ImageNet constants
])
    
test_transform = transforms.Compose([
    # transforms.CenterCrop((image_crop, image_crop)),
    CropBrainScan(),
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1167], std=[0.2228]) # ImageNet constants
])

train_dataset = ImageFolder(root + 'train', transform=train_transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_dataset = ImageFolder(root + 'test', transform=test_transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

patch_size = image_size // 8
channels = 1
embedding_dims = channels * patch_size**2
patches = (image_size // patch_size)**2
num_heads = embedding_dims // 64

assert image_size % patch_size == 0, print('Image size not divisible by patch size')

class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim,):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)
        self.class_token_embeddings = nn.Parameter(torch.rand((batch_size, 1, embedding_dims), requires_grad=True))
        self.position_embeddings = nn.Parameter(torch.rand((1, patches + 1, embedding_dims), requires_grad=True))

    def forward(self, x):
        output = torch.cat((self.class_token_embeddings, self.flatten_layer(self.conv_layer(x).permute((0, 2, 3, 1)))), dim=1) + self.position_embeddings
        return output
    

class MultiHeadSelfAttentionBlock(nn.Module):
  def __init__(self,
               embedding_dims = 256, # Hidden Size D in the ViT Paper Table 1
               num_heads = 12,  # Heads in the ViT Paper Table 1
               attn_dropout = 0.0 # Default to Zero as there is no dropout for the the MSA Block as per the ViT Paper
               ):
    super().__init__()

    self.embedding_dims = embedding_dims
    self.num_head = num_heads
    self.attn_dropout = attn_dropout

    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)

    self.multiheadattention =  nn.MultiheadAttention(num_heads = num_heads,
                                                     embed_dim = embedding_dims,
                                                     dropout = attn_dropout,
                                                     batch_first = True,
                                                    )

  def forward(self, x):
    x = self.layernorm(x)
    output,_ = self.multiheadattention(query=x, key=x, value=x,need_weights=False)
    return output


class MachineLearningPerceptronBlock(nn.Module):
  def __init__(self, embedding_dims, mlp_size, mlp_dropout):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.mlp_size = mlp_size
    self.dropout = mlp_dropout

    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)
    self.mlp = nn.Sequential(
        nn.Linear(in_features = embedding_dims, out_features = mlp_size),
        nn.GELU(),
        nn.Dropout(p = mlp_dropout),
        nn.Linear(in_features = mlp_size, out_features = embedding_dims),
        nn.Dropout(p = mlp_dropout)
    )

  def forward(self, x):
    return self.mlp(self.layernorm(x))


class TransformerBlock(nn.Module):
  def __init__(self, embedding_dims = 256,
               mlp_dropout=0.1,
               attn_dropout=0.0,
               mlp_size = 3072,
               num_heads = 12,
               ):
    super().__init__()

    self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims = embedding_dims,
                                                 num_heads = num_heads,
                                                 attn_dropout = attn_dropout)

    self.mlp_block = MachineLearningPerceptronBlock(embedding_dims = embedding_dims,
                                                    mlp_size = mlp_size,
                                                    mlp_dropout = mlp_dropout,
                                                    )

  def forward(self,x):
    x = self.msa_block(x) + x
    x = self.mlp_block(x) + x

    return x


class ViT(nn.Module):
  def __init__(self, img_size = 256,
               in_channels = 1,
               patch_size = 32,
               embedding_dims = 1024,
               num_transformer_layers = 12, # from table 1 above
               mlp_dropout = 0.1,
               attn_dropout = 0.0,
               mlp_size = 3072,
               num_heads = 16,
               num_classes = 2):
    super().__init__()

    self.patch_embedding_layer = PatchEmbeddingLayer(in_channels = in_channels,
                                                     patch_size=patch_size,
                                                     embedding_dim = embedding_dims)

    self.transformer_encoder = nn.Sequential(*[TransformerBlock(embedding_dims = embedding_dims,
                                              mlp_dropout = mlp_dropout,
                                              attn_dropout = attn_dropout,
                                              mlp_size = mlp_size,
                                              num_heads = num_heads) for _ in range(num_transformer_layers)])

    self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape = embedding_dims),
                                    nn.Linear(in_features = embedding_dims,
                                              out_features = num_classes))

  def forward(self, x):
    return self.classifier(self.transformer_encoder(self.patch_embedding_layer(x))[:, 0])


# Defining model and training options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
model = ViT(
    img_size=image_size,
    in_channels=channels,
    patch_size=patch_size,
    embedding_dims=embedding_dims,
    num_heads=num_heads
    ).to(device)
epochs = 10
learning_rate = 0.001
weight_decay = 0.0001

# Training loop
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = CrossEntropyLoss()
model.train() # training mode
for epoch in trange(epochs, desc="Training"):
    train_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(train_dataloader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")

model.eval() # evaluation mode
# Test loop
with torch.no_grad():
    correct, total = 0, 0
    test_loss = 0.0
    for batch in tqdm(test_dataloader, desc="Testing"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        test_loss += loss.detach().cpu().item() / len(test_dataloader)

        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total += len(x)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")
    