import time
from matplotlib import pyplot as plt
import torch
import dataset
import modules
from torch.utils.data import DataLoader
from torch import nn

# Hyperparameters
BATCH_SIZE = 10
LEARNING_RATE = 10e-4
EPOCHS = 20
DECAY = 10e-6

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set datasets
train_set, validate_set, test_set = dataset.get_datasets()

# Set dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=BATCH_SIZE, shuffle=False)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, output, mask):
        """
        Calculates the Dice Loss.
        @param output: The output of the model.
        @param mask: The ground truth mask.
        @return: The Dice Loss.
        """
        smooth = 1e-4
        
        output_flat = output.view(-1)
        mask_flat = mask.view(-1)
        
        intersection = torch.sum(output_flat * mask_flat)
        union = torch.sum(output_flat) + torch.sum(mask_flat)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return 1 - dice
    

def train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    """
    Trains the model.
    @param model: The model to train.
    @param criterion: The loss function.
    @param optimizer: The optimizer.
    @param scheduler: The learning rate scheduler.
    @param num_epochs: The number of epochs to train the model for.
    @return: The trained model and the training and validation losses.
    """
    print('Begin training model...')
    
    # Set model to training mode
    start = time.time()
    
    train_lossess = []
    validate_losses = []
    
    # Iterate over epochs
    for epoch in range(num_epochs):
        train_loss_sum = 0.0
        
        for image, mask in train_loader:
            model.train()
            
            image = image.to(device)
            mask = mask.to(device).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            output = model(image)
            
            loss = criterion(output, mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        train_lossess.append(avg_train_loss)
        
        print('Epoch: {}/{} | Training Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_train_loss))
        
        # Validate model
        validate_loss_sum = 0.0
        accuracy_sum = 0.0
        
        for image, mask in validate_loader:
            model.eval()
            
            image = image.to(device)
            mask = mask.to(device)
            
            output = model(image)
            
            with torch.no_grad():
                loss = criterion(output, mask)
                
                accuracy_sum += accuracy(output, mask)
                
                validate_loss_sum += loss.item()
                
        avg_validate_loss = validate_loss_sum / len(validate_loader)
        validate_losses.append(avg_validate_loss)
        
        print('Epoch: {}/{} | Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_validate_loss))
        print('Epoch: {}/{} | Validation Accuracy: {:.4f}'.format(epoch + 1, num_epochs, accuracy_sum / len(validate_loader)), end='\n\n')
        
        scheduler.step(avg_validate_loss)
        torch.save(model.state_dict(), f's4630051_improved_unet_segmentation/save/model_save_{epoch}.pth')
                
    end = time.time()
    print('Finished training model. Time taken: {:.4f} seconds'.format(end - start))
    
    torch.save(model.state_dict(), 's4630051_improved_unet_segmentation/save/model_save_final.pth')
    
    return model, train_lossess, validate_losses

def accuracy(model, mask):
  with torch.no_grad():
    model = (model > 0.5).float() #if a pixel has value > 0.5, we accept it as a skin lesion
    correct = (model == mask).sum()
    pixels = torch.numel(model)
    accuracy = correct / pixels + 1e-8

  return accuracy

def test_model(model):
    """
    Test the model.
    @param model: The model to test.
    """
    pass

model = modules.UNetImproved(3, 1)

model.to(device)

criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: LEARNING_RATE * (0.975 ** epoch))
train_model(model, criterion, optimizer, scheduler)