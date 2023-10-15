import torch
from dataset import *
from modules import *
import torch.optim as optim
import torch.nn
import torchvision.transforms as transforms
 

if __name__ == "__main__":
    
    dataroot = "./data/AD_NC/train"
    train_loader = load_data(dataroot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    
    model = SubPixel()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    model.to(device)

    transform = transforms.Compose([transforms.Resize((60,64))])

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            downsampled_inputs = transform(inputs)
            # Forward pass
            outputs = model(downsampled_inputs)
            loss = criterion(outputs, inputs)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    print("Training complete")

    torch.save(model.state_dict(), "subpixel_model.pth")

