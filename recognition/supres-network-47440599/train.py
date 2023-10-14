import torch
from dataset import *
from modules import *
import torch.optim as optim
import torch.nn
 

if __name__ == "__main__":
    
    train_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    

    model = SubPixel()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    print("Training complete")

    torch.save(model.state_dict(), "subpixel_model.pth")

