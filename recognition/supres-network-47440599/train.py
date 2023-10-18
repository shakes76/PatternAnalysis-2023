import torch
import torch.optim as optim
import torch.nn
from dataset import *
from modules import *
from utils import *
 

if __name__ == "__main__":
    #loading the data
    train_loader = load_data(train_root,train_batchsize)

    #Intialising the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #Initialising the model
    model = SubPixel()

    #Initialising the loss function and optimiser
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Setting the model to training mode
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            #Down sampling the image
            downsampled_inputs = down_sample(inputs).to(device)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(downsampled_inputs).to(device)
            loss = criterion(outputs, inputs)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    print("Training complete")

    torch.save(model.state_dict(), saved_path)

