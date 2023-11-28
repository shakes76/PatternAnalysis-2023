import torch
from dataset import load_data
from modules import ImprovedUNET
from utilities import DiceLoss, Wandb_logger
from torch.utils.data import DataLoader

#training, validating, testing and saving the model
def train(data, model, epochs):
    lr_init = 1e-3
    weight_decay = 1e-5
    dataset = data
    dataLoader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = model.to(device)
    
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_init * (0.995 ** epoch))

    logger = Wandb_logger(net, criterion, config={"lr_init": lr_init, "weight_decay": weight_decay})

    #Training the Network
    for epoch in range(epochs):  

        running_loss = 0.0
        for i, data in enumerate(dataLoader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            accuracy = criterion.accuracy(outputs, labels)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                logger.log(epoch, i, loss, accuracy)
                running_loss = 0.0
                
        scheduler.step()
        print('Finished Training epoch ', epoch + 1)
        if(epoch % 10 == 9):
            torch.save(net.state_dict(), 'model.pth')
            logger.print_images(outputs, labels)
            
train(load_data(), ImprovedUNET(3,16), 100)