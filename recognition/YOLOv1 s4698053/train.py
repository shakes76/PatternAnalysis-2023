from modules import *
from dataset import ISICDataloader, collate_split
from CONFIG import *
from torch.utils.data import DataLoader, random_split
from torch import optim
from torchvision import transforms as torchTransforms
import torch
import time



def check_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        exit("Warning CUDA not Found. Using CPU")
    return device


def main():
    model = YoloV1(split_size=1, num_boxes=1, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    loss_fn = YoloLoss()

    """transforms = torchTransforms.Compose([
        torchTransforms.Normalize(mean=[0.7079, 0.5916, 0.5469], std=[0.0925, 0.1103, 0.1247]),
    ])"""


    data = ISICDataloader(classify_file=classify_file, 
                                                 photo_dir=photo_dir, 
                                                 mask_dir=mask_dir,
                                                 mask_empty_dim=image_size)
    
    generator = torch.Generator().manual_seed(torch_seed)
    train_dataset, test_dataset = random_split(data, [train_size, test_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train(train_dataloader, model, optimizer, loss_fn)
    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, SAVE_MODEL_FILE)

def train(train_dataloader: DataLoader, model, optimizer, loss_fn):
    #--------------
    # Train the model
    device = check_cuda()
    model.to(device)
    model.train()
    total_step = len(train_dataloader)
    print("> Training")
    start = time.time() #time generation
    for epoch in range(NUM_EPOCHS):
        for batch_index, (image, label_matrix) in enumerate(train_dataloader): #load a batch
            image, label_matrix = image.to(device), label_matrix.to(device)
            
            # Forward pass
            out = model(image)
            loss = loss_fn(out, label_matrix)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_index+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                        .format(epoch+1, NUM_EPOCHS, batch_index+1, total_step, loss.item()))
        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                        .format(epoch+1, NUM_EPOCHS, batch_index+1, total_step, loss.item()))
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")



if __name__ == "__main__":
    main()