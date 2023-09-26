from utils import *
from dataset import *
from modules import *

def inference(model, dataloader, label:int, num_examples=1):
    print("Generating...")
    for i, data in enumerate(dataloader):
        input, label = data[0][0].to(device), data[1][0].to(device) # move tensor to gpu if available
        if label == label:  # got the label
            image = input   # got the img with the right label

    with torch.no_grad():
        mu, sigma = model.encode(image.view(1, 256*256)) # H:256 W:256

    for example in range(num_examples):
        z = mu + sigma * torch.randn_like(sigma) # mu + sigma * epsilon
        out = model.decode(z).view(-1, 1, 256, 256)
        save_image(out, f"{GENERATED_IMG_PATH}generated_{label}_ex{example}.png")

def main():
    start_time = time.time()
    print("Program Starts")
    print("Device:", device)

    # Data
    print("Loading Data...")
    testloader = load_data(batch_size=1,test=True)

    # Model
    model = VAE(INPUT_DIM, Z_DIM, H_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))   # load the model
    model.eval()
    
    # Test & visualize
    inference(model, testloader, 1, num_examples=10)
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))

if __name__ == "__main__":
    main()

