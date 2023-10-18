import torch
import torch.nn.parallel
import torch.utils.data
import torchvision

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def check_accuracy(loader, model):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images = batch['image']
            masks = batch['mask']
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            images, masks = images.cuda(), masks.cuda() 
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()
            num_correct += (preds == masks).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * masks).sum()) / (
                (preds + masks).sum() + 1e-8
            )
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])



def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    for idx, batch in enumerate(loader):
        images = batch['image']
        masks = batch['mask']
        #print("1",masks.max())        
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        images, masks = images.cuda(), masks.cuda() 
        with torch.no_grad():
            preds = model(images)
            print("1",preds.max())
            print("1",masks.max())
            preds = torch.round(preds)
            masks = torch.round(masks)
            print("2",preds.max())
            print("2",masks.max())
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(masks, f"{folder}{idx}.png")

    model.train()
    