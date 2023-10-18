import os
import gdown
import zipfile
import torch

def get_data_from_url(destination_dir, google_drive_id):

  if not os.path.exists(destination_dir):
    compressed_data = 'ISIC_data.zip'
    url = f'https://drive.google.com/uc?id={google_drive_id}'
    gdown.download(url, compressed_data, quiet=False)

    with zipfile.ZipFile(compressed_data, 'r') as zip_ref:
      zip_ref.extractall()
    os.remove(compressed_data)

  else:
    print('Data already loaded')

def train(model, loader, optimizer, criterion, DEVICE):
  model.train()
  for batch_idx, (images, masks) in enumerate(loader):
    optimizer.zero_grad()
    masks = masks.unsqueeze(0)
    images = images.to(device=DEVICE)
    masks = masks.to(device=DEVICE)

    preds = model(images)
    loss = criterion(preds, masks.float())
    print(loss.item())
    loss.backward()

    optimizer.step()
  
  return preds[0], masks[0]

def DSC(predictions, masks):
  return (2 * predictions * masks).sum() / (predictions + masks).sum() + 1e-8

def accuracy(model, loader, device):
  model.eval()
  correct = 0
  pixels = 0
  dice_score = 0
  with torch.no_grad(): 
    for _,(preds, masks) in enumerate(loader):
      preds = preds.to(device)
      masks = masks.to(device).unsqueeze(1)
      preds = torch.sigmoid(preds)
      preds = (preds > 0.5).float()
      correct += (preds==masks).sum()
      pixels += torch.numel(preds)
      dice_score += DSC(preds, masks)
      break
  
  dice_score = dice_score/len(loader)
  accuracy = correct / pixels*100
  print(
    f'Accuracy: {accuracy:.2f}, Dice Similarity Coefficient: {dice_score:.2f}, Loss: {1-dice_score:.2f}'
  )

