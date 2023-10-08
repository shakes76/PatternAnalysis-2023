
import os
from PIL import Image
from torch.utils.data import Dataset

class ADNIDataset(Dataset):
  """ADNI dataset"""

  def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)
        
  def __len__(self):
      return len(self.image_files)
      
  def __getitem__(self, idx):
      img_path = os.path.join(self.data_dir, self.image_files[idx])
      img = Image.open(img_path)
      label = 1 if 'AD' in self.image_files[idx] else 0  # Assuming filename contains class info
      
      if self.transform:
          img = self.transform(img)
            
      return img, label