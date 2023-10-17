import os
import gdown
import zipfile

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