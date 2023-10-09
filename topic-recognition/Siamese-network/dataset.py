import tensorflow as tf

def download_data ():

    #download data
    dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,fname='ad-nc' ,untar=True)
    data_dir = pathlib.Path(data_dir)

    # unzip data to current directory
    print (data_dir)
    ! unzip /root/.keras/datasets/ad-nc.tar.gz

download_data()


dataset_dir = '/content/AD_NC/train/AD'
image_list_train_ad = []
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpeg'):
        file_path = os.path.join(dataset_dir, filename)
        image = Image.open(file_path)
        image_list_train_ad.append(image)
print(f"Number of images loaded: {len(image_list_train_ad)}")


dataset_dir = '/content/AD_NC/train/NC'
image_list_train_nc = []
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpeg'):
        file_path = os.path.join(dataset_dir, filename)
        image = Image.open(file_path)
        image_list_train_nc.append(image)
print(f"Number of images loaded: {len(image_list_train_nc)}")


dataset_dir = '/content/AD_NC/test/AD'
image_list_test_ad = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpeg'):
        file_path = os.path.join(dataset_dir, filename)
        image = Image.open(file_path)
        image_list_test_ad.append(image)
print(f"Number of images loaded: {len(image_list_test_ad)}")


dataset_dir = '/content/AD_NC/test/NC'
image_list_test_nc = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpeg'):
        file_path = os.path.join(dataset_dir, filename)
        image = Image.open(file_path)
        image_list_test_nc.append(image)
print(f"Number of images loaded: {len(image_list_test_nc)}")


