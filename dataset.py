import os
import tensorflow as tf

class AlzheimerDataModule(tf.Module):

  def __init__(self, root_dir, preprocess_fn=None, num_ad=0, num_nc=0):
    
    self.root_dir = root_dir
    self.preprocess_fn = preprocess_fn
    
    ad_paths = self.get_image_paths("AD")
    nc_paths = self.get_image_paths("NC")
    
    self.ad_paths = ad_paths[:num_ad]
    self.nc_paths = nc_paths[:num_nc] 
    self.paths = self.ad_paths + self.nc_paths

  def get_image_paths(self, folder):
    path = os.path.join(self.root_dir, folder)
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return files

  def make_dataset(self):
    dataset = tf.data.Dataset.from_tensor_slices(self.paths)
    return dataset.map(self.load_image_and_label)

  def load_image_and_label(self, path):
    
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    
    if self.preprocess_fn:
      image = self.preprocess_fn(image)
      
    label = tf.cast(tf.strings.equal(os.path.basename(os.path.dirname(path)), 'AD'), tf.int32)
    return image, label
