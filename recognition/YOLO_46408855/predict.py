from train import model, device
from modules import *
from dataset import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def plot_boxes(image, vectors):
    """
    Plots detection boxes on image
    """
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image.cpu().permute(1, 2, 0))

    for i in range(vectors.shape(1)):
        vector = vectors[0,i,:]
        print(vector)
        # Create a Rectangle patch
        rect = patches.Rectangle((vector[0] - vector[2]/2, vector[1] - vector[2]/2), 
                                 vector[2], vector[3], linewidth=1, edgecolor='r', facecolor='none')
        #Get label
        if vector[5] > 0.8:
          label = 'melanoma'
        elif vector[6] > 0.8:
          label = 'seborrheic keratosis'
        else: 
          label = ''
        # Add the patch to the Axes
        ax.add_patch(rect)
        #Add labels
        plt.text(vector[0] - vector[2]/2, vector[1] - vector[2]/2, label, 
                 bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

def filter_boxes(pred, label):
  """
  Returns boxes that have a detection in them
  """
  boxes = []
  for i in range(pred.size(1)):
    box = pred[0][i]
    if box[4] >= 0.8:
      boxes.append(box)
  return boxes

def predict(path):
    """
    Displays image with the detection box and label
    """
    model.eval()
    image = cv2.imread(path)
    image = cv2.resize(image, (416, 416))
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image/255
    image = torch.from_numpy(image)
    image = torch.reshape(image, (1,3,416,416))
    image = image.to(device)
    pred = model(image)
    pred = filter_boxes(pred)
    plot_boxes(image, pred)