from modules import *
from dataset import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def plot_boxes(image, boxes):
    """
    Plots detection boxes on image
    """
    image = torch.reshape(image, (3,416,416))
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image.cpu().permute(1, 2, 0))

    if boxes is not None:
        vector = boxes.cpu()
        # Create a Rectangle patch
        rect = patches.Rectangle((vector[0] - vector[2]/2, vector[1] - vector[2]/2), vector[2], vector[3], linewidth=1, edgecolor='r', facecolor='none')
        #Get label
        if vector[5] > vector[6]:
            label = 'melanoma'
        else:
            label = 'seborrheic keratosis'

        # Add the patch to the Axes
        ax.add_patch(rect)
        #Add labels
        plt.text(vector[0] - vector[2]/2, vector[1] - vector[2]/2, label, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

def predict(path, model):
    """
    Displays image with the detection box and label
    """
    image = cv2.imread(path)
    image = cv2.resize(image, (416, 416))
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image/255
    image = torch.from_numpy(image)
    image = torch.reshape(image, (1,3,416,416))
    image = image.to(device)
    pred = model(image)
    box = filter_boxes(pred[0])
    plot_boxes(image, box)

model = YOLO()
checkpoint_path = "/content/drive/MyDrive/Uni/COMP3710/model.pt"
image_path = ""
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
predict(image_path, model)
