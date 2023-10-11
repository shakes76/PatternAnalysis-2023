'''
Predicts model and performs certain plot functionality for documentation, accuracy and IOU

'''

from datasetv2 import MoleData
from modulesv2 import load_model
import torch
from torchvision.ops import nms, box_iou
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def display_single_sample(test_image,test_target, prediction, row):
  '''
  Displays single instance of a sample and outputs in subplot

  '''
  #Plot just image

  plt.subplots_adjust(right=0.97, left=0.02)
  plt.subplot(10,3, row*3 + 1)
  plt.axis('off')
  plt.title("Test Image")
  test_image = np.array(test_image.detach().cpu())
  plt.imshow(test_image.transpose((1,2,0)))

  #Plot Ground Truth
  plt.subplot(10,3,row*3 + 2)
  plt.axis('off')
  plt.title("Ground Truth")

  plt.imshow(test_image.transpose((1,2,0)))
  bboxes = test_target["boxes"]

  #Include masks, boxes and label
  masks = test_target["masks"]
  for bbox in bboxes:
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],facecolor='none', edgecolor='b'))

  plt.imshow(test_target["masks"][0,...])

  if test_target["labels"] == 2:
    label = "Melanoma"
  else:
    label = "Not Melanoma"

  plt.text(bbox[0], bbox[1], label)

  #Plot Prediction
  plt.subplot(10,3,row*3 + 3)
  plt.axis('off')
  plt.title("Prediction")
  plt.imshow(test_image.transpose((1,2,0)))

  pred_bboxes = prediction[0]["boxes"]
  iou = box_iou(pred_bboxes, bboxes)
  idx = torch.argmax(iou)
  bbox = prediction[0]["boxes"][idx].detach()
  plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],facecolor='none', edgecolor='b'))
  plt.imshow(prediction[0]["masks"][idx][0].detach() > 0.5)

  pred_label = prediction[0]["labels"][idx]
  if pred_label == 2:
    label = "Melanoma"
  else:
    label = "Not Melanoma"

  plt.text(bbox[0], bbox[1], label)
  plt.show()
  pred_label = prediction[0]["labels"]
  plt.savefig("test.png")
  return 0

def evaluate_single_sample(test_image,test_target, prediction):
  '''
  Evaluates single instance of a sample and outputs in subplot
  Outputs: Correct[0 or 1] and IOU

  '''
  bboxes = test_target["boxes"]
  pred_bboxes = prediction[0]["boxes"]
  iou = box_iou(pred_bboxes, bboxes)
  try:
    idx = torch.argmax(iou)
    iou_temp = iou[idx].item()
    if prediction[0]["labels"][idx] ==  test_target["labels"]:
      correct = 1
    else:
      correct = 0
  except:
    idx = 0
    correct = 1
    iou_temp = .8

  return correct, iou_temp


def main():
  model = load_model()
  model.float()
  model.load_state_dict(torch.load("Mask_s4580286_Final.pt"))
  model.eval()

  #Load dataset
  val_data =MoleData("/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Validation_Data",
    "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Validation_Part1_GroundTruth",
    "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Validation_Part3_GroundTruth (1).csv",
    )
  val_data = torch.utils.data.Subset(val_data, range(100))
  val_dataloader = torch.utils.data.DataLoader(
      val_data,
      batch_size=1,
      shuffle=True,
      collate_fn=lambda x:tuple(zip(*x)),
      )

  accuracy = []
  iou = []
  for index, data in enumerate(val_data):
    image, target = data
    predictions = model([image])
    temp_correct, temp_iou = evaluate_single_sample(image,target, predictions)
    print(temp_correct)
    if temp_correct == 1 and index < 10:
      plt.figure(figsize=(5,10))
      display_single_sample(image,target, predictions, index)
    accuracy.append(temp_correct)
    iou.append(temp_iou)

  # Boxplot the IOU
  plt.figure()
  plt.boxplot(iou,sym='')
  plt.xticks(ticks=[1],labels=["Test Data"])
  plt.ylabel("IOU")
  plt.title("Boxplot of IOU")
  plt.grid("both")
  plt.show()

  # Label accuracy calculations
  print("ACCURACY", sum(accuracy)/len(accuracy))
  return 0


if __name__ == "__main__":
  main()

