import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import tensorflow as tf

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YOLO(nn.Module):

    #REFERENCE: yolov3-tiny.cfg from https://github.com/pjreddie/darknet/blob/master/cfg
    #Used as basis for what layers were needed 
    def __init__(self, num_classes):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        layers = []
        filters = [16,32,64,128,256,512]
        in_channels = 3
        #Convulution layers and maxpooling
        for i in filters:
            layers.append(nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1, bias=False))
            in_channels = i
            layers.append(nn.BatchNorm2d(i))
            layers.append(nn.LeakyReLU(0.1, True)) #might be false
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #Hopefully works
        layers.append(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(512, 255, kernel_size=1, stride=1, padding=1, bias=True))
        self.conv_start = nn.Sequential(*layers)

        #Detection layer - given anchors
        self.anchor1 =  [(81,82), (135,169), (344,319)] #Anchors depends on image?
        
        #Route layer could go here
        self.conv_mid = nn.Sequential(
            nn.Conv2d(255, 128, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode="bilinear"))
        #Another route layer maybe
        self.conv_end = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=1, bias=True))
        
        #Another detection layer
        self.anchor2 = [(10,14), (23,27), (37,58)]

    def forward(self, x):
        out = self.conv_start(x)
        out = out.data
        a = self.predict_transform(out, 416, self.anchor1, self.num_classes)
        out = self.conv_mid(out)
        out = self.conv_end(out)
        out = out.data
        b = self.predict_transform(out, 416, self.anchor2, self.num_classes)
        return torch.cat((a, b), 1)

    def predict_transform(self, prediction, inp_dim, anchors, num_classes):
        """
        This is everything I need to understand but don't
        """
        batch_size = prediction.size(0)
        stride =  inp_dim // prediction.size(2)
        grid_size = inp_dim // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)
        
        prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
         #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
         #Add the center offsets
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        x_offset = x_offset.to(device)
        y_offset = y_offset.to(device)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

        prediction[:,:,:2] += x_y_offset
        #log space transform height and the width
        anchors = torch.FloatTensor(anchors)
        anchors = anchors.to(device)

        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
        prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
        prediction[:,:,:4] *= stride
        return prediction


def calculate_iou(pred, label):
    px, py, pw, ph = pred[0], pred[1], pred[2], pred[3]
    lx, ly, lw, lh = label[0], label[1], label[2], label[3]
    box_a = [px-(pw/2), py-(ph/2), px+(pw/2), py+(ph/2)]
    box_b = [lx-(lw/2), ly-(lh/2), lx+(lw/2), ly+(lh/2)]

    # determine the (x, y) of the corners of intersection area 
    ax = max(box_a[0], box_b[0])
    ay = max(box_a[1], box_b[1])
    bx = min(box_a[2], box_b[2])
    by = min(box_a[3], box_b[3])

    # compute the area of intersection
    intersect = abs(max((bx - ax, 0)) * max((by - ay), 0))
    if intersect == 0:
      return 0

    # compute the area of both the prediction and ground-truth
    area_a = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    # compute the iou
    iou = intersect / float(area_a + area_b - intersect)
    return iou

def compute_loss(pred, label, batch_size):

    pred_xywh = pred[:,:,0:4]

    label_xywh = label[:,0:4]
    label_prob = label[:,5:]
    iou = torch.zeros(batch_size, pred.shape[1])

    #IoU
    for i in range(batch_size):
      for j in range(pred.shape[1]):
        iou[i][j] = calculate_iou(pred_xywh[i][j], label_xywh[i])
    iou, best_boxes = torch.max(iou, dim=1)

    best_box_conf = torch.zeros(batch_size)
    best_box_class1 = torch.zeros(batch_size)
    best_box_class2 = torch.zeros(batch_size)
    best_box_w = torch.zeros(batch_size)
    best_box_h = torch.zeros(batch_size)
    best_box_x = torch.zeros(batch_size)
    best_box_y = torch.zeros(batch_size)


    for i in range(batch_size):
      best_box_conf[i] = pred[i, best_boxes[i], 4]
      best_box_class1[i] = pred[i, best_boxes[i], 5]
      best_box_class2[i] = pred[i, best_boxes[i], 6]
      best_box_w[i] = pred[i, best_boxes[i], 2]
      best_box_h[i] = pred[i, best_boxes[i], 3]
      best_box_x[i] = pred[i, best_boxes[i], 0]
      best_box_y[i] = pred[i, best_boxes[i], 1]

    ones = torch.ones(10)
    conf_loss = torch.zeros(10)

    #confidence loss
    for box in range(len(best_boxes)):
      conf_loss[box] = best_box_conf[box]*iou[box]
    conf_loss = ones - conf_loss

    #classification loss
    step1 = torch.square(label_prob[:,0] - best_box_class1)
    step2 = torch.square(label_prob[:,1] - best_box_class2)
    class_loss = step1 + step2

    #coordinate loss
    step1 = torch.square(label[:,0] - best_box_x) + torch.square(label[:,1] - best_box_y)
    step2 = torch.square(torch.sqrt(label[:,2]) - torch.sqrt(best_box_w)) + torch.square(torch.sqrt(label[:,3]) - torch.sqrt(best_box_h))
    coord_loss = step1 + step2

    total_loss = torch.sum(conf_loss) + torch.sum(class_loss)

    return total_loss

        