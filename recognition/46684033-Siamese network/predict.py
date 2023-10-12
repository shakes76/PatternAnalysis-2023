#predict.py
import torch
import dataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

model = torch.load(r"C:/Users/wongm/Desktop/COMP3710/project/siamese_triplet_epoch_20.pth")
model.eval()



transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),

])
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
paired_testset = dataset.SiameseDatset_test(trainset,testset)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(paired_testset, batch_size=64,shuffle=True)
correct = 0
total = 0

# contrastive loss
# with torch.no_grad():
#     for i, (test_images,pos_image1,pos_image2,pos_image3,pos_image4,pos_image5, neg_image1,neg_image2,neg_image3,neg_image4,neg_image5,test_labels) in enumerate(test_loader):
#         test_images = test_images.to(device)
#         pos_image1 = pos_image1.to(device)
#         neg_image1 = neg_image1.to(device)
#         pos_image2 = pos_image2.to(device)
#         neg_image2 = neg_image2.to(device)
#         pos_image3 = pos_image3.to(device)
#         neg_image3 = neg_image3.to(device)
#         pos_image4 = pos_image4.to(device)
#         neg_image4 = neg_image4.to(device)
#         pos_image5 = pos_image5.to(device)
#         neg_image5 = neg_image5.to(device)
#         test_labels = test_labels.to(device)
#
#         # accuracy test
#         px1, py1,t = model(test_images, pos_image1)
#         px2, py2,t = model(test_images, pos_image2)
#         px3, py3,t = model(test_images, pos_image3)
#         px4, py4 ,t= model(test_images, pos_image4)
#         px5, py5 ,t= model(test_images, pos_image5)
#         pos_distances = []
#         pos_distances.append((py1 - px1).pow(2).sum(1))
#         pos_distances.append((py2 - px2).pow(2).sum(1))
#         pos_distances.append((py3 - px3).pow(2).sum(1))
#         pos_distances.append((py4 - px4).pow(2).sum(1))
#         pos_distances.append((py5 - px5).pow(2).sum(1))
#         pos = torch.cat((pos_distances[0].unsqueeze(1), pos_distances[1].unsqueeze(1), pos_distances[2].unsqueeze(1),
#                    pos_distances[3].unsqueeze(1), pos_distances[4].unsqueeze(1)), dim=1)
#         # Find the indices of the minimum and maximum values in each row
#         min_indices = torch.argmin(pos, dim=1)
#         max_indices = torch.argmax(pos, dim=1)
#
#         # Create a mask to exclude the min and max values
#         mask = torch.ones(pos.size()).to(device)
#         for j in range(64):
#             mask[j, min_indices[i]] = 0
#             mask[j, max_indices[i]] = 0
#         filtered_tensor = pos * mask
#
#         mean_pos_distances = filtered_tensor.sum(dim=1) / 3
#
#         nx1, ny1,t = model(test_images, neg_image1)
#         nx2, ny2,t = model(test_images, neg_image2)
#         nx3, ny3,t = model(test_images, neg_image3)
#         nx4, ny4,t = model(test_images, neg_image4)
#         nx5, ny5,t = model(test_images, neg_image5)
#
#         neg_distances1 = (ny1 - nx1).pow(2).sum(1).unsqueeze(1)
#         neg_distances2 = (ny2 - nx2).pow(2).sum(1).unsqueeze(1)
#         neg_distances3 = (ny3 - nx3).pow(2).sum(1).unsqueeze(1)
#         neg_distances4 = (ny4 - nx4).pow(2).sum(1).unsqueeze(1)
#         neg_distances5 = (ny5 - nx5).pow(2).sum(1).unsqueeze(1)
#         neg = torch.cat((neg_distances1,neg_distances2,neg_distances3,neg_distances4,neg_distances5),dim=1)
#
#         min_indices = torch.argmin(neg, dim=1)
#         max_indices = torch.argmax(neg, dim=1)
#
#         # Create a mask to exclude the min and max values
#         mask = torch.ones(pos.size()).to(device)
#         for j in range(64):
#             mask[j, min_indices[i]] = 0
#             mask[j, max_indices[i]] = 0
#         filtered_tensor = neg * mask
#
#         mean_neg_distances = filtered_tensor.sum(dim=1) / 3
#
#         pred = torch.where(mean_pos_distances < mean_neg_distances, 1, 0)
#
#         correct += (pred == 1).sum().item()
#         total += test_labels.size(0)
#         print(f"progress [{i}/{len(test_loader)}]")
#         print(f"test accuracy {100*correct/total}%")

# for triplet loss
with torch.no_grad():
    for i, (test_images,pos_image1,pos_image2,pos_image3,pos_image4,pos_image5, neg_image1,neg_image2,neg_image3,neg_image4,neg_image5,test_labels) in enumerate(test_loader):
        test_images = test_images.to(device)
        pos_image1 = pos_image1.to(device)
        neg_image1 = neg_image1.to(device)
        pos_image2 = pos_image2.to(device)
        neg_image2 = neg_image2.to(device)
        pos_image3 = pos_image3.to(device)
        neg_image3 = neg_image3.to(device)
        pos_image4 = pos_image4.to(device)
        neg_image4 = neg_image4.to(device)
        pos_image5 = pos_image5.to(device)
        neg_image5 = neg_image5.to(device)
        test_labels = test_labels.to(device)

        # accuracy test
        t1, p1,n1 = model(test_images, pos_image1 , neg_image1)
        t2, p2,n2 = model(test_images, pos_image2 , neg_image2)
        t3, p3,n3 = model(test_images, pos_image3 , neg_image3)
        t4, p4 ,n4= model(test_images, pos_image4 , neg_image4)
        t5, p5 ,n5= model(test_images, pos_image5, neg_image5)
        pos_distances = []
        pos_distances.append((p1 - t1).pow(2).sum(1))
        pos_distances.append((p2 - t2).pow(2).sum(1))
        pos_distances.append((p3 - t3).pow(2).sum(1))
        pos_distances.append((p4 - t4).pow(2).sum(1))
        pos_distances.append((p5 - t5).pow(2).sum(1))
        pos = torch.cat((pos_distances[0].unsqueeze(1), pos_distances[1].unsqueeze(1), pos_distances[2].unsqueeze(1),
                   pos_distances[3].unsqueeze(1), pos_distances[4].unsqueeze(1)), dim=1)
        # Find the indices of the minimum and maximum values in each row
        min_indices = torch.argmin(pos, dim=1)
        max_indices = torch.argmax(pos, dim=1)

        # Create a mask to exclude the min and max values
        mask = torch.ones(pos.size()).to(device)
        for j in range(64):
            mask[j, min_indices[i]] = 0
            mask[j, max_indices[i]] = 0
        filtered_tensor = pos * mask

        mean_pos_distances = filtered_tensor.sum(dim=1) / 3

        neg_distances1 = (t1 - n1).pow(2).sum(1).unsqueeze(1)
        neg_distances2 = (t2 - n2).pow(2).sum(1).unsqueeze(1)
        neg_distances3 = (t3 - n3).pow(2).sum(1).unsqueeze(1)
        neg_distances4 = (t4 - n4).pow(2).sum(1).unsqueeze(1)
        neg_distances5 = (t5 - n5).pow(2).sum(1).unsqueeze(1)
        neg = torch.cat((neg_distances1,neg_distances2,neg_distances3,neg_distances4,neg_distances5),dim=1)

        min_indices = torch.argmin(neg, dim=1)
        max_indices = torch.argmax(neg, dim=1)

        # Create a mask to exclude the min and max values
        mask = torch.ones(pos.size()).to(device)
        for j in range(64):
            mask[j, min_indices[i]] = 0
            mask[j, max_indices[i]] = 0
        filtered_tensor = neg * mask

        mean_neg_distances = filtered_tensor.sum(dim=1) / 3

        pred = torch.where(mean_pos_distances < mean_neg_distances, 1, 0)

        correct += (pred == 1).sum().item()
        total += test_labels.size(0)
        print(f"progress [{i}/{len(test_loader)}]")
        print(f"test accuracy {100*correct/total}%")

    #TODO
    # test all model in certain epoch
    # include average result