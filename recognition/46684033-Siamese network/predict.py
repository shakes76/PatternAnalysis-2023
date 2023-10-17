#predict.py
import torch
import dataset
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import modules
import dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),


])
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
paired_testset = dataset.SiameseDatset_test(trainset,testset)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64)
correct = 0
total = 0
best_epoch = 0
best_acc = 0
#or epoch in range(1,60):
model = torch.load(f"C:/Users/wongm/Desktop/COMP3710/project/siamese_triplet_epoch_50.pth")
model = model.to(device)
model.eval()

learning_rate = 0.005
classifier = modules.Classifier(model)
classifier = classifier.to(device)
classifier_loss = nn.BCELoss()
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-3)

# from torchview import draw_graph
# import graphviz
# model_graph = draw_graph(model, input_size=(64,1,128,128), device='meta')
# model_graph.visual_graph
correct = 0
total = 0
num_epoch = 60
classifier.train()
print(classifier)
print("classifier training start")
for epoch in range(num_epoch):
    c_correct = 0
    c_total = 0
    t_correct = 0
    t_total = 0
    classifier.train()
    c_loss = 0
    tc_loss = 0
    train_acc = 0
    test_acc = 0
    for i,(image,label) in enumerate(train_loader):
        image=image.to(device)
        label=label.to(device)
        # train classifier
        classifier_optimizer.zero_grad()
        test_label = label.to(device)
        output = classifier(image).squeeze()
        c_loss = classifier_loss(output, label.float())
        c_loss.backward()
        classifier_optimizer.step()

        pred = torch.where(output > 0.5, 1, 0)
        c_correct += (pred == label).sum().item()
        c_total += label.size(0)
        train_acc = 100 * c_correct / c_total
        # if (i + 1) % 100 == 0:
        #         print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: {}% "
        #               .format(epoch + 1, num_epoch, i + 1, len(train_loader), c_loss.item(), acc))

    classifier.eval()
    with torch.no_grad():
        for i, (test_image, test_label) in enumerate(test_loader):
            test_image = test_image.to(device)
            test_label = test_label.to(device)
            output = classifier(test_image).squeeze()
            tc_loss = classifier_loss(output,test_label.float())

            pred = torch.where(output > 0.5, 1, 0)
            t_correct += (pred == test_label).sum().item()
            t_total += test_label.size(0)
            test_acc = 100 * t_correct / t_total
    print("Epoch [{}/{}], Training Loss: {:.5f} Training Accuracy: {}% Testing Loss: {:.5f} Testing accuracy: {}%"
          .format(epoch + 1, num_epoch, c_loss.item(), train_acc, tc_loss.item(),test_acc))





















# contrastive loss
# with torch.no_grad():
#     for i, (test_images, pos_image1,pos_image2,pos_image3,pos_image4,pos_image5,pos_image6,pos_image7,pos_image8,pos_image9,pos_image10, neg_image1,neg_image2,neg_image3,neg_image4, neg_image5,neg_image6,neg_image7,neg_image8,neg_image9,neg_image10,test_labels) in enumerate(test_loader):
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
#         pos_image6 = pos_image6.to(device)
#         neg_image6 = neg_image6.to(device)
#         pos_image7 = pos_image7.to(device)
#         neg_image7 = neg_image7.to(device)
#         pos_image8 = pos_image8.to(device)
#         neg_image8 = neg_image8.to(device)
#         pos_image9 = pos_image9.to(device)
#         neg_image9 = neg_image9.to(device)
#         pos_image10 = pos_image10.to(device)
#         neg_image10 = neg_image10.to(device)
#
#         test_labels = test_labels.to(device)
#
#         # accuracy test
#         # px1, py1 = model(test_images, pos_image1)
#         # px2, py2 = model(test_images, pos_image2)
#         # px3, py3 = model(test_images, pos_image3)
#         # px4, py4 = model(test_images, pos_image4)
#         # px5, py5 = model(test_images, pos_image5)
#
#         # px6, py6 = model(test_images, pos_image6)
#         # px7, py7 = model(test_images, pos_image7)
#         # px8, py8 = model(test_images, pos_image8)
#         # px9, py9 = model(test_images, pos_image9)
#         # px10, py10 = model(test_images, pos_image10)
#         test = model.forward_once(test_images)
#         py1 = model.forward_once(pos_image1)
#         py2 = model.forward_once(pos_image2)
#         py3 = model.forward_once(pos_image3)
#         py4 = model.forward_once(pos_image4)
#         py5 = model.forward_once(pos_image5)
#         pos_distances = []
#         pos_distances.append((py1 - test).pow(2).sum(1))
#         pos_distances.append((py2 - test).pow(2).sum(1))
#         pos_distances.append((py3 - test).pow(2).sum(1))
#         pos_distances.append((py4 - test).pow(2).sum(1))
#         pos_distances.append((py5 - test).pow(2).sum(1))
#         # pos_distances.append((py6 - px6).pow(2).sum(1))
#         # pos_distances.append((py7 - px7).pow(2).sum(1))
#         # pos_distances.append((py8 - px8).pow(2).sum(1))
#         # pos_distances.append((py9 - px9).pow(2).sum(1))
#         # pos_distances.append((py10 - px10).pow(2).sum(1))
#         pos = torch.cat((pos_distances[0].unsqueeze(1), pos_distances[1].unsqueeze(1), pos_distances[2].unsqueeze(1),
#                    pos_distances[3].unsqueeze(1), pos_distances[4].unsqueeze(1)), dim=1)
#         # Find the indices of the minimum and maximum values in each row
#         # min_indices = torch.argmin(pos, dim=1)
#         # max_indices = torch.argmax(pos, dim=1)
#         #
#         # # Create a mask to exclude the min and max values
#         # mask = torch.ones(pos.size()).to(device)
#         # for j in range(len(min_indices)):
#         #     mask[j, min_indices[j]] = 0
#         #     mask[j, max_indices[j]] = 0
#         # filtered_tensor = pos * mask
#
#         # mean_pos_distances = filtered_tensor.sum(dim=1) / 3
#         mean_pos_distances = pos.sum(dim=1)/5
#
#         # nx1, ny1 = model(test_images, neg_image1)
#         # nx2, ny2 = model(test_images, neg_image2)
#         # nx3, ny3 = model(test_images, neg_image3)
#         # nx4, ny4 = model(test_images, neg_image4)
#         # nx5, ny5 = model(test_images, neg_image5)
#
#         ny1 = model.forward_once(neg_image1)
#         ny2 = model.forward_once(neg_image2)
#         ny3 = model.forward_once(neg_image3)
#         ny4 = model.forward_once(neg_image4)
#         ny5 = model.forward_once(neg_image5)
#         # nx6, ny6 = model(test_images, neg_image6)
#         # nx7, ny7 = model(test_images, neg_image7)
#         # nx8, ny8 = model(test_images, neg_image8)
#         # nx9, ny9 = model(test_images, neg_image9)
#         # nx10, ny10 = model(test_images, neg_image10)
#
#         neg_distances1 = (ny1 - test).pow(2).sum(1).unsqueeze(1)
#         neg_distances2 = (ny2 - test).pow(2).sum(1).unsqueeze(1)
#         neg_distances3 = (ny3 - test).pow(2).sum(1).unsqueeze(1)
#         neg_distances4 = (ny4 - test).pow(2).sum(1).unsqueeze(1)
#         neg_distances5 = (ny5 - test).pow(2).sum(1).unsqueeze(1)
#         # neg_distances6 = (ny6 - nx6).pow(2).sum(1).unsqueeze(1)
#         # neg_distances7 = (ny7 - nx7).pow(2).sum(1).unsqueeze(1)
#         # neg_distances8 = (ny8 - nx8).pow(2).sum(1).unsqueeze(1)
#         # neg_distances9 = (ny9 - nx9).pow(2).sum(1).unsqueeze(1)
#         # neg_distances10 = (ny10 - nx10).pow(2).sum(1).unsqueeze(1)
#         neg = torch.cat((neg_distances1,neg_distances2,neg_distances3,neg_distances4,neg_distances5),dim=1)
#
#         # min_indices = torch.argmin(neg, dim=1)
#         # max_indices = torch.argmax(neg, dim=1)
#         #
#         # # Create a mask to exclude the min and max values
#         # mask = torch.ones(pos.size()).to(device)
#         # for j in range(len(max_indices)):
#         #     mask[j, min_indices[j]] = 0
#         #     mask[j, max_indices[j]] = 0
#         # filtered_tensor = neg * mask
#
#         # mean_neg_distances = filtered_tensor.sum(dim=1) / 3
#         mean_neg_distances = neg.sum(dim=1) / 5
#
#         pred = torch.where(mean_pos_distances < mean_neg_distances, 1, 0)
#
#         correct += (pred == 1).sum().item()
#         total += test_labels.size(0)
#         acc= 100*correct/total
#         print(f"progress [{i}/{len(test_loader)}]")
#         print(f"test accuracy {acc}%")
# print(f"progress [{epoch}/60]")
# if acc > best_acc:
#     best_acc = acc
#     best_epoch = epoch
# print(f"best accuracy is {best_acc}% and the epoch is {best_epoch}")
# for triplet loss
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
#         t1, p1,n1 = model(test_images, pos_image1 , neg_image1)
#         t2, p2,n2 = model(test_images, pos_image2 , neg_image2)
#         t3, p3,n3 = model(test_images, pos_image3 , neg_image3)
#         t4, p4 ,n4= model(test_images, pos_image4 , neg_image4)
#         t5, p5 ,n5= model(test_images, pos_image5, neg_image5)
#         pos_distances = []
#         pos_distances.append((p1 - t1).pow(2).sum(1))
#         pos_distances.append((p2 - t2).pow(2).sum(1))
#         pos_distances.append((p3 - t3).pow(2).sum(1))
#         pos_distances.append((p4 - t4).pow(2).sum(1))
#         pos_distances.append((p5 - t5).pow(2).sum(1))
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
#         neg_distances1 = (t1 - n1).pow(2).sum(1).unsqueeze(1)
#         neg_distances2 = (t2 - n2).pow(2).sum(1).unsqueeze(1)
#         neg_distances3 = (t3 - n3).pow(2).sum(1).unsqueeze(1)
#         neg_distances4 = (t4 - n4).pow(2).sum(1).unsqueeze(1)
#         neg_distances5 = (t5 - n5).pow(2).sum(1).unsqueeze(1)
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

    #TODO
    # test all model in certain epoch
    # include average result