import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from dataset import gen_loaders
from modules import TripletSiameseNetwork, TripletLoss, TripletLossWithRegularization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from itertools import cycle

loaders = gen_loaders()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
trip_model = TripletSiameseNetwork()
trip_criterion = TripletLossWithRegularization(margin=1.0)
val_criterion = TripletLoss(margin=1.0)
total_step = len(loaders['train'])
epochs = 20

optimizer = torch.optim.SGD(trip_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=learning_rate, step_size_up=epochs // 2, step_size_down=epochs // 2, mode="triangular", verbose=False)
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0001/learning_rate, end_factor=0.0001/learning_rate, verbose=False)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[epochs])



test_iter = cycle(iter(loaders['test']))
trip_model.to(device)
losses = []
val_losses = []

transform_train = transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))

prev_loss = float('inf')
epochs_without_improvement = 0
max_epochs_without_improvement = 3

for epoch in range(epochs):
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    losses.append([])
    val_losses.append([])
    for i, (img1, img2, img3, _) in enumerate(loaders['train']):
        size = img1.size(0)
        img1, img2, img3 = img1[torch.randperm(size)], img2[torch.randperm(size)], img3[torch.randperm(size)]
        img1_trans = transform_train(img1)
        img2_trans = transform_train(img2)
        img3_trans = transform_train(img3)
        img1_trans, img2_trans, img3_trans = img1_trans.to(device), img2_trans.to(device), img3_trans.to(device)


        out1, out2, out3 = trip_model(img1_trans, img2_trans, img3_trans)

        loss = trip_criterion(out1, out2, out3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        no_loss = val_criterion(out1, out2, out3)
        losses[epoch].append(no_loss.item())
        if (i + 1) % (total_step // 10) == 0:

            with torch.no_grad():
                val_img1, val_img2, val_img3, _ = next(test_iter)
                val_img1, val_img2, val_img3 = val_img1.to(device), val_img2.to(device), val_img3.to(device)
                val_out1, val_out2, val_out3 = trip_model(val_img1, val_img2, val_img3)
                val_loss = val_criterion(val_out1, val_out2, val_out3)
                val_losses[epoch].append(val_loss.item())

                print(f"Epoch [{epoch + 1} / {epochs}], Step [{i + 1} / {total_step}], Loss: {np.mean(losses[epoch])}, Validation Loss: {np.mean(val_losses[epoch])}")

    scheduler.step()
    mean_list1 = np.mean(losses, axis=1)
    mean_list2 = np.mean(val_losses, axis=1)

    # Plot the means
    plt.plot(mean_list1, label='Train Loss', marker='o')
    plt.plot(mean_list2, label='Val Loss', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.show()

    if np.mean(losses[epoch]) < prev_loss:
        epochs_without_improvement = 0
        prev_loss = np.mean(losses[epoch])
    else:
        epochs_without_improvement += 1
        print(f'{epochs_without_improvement} / {max_epochs_without_improvement} before early stop.')

    if epochs_without_improvement >= max_epochs_without_improvement:
        print(f'Early stopping after {epoch + 1} epochs.')
        break
    




output_dict = {}
label_dict = {}
for stage in ['train', 'test']:
    outputs = None
    labels = None
    for i, (img, _, _, label) in enumerate(loaders[stage]):
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            features = trip_model.forward_once(img)

        if outputs is None:
            outputs = features.cpu()
            labels = label.cpu()
        else:
            outputs = torch.cat((outputs, features.cpu()), dim=0)
            labels = torch.cat((labels, label.cpu()), dim=0)
    output_dict[stage] = outputs
    label_dict[stage] = labels


# Example usage:
clf = RandomForestClassifier(n_estimators=400, min_samples_split=300, max_depth=8, critereon='entropy')
clf.fit(output_dict['train'], label_dict['train'])

y_pred = clf.predict(output_dict['test'])
y_pred_train = clf.predict(output_dict['train'])


# Calculate the accuracy of the classifier
print(f"Test accuracy: {accuracy_score(label_dict['test'], y_pred)}, Train accuracy: {accuracy_score(label_dict['train'], y_pred_train)}")


trip_model.save('trip_model.pth')
with open('classifier.pickle', 'wb') as f:
    pickle.dump(clf, f)

print('Saved siamese to trip_model.pth and classifier to classifier.pickle')