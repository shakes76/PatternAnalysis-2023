import torch
import time
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import utils
import random
from modules import RawSiameseModel, ContrastiveLossFunction, BinaryModelClassifier
from dataset import LoadData

from predict import random_test_model, save_plot_image

SIAMESE_LOSS_SAVE_PATH = utils.siamese_loss
SIAMESE_MODEL_SAVE_PATH = utils.siamese

CLASSIFIER_LOSS_SAVE_PATH = utils.classifer_loss
CLASSIFIER_MODEL_SAVE_PATH = utils.classifier

VALIDATE_ACCURACY_SAVE_PATH = utils.train_accuracy

def train_siamese(model, train_loader, criterion, optimizer, loss_list):
    model.train()

    for i, val in enumerate(train_loader):
        img0, img1 , labels = val
        img0 = img0.to(device)
        img1 = img1.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output1 = model(img0)
        output2 = model(img1)
        loss = criterion(output1, output2, labels)

        loss.backward()

        optimizer.step()

        # save loss for graph
        loss_list.append(loss.item())

        if (i + 1) % 40 == 0:
            print(">>>>> Step [{}/{}] Loss: {:.5f}"
                    .format(i+1, len(train_loader), loss.item()))
    
    return model

def validate_siamese(model, val_loader, criterion, val_loss_list):
    model.eval()

    with torch.no_grad():
        for i, val in enumerate(val_loader):
            img0, img1, label = val
            img0 = img0.to(device)
            img1 = img1.to(device)
            label = label.to(device)

            output1 = model(img0)
            output2 = model(img1)
            loss = criterion(output1, output2, label)
            val_loss_list.append(loss.item())

            if (i+1) % 10 == 0:
                print (">>>>> Step [{}/{}] Validate Loss: {:.5f}"
                        .format(i+1, len(val_loader), loss.item()))

def plot_tsne(fv, label, name, file):
    fv = fv.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    val_tsne = tsne.fit_transform(fv)
    plt.scatter(val_tsne[:, 0], val_tsne[:, 1], c=label)
    plt.title(name)
    plt.savefig(file)
    plt.close()

def train_classifier(sModel, cModel, train_loader, criterion, optimizer, loss_list, lastEpoch=False):
    sModel.eval() # siamese
    cModel.train() # classifier

    correct_predict = 0
    total_test = 0
    for i, val in enumerate(train_loader):
        img, label = val
        img = img.to(device)
        label = label.to(device).float()
        optimizer.zero_grad()

        fv1 = sModel(img)
        if lastEpoch and i + 2 == len(train_loader):
            plot_tsne(fv1, label, "t-SNE train", utils.tsne_train)
        output = cModel(fv1)
        loss = criterion(output.view(-1), label)

        # train test
        predicted = (output.view(-1) > 0.5).float()
        total_test += label.size(0)
        correct_predict += (predicted == label).sum().item()

        loss.backward()
        optimizer.step()

        # save loss for graph
        loss_list.append(loss.item())
        if (i+1) % 40 == 0:
            print (">>>>> Step [{}/{}] Loss: {:.5f}"
                    .format(i+1, len(train_loader), loss.item()))
    
    accuracy = 100 * correct_predict / total_test
    return cModel, accuracy

def validate_classifier(sModel, cModel, val_loader, criterion, val_loss_list, lastEpoch=False):
    sModel.eval()
    cModel.eval()

    with torch.no_grad():
        correct_predict = 0
        total_test = 0
        for i, (img, label) in enumerate(val_loader):
            img = img.to(device)
            label = label.to(device).float()

            fv = sModel(img) 
            output = cModel(fv)
            if lastEpoch and i + 2 == len(val_loader):
                plot_tsne(output, label, "t-SNE validate", utils.tsne_validate)
            output = output.view(-1)
            loss = criterion(output, label)
            val_loss_list.append(loss.item())

            predicted = (output > 0.5).float()
            total_test += label.size(0)
            correct_predict += (predicted == label).sum().item()

            if (i+1) % 10 == 0:
                print (">>>>> Step [{}/{}] Classifier Validate Loss: {:.5f}"
                        .format(i+1, len(val_loader), loss.item()))

        accuracy = 100 * correct_predict / total_test
        print(f"Validate predict >>>> {accuracy}%")
        return accuracy

def test_model(model, cModel, test_loader):
    # evaluate the model
    model.eval() # disable drop out, batch normalization
    cModel.eval()
    with torch.no_grad(): # disable gradient computation
        correct_predict = 0
        total_test = 0
        count = 0
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device).float()

            fv = model(img)
            if count + 2 == len(test_loader):
                plot_tsne(fv, label, "t-SNE test", utils.tsne_test)
            output = cModel(fv)
            output = output.view(-1)

            predicted = (output > 0.5).float()
            total_test += label.size(0)
            correct_predict += (predicted == label).sum().item()
            count += 1
    
    return correct_predict/total_test

def save_loss_plot(loss_list, val_loss_list, epoch=0, siamese=True):
    """
        Save loss for siamese and classifier
    """
    plt.figure(figsize=(12, 8))
    plt.plot(loss_list, label="train loss")
    plt.plot(val_loss_list, label="validate loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if siamese:
        plt.title(f"Siamese losses -> Total Epoch {epoch + 1}")
        plt.savefig(SIAMESE_LOSS_SAVE_PATH)
    else:
        plt.title(f"Classifier losses -> Total epoch {epoch + 1}")
        plt.savefig(CLASSIFIER_LOSS_SAVE_PATH)

    plt.close()

def save_val_accuracy_plot(accuracy_list, train_accuracy_list, epoch=0):
    """
        Save accuracy list during validate for classifier
    """
    plt.figure(figsize=(12, 8))
    plt.plot(accuracy_list, label="Accuracy")
    plt.plot(train_accuracy_list, label="Train Accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.title(f"Classifier validate accuracy -> Total epoch {epoch + 1}")
    plt.savefig(VALIDATE_ACCURACY_SAVE_PATH)

    plt.close()

def save_model(epochs, m_state_dict, criterion, o_state_dict, model_type=0):
    if model_type == 0:
        PATH = SIAMESE_MODEL_SAVE_PATH
    else:
        PATH = CLASSIFIER_MODEL_SAVE_PATH

    torch.save({
        "epochs": epochs,
        "model": m_state_dict,
        "criterion": criterion,
        "optimizer": o_state_dict,
    }, PATH)

    print("Save model")

def load_model(model_type=0):
    if model_type == 0:
        PATH = SIAMESE_MODEL_SAVE_PATH
    else:
        PATH = CLASSIFIER_MODEL_SAVE_PATH

    if os.path.exists(PATH):
        load_model = torch.load(PATH)
        epoch = load_model['epochs']
        model = load_model['model']
        criterion = load_model['criterion']
        optimizer = load_model['optimizer']

        return epoch, model, criterion, optimizer
    else:
        return None

def execute_sTrain(train_loader, val_loader):
    model = RawSiameseModel().to(device)

    # hyper-parameters
    num_epochs = 15
    learning_rate = 0.0001

    criterion = ContrastiveLossFunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999)) # Optimize model parameter

    # training
    loss_list = []
    avg_loss_list = []

    # validating
    val_loss_list = []
    avg_val_loss = []

    # save model based on validate
    best_val_loss = 100 # should not be 100
    best_model = None


    start = time.time() #time generation
    print("Start training")

    for epoch in range(num_epochs):
        print ("Epoch [{}/{}]".format(epoch + 1, num_epochs))
        model = train_siamese(model, train_loader, criterion, optimizer, loss_list)
        validate_siamese(model, val_loader, criterion, val_loss_list)

        current_val_loss = np.mean(val_loss_list)

        avg_loss_list.append(np.mean(loss_list))
        avg_val_loss.append(current_val_loss)

        if best_model is None:
            best_model = model.state_dict()
            best_val_loss = current_val_loss
            print(f"Current best validate loss is {best_val_loss} at epoch {epoch + 1}")
            save_model(epoch, best_model, criterion, optimizer.state_dict())
        elif current_val_loss < best_val_loss:
            best_model = model.state_dict()
            print(f"The previous best validate loss is {best_val_loss}")
            best_val_loss = current_val_loss
            print(f"Current best validate loss is {best_val_loss} at epoch {epoch + 1}")
            save_model(epoch, best_model, criterion, optimizer.state_dict())

        loss_list = []
        val_loss_list = []
        save_loss_plot(avg_loss_list, avg_val_loss, epoch)

    # test save the last model
    # save_model(epoch, best_model, criterion, optimizer.state_dict())

    end = time.time() #time generation

    print("Finish training")
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total \n")
    
    # don't need to return model because will load from pt
    # return model

def execute_cTrain(sModel, train_loader_classifier, val_loader_classifier):
    cModel = BinaryModelClassifier().to(device)

    # hyper parameters for classifier
    num_epochs = 25
    learning_rate = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(cModel.parameters(), lr=learning_rate)

    # training
    classifier_loss_list = []
    avg_classifier_loss_list = []

    # validating
    classifier_val_loss_list = []
    avg_classifier_val_loss = []

    # accuracy
    accuracy_list = []
    train_accuracy_list = []

    # save model based on validate
    # best_val_loss = 100 # should not be 100
    best_model = None

    # save model based on accuracy
    best_acc = 0

    print("Start classifier training")
    
    start = time.time()
    # train classifier
    for epoch in range(num_epochs):
        print ("Epoch [{}/{}]".format(epoch + 1, num_epochs))
        if epoch + 1 == num_epochs:
            cModel, t_accuracy = train_classifier(sModel, cModel, train_loader_classifier, criterion, optimizer, classifier_loss_list, True)
            accuracy = validate_classifier(sModel, cModel, val_loader_classifier, criterion, classifier_val_loss_list, True)
        else:
            cModel, t_accuracy = train_classifier(sModel, cModel, train_loader_classifier, criterion, optimizer, classifier_loss_list)
            accuracy = validate_classifier(sModel, cModel, val_loader_classifier, criterion, classifier_val_loss_list)
        
        accuracy_list.append(accuracy)
        train_accuracy_list.append(t_accuracy)
        
        current_classifier_val_loss = np.mean(classifier_val_loss_list)

        avg_classifier_loss_list.append(np.mean(classifier_loss_list))
        avg_classifier_val_loss.append(current_classifier_val_loss)
        
        if best_model is None:
            best_model = cModel.state_dict()
            best_acc = current_classifier_val_loss
            print(f"Current best classifier validate loss is {current_classifier_val_loss} at epoch {epoch + 1}")
            save_model(epoch, best_model, criterion, optimizer.state_dict(), 1)
        elif current_classifier_val_loss < best_acc:
            best_model = cModel.state_dict()
            print(f"The previous best classifier validate loss is {best_acc}")
            best_acc = current_classifier_val_loss
            print(f"Current best classifier validate loss is {best_acc} at epoch {epoch + 1}")
            save_model(epoch, best_model, criterion, optimizer.state_dict(), 1)
        
        classifier_loss_list = []
        classifier_val_loss_list = []

        save_loss_plot(avg_classifier_loss_list, avg_classifier_val_loss, epoch, siamese=False)
        save_val_accuracy_plot(accuracy_list, train_accuracy_list, epoch)
    
    # test save the last model
    # save_model(epoch, best_model, criterion, optimizer.state_dict(), 1)

    end = time.time() #time generation
    print("Finish classifier training")
    elapsed = end - start
    print("Training classifier took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total \n")

    # don't need to return model because will load from pt
    # return cModel

if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    ######### LOADING DATA ##########
    print("Begin loading data")
    train_loader, val_loader = LoadData(train=True, siamese=True).load_data()
    train_loader_classifier, val_loader_classifier = LoadData(train=True, siamese=False).load_data()
    test_loader = LoadData(train=False).load_data()
    print("Finish loading data \n")

    #########  TRAINING SIAMASE MODEL ##########
    execute_sTrain(train_loader, val_loader)

    siamese_model = RawSiameseModel().to(device)
    load_save_model = load_model()

    if load_save_model is not None:
        siamese_model.load_state_dict(load_save_model[1])
        print("load siamese model from save")
    else:
        print("error, unable to load, cannot find save file")

    #########  TRAINING BINARY CLASSIFIER MODEL ########## 
    execute_cTrain(siamese_model, train_loader_classifier, val_loader_classifier)

    classifier_model = BinaryModelClassifier().to(device)
    load_classifier_save_model = load_model(1)

    if load_classifier_save_model is not None:
        classifier_model.load_state_dict(load_classifier_save_model[1])
        print("Load classifier model from save")
    else:
        print("error, unable to load cannot find save file")

    #########  TESTING MODEL ##########
    print("Start Testing")
    start = time.time()
    accuracy = test_model(siamese_model, classifier_model, test_loader)
    print('Test Accuracy: {} %'.format(100 * accuracy))
    end = time.time() #time generation
    print("Finish Testing")
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total")

    #########  TESTING MODEL ##########
    img, label, predicted = random_test_model(device, siamese_model, classifier_model, test_loader)

    save_plot_image(img, label, predicted)
