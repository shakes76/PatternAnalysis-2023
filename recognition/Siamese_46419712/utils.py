# define data load path
train_path = "/home/groups/comp3710/ADNI/AD_NC/train"
test_path = "/home/groups/comp3710/ADNI/AD_NC/test"

# define result folder
folder = "/home/Student/s4641971/project/result/"

# define model save path
siamese = folder + "siamese.pt"
classifier = folder + "classifier.pt"

# define loss plot and tsne
siamese_loss = folder + "siamese_loss_plot.png"
classifer_loss = folder + "classifier_loss_plot.png"
tsne_train = folder + "tsne_train.png"
tsne_validate = folder + "tsne_validate.png"
tsne_test = folder + "tsne_test.png"

# define accuracy plot and predict image
train_accuracy = folder + "Accuracy_plot.png"
image_plot = folder + "predict_result.png"