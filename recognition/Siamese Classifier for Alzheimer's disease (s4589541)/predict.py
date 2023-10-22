"""
    predict.py - example usage of trained model
"""
from dataset import *
from modules import *
import torch
import argparse


def predict(classifier: BinaryClassifier, siamese: TripletNetwork, device, 
            test_loader: DataLoader):
    """Predicts class of images in the test set.

    Args:
        classifier (BinaryClassifier): trained classifier
        siamese (TripletNetwork): trained siamese model
        device (_type_): cpu or cuda
        test_loader (DataLoader): data loader for test data

    Returns:
        float: average classification accuracy
    """
    classifier.eval()
    siamese.eval()
    accuracies = []
    print(f"Total batches: {len(test_loader)}")
    try:
        for _, (a, label) in enumerate(test_loader):
            # move the data to the GPU
            a, label = a.to(device), torch.unsqueeze(label.to(device), dim=1).float()
            # input image into siamese model to generate embedding
            a_embed = siamese.single_foward(a)
            # pass into classifier
            a_out = nn.Sigmoid()(classifier(a_embed))
            pred = torch.round(a_out)
            correct = torch.eq(pred, label)
            print(f"Correctly predicted {torch.sum(correct)}/{BATCH_SIZE}")
            accuracies.append(torch.sum(correct).item()/BATCH_SIZE)

    except Exception as e:
        print(e)

    average_acc = sum(accuracies) / len(accuracies)
    print(f"Average accuracy: {average_acc}")

    return average_acc
    


def parse_user_args():
    """Parse user CLI args"""
    parser = argparse.ArgumentParser(description="Training/testing model")

    parser.add_argument(
        "--path-s",
        type=str,
        help="Path to saved siamese model",
        metavar="SIAM_FILE_PATH",
        required=True
    )

    parser.add_argument(
        "--path-c",
        type=str,
        help="Path to saved classifier",
        metavar="CLSF_FILE_PATH",
        required=True
    )

    args = parser.parse_args()
    return args.path_s, args.path_c


def main():
    """Main function"""
    # parse args
    path_s, path_c = parse_user_args()
    
    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese = TripletNetwork().to(device)
    classifier = BinaryClassifier(siamese.embedding_dim).to(device)
    siamese.load_state_dict(torch.load(path_s, map_location=torch.device('cpu')))
    classifier.load_state_dict(torch.load(path_c, map_location=torch.device('cpu')))
    criterion = torch.nn.BCEWithLogitsLoss()

    # setup the transforms for the images
    transform = transforms.Compose([
        transforms.Resize((256, 240)),
        transforms.ToTensor(),
        OneChannel()
    ])
    test_set = TripletDataset(root="data/test", transform=transform, triplet=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    predict(classifier, siamese, device, test_loader)


if __name__ == '__main__':
    main()